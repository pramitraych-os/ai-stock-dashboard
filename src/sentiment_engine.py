"""News-sentiment engine for the AI Stock Dashboard.

This module turns recent news coverage of a ticker into a single, normalized
sentiment signal that can be blended with the technical/ML signal produced by
:mod:`ml_engine`:

1. Recent headlines and summaries are fetched with :func:`fetch_stock_news`,
   which tries :mod:`yfinance` first and falls back to the Yahoo Finance RSS
   feed (neither needs an API key).
2. Those articles are rendered into a structured prompt that asks an LLM to
   return strict JSON.
3. The LLM's reply is parsed defensively by :func:`get_llm_sentiment_score`
   into a ``sentiment_score`` in ``[-1.0, +1.0]`` -- the same scale as
   ``ml_engine.get_ml_score`` -- plus a short bullet-point rationale.

Two providers are supported over plain HTTPS (via :mod:`requests`, so no extra
vendor SDK is required):

* **Google Gemini** -- set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
* **OpenAI** -- set ``OPENAI_API_KEY``.

Whichever key is present is used; ``SENTIMENT_LLM_PROVIDER`` (``"gemini"`` or
``"openai"``) forces a choice when both are set. Keys are read from the
environment, with a ``.env`` file at the project root loaded automatically if
:mod:`dotenv` is installed.

Every failure mode is non-fatal by design: a missing key, no available news, a
rate limit, or an unparseable reply all return a neutral ``0.0`` score with a
``status`` field explaining why, so the dashboard never crashes on a bad news
day.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

try:  # Optional: load ``.env`` so API keys don't have to be exported manually.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is declared in requirements.txt
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Yahoo Finance RSS feed used when yfinance returns nothing. Keyed by symbol.
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# Yahoo rejects requests without a browser-like User-Agent.
_RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-stock-dashboard/1.0)"}

# Default models, overridable via ``GEMINI_MODEL`` / ``OPENAI_MODEL``.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Network + retry budget. Retries cover HTTP 429 and 5xx as well as transport
# errors, with exponential backoff (2s, 4s, 8s) unless the server sends a
# ``Retry-After`` header we should honour instead.
REQUEST_TIMEOUT_SECONDS = 45
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0

# HTTP statuses worth retrying: rate limits plus transient server errors.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Per-article summary budget, in characters. Keeps the prompt small (and cheap)
# without losing the gist of a story.
MAX_SUMMARY_CHARS = 400

# Hard ceiling on articles sent to the LLM, regardless of the requested limit.
MAX_ARTICLES = 25

# Returned when there is nothing to score or the call fails.
NEUTRAL_SCORE = 0.0


# ---------------------------------------------------------------------------
# News retrieval
# ---------------------------------------------------------------------------


def _clean_text(value: object, max_chars: int | None = None) -> str:
    """Normalizes a raw news field into clean, single-line plain text.

    Strips HTML tags and entities (RSS descriptions frequently contain both)
    and collapses runs of whitespace.

    Parameters:
    - value (object): Raw field from a news payload; non-strings become ''.
    - max_chars (int | None): Optional truncation length, with an ellipsis
      appended when the text is cut.

    Returns:
    - str: Cleaned text, possibly empty.
    """
    if not isinstance(value, str):
        return ""

    # Unescape entities first so tag stripping also catches encoded markup.
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text


def _normalize_timestamp(value: object) -> str:
    """Converts the assorted timestamp formats Yahoo returns into ISO-8601.

    Handles ISO strings (current yfinance), Unix epoch seconds (legacy
    yfinance ``providerPublishTime``), and RFC 822 dates (RSS ``pubDate``).

    Parameters:
    - value (object): The raw timestamp in any of the above forms.

    Returns:
    - str: An ISO-8601 timestamp, or '' if the value could not be parsed.
    """
    if value in (None, ""):
        return ""

    # Legacy yfinance exposed epoch seconds.
    if isinstance(value, (int, float)):
        try:
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
        except (OSError, OverflowError, ValueError):
            return ""

    if not isinstance(value, str):
        return ""

    text = value.strip()

    # Current yfinance already returns ISO-8601 (e.g. '2026-07-30T15:06:31Z').
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text

    # RSS feeds use RFC 822 (e.g. 'Wed, 30 Jul 2026 15:06:31 +0000').
    try:
        return parsedate_to_datetime(text).isoformat()
    except (TypeError, ValueError):
        return ""


def _normalize_article(raw: dict) -> dict | None:
    """Flattens one yfinance news entry into the module's article shape.

    yfinance changed its payload: current versions nest the story under a
    ``content`` key, older ones were flat with ``publisher`` /
    ``providerPublishTime``. Both layouts are accepted here.

    Parameters:
    - raw (dict): A single element of ``yfinance.Ticker.news``.

    Returns:
    - dict | None: An article with 'title', 'summary', 'publisher',
      'published' and 'link' keys, or None if it carries no usable title.
    """
    if not isinstance(raw, dict):
        return None

    # Current yfinance nests the story; fall back to the flat legacy layout.
    content = raw.get("content") if isinstance(raw.get("content"), dict) else raw

    title = _clean_text(content.get("title"))
    if not title:
        # A story without a headline contributes nothing to the analysis.
        return None

    # 'summary' is the richer field; 'description' is the legacy/RSS spelling.
    summary = _clean_text(content.get("summary"), MAX_SUMMARY_CHARS) or _clean_text(
        content.get("description"), MAX_SUMMARY_CHARS
    )

    # Publisher lives under provider.displayName now, 'publisher' previously.
    provider = content.get("provider")
    if isinstance(provider, dict):
        publisher = _clean_text(provider.get("displayName"))
    else:
        publisher = _clean_text(content.get("publisher"))

    published = _normalize_timestamp(
        content.get("pubDate")
        or content.get("displayTime")
        or content.get("providerPublishTime")
    )

    link = content.get("canonicalUrl")
    if isinstance(link, dict):
        link = link.get("url")
    link = _clean_text(link or content.get("link"))

    return {
        "title": title,
        "summary": summary,
        "publisher": publisher,
        "published": published,
        "link": link,
    }


def _fetch_news_via_yfinance(ticker: str, limit: int) -> list[dict]:
    """Pulls recent news for a ticker from yfinance.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - limit (int): Maximum number of articles to return.

    Returns:
    - list[dict]: Normalized articles; empty on any failure, so the caller can
      fall through to the RSS source.
    """
    try:
        # Imported lazily so a broken/absent yfinance still leaves RSS working.
        import yfinance as yf

        raw_items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"yfinance news lookup failed for '{ticker}': {e}")
        return []

    articles = []
    for raw in raw_items:
        article = _normalize_article(raw)
        if article is not None:
            articles.append(article)
        if len(articles) >= limit:
            break

    return articles


def _fetch_news_via_rss(ticker: str, limit: int) -> list[dict]:
    """Pulls recent news for a ticker from the Yahoo Finance RSS feed.

    Used as a fallback because it needs no API key and stays available when
    yfinance's news endpoint returns an empty list.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - limit (int): Maximum number of articles to return.

    Returns:
    - list[dict]: Normalized articles; empty on any network or parse failure.
    """
    url = YAHOO_RSS_URL.format(ticker=requests.utils.quote(ticker))

    try:
        response = requests.get(url, headers=_RSS_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as e:
        print(f"RSS news lookup failed for '{ticker}': {e}")
        return []

    articles = []
    for item in root.iter("item"):
        title = _clean_text(item.findtext("title"))
        if not title:
            continue

        articles.append(
            {
                "title": title,
                "summary": _clean_text(item.findtext("description"), MAX_SUMMARY_CHARS),
                # RSS has no publisher element; the feed itself is the source.
                "publisher": "Yahoo Finance",
                "published": _normalize_timestamp(item.findtext("pubDate")),
                "link": _clean_text(item.findtext("link")),
            }
        )

        if len(articles) >= limit:
            break

    return articles


def fetch_stock_news(ticker: str, limit: int = 10) -> list[dict]:
    """Fetches recent news headlines and summaries for a given ticker symbol.

    Tries yfinance first and falls back to the Yahoo Finance RSS feed if it
    yields nothing. Neither source requires an API key.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - limit (int): Maximum number of articles to return. Values above
      ``MAX_ARTICLES`` are capped; values below 1 return an empty list.

    Returns:
    - list[dict]: Articles ordered newest-first as reported by the source,
      each with 'title', 'summary', 'publisher', 'published' and 'link' keys.
      Empty if the ticker is invalid or both sources fail.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        print("Warning: fetch_stock_news requires a non-empty ticker symbol.")
        return []

    ticker = ticker.strip().upper()
    limit = min(int(limit), MAX_ARTICLES)

    if limit < 1:
        return []

    articles = _fetch_news_via_yfinance(ticker, limit)

    if not articles:
        print(f"No yfinance news for '{ticker}'; trying the Yahoo Finance RSS feed...")
        articles = _fetch_news_via_rss(ticker, limit)

    if not articles:
        print(f"Warning: No recent news found for '{ticker}'.")

    return articles


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a disciplined sell-side equity analyst. You judge the likely "
    "short-term (1-2 week) market impact of news on a single stock. You reply "
    "with JSON only -- no prose, no markdown fences."
)

PROMPT_TEMPLATE = """Analyze the market sentiment for {ticker} based on the {count} recent news items below.

{articles}

Scoring guidance:
- Weight material, company-specific news (earnings, guidance, regulation, litigation, M&A, product cycles) far above routine commentary, listicles, or price-recap articles.
- Treat older items as less relevant than newer ones.
- Judge likely short-term price impact, not whether the news is pleasant.
- Use the full range; reserve values beyond +/-0.7 for genuinely decisive news.

Score scale:
 -1.0 extremely negative | -0.5 negative | 0.0 neutral/mixed | +0.5 positive | +1.0 extremely positive

Respond with a single JSON object and nothing else, in exactly this shape:
{{
  "sentiment_score": <float between -1.0 and 1.0>,
  "summary": ["<key market driver>", "<key market driver>", "<key market driver>"]
}}

Rules for "summary": 2 to 4 concise bullet points, each under 25 words, naming the
concrete driver behind the score. Do not restate the score itself."""


def build_sentiment_prompt(ticker: str, articles: list[dict]) -> str:
    """Renders the news list into the structured LLM prompt.

    Parameters:
    - ticker (str): The stock symbol the articles relate to.
    - articles (list[dict]): Articles as returned by :func:`fetch_stock_news`.

    Returns:
    - str: The full prompt, with articles numbered newest-first and annotated
      with publisher and publication date so the model can weight recency.
    """
    lines = []
    for index, article in enumerate(articles, start=1):
        # Attribution and date give the model the context the scoring
        # guidance asks it to weigh (source quality and recency).
        meta = " | ".join(
            part for part in (article.get("publisher"), article.get("published")) if part
        )
        header = f"{index}. {article.get('title', '')}"
        if meta:
            header += f"  [{meta}]"
        lines.append(header)

        summary = article.get("summary")
        if summary:
            lines.append(f"   Summary: {summary}")

    return PROMPT_TEMPLATE.format(
        ticker=ticker, count=len(articles), articles="\n".join(lines)
    )


# ---------------------------------------------------------------------------
# LLM provider plumbing
# ---------------------------------------------------------------------------


class SentimentAPIError(RuntimeError):
    """Raised when an LLM call fails in a way the caller should report.

    Attributes:
    - status (str): Machine-readable reason, surfaced as the result's
      ``status`` field (e.g. 'rate_limited', 'api_error', 'no_api_key').
    """

    def __init__(self, message: str, status: str = "api_error") -> None:
        super().__init__(message)
        self.status = status


def resolve_provider() -> tuple[str, str]:
    """Determines which LLM provider and API key to use from the environment.

    Honours ``SENTIMENT_LLM_PROVIDER`` when set; otherwise prefers Gemini and
    falls back to OpenAI, based on which key is present.

    Returns:
    - tuple[str, str]: The provider name ('gemini' or 'openai') and its key.

    Raises:
    - SentimentAPIError: If the requested provider is unknown, or no usable
      API key is configured (status 'no_api_key').
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    openai_key = os.getenv("OPENAI_API_KEY") or ""

    requested = (os.getenv("SENTIMENT_LLM_PROVIDER") or "").strip().lower()

    if requested:
        if requested not in ("gemini", "openai"):
            raise SentimentAPIError(
                f"Unknown SENTIMENT_LLM_PROVIDER '{requested}'. Use 'gemini' or 'openai'.",
                status="config_error",
            )

        key = gemini_key if requested == "gemini" else openai_key
        if not key:
            env_name = "GEMINI_API_KEY" if requested == "gemini" else "OPENAI_API_KEY"
            raise SentimentAPIError(
                f"SENTIMENT_LLM_PROVIDER is '{requested}' but {env_name} is not set.",
                status="no_api_key",
            )
        return requested, key

    # No explicit preference: use whichever key is available.
    if gemini_key:
        return "gemini", gemini_key
    if openai_key:
        return "openai", openai_key

    raise SentimentAPIError(
        "No LLM API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) or "
        "OPENAI_API_KEY in your environment or in a .env file at the project root.",
        status="no_api_key",
    )


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    """Computes how long to wait before retrying a failed LLM request.

    Prefers the server's ``Retry-After`` hint (common on HTTP 429) and
    otherwise backs off exponentially.

    Parameters:
    - response (requests.Response | None): The failed response, if any.
    - attempt (int): 1-based attempt number that just failed.

    Returns:
    - float: Seconds to sleep, capped at ``MAX_BACKOFF_SECONDS``.
    """
    if response is not None:
        retry_after = response.headers.get("Retry-After", "")
        try:
            # Retry-After may be given in seconds (the common case for 429s).
            return min(float(retry_after), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass

    return min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)


def _post_with_retries(url: str, headers: dict, payload: dict, provider: str) -> dict:
    """POSTs a JSON payload to an LLM endpoint, retrying transient failures.

    Retries on HTTP 429/5xx and on transport errors, sleeping per
    :func:`_retry_delay` between attempts.

    Parameters:
    - url (str): Fully-qualified endpoint URL.
    - headers (dict): Request headers, including authentication.
    - payload (dict): The JSON request body.
    - provider (str): Provider name, used only in messages.

    Returns:
    - dict: The decoded JSON response body.

    Raises:
    - SentimentAPIError: On a non-retryable error, or once retries are
      exhausted. Rate-limit exhaustion carries status 'rate_limited'.
    """
    last_error = "unknown error"
    last_status = "api_error"

    for attempt in range(1, MAX_RETRIES + 1):
        response = None
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )

            if response.status_code == 200:
                return response.json()

            # Truncate provider error bodies; they can be verbose.
            detail = _clean_text(response.text, 300)
            last_error = f"{provider} HTTP {response.status_code}: {detail}"

            if response.status_code in (401, 403):
                raise SentimentAPIError(
                    f"{provider} rejected the API key (HTTP {response.status_code}). "
                    "Check that the key is valid and has access to the model.",
                    status="auth_error",
                )

            if response.status_code not in RETRYABLE_STATUS_CODES:
                raise SentimentAPIError(last_error, status="api_error")

            last_status = "rate_limited" if response.status_code == 429 else "api_error"

        except requests.RequestException as e:
            last_error = f"{provider} request failed: {e}"
            last_status = "network_error"
        except ValueError as e:
            # 200 response whose body wasn't JSON: not worth retrying.
            raise SentimentAPIError(
                f"{provider} returned a non-JSON response: {e}", status="api_error"
            ) from e

        if attempt < MAX_RETRIES:
            delay = _retry_delay(response, attempt)
            print(
                f"{last_error} -- retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(delay)

    raise SentimentAPIError(
        f"{last_error} (gave up after {MAX_RETRIES} attempts)", status=last_status
    )


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Sends the prompt to the Google Gemini REST API.

    Parameters:
    - prompt (str): The rendered sentiment prompt.
    - api_key (str): Gemini API key.
    - model (str): Gemini model id (e.g. 'gemini-2.5-flash').

    Returns:
    - str: The model's raw text reply.

    Raises:
    - SentimentAPIError: On an API failure or an empty/blocked response.
    """
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            # Ask for JSON natively so the reply needs minimal cleanup.
            "responseMimeType": "application/json",
            # Low temperature keeps scores stable across repeated runs.
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    body = _post_with_retries(GEMINI_URL.format(model=model), headers, payload, "Gemini")

    candidates = body.get("candidates") or []
    if not candidates:
        # Safety filters and prompt blocks come back with no candidates.
        reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise SentimentAPIError(f"Gemini returned no usable content ({reason}).")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))

    if not text.strip():
        finish = candidates[0].get("finishReason", "unknown")
        raise SentimentAPIError(f"Gemini returned an empty reply (finishReason={finish}).")

    return text


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    """Sends the prompt to the OpenAI chat completions API.

    Parameters:
    - prompt (str): The rendered sentiment prompt.
    - api_key (str): OpenAI API key.
    - model (str): OpenAI model id (e.g. 'gpt-4o-mini').

    Returns:
    - str: The model's raw text reply.

    Raises:
    - SentimentAPIError: On an API failure or an empty response.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        # JSON mode guarantees syntactically valid JSON in the reply.
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    body = _post_with_retries(OPENAI_URL, headers, payload, "OpenAI")

    choices = body.get("choices") or []
    if not choices:
        raise SentimentAPIError("OpenAI returned no choices.")

    text = (choices[0].get("message") or {}).get("content") or ""

    if not text.strip():
        finish = choices[0].get("finish_reason", "unknown")
        raise SentimentAPIError(f"OpenAI returned an empty reply (finish_reason={finish}).")

    return text


def _resolve_model(provider: str) -> str:
    """Returns the model id to use for a provider, honouring env overrides.

    Parameters:
    - provider (str): Either 'gemini' or 'openai'.

    Returns:
    - str: The configured model id, or the provider's documented default.
    """
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    return os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def _call_llm(prompt: str, provider: str, api_key: str, model: str) -> str:
    """Dispatches the prompt to the selected provider.

    Parameters:
    - prompt (str): The rendered sentiment prompt.
    - provider (str): Either 'gemini' or 'openai'.
    - api_key (str): The provider's API key.
    - model (str): The model id to call.

    Returns:
    - str: The model's raw text reply.

    Raises:
    - SentimentAPIError: On an unsupported provider or an API failure.
    """
    if provider == "gemini":
        return _call_gemini(prompt, api_key, model)
    if provider == "openai":
        return _call_openai(prompt, api_key, model)

    raise SentimentAPIError(f"Unsupported provider '{provider}'.", status="config_error")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict:
    """Pulls the first complete JSON object out of a raw LLM reply.

    Tolerates the two things models do even when told not to: wrapping JSON in
    ```json fences, and adding a sentence of prose around it.

    Parameters:
    - text (str): The model's raw reply.

    Returns:
    - dict: The decoded JSON object.

    Raises:
    - SentimentAPIError: If no JSON object can be decoded (status
      'parse_error').
    """
    cleaned = (text or "").strip()

    # Strip a leading ```json / ``` fence and its closing counterpart.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Fast path: the whole reply is the object we want.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Slow path: scan for a balanced {...} span, ignoring braces inside
    # strings, and try to decode each candidate.
    start = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(cleaned):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        candidate = json.loads(cleaned[start : index + 1])
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(candidate, dict):
                        return candidate
                    start = None

    raise SentimentAPIError(
        f"Could not parse JSON from the model reply: {_clean_text(text, 300)!r}",
        status="parse_error",
    )


def _coerce_score(value: object) -> float:
    """Converts a model-supplied score into a float clipped to [-1.0, +1.0].

    Parameters:
    - value (object): The raw ``sentiment_score`` field. Numeric strings such
      as '+0.4' are accepted, since models sometimes quote the number.

    Returns:
    - float: The clipped score, or ``NEUTRAL_SCORE`` if it is not numeric.
    """
    if isinstance(value, bool):
        # bool is an int subclass; treating True as 1.0 would be nonsense.
        return NEUTRAL_SCORE

    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if not match:
            return NEUTRAL_SCORE
        score = float(match.group())
    else:
        return NEUTRAL_SCORE

    if score != score:  # NaN
        return NEUTRAL_SCORE

    # Clip so downstream blending can rely on the declared bounds.
    return round(max(-1.0, min(1.0, score)), 4)


def _coerce_summary(value: object) -> list[str]:
    """Normalizes the model's ``summary`` field into a list of bullet strings.

    Accepts a list of strings (the requested shape) or a single newline- or
    bullet-delimited string, which models occasionally return instead.

    Parameters:
    - value (object): The raw ``summary`` field.

    Returns:
    - list[str]: Non-empty bullet points with list markers stripped; empty if
      nothing usable was provided.
    """
    if isinstance(value, str):
        # Split a prose blob on newlines or inline bullet markers.
        raw_items = re.split(r"\n+|(?:^|\s)[-*•]\s+", value)
    elif isinstance(value, list):
        raw_items = value
    elif isinstance(value, dict):
        # Some models return {"driver_1": "...", ...}.
        raw_items = list(value.values())
    else:
        return []

    bullets = []
    for item in raw_items:
        if isinstance(item, (int, float)):
            item = str(item)
        if not isinstance(item, str):
            continue

        # Strip leading bullet glyphs and numbering, e.g. '- ', '1. '.
        bullet = _clean_text(re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", item))
        if bullet:
            bullets.append(bullet)

    return bullets


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------


def _build_result(
    ticker: str,
    sentiment_score: float,
    summary: list[str],
    status: str,
    article_count: int = 0,
    provider: str | None = None,
    model: str | None = None,
    error: str | None = None,
    headlines: list[str] | None = None,
) -> dict:
    """Assembles the result dict returned by :func:`get_llm_sentiment_score`.

    Every code path goes through here so callers always see the same keys,
    whether the analysis succeeded or fell back to neutral.

    Parameters:
    - ticker (str): The symbol that was analyzed.
    - sentiment_score (float): Score in [-1.0, +1.0].
    - summary (list[str]): Bullet-point market drivers.
    - status (str): 'ok' on success, otherwise the failure reason.
    - article_count (int): How many articles were sent to the LLM.
    - provider (str | None): Provider used, if one was reached.
    - model (str | None): Model id used, if one was reached.
    - error (str | None): Human-readable failure detail.
    - headlines (list[str] | None): The headlines behind the score.

    Returns:
    - dict: The full result payload.
    """
    return {
        "ticker": ticker,
        "sentiment_score": sentiment_score,
        "summary": summary,
        "status": status,
        "article_count": article_count,
        "provider": provider,
        "model": model,
        "error": error,
        "headlines": headlines or [],
    }


def get_llm_sentiment_score(ticker: str, limit: int = 10) -> dict:
    """Scores recent news sentiment for a ticker using an LLM.

    Fetches news via :func:`fetch_stock_news`, prompts the configured provider
    (Gemini or OpenAI) for strict JSON, and parses the reply defensively. The
    resulting ``sentiment_score`` shares the ``[-1.0, +1.0]`` scale used by
    ``ml_engine.get_ml_score``, so the two signals can be blended directly.

    This function does not raise: a missing API key, absent news, a rate limit,
    or an unparseable reply all yield a neutral ``0.0`` score with a ``status``
    describing what happened. Inspect ``status`` to tell a real neutral reading
    ('ok') from a fallback.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - limit (int): Maximum number of recent articles to analyze.

    Returns:
    - dict: With keys:
      - 'ticker' (str): The normalized symbol.
      - 'sentiment_score' (float): Score in [-1.0, +1.0]; 0.0 on fallback.
      - 'summary' (list[str]): Bullet points on the key market drivers.
      - 'status' (str): 'ok', 'no_news', 'no_api_key', 'auth_error',
        'rate_limited', 'api_error', 'network_error', 'parse_error',
        'config_error' or 'invalid_ticker'.
      - 'article_count' (int): Number of articles analyzed.
      - 'provider' (str | None): 'gemini' or 'openai'.
      - 'model' (str | None): The model id used.
      - 'error' (str | None): Failure detail when status is not 'ok'.
      - 'headlines' (list[str]): The headlines the score is based on.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        return _build_result(
            str(ticker),
            NEUTRAL_SCORE,
            [],
            status="invalid_ticker",
            error="A non-empty ticker symbol is required.",
        )

    ticker = ticker.strip().upper()

    # 1. Gather the news. No articles means there is nothing to score.
    articles = fetch_stock_news(ticker, limit=limit)
    headlines = [article["title"] for article in articles]

    if not articles:
        return _build_result(
            ticker,
            NEUTRAL_SCORE,
            [f"No recent news found for {ticker}; defaulting to a neutral score."],
            status="no_news",
        )

    # 2. Resolve credentials before spending effort on the prompt.
    try:
        provider, api_key = resolve_provider()
    except SentimentAPIError as e:
        print(f"Sentiment analysis unavailable: {e}")
        return _build_result(
            ticker,
            NEUTRAL_SCORE,
            [f"LLM sentiment unavailable ({e.status}); defaulting to a neutral score."],
            status=e.status,
            article_count=len(articles),
            error=str(e),
            headlines=headlines,
        )

    model = _resolve_model(provider)
    prompt = build_sentiment_prompt(ticker, articles)

    # 3. Call the LLM and parse its reply. Any failure degrades to neutral.
    try:
        raw_reply = _call_llm(prompt, provider, api_key, model)
        parsed = _extract_json_object(raw_reply)
    except SentimentAPIError as e:
        print(f"Sentiment analysis failed for '{ticker}': {e}")
        return _build_result(
            ticker,
            NEUTRAL_SCORE,
            [f"Sentiment analysis failed ({e.status}); defaulting to a neutral score."],
            status=e.status,
            article_count=len(articles),
            provider=provider,
            model=model,
            error=str(e),
            headlines=headlines,
        )

    score = _coerce_score(parsed.get("sentiment_score"))
    summary = _coerce_summary(parsed.get("summary"))

    if not summary:
        # The score is still usable even if the rationale came back malformed.
        summary = [f"Model returned no usable summary for {ticker}."]

    return _build_result(
        ticker,
        score,
        summary,
        status="ok",
        article_count=len(articles),
        provider=provider,
        model=model,
        headlines=headlines,
    )


def get_sentiment_score(ticker: str, limit: int = 10) -> float:
    """Returns just the sentiment score for a ticker, for signal blending.

    A thin convenience wrapper for callers that only need the number, mirroring
    the shape of ``ml_engine.get_ml_score``.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - limit (int): Maximum number of recent articles to analyze.

    Returns:
    - float: The ``sentiment_score`` in [-1.0, +1.0]; 0.0 on any fallback.
    """
    return get_llm_sentiment_score(ticker, limit=limit)["sentiment_score"]


if __name__ == "__main__":
    print("--- News Sentiment Engine (fetch -> prompt -> score) ---\n")

    for symbol in ["AAPL", "RELIANCE.NS"]:
        print(f"=== {symbol} ===")

        news = fetch_stock_news(symbol, limit=5)
        print(f"Fetched {len(news)} article(s).")
        for item in news[:3]:
            source = item["publisher"] or "unknown source"
            date = item["published"] or "undated"
            print(f"  - {item['title']}  [{source} | {date}]")

        result = get_llm_sentiment_score(symbol, limit=5)

        print(f"\n  status          = {result['status']}")
        print(f"  provider/model  = {result['provider']}/{result['model']}")
        print(f"  sentiment_score = {result['sentiment_score']:+.4f}")
        print("  summary:")
        for bullet in result["summary"]:
            print(f"    * {bullet}")
        if result["error"]:
            print(f"  error           = {result['error']}")

        # The score must always honour the declared bounds, fallback included.
        assert -1.0 <= result["sentiment_score"] <= 1.0, "score out of bounds"
        print("-" * 60)
