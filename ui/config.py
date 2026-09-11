"""Selectable options and input rules for the dashboard sidebar.

Everything the user can pick lives here rather than inline in the widget calls,
so the preset list and the lookback windows can grow without touching layout
code. Symbols follow the convention ``data_loader`` already fetches with: bare
tickers for US listings (``AAPL``) and a ``.NS`` suffix for NSE-listed Indian
ones (``RELIANCE.NS``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ticker:
    """One entry in the preset dropdown.

    Attributes:
    - symbol (str): The yfinance symbol, e.g. ``'RELIANCE.NS'``.
    - name (str): Company name shown beside the symbol.
    - market (str): Short market tag, ``'US'`` or ``'India'``.
    """

    symbol: str
    name: str
    market: str

    @property
    def label(self) -> str:
        """The dropdown text for this entry, e.g. ``'AAPL - Apple Inc. (US)'``."""
        return f"{self.symbol} - {self.name} ({self.market})"


# Large, liquid names on both exchanges: enough news flow for the sentiment
# stage and enough history for the model to train on.
#
# Deliberately longer than :data:`MARKET_TABLE_SIZE`. The overview table ranks
# this whole list and shows the strongest slice of it, so the universe has to
# exceed the slice for the ranking -- and the "show all" control under the table
# -- to mean anything.
TICKER_UNIVERSE: tuple[Ticker, ...] = (
    Ticker("AAPL", "Apple Inc.", "US"),
    Ticker("MSFT", "Microsoft Corp.", "US"),
    Ticker("GOOGL", "Alphabet Inc.", "US"),
    Ticker("AMZN", "Amazon.com Inc.", "US"),
    Ticker("NVDA", "NVIDIA Corp.", "US"),
    Ticker("META", "Meta Platforms Inc.", "US"),
    Ticker("TSLA", "Tesla Inc.", "US"),
    Ticker("JPM", "JPMorgan Chase & Co.", "US"),
    Ticker("AVGO", "Broadcom Inc.", "US"),
    Ticker("AMD", "Advanced Micro Devices", "US"),
    Ticker("NFLX", "Netflix Inc.", "US"),
    Ticker("WMT", "Walmart Inc.", "US"),
    Ticker("RELIANCE.NS", "Reliance Industries", "India"),
    Ticker("TCS.NS", "Tata Consultancy Services", "India"),
    Ticker("INFY.NS", "Infosys Ltd.", "India"),
    Ticker("HDFCBANK.NS", "HDFC Bank Ltd.", "India"),
    Ticker("ICICIBANK.NS", "ICICI Bank Ltd.", "India"),
    Ticker("ITC.NS", "ITC Ltd.", "India"),
    Ticker("SBIN.NS", "State Bank of India", "India"),
    Ticker("BHARTIARTL.NS", "Bharti Airtel Ltd.", "India"),
    Ticker("LT.NS", "Larsen & Toubro Ltd.", "India"),
    Ticker("AXISBANK.NS", "Axis Bank Ltd.", "India"),
    Ticker("KOTAKBANK.NS", "Kotak Mahindra Bank", "India"),
    Ticker("MARUTI.NS", "Maruti Suzuki India", "India"),
)

# Symbol lookup for the overview table, which starts from a scan row -- a
# pipeline payload that knows the symbol and nothing about company names.
TICKER_BY_SYMBOL: dict[str, Ticker] = {entry.symbol: entry for entry in TICKER_UNIVERSE}

# Both tickers in ``data/`` are already cached, so the default selection loads
# without a cold fetch.
DEFAULT_TICKER = "AAPL"

# Rows the overview table shows before the "show all" control is used. Twenty is
# a screenful on a laptop: long enough that both ends of the signal ladder are
# represented, short enough to read without scrolling past the detail sections.
MARKET_TABLE_SIZE = 20


def ticker_for(symbol: str) -> Ticker | None:
    """Looks up a preset entry by symbol.

    Parameters:
    - symbol (str): A yfinance symbol, e.g. ``'RELIANCE.NS'``.

    Returns:
    - Ticker | None: The universe entry, or None for a symbol that is not a
      preset -- which is the normal case for a hand-typed ticker.
    """
    return TICKER_BY_SYMBOL.get(str(symbol).strip().upper())


# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

# Calendar days, not trading days -- ``fetch_price_history`` takes a calendar
# window and lands near 70% of it in actual bars.
DATE_RANGE_OPTIONS: tuple[tuple[str, int], ...] = (
    ("Last 1 Month", 30),
    ("Last 3 Months", 91),
    ("Last 6 Months", 182),
    ("Last 1 Year", 365),
    ("Last 2 Years", 730),
)

_DATE_RANGE_DAYS = dict(DATE_RANGE_OPTIONS)

# Two years matches ``run_analysis.DEFAULT_LOOKBACK_DAYS``: the shortest window
# that reliably clears the indicator warm-up plus the 50-row training minimum.
DEFAULT_DATE_RANGE = "Last 2 Years"

# Below this the warm-up window leaves too few labelled rows to train a model
# from scratch, so a first-time ticker degrades to a neutral ML score. Shorter
# windows still work for a ticker whose model is already saved in ``models/``.
MIN_LOOKBACK_FOR_TRAINING = 365


def lookback_days_for(range_label: str) -> int:
    """Translates a date-range label into calendar days of history.

    Parameters:
    - range_label (str): One of the labels in :data:`DATE_RANGE_OPTIONS`.

    Returns:
    - int: Calendar days to request, falling back to the default window when
      the label is unrecognised.
    """
    return _DATE_RANGE_DAYS.get(range_label, _DATE_RANGE_DAYS[DEFAULT_DATE_RANGE])


# ---------------------------------------------------------------------------
# Ticker validation
# ---------------------------------------------------------------------------

# Accepts what yfinance actually resolves: an optional '^' index prefix, an
# alphanumeric root, then dot- or dash-separated suffixes for exchange and
# share-class codes (RELIANCE.NS, BRK-B, ^GSPC). Anything else -- spaces,
# punctuation, a pasted company name -- is rejected before a network call.
TICKER_PATTERN = re.compile(r"^\^?[A-Z0-9]{1,12}(?:[.\-][A-Z0-9]{1,6})*$")

MAX_TICKER_LENGTH = 20


def validate_ticker(raw: str | None) -> tuple[str | None, str | None]:
    """Normalizes and validates a hand-typed ticker symbol.

    Whitespace is trimmed and the symbol upper-cased, matching the
    normalization ``analyze_ticker`` applies, so the value handed downstream is
    the same one the cache and model filenames are keyed on.

    Parameters:
    - raw (str | None): The raw text-input contents; may be blank.

    Returns:
    - tuple[str | None, str | None]: ``(symbol, error)``. Exactly one is
      populated for non-blank input; blank input yields ``(None, None)`` since
      an empty box means "use the dropdown", not a mistake.
    """
    if not raw or not raw.strip():
        return None, None

    candidate = raw.strip().upper()

    if len(candidate) > MAX_TICKER_LENGTH:
        return None, f"Ticker is too long (max {MAX_TICKER_LENGTH} characters)."

    if not TICKER_PATTERN.match(candidate):
        return None, (
            "Use letters, digits, dots or dashes only - e.g. AAPL, RELIANCE.NS, BRK-B."
        )

    return candidate, None
