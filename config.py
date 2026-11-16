"""Centralized configuration and secrets loading for the AI Stock Dashboard.

Every credential the app needs -- the Anthropic/OpenAI keys used by
``sentiment_engine`` and the optional yfinance network settings used by
``data_loader`` -- is resolved here, in one place, through :func:`get`.

Two deployment shapes are supported without any call site needing to know
which one it's running under:

1. **Local development** -- values come from real environment variables or a
   ``.env`` file at the project root, loaded once via :mod:`python-dotenv`.
2. **Streamlit Community Cloud** (or any host where secrets are configured
   through Streamlit) -- ``.env`` won't exist, so :func:`get` falls back to
   ``st.secrets``.

The environment always wins when both are present, since that's what lets a
deployed app be overridden by a platform-level env var without touching
``st.secrets``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Explicit path rather than a bare load_dotenv(): Streamlit can launch the
# app with a different working directory than the project root, and a bare
# call only searches upward from the CWD.
load_dotenv(PROJECT_ROOT / ".env")

try:
    import streamlit as st

    _STREAMLIT_AVAILABLE = True
except ImportError:  # pragma: no cover - streamlit is declared in requirements.txt
    _STREAMLIT_AVAILABLE = False


def _secret(name: str) -> str | None:
    """Reads one key from ``st.secrets``, or None if unavailable.

    ``st.secrets`` raises when no ``secrets.toml`` exists at all (the normal
    case for local dev without Streamlit Cloud), so that has to be caught
    rather than checked for up front.

    Note: Streamlit copies every key in ``secrets.toml`` into
    ``os.environ`` the first time ``st.secrets`` is touched at all (its own
    documented behaviour, not this module's). That's harmless here because
    every value below is read once into a module-level constant at import
    time, before any such copy could overwrite a real environment variable
    -- but it means calling :func:`get` again later isn't guaranteed to see
    the same precedence.
    """
    if not _STREAMLIT_AVAILABLE:
        return None

    try:
        return st.secrets.get(name)  # type: ignore[union-attr]
    except Exception:
        return None


def get(name: str, default: str | None = None) -> str | None:
    """Resolves a config value: real environment/``.env`` first, then ``st.secrets``.

    Parameters:
    - name (str): The variable name, e.g. ``'ANTHROPIC_API_KEY'``.
    - default (str | None): Returned if the value isn't set anywhere.

    Returns:
    - str | None: The resolved value, or ``default``.
    """
    return os.getenv(name) or _secret(name) or default


def require(name: str) -> str:
    """Like :func:`get`, but raises if the value is missing or blank.

    Parameters:
    - name (str): The variable name, e.g. ``'ANTHROPIC_API_KEY'``.

    Returns:
    - str: The resolved value.

    Raises:
    - RuntimeError: If nothing set ``name`` in the environment, ``.env``, or
      ``st.secrets``.
    """
    value = get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to a .env file at the project root "
            "for local development, or to Streamlit's secrets when deployed."
        )
    return value


# ---------------------------------------------------------------------------
# Anthropic / Claude sentiment scoring
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = get("ANTHROPIC_API_KEY") or get("CLAUDE_API_KEY")
CLAUDE_MODEL = get("CLAUDE_MODEL") or get("ANTHROPIC_MODEL")
CLAUDE_EFFORT = get("CLAUDE_EFFORT")

# OpenAI is sentiment_engine's alternate provider; centralized here for the
# same seamless local/cloud lookup even though this task is Claude-focused.
OPENAI_API_KEY = get("OPENAI_API_KEY")
OPENAI_MODEL = get("OPENAI_MODEL")

SENTIMENT_LLM_PROVIDER = get("SENTIMENT_LLM_PROVIDER")

# ---------------------------------------------------------------------------
# yfinance network settings
# ---------------------------------------------------------------------------

# Both optional: unset means "use yfinance's own defaults" (no proxy, its
# built-in retry count). Set YFINANCE_PROXY if requests need to go through a
# corporate/self-hosted proxy; set YFINANCE_RETRIES to tune resilience against
# Yahoo's rate limiting.
YFINANCE_PROXY = get("YFINANCE_PROXY")
_yfinance_retries_raw = get("YFINANCE_RETRIES")
YFINANCE_RETRIES = int(_yfinance_retries_raw) if _yfinance_retries_raw else None
