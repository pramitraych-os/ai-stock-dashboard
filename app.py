"""Streamlit entry point for the AI Stock Dashboard.

Run with::

    streamlit run app.py

The file stays deliberately thin: it configures the page, renders the sidebar,
lays out the main area, and hands the resulting containers to whatever draws
the content. The pipeline behind those containers lives in ``run_analysis`` and
the ``src`` modules; the widgets live in the ``ui`` package.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Running via ``streamlit run app.py`` does not put the project root on the
# path the way ``python app.py`` would, so ``import ui`` needs the same fix-up
# ``run_analysis`` applies for ``src``.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.layout import render_layout, render_placeholders  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402


def configure_page() -> None:
    """Sets the page-level Streamlit options.

    Must run before any other Streamlit call, so this is the first thing
    :func:`main` does. Wide layout gives the price chart the horizontal room a
    multi-month candlestick series needs.
    """
    st.set_page_config(
        page_title="AI Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def main() -> None:
    """Renders one pass of the dashboard.

    Streamlit re-runs this top to bottom on every widget interaction, so the
    body has to stay cheap: it reads the sidebar selection and builds the empty
    section containers, nothing that touches the network.
    """
    configure_page()

    selection = render_sidebar()
    slots = render_layout(selection)

    # Placeholder copy until the signal card, chart and breakdown renderers
    # land; drop these calls one at a time as each section is implemented.
    render_placeholders(slots, selection)


if __name__ == "__main__":
    main()
