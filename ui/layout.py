"""Main-area layout for the dashboard.

The page is three stacked sections, read top to bottom as the argument for a
signal: the verdict, then the price action behind it, then the two scores that
produced it. This module only builds the shells and hands back the containers;
what goes inside them is the next phase's job. Because Streamlit containers
hold their position in the page regardless of when they are written to, later
code can fill the bottom section before the top one without reordering the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:  # Import cycle otherwise: sidebar has no need for layout.
    from ui.sidebar import Selection


@dataclass(frozen=True)
class DashboardSlots:
    """The empty containers making up the main page area.

    Attributes:
    - header (DeltaGenerator): Page title and the current selection line.
    - signal_summary (DeltaGenerator): Top section -- the signal metric card.
    - price_chart (DeltaGenerator): Middle section -- the interactive chart.
    - breakdown (DeltaGenerator): Bottom section, wrapping the two columns below.
    - breakdown_technical (DeltaGenerator): Left column -- indicators and the
      ML score.
    - breakdown_sentiment (DeltaGenerator): Right column -- headlines and the
      sentiment rationale.
    """

    header: "st.delta_generator.DeltaGenerator"
    signal_summary: "st.delta_generator.DeltaGenerator"
    price_chart: "st.delta_generator.DeltaGenerator"
    breakdown: "st.delta_generator.DeltaGenerator"
    breakdown_technical: "st.delta_generator.DeltaGenerator"
    breakdown_sentiment: "st.delta_generator.DeltaGenerator"


def render_layout(selection: "Selection") -> DashboardSlots:
    """Builds the page skeleton and returns its containers.

    Parameters:
    - selection (Selection): The resolved sidebar selection, used for the
      header line describing what is on screen.

    Returns:
    - DashboardSlots: The containers each section's content is written into.
    """
    header = st.container()
    with header:
        st.title(f"{selection.ticker} - AI Stock Analysis")
        st.caption(
            f"{selection.range_label} of daily bars - "
            f"{'custom' if selection.is_custom else 'preset'} ticker"
        )

    # Top: the one number a user came for.
    st.subheader("Signal Summary")
    signal_summary = st.container(border=True)

    # Middle: the price action the signal is a claim about.
    st.subheader("Price Chart")
    price_chart = st.container(border=True)

    # Bottom: the two component scores, side by side so neither reads as the
    # headline.
    st.subheader("Analysis Breakdown")
    breakdown = st.container(border=True)
    with breakdown:
        technical_col, sentiment_col = st.columns(2, gap="large")

    return DashboardSlots(
        header=header,
        signal_summary=signal_summary,
        price_chart=price_chart,
        breakdown=breakdown,
        breakdown_technical=technical_col,
        breakdown_sentiment=sentiment_col,
    )


def render_placeholders(slots: DashboardSlots, selection: "Selection") -> None:
    """Fills the empty sections with a note on what will land there.

    A scaffolding aid only: each call is replaced by the real renderer as the
    corresponding phase is built.

    Parameters:
    - slots (DashboardSlots): The containers from :func:`render_layout`.
    - selection (Selection): The selection the eventual content will describe.
    """
    with slots.signal_summary:
        st.info(
            f"The blended buy/sell signal for **{selection.ticker}**, its score "
            "and the latest close will render here."
        )

    with slots.price_chart:
        st.info(
            f"An interactive candlestick chart of the {selection.range_label.lower()} "
            "of bars, with moving averages overlaid, will render here."
        )

    with slots.breakdown_technical:
        st.markdown("**Technical**")
        st.caption("RSI, MACD, moving averages and the ML score behind them.")

    with slots.breakdown_sentiment:
        st.markdown("**Sentiment**")
        st.caption("Recent headlines and the LLM's rationale for its score.")
