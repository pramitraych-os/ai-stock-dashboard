"""Main-area layout for the dashboard.

The page is a ranked table of the whole universe followed by four stacked
sections about one stock in it, read top to bottom as a narrowing: where to
look, then the verdict, then the price action behind it, then the two scores
that produced it. This module only builds the shells and hands back the
containers; what goes inside them belongs to the renderers in ``market_table``,
``signal_card``, ``price_chart`` and ``breakdown``. Because Streamlit containers
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
    - header (DeltaGenerator): Page title and the one-line description.
    - market_table (DeltaGenerator): Top section -- the ranked overview table
      and the control that expands it.
    - signal_summary (DeltaGenerator): The selected stock's signal metric card.
    - price_chart (DeltaGenerator): Its interactive chart.
    - breakdown (DeltaGenerator): The section wrapping the two columns below.
    - breakdown_technical (DeltaGenerator): Left column -- indicators and the
      ML score.
    - breakdown_sentiment (DeltaGenerator): Right column -- headlines and the
      sentiment rationale.
    """

    header: "st.delta_generator.DeltaGenerator"
    market_table: "st.delta_generator.DeltaGenerator"
    signal_summary: "st.delta_generator.DeltaGenerator"
    price_chart: "st.delta_generator.DeltaGenerator"
    breakdown: "st.delta_generator.DeltaGenerator"
    breakdown_technical: "st.delta_generator.DeltaGenerator"
    breakdown_sentiment: "st.delta_generator.DeltaGenerator"


def render_layout(selection: "Selection") -> DashboardSlots:
    """Builds the page skeleton and returns its containers.

    Parameters:
    - selection (Selection): The resolved sidebar selection, used for the
      heading of the single-stock half of the page.

    Returns:
    - DashboardSlots: The containers each section's content is written into.
    """
    header = st.container()
    with header:
        # App-level, not ticker-level: the market table above the detail
        # sections covers the whole universe, so titling the page after one
        # symbol would misdescribe most of what is on it.
        st.title("AI Stock Analysis")
        st.caption(
            "A technical ML score blended with LLM news sentiment, ranked across "
            "the universe and unpacked one stock at a time."
        )

    # Top: where to look. Ranked so the strongest claims are the first thing read.
    st.subheader("Market Overview")
    market_table = st.container(border=True)

    # Everything below is about one stock, so the divider and heading make the
    # change of scope explicit rather than leaving the reader to infer it.
    st.divider()
    st.subheader(f"{selection.ticker} - Detail")
    st.caption(
        f"{selection.range_label} of daily bars - "
        f"{'custom' if selection.is_custom else 'preset'} ticker. "
        "Pick another from the sidebar, or click a row above."
    )

    # The one number a user came for.
    signal_summary = st.container(border=True)

    # The price action the signal is a claim about.
    st.subheader("Price Chart")
    price_chart = st.container(border=True)

    # The two component scores, side by side so neither reads as the headline.
    st.subheader("Analysis Breakdown")
    breakdown = st.container(border=True)
    with breakdown:
        technical_col, sentiment_col = st.columns(2, gap="large")

    return DashboardSlots(
        header=header,
        market_table=market_table,
        signal_summary=signal_summary,
        price_chart=price_chart,
        breakdown=breakdown,
        breakdown_technical=technical_col,
        breakdown_sentiment=sentiment_col,
    )
