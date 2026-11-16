"""The interactive candlestick chart that occupies the middle of the dashboard.

The signal card above states a verdict; this chart is the evidence for it. It
stacks two panels on a shared date axis -- daily OHLC candles with the moving
averages overlaid (70% of the height), and daily traded volume below (30%) --
so a reader can check the claim "momentum is turning" against the bars that
produced it without re-reading the axis.

Three decisions here are worth knowing about before editing.

**Up and down days are encoded twice, on purpose.** Green-vs-red is the
entrenched convention for candles and matches the tints ``signal_blender``
already uses, but it is also the textbook colour-vision failure: measured
against the dark surface this module renders on, ``#0ca30c`` and ``#d03b3b``
separate by only ΔE 4.1 under deuteranopia (OKLab x100; 8 is the target). No
choice of green and red fixes that. So direction does not ride on hue alone --
it also rides on body fill, the traditional Japanese candlestick channel: an up
day is **hollow** (outline only), a down day is **solid**. That reads with no
colour vision at all, and in greyscale print.

**The two moving averages are an ordinal ramp, not two categorical hues.**
SMA_20 and SMA_50 are the same measurement at two window lengths, so they are
drawn as two steps of one blue -- lighter for the shorter, faster window,
darker for the longer, structural one -- rather than as two unrelated colours
competing for attention over the candles. Blue also keeps the overlays clear of
the green/red the candles have already claimed.

**Only SMA_20 and SMA_50 are overlaid**, though ``feature_engineering`` also
supplies EMA_12/26 and Bollinger Bands. Six lines over a candle series is
unreadable, and each extra overlay needs its own validated colour. Adding one
is a single entry in :data:`OVERLAYS`; the cap is a judgement about legibility,
not a limitation of the code.

The date axis skips weekends and market holidays. Daily bars are naturally
gappy, and left alone Plotly renders those gaps as blank stripes every five
candles -- :func:`_non_trading_days` derives the absent sessions from the data
itself and hands them to the axis as range breaks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Columns the chart cannot be drawn without.
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
#
# A deliberately selected dark theme rather than an inverted light one, so
# every colour below was validated against this surface specifically. A light
# mode would need its own steps chosen and re-validated against the light
# surface -- flipping these values is not a substitute.

SURFACE = "#1a1a19"  # Plot surface; the contrast figures below are against it.
PAGE = "#0d0d0d"  # Paper behind the plot, one step further back.

INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"  # Axis ticks and labels.
GRIDLINE = "#2c2c2a"  # Hairline, one shade off the surface.
AXIS_LINE = "#383835"

# Candle direction. Reserved status colours, used here for exactly what they
# mean -- a gain and a loss -- and paired with the hollow/solid body channel
# because this hue pair cannot be made colour-vision safe (see module docstring).
GAIN = "#0ca30c"  # 5.19:1 on SURFACE
LOSS = "#d03b3b"  # 3.62:1 on SURFACE

# Volume is one series measuring activity, not identity, so it takes a neutral
# and stays out of the hue vocabulary the candles and overlays are using.
VOLUME_FILL = "#898781"  # 3.44:1 on SURFACE

# Latest-close callout: ink tokens, not a series colour, so the label never
# looks like a third data series.
CALLOUT_BG = "#2c2c2a"


@dataclass(frozen=True)
class Overlay:
    """One technical indicator drawn as a line over the candles.

    Attributes:
    - column (str): Column in the input frame holding the values.
    - label (str): Legend and tooltip name.
    - color (str): Line colour as ``'#RRGGBB'``.
    - width (float): Stroke width in px.
    """

    column: str
    label: str
    color: str
    width: float


# Two steps of one blue: light for the fast window, dark for the slow one.
# Validated as an ordinal ramp against SURFACE -- monotone lightness, adjacent
# ΔL clear of the 0.06 floor, single hue (2° spread), dark end at 4.79:1. The
# two steps sit ΔE 20.7 apart to normal vision, comfortably over the 15 floor:
# a closer pair passed the same checks but the swatches read as one colour in
# the legend, which is where the two lines are actually told apart.
OVERLAYS: tuple[Overlay, ...] = (
    Overlay("SMA_20", "SMA 20", "#9ec5f4", 1.8),
    Overlay("SMA_50", "SMA 50", "#3987e5", 1.8),
)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# The 70/30 split the brief calls for. Volume is a supporting read, so it gets
# enough height to show relative spikes and no more.
ROW_HEIGHTS = (0.7, 0.3)

# Total figure height. Chosen so the price panel stays tall enough to read
# candle bodies once the axis band and range slider have taken their share.
FIGURE_HEIGHT = 720

# Fraction of the plot the range slider occupies.
RANGESLIDER_THICKNESS = 0.07

# Above this share of missing sessions the frame is not a daily series with
# holidays in it -- it is sparse or synthetic -- and collapsing the gaps would
# distort rather than tidy the axis.
MAX_RANGEBREAK_SHARE = 0.4


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_compact(value: float | None) -> str:
    """Formats a large number with a magnitude suffix, e.g. ``'50.04M'``.

    Volume routinely runs into the tens of millions, which is unreadable at
    full width in a tooltip and misleading when rounded to a bare integer.

    Parameters:
    - value (float | None): The number to format; None or non-finite is treated
      as missing.

    Returns:
    - str: The abbreviated number, or ``'n/a'``.
    """
    if value is None or not math.isfinite(value):
        return "n/a"

    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:,.2f}{suffix}"
    return f"{value:,.0f}"


def _format_signed_pct(value: float | None) -> str:
    """Formats a percentage change with an explicit sign, e.g. ``'+1.23%'``.

    Parameters:
    - value (float | None): The change in percent; None or non-finite is
      treated as missing, which is the normal state for the first bar in a
      window since it has no predecessor.

    Returns:
    - str: The signed percentage, or ``'n/a'``.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:+.2f}%"


def currency_symbol_for(ticker: str) -> str:
    """Guesses the trading currency symbol from a ticker's exchange suffix.

    Only the two markets ``ui.config`` offers are distinguished: NSE-listed
    symbols carry a ``.NS`` suffix and trade in rupees, everything else is
    assumed to be a US listing. A wrong guess costs an axis prefix, not a
    number, so an unrecognised suffix falls back to the dollar rather than
    dropping the prefix entirely.

    Parameters:
    - ticker (str): The symbol, e.g. ``'AAPL'`` or ``'RELIANCE.NS'``.

    Returns:
    - str: ``'₹'`` for NSE listings, otherwise ``'$'``.
    """
    return "₹" if str(ticker).upper().endswith(".NS") else "$"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _extract_dates(df: pd.DataFrame) -> pd.Series:
    """Pulls the bar dates out of a frame that may carry them either way.

    ``clean_stock_data`` returns ``Date`` as a column and
    ``add_technical_indicators`` preserves that, but the same functions leave a
    ``DatetimeIndex`` in place when handed one, and callers building frames by
    hand tend to index by date. Both shapes are accepted so the chart is not
    the reason a caller has to reshape its data.

    Parameters:
    - df (pd.DataFrame): The price frame.

    Returns:
    - pd.Series: The dates as datetimes, positionally aligned with ``df``.

    Raises:
    - ValueError: If the frame carries no usable dates in either place.
    """
    if "Date" in df.columns:
        return pd.to_datetime(df["Date"]).reset_index(drop=True)

    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(df.index), name="Date").reset_index(drop=True)

    raise ValueError(
        "Price frame carries no dates: expected a 'Date' column (as "
        "data_loader.clean_stock_data produces) or a DatetimeIndex."
    )


def _prepare(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validates and normalizes a price frame into what the traces need.

    Produces a positionally indexed frame with a ``Date`` column, numeric
    OHLCV, whichever overlay columns were present, and the two pre-formatted
    tooltip strings. Formatting happens here rather than in the hover template
    because a missing change or volume has to render as ``'n/a'``, which a
    numeric Plotly format string cannot express.

    Parameters:
    - df (pd.DataFrame): Daily bars with OHLCV columns and dates in either the
      index or a ``Date`` column.
    - ticker (str): The symbol, used only in error messages.

    Returns:
    - pd.DataFrame: The normalized frame, oldest bar first.

    Raises:
    - ValueError: If a required OHLCV column or the dates are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot chart '{ticker}': price frame is missing required "
            f"column(s) {missing}. Expected {list(REQUIRED_COLUMNS)}."
        )

    prepared = pd.DataFrame({"Date": _extract_dates(df)})

    source = df.reset_index(drop=True)
    for col in REQUIRED_COLUMNS:
        prepared[col] = pd.to_numeric(source[col], errors="coerce")

    for overlay in OVERLAYS:
        if overlay.column in source.columns:
            prepared[overlay.column] = pd.to_numeric(
                source[overlay.column], errors="coerce"
            )

    # Prefer the pipeline's own return column when it is there, so the tooltip
    # cannot disagree with the number the ML features were built from. It has to
    # be attached before the drop/sort below, so it travels with its own row --
    # reading it off the input afterwards would align it against a reindexed
    # frame and shift every change label by however many rows were dropped.
    if "Daily_Return_Pct" in source.columns:
        prepared["_change"] = pd.to_numeric(
            source["Daily_Return_Pct"], errors="coerce"
        )

    # A bar with no close cannot be drawn as a candle at all; one with a stray
    # NaN elsewhere still can, and Plotly renders the gap.
    prepared = prepared.dropna(subset=["Date", "Close"]).sort_values("Date")
    prepared = prepared.reset_index(drop=True)

    if "_change" in prepared.columns:
        change = prepared.pop("_change")
    else:
        change = prepared["Close"].pct_change() * 100

    prepared["ChangeLabel"] = [_format_signed_pct(v) for v in change]
    prepared["VolumeLabel"] = [_format_compact(v) for v in prepared["Volume"]]

    return prepared


def _non_trading_days(dates: pd.Series) -> list[str]:
    """Finds the business days inside the window that have no bar.

    Weekends are excluded by a separate axis rule, so what this returns is
    effectively the market holidays -- derived from the data rather than from a
    hard-coded exchange calendar, which keeps it correct for both the US and
    Indian listings the dashboard offers.

    Dates come back as ``'YYYY-MM-DD'`` strings rather than Timestamps: the
    figure has to survive ``to_json`` and ``write_image``, and the image
    backend serializes layout values with a plain JSON encoder that has no
    handler for a pandas Timestamp.

    Parameters:
    - dates (pd.Series): The dates actually present, ascending.

    Returns:
    - list[str]: Absent weekdays as ISO date strings, or an empty list when the
      frame is too sparse for collapsing gaps to be honest (see
      :data:`MAX_RANGEBREAK_SHARE`).
    """
    if len(dates) < 2:
        return []

    weekdays = pd.bdate_range(dates.iloc[0], dates.iloc[-1])
    absent = weekdays.difference(pd.DatetimeIndex(dates.dt.normalize()))

    if len(absent) > MAX_RANGEBREAK_SHARE * len(weekdays):
        return []
    return [day.strftime("%Y-%m-%d") for day in absent]


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def _add_candles(fig: go.Figure, data: pd.DataFrame) -> None:
    """Adds the OHLC candlestick trace to the upper panel.

    Up days are drawn hollow and down days solid, which is what carries
    direction for a reader who cannot separate the two hues. Hover is turned
    off on this trace: the tooltip is assembled once by
    :func:`_add_hover_carrier` so the OHLC block, the change and the volume
    arrive in one box instead of three.

    Parameters:
    - fig (go.Figure): The two-row figure being built.
    - data (pd.DataFrame): The prepared frame from :func:`_prepare`.
    """
    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            # Hollow body, coloured outline: direction survives without hue.
            increasing=dict(line=dict(color=GAIN, width=1), fillcolor="rgba(0,0,0,0)"),
            decreasing=dict(line=dict(color=LOSS, width=1), fillcolor=LOSS),
            line=dict(width=1),
            whiskerwidth=0,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


def _add_overlays(fig: go.Figure, data: pd.DataFrame) -> list[Overlay]:
    """Adds a line for each indicator in :data:`OVERLAYS` that is present.

    Parameters:
    - fig (go.Figure): The two-row figure being built.
    - data (pd.DataFrame): The prepared frame from :func:`_prepare`.

    Returns:
    - list[Overlay]: The overlays actually drawn, for the caller to report.
    """
    drawn: list[Overlay] = []

    for overlay in OVERLAYS:
        if overlay.column not in data.columns:
            continue
        if data[overlay.column].isna().all():
            continue

        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data[overlay.column],
                name=overlay.label,
                mode="lines",
                line=dict(color=overlay.color, width=overlay.width),
                hovertemplate="%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        drawn.append(overlay)

    return drawn


def _add_hover_carrier(fig: go.Figure, data: pd.DataFrame, symbol: str) -> None:
    """Adds the invisible trace that carries the OHLCV tooltip.

    ``go.Candlestick`` has a fixed hover layout and no ``hovertemplate``, so
    the custom tooltip the brief asks for rides on a fully transparent line
    trace pinned to the closing price instead. Under ``hovermode='x unified'``
    Plotly gathers every trace at the hovered date into one box, so this
    contributes the OHLC block while the moving averages contribute their own
    rows -- and the date is supplied once, as the box's heading.

    Parameters:
    - fig (go.Figure): The two-row figure being built.
    - data (pd.DataFrame): The prepared frame from :func:`_prepare`.
    - symbol (str): Currency symbol to prefix prices with.
    """
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            name="OHLC",
            mode="lines",
            # Transparent rather than ``opacity=0``: the trace has to stay
            # hoverable, and a zero-opacity trace is not reliably hit-tested.
            line=dict(color="rgba(0,0,0,0)", width=0),
            customdata=data[
                ["Open", "High", "Low", "Close", "ChangeLabel", "VolumeLabel"]
            ].to_numpy(),
            hovertemplate=(
                f"<b>Open</b> {symbol}%{{customdata[0]:,.2f}}"
                f"   <b>High</b> {symbol}%{{customdata[1]:,.2f}}<br>"
                f"<b>Low</b> {symbol}%{{customdata[2]:,.2f}}"
                f"   <b>Close</b> {symbol}%{{customdata[3]:,.2f}}<br>"
                "<b>Change</b> %{customdata[4]}<br>"
                "<b>Volume</b> %{customdata[5]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )


def _add_volume(fig: go.Figure, data: pd.DataFrame) -> None:
    """Adds the daily volume bars to the lower panel.

    Parameters:
    - fig (go.Figure): The two-row figure being built.
    - data (pd.DataFrame): The prepared frame from :func:`_prepare`.
    """
    fig.add_trace(
        go.Bar(
            x=data["Date"],
            y=data["Volume"],
            name="Volume",
            marker=dict(color=VOLUME_FILL, line=dict(width=0), cornerradius=2),
            customdata=data["VolumeLabel"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )


def _add_latest_close_label(fig: go.Figure, data: pd.DataFrame, symbol: str) -> None:
    """Direct-labels the most recent close at the right edge of the price panel.

    The one value a reader wants without hovering. Labelling only the endpoint
    keeps the plot free of the per-point numbers that make a dense series
    unreadable, and it means the latest price is legible without the tooltip.
    The label wears ink colours, not a candle colour, so it cannot be mistaken
    for a series of its own.

    Parameters:
    - fig (go.Figure): The two-row figure being built.
    - data (pd.DataFrame): The prepared frame from :func:`_prepare`.
    - symbol (str): Currency symbol to prefix the price with.
    """
    latest = data.iloc[-1]

    fig.add_annotation(
        # ISO string, not the Timestamp itself, for the same serialization
        # reason as the axis range breaks -- see :func:`_non_trading_days`.
        x=latest["Date"].strftime("%Y-%m-%d"),
        y=float(latest["Close"]),
        text=f"{symbol}{latest['Close']:,.2f}",
        row=1,
        col=1,
        xanchor="left",
        yanchor="middle",
        xshift=8,
        showarrow=False,
        font=dict(color=INK_PRIMARY, size=12),
        bgcolor=CALLOUT_BG,
        borderpad=4,
    )


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def _style_axes(fig: go.Figure, data: pd.DataFrame, symbol: str) -> None:
    """Applies the shared axis treatment to both panels.

    Parameters:
    - fig (go.Figure): The figure to style.
    - data (pd.DataFrame): The prepared frame, for the range-break calculation.
    - symbol (str): Currency symbol for the price axis prefix.
    """
    # Weekends by rule, holidays by observation. Without these the axis shows a
    # blank stripe wherever the market was shut.
    rangebreaks = [dict(bounds=["sat", "mon"])]
    holidays = _non_trading_days(data["Date"])
    if holidays:
        rangebreaks.append(dict(values=holidays))

    fig.update_xaxes(
        rangebreaks=rangebreaks,
        showgrid=False,
        showline=True,
        linecolor=AXIS_LINE,
        linewidth=1,
        ticks="outside",
        tickcolor=AXIS_LINE,
        tickfont=dict(color=INK_MUTED, size=11),
        # Crosshair: a solid hairline spanning both panels, so a date lines up
        # between a candle and its volume bar by eye.
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikedash="solid",
        spikecolor=INK_MUTED,
        hoverformat="%a %d %b %Y",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRIDLINE,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickcolor=AXIS_LINE,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_SECONDARY, size=12),
    )

    # Price is a range, not a magnitude, so its axis must not be forced to zero
    # -- doing so would flatten every candle into the top of the panel.
    fig.update_yaxes(
        title_text="Price",
        tickprefix=symbol,
        tickformat=",.2f",
        row=1,
        col=1,
    )

    # Volume is a magnitude read off bar length, so it is anchored at zero.
    fig.update_yaxes(
        title_text="Volume",
        rangemode="tozero",
        tickformat="~s",
        row=2,
        col=1,
    )

    # Adding a Candlestick trace makes Plotly attach a range slider to that
    # trace's own axis, which would wedge it between the two panels. Turn it
    # off there and mount it under the bottom panel instead.
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=RANGESLIDER_THICKNESS,
            bgcolor=PAGE,
            bordercolor=AXIS_LINE,
            borderwidth=1,
        ),
        row=2,
        col=1,
    )

    # Quick presets, the control most readers reach for before dragging the
    # slider handles. Mounted on the top panel's axis so they sit above the
    # plot; the shared axis applies the zoom to both panels.
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            x=1,
            xanchor="right",
            y=1.0,
            yanchor="bottom",
            bgcolor=CALLOUT_BG,
            activecolor=AXIS_LINE,
            bordercolor=AXIS_LINE,
            borderwidth=1,
            font=dict(color=INK_SECONDARY, size=11),
        ),
        row=1,
        col=1,
    )


def _style_layout(fig: go.Figure, ticker: str, data: pd.DataFrame) -> None:
    """Applies the figure-level dark theme, title, legend and hover styling.

    Parameters:
    - fig (go.Figure): The figure to style.
    - ticker (str): The symbol, for the title.
    - data (pd.DataFrame): The prepared frame, for the title's date range.
    """
    first = data["Date"].iloc[0].strftime("%d %b %Y")
    last = data["Date"].iloc[-1].strftime("%d %b %Y")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAGE,
        plot_bgcolor=SURFACE,
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            color=INK_SECONDARY,
            size=12,
        ),
        title=dict(
            text=f"{ticker} - Daily price & volume",
            subtitle=dict(
                text=f"{len(data):,} sessions - {first} to {last}",
                font=dict(color=INK_MUTED, size=12),
            ),
            font=dict(color=INK_PRIMARY, size=17),
            x=0,
            xref="paper",
            pad=dict(l=4),
        ),
        height=FIGURE_HEIGHT,
        # Right margin leaves room for the latest-close callout; the top band
        # holds the title, legend and range-selector row.
        margin=dict(l=64, r=88, t=112, b=48),
        bargap=0.3,
        # One box per date rather than one per trace, with a crosshair, so the
        # OHLC block and the moving averages are read together.
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=CALLOUT_BG,
            bordercolor=AXIS_LINE,
            font=dict(
                color=INK_PRIMARY,
                size=12,
                family='system-ui, -apple-system, "Segoe UI", sans-serif',
            ),
            align="left",
        ),
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK_SECONDARY, size=11),
        ),
        dragmode="pan",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def empty_chart(ticker: str, message: str) -> go.Figure:
    """Builds a correctly themed figure that says why there is nothing to draw.

    Returning a figure rather than None keeps the caller's layout stable: the
    chart section holds its height and its styling instead of collapsing.

    Parameters:
    - ticker (str): The symbol, for the title.
    - message (str): The explanation to centre in the empty plot.

    Returns:
    - go.Figure: A blank, themed figure carrying the message.
    """
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAGE,
        plot_bgcolor=SURFACE,
        height=260,
        margin=dict(l=64, r=64, t=64, b=48),
        title=dict(
            text=f"{ticker} - Daily price & volume",
            font=dict(color=INK_PRIMARY, size=17),
            x=0,
            xref="paper",
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=INK_MUTED, size=13),
            )
        ],
    )
    return fig


def create_stock_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Builds the two-panel candlestick and volume figure for one ticker.

    The upper panel (70% of the height) carries OHLC candles plus whichever of
    :data:`OVERLAYS` the frame supplies; the lower panel (30%) carries daily
    volume. Both share one date axis, with a range slider and preset buttons
    for filtering and a unified hover box reporting date, OHLC, change and
    volume together.

    Pure apart from the figure it returns -- the input frame is not mutated,
    and nothing is drawn to the page. :func:`render_stock_chart` does that.

    Parameters:
    - df (pd.DataFrame): Daily bars, oldest first, with ``Open``, ``High``,
      ``Low``, ``Close`` and ``Volume`` columns and dates in either a ``Date``
      column or a ``DatetimeIndex``. Indicator columns named in
      :data:`OVERLAYS` are overlaid when present and ignored when absent, so
      both a raw ``clean_stock_data`` frame and an
      ``add_technical_indicators`` one are valid input.
    - ticker (str): The symbol, used in the title and to pick the currency
      prefix for the price axis.

    Returns:
    - go.Figure: The styled figure, ready for ``st.plotly_chart``. An empty or
      all-NaN input yields the themed empty-state figure from
      :func:`empty_chart` rather than a broken plot.

    Raises:
    - ValueError: If a required OHLCV column is missing, or the frame carries
      no dates in either supported place. Both are caller bugs rather than data
      conditions, so they surface loudly instead of rendering an empty chart.
    """
    if df is None or df.empty:
        return empty_chart(ticker, "No price history to chart.")

    data = _prepare(df, ticker)

    if data.empty:
        return empty_chart(ticker, "No usable price bars in this window.")

    symbol = currency_symbol_for(ticker)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=list(ROW_HEIGHTS),
        vertical_spacing=0.04,
    )

    _add_candles(fig, data)
    _add_overlays(fig, data)
    # Added after the overlays so its tooltip row leads the unified box, and
    # last on the price panel so nothing is drawn over the candles.
    _add_hover_carrier(fig, data, symbol)
    _add_volume(fig, data)
    _add_latest_close_label(fig, data, symbol)

    _style_axes(fig, data, symbol)
    _style_layout(fig, ticker, data)

    return fig


def render_stock_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Renders the price chart into the current Streamlit container.

    Alongside the figure this writes the two things the plot cannot carry on
    its own: a caption naming the hollow/solid convention, since the candle
    colours are not colour-vision safe and the fill channel is what makes them
    readable, and a collapsed table of the same bars, so every value the
    tooltip shows is also reachable without hovering.

    Parameters:
    - df (pd.DataFrame): Daily bars, as documented on :func:`create_stock_chart`.
    - ticker (str): The symbol being charted.

    Returns:
    - go.Figure: The figure that was drawn, for callers that want to reuse it.

    Raises:
    - ValueError: Propagated from :func:`create_stock_chart` for a malformed
      frame.
    """
    fig = create_stock_chart(df, ticker)

    # ``width='stretch'`` is the current spelling of ``use_container_width=True``
    # and behaves identically. The older kwarg still works but Streamlit has
    # deprecated it past a removal date that has now passed, so it logs a notice
    # on every rerun; swap this line back if you are pinned below Streamlit 1.49.
    st.plotly_chart(fig, width="stretch")

    if df is None or df.empty:
        return fig

    st.caption(
        "Hollow candles are up days, solid candles are down days - the fill, "
        "not just the colour, carries direction. Drag the slider or use the "
        "presets to change the window."
    )

    with st.expander("View as table"):
        table = _prepare(df, ticker).drop(columns=["ChangeLabel", "VolumeLabel"])
        st.dataframe(
            # Newest first: the opposite of plotting order, because a table is
            # scanned from the top and the latest bar is the one wanted.
            table.iloc[::-1].reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )

    return fig


if __name__ == "__main__":
    # Visual self-test: ``streamlit run ui/price_chart.py`` draws the chart for
    # a cached ticker, which is the only practical way to check the hover box,
    # the range slider and the candle fills actually behave.
    import os
    import sys

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    import feature_engineering  # noqa: E402

    st.set_page_config(page_title="Price chart preview", layout="wide")
    st.title("Price chart - preview")

    _probe = st.selectbox("Cached ticker", ("AAPL", "RELIANCE.NS"))
    _csv = os.path.join(_PROJECT_ROOT, "data", f"{_probe}_daily.csv")

    _bars = pd.read_csv(_csv, parse_dates=["Date"])
    _bars = feature_engineering.add_technical_indicators(_bars)
    render_stock_chart(_bars, _probe)

    st.divider()
    st.subheader("Edge cases")
    render_stock_chart(pd.DataFrame(), "EMPTY")
