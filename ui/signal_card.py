"""The Daily Signal card that headlines the dashboard.

This is the one element a user reads before anything else, so it answers a
single question -- buy or sell, and how strongly? -- as a solid colour-tinted
panel: the signal name, the blended score behind it, where that score sits on
the ``[-1, +1]`` scale, and the two component scores that produced it.

The signal name and its tint are not defined here. They come from
:func:`signal_blender.get_dashboard_signal`, the same call ``run_analysis``
makes for its CLI output, so the card can never disagree with the blending
logic it reports on. What this module owns is the *presentation* of that
verdict, and one detail the palette forces: three of the six tints are dark
(``#006400``, ``#228B22``, ``#8B0000``) and three are pale (``#90EE90``,
``#FFD700``, ``#FF6347``), so white text is unreadable on half of them. Rather
than hard-code an ink colour per band -- a table that silently rots the next
time a hex changes -- :func:`_ink_for` measures each tint's WCAG relative
luminance and picks whichever ink actually contrasts against it.

Rendering goes through ``st.markdown(unsafe_allow_html=True)`` rather than
``st.metric``, which cannot be tinted per value. Layout and typography live in
one ``<style>`` block; the per-signal colours are written as inline ``style``
attributes on the elements that need them, so two cards with different signals
can coexist on a page without their styles colliding.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import streamlit as st

# ``signal_blender`` imports its siblings by bare name, so ``src`` goes on the
# path directly -- the same fix-up ``run_analysis`` applies for the same reason.
_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import signal_blender  # noqa: E402  (import after the sys.path fix-up)

# ---------------------------------------------------------------------------
# Ink selection
# ---------------------------------------------------------------------------

# The two candidate foregrounds. Both are near-neutral: a green- or red-cast
# ink would read as a second signal colour on the gold Weak Sell tint.
INK_ON_DARK = "#FFFFFF"
INK_ON_LIGHT = "#161616"

# WCAG AA for body-size text. Captions and tile labels are small enough that
# this is the bar they have to clear, so the muted ink is not a fixed opacity:
# :func:`_muted_for` fades the primary ink only as far as the tint can afford.
MUTED_MIN_CONTRAST = 4.5

# Opacities tried, faintest first, so a tint with contrast to spare gets a
# softer caption and only the tight ones get pushed towards full strength.
MUTED_OPACITY_STEPS = (0.62, 0.70, 0.78, 0.86, 0.94, 1.0)

# How far the inset sub-metric tiles darken the tint. Dark tints take the
# heavier value: it deepens them away from the white ink instead of washing
# them towards it, so the tile interior reads better than the card surface.
TILE_OPACITY_ON_DARK = 0.18
TILE_OPACITY_ON_LIGHT = 0.07

# Tint used when there is no score to show at all, which is a different
# statement from a neutral score and should not borrow a band's colour.
UNAVAILABLE_TINT = "#6B7280"


@dataclass(frozen=True)
class _Ink:
    """A foreground treatment for one background tint.

    Attributes:
    - text (str): Hex colour for primary text on this tint.
    - muted (str): Same hue at reduced opacity, for captions and labels.
    - tile (str): Fill for the inset sub-metric tiles.
    - line (str): Border/rule colour, one step stronger than ``tile``.
    """

    text: str
    muted: str
    tile: str
    line: str


def _relative_luminance(hex_color: str) -> float:
    """Computes a colour's WCAG 2.1 relative luminance.

    Parameters:
    - hex_color (str): A colour as ``'#RRGGBB'``; the leading ``#`` is optional
      and the digits are case-insensitive.

    Returns:
    - float: Luminance in ``[0.0, 1.0]`` -- ``0`` for black, ``1`` for white.
    """
    raw = hex_color.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    # Undo the sRGB transfer function before weighting, per the WCAG formula.
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(hex_a: str, hex_b: str) -> float:
    """Computes the WCAG contrast ratio between two colours.

    Parameters:
    - hex_a (str): First colour as ``'#RRGGBB'``.
    - hex_b (str): Second colour as ``'#RRGGBB'``.

    Returns:
    - float: The ratio, from ``1.0`` (identical) to ``21.0`` (black on white).
    """
    lum_a, lum_b = _relative_luminance(hex_a), _relative_luminance(hex_b)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def _mix(background: str, foreground: str, opacity: float) -> str:
    """Flattens ``foreground`` at ``opacity`` over ``background``.

    Compositing here rather than emitting ``rgba(...)`` means the resulting
    colour can be measured with :func:`_contrast_ratio` before it ships, which
    is the whole point of :func:`_muted_for`.

    Parameters:
    - background (str): The colour underneath, as ``'#RRGGBB'``.
    - foreground (str): The colour on top, as ``'#RRGGBB'``.
    - opacity (float): The foreground's alpha in ``[0, 1]``.

    Returns:
    - str: The flattened colour as ``'#RRGGBB'``.
    """
    back = background.lstrip("#")
    front = foreground.lstrip("#")
    mixed = [
        round(int(front[i : i + 2], 16) * opacity + int(back[i : i + 2], 16) * (1 - opacity))
        for i in (0, 2, 4)
    ]
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def _muted_for(surfaces: tuple[str, ...], ink: str) -> str:
    """Picks the faintest caption ink that clears AA on every surface given.

    Captions appear both directly on the card tint and inside the darkened
    sub-metric tiles, and one hex has to work on both, so a candidate is only
    accepted if it reaches :data:`MUTED_MIN_CONTRAST` against the harder of
    them. Steps run faintest first, so a tint with contrast to spare keeps a
    soft caption and only the tight ones get pushed towards full strength.

    When no step qualifies -- ``#228B22`` (Buy) tops out at 4.39:1 against
    white, a ceiling of the hex itself rather than of the styling -- the fully
    opaque ink is returned, which is the most contrast that tint can give.

    Parameters:
    - surfaces (tuple[str, ...]): Every background the caption sits on, as
      ``'#RRGGBB'``. The first is the one the ink is composited over.
    - ink (str): The primary ink chosen for that tint.

    Returns:
    - str: The caption colour as ``'#RRGGBB'``.
    """
    for opacity in MUTED_OPACITY_STEPS:
        candidate = _mix(surfaces[0], ink, opacity)
        if all(_contrast_ratio(s, candidate) >= MUTED_MIN_CONTRAST for s in surfaces):
            return candidate
    return ink


def _ink_for(background: str) -> _Ink:
    """Derives a readable foreground treatment for a background tint.

    Both candidate inks are measured against the tint and the higher-contrast
    one wins, so a palette edit cannot leave the card with white text on pale
    green. The sub-metric tiles then darken the tint regardless of which ink
    won: an inset panel reading as recessed is the conventional cue, and on the
    three dark tints it also buys the white text a little extra contrast rather
    than washing the background out towards it.

    Parameters:
    - background (str): The card's tint as ``'#RRGGBB'``.

    Returns:
    - _Ink: The text, caption, tile and border colours to render with.
    """
    on_dark = _contrast_ratio(background, INK_ON_DARK)
    on_light = _contrast_ratio(background, INK_ON_LIGHT)
    ink = INK_ON_DARK if on_dark >= on_light else INK_ON_LIGHT

    # Pale tints have contrast to spare but tolerate less darkening before the
    # tile stops looking like the same colour family.
    tile_opacity = TILE_OPACITY_ON_DARK if ink == INK_ON_DARK else TILE_OPACITY_ON_LIGHT

    return _Ink(
        text=ink,
        # The tile interior is the second surface captions land on, so it has
        # to be flattened here rather than left to the browser: the muted ink
        # is chosen to clear AA on the card and inside a tile alike.
        muted=_muted_for((background, _mix(background, "#000000", tile_opacity)), ink),
        tile=f"rgba(0,0,0,{tile_opacity})",
        line="rgba(255,255,255,0.22)" if ink == INK_ON_DARK else "rgba(0,0,0,0.14)",
    )


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _format_score(value: float | None) -> str:
    """Formats a ``[-1, +1]`` score for display, e.g. ``'+0.82'``.

    Parameters:
    - value (float | None): The score, or None/non-finite if unavailable.

    Returns:
    - str: The signed two-decimal score, or ``'n/a'``.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:+.2f}"


def _format_probability(value: float | None) -> str:
    """Formats a ``[0, 1]`` probability as a percentage, e.g. ``'68.4%'``.

    Parameters:
    - value (float | None): The probability, or None/non-finite if the model
      could not be scored.

    Returns:
    - str: The percentage to one decimal place, or ``'n/a'``.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{min(max(value, 0.0), 1.0) * 100:.1f}%"


def _scale_position(score: float) -> float:
    """Maps a score onto its left offset on the ``[-1, +1]`` scale track.

    Parameters:
    - score (float): A score in ``[-1, +1]``.

    Returns:
    - float: Position as a percentage from the track's left edge, ``0``-``100``.
    """
    clamped = min(max(score, signal_blender.SCORE_MIN), signal_blender.SCORE_MAX)
    return (clamped + 1.0) / 2.0 * 100.0


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

# Structure and type only -- every colour is applied inline per card, so this
# block is identical for all six signals and safe to emit more than once.
# ``clamp()`` keeps the headline readable from a laptop to a wide monitor
# without a media query.
_CARD_CSS = """
<style>
.signal-card {
  border-radius: 14px;
  padding: 1.15rem 1.35rem 1.25rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.16), 0 6px 18px rgba(0,0,0,0.10);
}
.signal-card__head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}
.signal-card__eyebrow {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.15rem;
}
.signal-card__label {
  font-size: clamp(1.85rem, 4.2vw, 2.85rem);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.015em;
}
.signal-card__score {
  text-align: right;
  flex-shrink: 0;
}
.signal-card__score-value {
  font-size: clamp(1.85rem, 4.2vw, 2.85rem);
  font-weight: 800;
  line-height: 1.05;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}
.signal-card__score-caption {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.signal-card__scale {
  margin: 1.1rem 0 0.2rem;
}
.signal-card__track {
  position: relative;
  height: 6px;
  border-radius: 999px;
}
.signal-card__zero {
  position: absolute;
  top: -3px;
  bottom: -3px;
  left: 50%;
  width: 1px;
}
.signal-card__marker {
  position: absolute;
  top: 50%;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
}
.signal-card__ends {
  display: flex;
  justify-content: space-between;
  margin-top: 0.4rem;
  font-size: 0.68rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.signal-card__subs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  margin-top: 1.05rem;
}
.signal-card__tile {
  flex: 1 1 180px;
  border-radius: 10px;
  border: 1px solid transparent;
  padding: 0.6rem 0.75rem 0.65rem;
}
.signal-card__tile-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}
.signal-card__tile-value {
  font-size: 1.35rem;
  font-weight: 750;
  line-height: 1.2;
  margin-top: 0.1rem;
  font-variant-numeric: tabular-nums;
}
.signal-card__tile-note {
  font-size: 0.7rem;
  line-height: 1.3;
  margin-top: 0.1rem;
}
</style>
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_tile(label: str, value: str, note: str, ink: _Ink) -> str:
    """Builds the HTML for one inset sub-metric tile.

    Parameters:
    - label (str): The metric name, e.g. ``'ML Probability Score'``.
    - value (str): The pre-formatted value, e.g. ``'68.4%'``.
    - note (str): One-line explanation shown under the value.
    - ink (_Ink): The foreground treatment for the card's tint.

    Returns:
    - str: A ``div.signal-card__tile`` fragment.
    """
    return (
        f'<div class="signal-card__tile" style="background:{ink.tile};'
        f'border-color:{ink.line};color:{ink.text}">'
        f'<div class="signal-card__tile-label" style="color:{ink.muted}">{label}</div>'
        f'<div class="signal-card__tile-value">{value}</div>'
        f'<div class="signal-card__tile-note" style="color:{ink.muted}">{note}</div>'
        "</div>"
    )


def render_signal_card(
    blended_score: float | None,
    *,
    ml_probability: float | None = None,
    sentiment_score: float | None = None,
) -> dict:
    """Renders the Daily Signal card into the current Streamlit container.

    The score is mapped through :func:`signal_blender.get_dashboard_signal`, so
    the label and tint match what every other consumer of that score sees. Both
    sub-metrics are optional and render as ``n/a`` when absent, which is the
    normal state when the model could not be scored or the sentiment stage fell
    back -- the headline signal is still worth showing in that case.

    Parameters:
    - blended_score (float): The blended score in ``[-1.00, +1.00]``. Values
      outside the range are clamped; ``None`` or a non-finite value renders a
      neutral "unavailable" card rather than a misleading verdict.
    - ml_probability (float | None): The model's probability in ``[0, 1]`` that
      the next close rises -- ``diagnostics['ml_probability']`` from
      ``run_analysis.analyze_ticker``.
    - sentiment_score (float | None): The LLM's news score in ``[-1, +1]`` --
      ``ai_score`` from the same payload.

    Returns:
    - dict: ``{'blended_score': float | None, 'signal': str, 'color': str}``,
      the resolved verdict the card drew, for callers that want to reuse it.
    """
    unavailable = blended_score is None or not math.isfinite(float(blended_score))

    if unavailable:
        verdict = {"blended_score": None, "signal": "No Signal", "color": UNAVAILABLE_TINT}
    else:
        verdict = signal_blender.get_dashboard_signal(float(blended_score))

    tint = verdict["color"]
    ink = _ink_for(tint)
    score = verdict["blended_score"]

    # The scale is meaningless without a score, so the unavailable card drops
    # it rather than parking a marker at a position it does not have.
    scale_html = ""
    if not unavailable:
        scale_html = (
            '<div class="signal-card__scale">'
            f'<div class="signal-card__track" style="background:{ink.tile}">'
            f'<div class="signal-card__zero" style="background:{ink.line}"></div>'
            f'<div class="signal-card__marker" style="left:{_scale_position(score):.2f}%;'
            f'background:{ink.text};border:2px solid {tint}"></div>'
            "</div>"
            f'<div class="signal-card__ends" style="color:{ink.muted}">'
            "<span>-1.00 Strong Sell</span><span>0.00</span>"
            "<span>Strong Buy +1.00</span>"
            "</div>"
            "</div>"
        )

    card_html = (
        f'<div class="signal-card" style="background:{tint};color:{ink.text}">'
        '<div class="signal-card__head">'
        "<div>"
        f'<div class="signal-card__eyebrow" style="color:{ink.muted}">Daily Signal</div>'
        f'<div class="signal-card__label">{verdict["signal"]}</div>'
        "</div>"
        '<div class="signal-card__score">'
        f'<div class="signal-card__score-value">{_format_score(score)}</div>'
        f'<div class="signal-card__score-caption" style="color:{ink.muted}">'
        "Blended score</div>"
        "</div>"
        "</div>"
        f"{scale_html}"
        '<div class="signal-card__subs">'
        + _render_tile(
            "ML Probability Score",
            _format_probability(ml_probability),
            "Model's chance the next close rises",
            ink,
        )
        + _render_tile(
            "Sentiment Score",
            _format_score(sentiment_score),
            "LLM read of recent headlines",
            ink,
        )
        + "</div>"
        "</div>"
    )

    st.markdown(_CARD_CSS + card_html, unsafe_allow_html=True)
    return verdict


if __name__ == "__main__":
    # Visual self-test: ``streamlit run ui/signal_card.py`` draws one card per
    # band, which is the only practical way to check that every tint stays
    # legible and that the scale marker tracks the score.
    st.set_page_config(page_title="Signal card preview", layout="wide")
    st.title("Daily Signal card - all bands")

    previews = (
        (0.82, 0.91, 0.73),
        (0.48, 0.74, 0.22),
        (0.18, 0.59, -0.23),
        (-0.10, 0.45, -0.65),
        (-0.52, 0.24, -0.80),
        (-0.88, 0.06, -1.00),
        (0.31, None, None),
        (float("nan"), None, None),
    )
    for probe, probability, sentiment in previews:
        drawn = render_signal_card(
            probe, ml_probability=probability, sentiment_score=sentiment
        )
        contrast = _contrast_ratio(drawn["color"], _ink_for(drawn["color"]).text)
        st.caption(
            f"input {probe:+.2f} -> {drawn['signal']} {drawn['color']} "
            f"- ink contrast {contrast:.2f}:1"
        )
        st.write("")
