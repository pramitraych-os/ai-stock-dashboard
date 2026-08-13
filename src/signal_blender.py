"""Signal blender for the AI Stock Dashboard.

This module is the last step of the scoring chain. It takes the two normalized
signals produced upstream -- both already on the same ``[-1.0, +1.0]`` scale --
and turns them into what the dashboard actually renders:

1. :func:`blend_scores` combines ``ml_score`` (from
   :func:`ml_engine.get_ml_score`) and ``ai_score`` (from
   :func:`sentiment_engine.get_llm_sentiment_score`) into a single weighted
   ``blended_score``, clamped to ``[-1.0, +1.0]``.
2. :func:`get_dashboard_signal` buckets that blended score into one of six
   human-readable recommendations, each paired with the hex colour the UI uses
   to tint the ticker's card.

The default weighting is an even 50/50 split: the quantitative model and the
news narrative get an equal vote. Callers can tilt the balance with
``ml_weight`` (e.g. ``0.7`` to lean on the model and treat sentiment as a
tie-breaker).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bounds every score in this module lives inside, shared with ``ml_engine`` and
# ``sentiment_engine`` so the three stages are directly comparable.
SCORE_MIN = -1.0
SCORE_MAX = 1.0

# The six dashboard buckets, ordered from most bullish to most bearish as
# ``(lower_bound, signal, colour_name, hex_colour)``.
#
# ``lower_bound`` is inclusive and each bucket runs up to the next one's lower
# bound, so the ladder covers ``[-1, +1]`` with no gaps. The published bands are
# quoted to two decimals (e.g. Sell tops out at ``-0.35`` and Weak Sell starts
# at ``-0.34``); a score landing in such a gap falls into the more bearish
# bucket, which keeps the mapping conservative.
SIGNAL_BANDS = (
    (0.75, "Strong Buy", "Dark Green", "#006400"),
    (0.35, "Buy", "Light Green", "#90EE90"),
    (0.05, "Weak Buy", "Pale Green", "#98FB98"),
    (-0.34, "Weak Sell", "Light Orange", "#FFD700"),
    (-0.74, "Sell", "Light Red", "#FFA07A"),
    (SCORE_MIN, "Strong Sell", "Dark Red", "#8B0000"),
)


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


def blend_scores(ml_score: float, ai_score: float, ml_weight: float = 0.5) -> float:
    """Blends the ML and AI-sentiment scores into one weighted score.

    The two inputs are combined as
    ``ml_score * ml_weight + ai_score * (1 - ml_weight)``, so the weights always
    sum to 1 and the result stays on the same ``[-1, +1]`` scale as its inputs.
    It is clamped anyway, so a slightly out-of-range input (or a rounding
    artefact) can never leak a score the signal ladder cannot classify.

    Parameters:
    - ml_score (float): Directional score from the ML model, in ``[-1, +1]``.
    - ai_score (float): News-sentiment score from the LLM, in ``[-1, +1]``.
    - ml_weight (float): Share of the blend given to ``ml_score``; the remainder
      goes to ``ai_score``. Defaults to ``0.5`` (an even split).

    Returns:
    - float: The ``blended_score``, clamped to ``[-1.0, +1.0]``.
    """
    blended_score = (ml_score * ml_weight) + (ai_score * (1.0 - ml_weight))
    # Clamp so downstream consumers -- above all ``get_dashboard_signal`` -- can
    # rely on the declared bounds.
    return float(min(max(blended_score, SCORE_MIN), SCORE_MAX))


# ---------------------------------------------------------------------------
# Signal mapping
# ---------------------------------------------------------------------------


def get_dashboard_signal(blended_score: float) -> dict:
    """Maps a blended score onto the dashboard's signal and colour.

    Walks :data:`SIGNAL_BANDS` from the most bullish bucket down and returns the
    first one whose lower bound the score clears. The incoming score is clamped
    first, so any finite input yields a bucket.

    Parameters:
    - blended_score (float): A score in ``[-1, +1]``, normally from
      :func:`blend_scores`.

    Returns:
    - dict: ``{"blended_score": float, "signal": str, "color": str}``, where
      ``color`` is a hex string such as ``"#006400"``.
    """
    score = float(min(max(blended_score, SCORE_MIN), SCORE_MAX))

    for lower_bound, signal, _color_name, hex_color in SIGNAL_BANDS:
        if score >= lower_bound:
            return {"blended_score": score, "signal": signal, "color": hex_color}

    # Unreachable: the final band's lower bound is ``SCORE_MIN`` and ``score``
    # is clamped to it, but fall back to the most bearish bucket rather than
    # returning ``None`` if ``SIGNAL_BANDS`` is ever edited badly.
    _, signal, _color_name, hex_color = SIGNAL_BANDS[-1]
    return {"blended_score": score, "signal": signal, "color": hex_color}


if __name__ == "__main__":
    print("--- Testing signal_blender ---\n")

    print("Blending (ml_score, ai_score, ml_weight) -> blended_score:")
    blend_cases = (
        (0.90, 0.80, 0.5),  # both engines strongly bullish
        (0.60, -0.20, 0.5),  # model bullish, news lukewarm -> a plain Buy
        (0.60, -0.20, 0.9),  # same inputs, model-heavy weighting
        (0.00, 0.00, 0.5),  # no conviction either way
        (-0.50, -0.10, 0.5),  # model bearish, news mildly bearish
        (-0.95, -0.85, 0.5),  # both engines strongly bearish
        (1.00, 1.00, 0.5),  # upper bound holds
        (-1.00, -1.00, 0.5),  # lower bound holds
    )
    for ml, ai, weight in blend_cases:
        blended = blend_scores(ml, ai, weight)
        result = get_dashboard_signal(blended)
        print(
            f"  ml={ml:+.2f}  ai={ai:+.2f}  w={weight:.1f}"
            f"  -> {blended:+.4f}  {result['signal']:<12} {result['color']}"
        )

    print("\nBand boundaries (one score per bucket, plus each edge):")
    boundary_scores = (
        1.00,
        0.75,
        0.74,
        0.35,
        0.34,
        0.05,
        0.04,
        -0.34,
        -0.35,
        -0.74,
        -0.75,
        -1.00,
    )
    for probe in boundary_scores:
        mapped = get_dashboard_signal(probe)
        print(f"  {probe:+.2f} -> {mapped['signal']:<12} {mapped['color']}")

    # Sanity checks: bounds are respected and every score maps to a bucket.
    steps = 401
    sweep = [SCORE_MIN + (2.0 * i / (steps - 1)) for i in range(steps)]
    blended_sweep = [blend_scores(s, s) for s in sweep]
    print(
        f"\nSweep of {steps} scores within [-1, 1]: "
        f"{all(SCORE_MIN <= s <= SCORE_MAX for s in blended_sweep)}"
    )
    clamped = blend_scores(5.0, 5.0) == SCORE_MAX and blend_scores(-5.0, -5.0) == SCORE_MIN
    print(f"Out-of-range inputs clamped: {clamped}")
    signals_seen = {get_dashboard_signal(s)["signal"] for s in sweep}
    expected_signals = {band[1] for band in SIGNAL_BANDS}
    print(f"All 6 signals reachable: {signals_seen == expected_signals}")
    print(f"Every score mapped: {all(get_dashboard_signal(s)['signal'] for s in sweep)}")
    print("\n--- Self-test complete ---")
