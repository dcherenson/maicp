"""Exact binomial calibration rank for a finite sample."""
from math import isfinite
from scipy.stats import binom


def binomial_calibration_rank(count: int, alpha: float, delta_cal: float) -> int:
    """Return the smallest rank k with P[Binomial(count, 1-alpha) ≤ k-1] ≥ 1-delta_cal. Raise ValueError if no finite sample rank qualifies."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('Calibration sample count must be a positive integer')
    if not isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1)')
    if not isfinite(delta_cal) or not 0 < delta_cal < 1:
        raise ValueError('delta_cal must lie in (0,1)')
    rank = int(binom.ppf(1 - delta_cal, count, 1 - alpha)) + 1
    if not 1 <= rank <= count:
        raise ValueError('Binomial calibration rank exceeds available samples; no finite threshold')
    return rank
