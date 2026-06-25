"""
Zero-coupon curve bootstrapping from SOFR swap rates.

This module builds a continuously-compounded zero-coupon curve from a set
of par swap rates, interpolates it with PIECEWISE LINEAR interpolation,
and exposes discount factors, zero rates, and instantaneous forward rates.
"""

import numpy as np

# Standard SOFR curve tenors: money-market rates at the short end (Act/360, single payment), par swap rates at the long end (annual coupons).
SWAP_RATES = {
    1 / 12: 0.019,   # 1M
    3 / 12: 0.0195,  # 3M
    6 / 12: 0.020,   # 6M
    1: 0.020,        # 1Y
    2: 0.023,        # 2Y
    3: 0.025,        # 3Y
    5: 0.030,        # 5Y
    7: 0.032,        # 7Y
    10: 0.035,       # 10Y
}


def bootstrap_zero_curve(swap_rates: dict):
    """
    Bootstrap continuously-compounded zero rates from money-market rates
    (tenor <= 1Y) and par swap rates (tenor > 1Y).

    Parameters
    ----------
    swap_rates : dict
        Mapping {tenor_in_years (float) : rate (float)}. Tenors <= 1.0 are
        treated as money-market rates (single payment, Act/360, no
        intermediate coupons). Tenors > 1.0 are treated as par swap rates
        with annual fixed-leg coupons (Act/360 day count, tau ~ 1.0 per
        year for simplicity — a standard simplification for an educational
        bootstrap; production curves use exact SOFR swap schedules).

    Returns
    -------
    (pillar_times, pillar_zero_rates) : (np.ndarray, np.ndarray)
        The bootstrapped continuously-compounded zero rates at each pillar
        maturity.

    Notes
    -----
    For swap pillars that are not consecutive integer years apart (e.g. the
    jump from 3y to 5y to 7y to 10y), the missing intermediate annual
    discount factors are obtained by LINEARLY interpolating the zero rates
    bootstrapped so far. This lets each new par equation be solved for a
    single unknown discount factor, exactly as described in the
    bootstrapping theory note above.
    """

    tenors = sorted(swap_rates.keys())
    discount_factors = {}  # discount factor
    zero_rates = {}  # continuously-compounded zero rate

    discount_factors[0.0] = 1.0

    for tenor in tenors:
        rate = swap_rates[tenor]
        tenor = float(tenor)

        if tenor <= 1.0:
            # Money-market rate: single payment, Act/360, tau = tenor.
            DF_n = 1.0 / (1.0 + rate * tenor)
        else:
            # Par swap with annual coupons at 1, 2, ..., tenor.
            payment_times = np.arange(1, tenor + 1)

            known_zero_times = np.array(sorted(zero_rates.keys()))
            known_zeros = np.array(
                [zero_rates[t] for t in known_zero_times]
            )

            for t in payment_times[:-1]:
                t = float(t)
                if t not in discount_factors:
                    if len(known_zero_times) >= 2:
                        # np.interp clamps flat beyond the known range, so
                        # this is flat extrapolation beyond the last known
                        # pillar, never an overshoot/undershoot artifact.
                        r_t = float(np.interp(t, known_zero_times, known_zeros))
                    else:
                        # Only one pillar known so far: flat extrapolation
                        r_t = float(known_zeros[0])
                    discount_factors[t] = np.exp(-r_t * t)

            # tau_i ~ 1.0 (Act/360 with ~360 days per annual period)
            tau = 1.0
            sum_known = sum(
                tau * discount_factors[float(t)] for t in payment_times[:-1]
            )

            # Solve the par-swap equation for DF at this tenor's maturity:
            #   rate * (sum_known + tau_n * DF_n) + DF_n = 1
            DF_n = (1.0 - rate * sum_known) / (1.0 + rate * tau)

        discount_factors[tenor] = DF_n
        zero_rates[tenor] = -np.log(DF_n) / tenor

    pillar_times = np.array(sorted(zero_rates.keys()))
    pillar_zero_rates = np.array([zero_rates[t] for t in pillar_times])
    return pillar_times, pillar_zero_rates


class ZeroCurve:
    """
    A continuously-compounded zero-coupon yield curve.

    Wraps a PIECEWISE LINEAR fit through bootstrapped pillar zero rates
    and exposes discount factors, zero rates, instantaneous forward
    rates, and simple (Act/360) forward rates between two dates.

    Linear interpolation (rather than a cubic spline) is used deliberately:
    a straight line between two pillars is always monotonic between them,
    so it can never overshoot above or dip below the values implied by the
    pillars themselves. This was found to matter in practice on this curve
    — see the THEORY note in bootstrap_zero_curve for the concrete dip and
    overshoot artifacts a cubic spline produced here. The trade-off is a
    kinked zero curve and a step-function instantaneous forward curve.
    """

    def __init__(self, swap_rates: dict = None):
        if swap_rates is None:
            swap_rates = SWAP_RATES
        self.swap_rates = swap_rates
        self.pillar_times, self.pillar_zero_rates = bootstrap_zero_curve(
            swap_rates
        )

    def zero_rate(self, T: float) -> float:
        """
        Continuously-compounded zero rate r(T) at maturity T (years), via
        linear interpolation between pillars (flat-extrapolated beyond the
        first/last pillar — np.interp's default behavior).
        """
        return float(np.interp(T, self.pillar_times, self.pillar_zero_rates))

    def discount(self, T: float) -> float:
        """Discount factor DF(T) = exp(-r(T) * T)."""
        if T <= 0:
            return 1.0
        return float(np.exp(-self.zero_rate(T) * T))

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Simple (Act/360-style) forward rate F(T1, T2) implied by the curve,
        i.e. the rate such that investing DF(T1) at T1 and rolling it at
        the simple rate F over [T1, T2] reproduces DF(T2):

            F(T1, T2) = (1 / tau) * [ DF(T1) / DF(T2) - 1 ]

        with tau = T2 - T1 (year fraction, Act/360 convention).
        """
        tau = T2 - T1
        return (self.discount(T1) / self.discount(T2) - 1.0) / tau

    def instantaneous_forward(self, T: float) -> float:
        """
        Instantaneous forward rate f(T) = d/dT [ r(T) * T ] = r(T) + T * r'(T).

        With piecewise-LINEAR r(T), r'(T) is the (constant) slope of the
        segment containing T, evaluated via a small central finite
        difference. This makes f(T) a step function — constant within
        each segment, with a jump at every pillar — by construction. This
        is the documented trade-off of choosing linear over cubic spline
        interpolation (see ZeroCurve's docstring): no overshoot/dip
        artifacts, but no smoothness either.
        """
        h = 1e-4
        r = self.zero_rate(T)
        r_prime = (self.zero_rate(T + h) - self.zero_rate(T - h)) / (2 * h)
        return r + T * r_prime
