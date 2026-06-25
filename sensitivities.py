"""
Risk sensitivities for the cap: DV01 (rate risk) and Vega (volatility risk).

Both are computed by FINITE DIFFERENCE / "bump and reprice": perturb a
single market input, rebuild the curve (or reprice), and measure the
change in cap value.
"""

from curve import ZeroCurve
from cap_pricing import cap_price

def compute_dv01(
    swap_rates: dict,
    strike: float,
    sigma: float,
    notional: float = 1_000_000.0,
    maturity_years: float = 5.0,
    bump: float = 0.0001,
):
    """
    DV01: change in cap value for a +1bp parallel shift in all swap rates.

    Rebuilds the entire zero curve from the bumped swap rates (a true
    "bump and re-bootstrap", not a discount-factor-level shortcut) so the
    result reflects the full re-pricing a trading desk would see.
    """
    base_curve = ZeroCurve(swap_rates)
    base_price, _ = cap_price(base_curve, strike, sigma, notional, maturity_years)

    bumped_rates = {tenor: rate + bump for tenor, rate in swap_rates.items()}
    bumped_curve = ZeroCurve(bumped_rates)
    bumped_price, _ = cap_price(
        bumped_curve, strike, sigma, notional, maturity_years
    )

    return bumped_price - base_price


def compute_vega(
    curve: ZeroCurve,
    strike: float,
    sigma: float,
    notional: float = 1_000_000.0,
    maturity_years: float = 5.0,
    bump: float = 0.01,
):
    """
    Vega: change in cap value for a +1% (absolute) bump in flat Black vol.

    No re-bootstrapping is needed here — only the option pricing step
    (Black's formula) depends on sigma, so the curve/forward rates are
    reused unchanged.
    """
    base_price, _ = cap_price(curve, strike, sigma, notional, maturity_years)
    bumped_price, _ = cap_price(
        curve, strike, sigma + bump, notional, maturity_years
    )
    return bumped_price - base_price
