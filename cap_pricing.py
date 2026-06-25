"""
Interest rate cap pricing using Black's formula on SOFR forward rates.

A cap is priced as a strip of caplets, each valued with Black's model using
forward rates and discount factors extracted from the ZeroCurve built in
curve.py.
"""

import numpy as np
from scipy.stats import norm

from curve import ZeroCurve

def black_caplet_price(
    forward_rate: float,
    strike: float,
    sigma: float,
    T1: float,
    tau: float,
    discount_factor_T2: float,
    notional: float = 1.0,
) -> float:
    """
    Price a single caplet using Black's model.

    Parameters
    ----------
    forward_rate : float
        F(T1, T2), the forward rate for the caplet's accrual period.
    strike : float
        Cap strike K.
    sigma : float
        Flat lognormal volatility of the forward rate.
    T1 : float
        Time (years) to the rate fixing date — the option's expiry.
    tau : float
        Act/360 accrual year fraction of the period [T1, T2].
    discount_factor_T2 : float
        Today's discount factor to the payment date T2.
    notional : float
        Notional amount.

    Returns
    -------
    float
        Present value of the caplet.

    Notes
    -----
    If T1 <= 0 (the period has already started/fixed) or sigma <= 0, the
    caplet has no optionality left and is priced at its intrinsic value.
    """
    if T1 <= 0 or sigma <= 0:
        intrinsic = max(forward_rate - strike, 0.0)
        return notional * tau * discount_factor_T2 * intrinsic

    d1 = (np.log(forward_rate / strike) + 0.5 * sigma ** 2 * T1) / (
        sigma * np.sqrt(T1)
    )
    d2 = d1 - sigma * np.sqrt(T1)

    price = notional * tau * discount_factor_T2 * (
        forward_rate * norm.cdf(d1) - strike * norm.cdf(d2)
    )
    return price


def cap_price(
    curve: ZeroCurve,
    strike: float,
    sigma: float,
    notional: float = 1_000_000.0,
    maturity_years: float = 5.0,
    freq_per_year: int = 4,
) -> tuple:
    """
    Price a cap as the sum of its constituent caplets.

    Parameters
    ----------
    curve : ZeroCurve
        Bootstrapped zero curve providing discount factors / forward rates.
    strike : float
        Cap strike K.
    sigma : float
        Flat Black volatility applied to every caplet.
    notional : float
        Notional amount.
    maturity_years : float
        Cap maturity in years.
    freq_per_year : int
        Number of resets per year (4 = quarterly).

    Returns
    -------
    (total_price, caplet_details) : (float, list[tuple])
        total_price is the sum of all caplet prices. caplet_details is a
        list of (T1, T2, forward_rate, caplet_price) for each period.

        The FIRST period [0, tau] is excluded by market convention: its
        floating rate is already fixed/known at trade inception (there is
        no optionality on a rate that is already set), so a standard cap
        starts accruing optionality from the SECOND period onward.
    """
    tau = 1.0 / freq_per_year
    n_periods = int(round(maturity_years * freq_per_year))

    period_times = [round(i * tau, 6) for i in range(n_periods + 1)]

    caplet_details = []
    total_price = 0.0

    # Skip the first period (i=0 -> i=1): already fixed, no optionality.
    for i in range(1, n_periods):
        T1 = period_times[i]
        T2 = period_times[i + 1]

        F = curve.forward_rate(T1, T2)
        DF2 = curve.discount(T2)

        price = black_caplet_price(
            forward_rate=F,
            strike=strike,
            sigma=sigma,
            T1=T1,
            tau=tau,
            discount_factor_T2=DF2,
            notional=notional,
        )

        caplet_details.append((T1, T2, F, price))
        total_price += price

    return total_price, caplet_details
