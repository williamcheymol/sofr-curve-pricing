"""
Builds the SOFR zero curve, prices a 5y cap, computes risk sensitivities, prints a summary, and produces diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from curve import ZeroCurve, SWAP_RATES
from cap_pricing import cap_price
from sensitivities import compute_dv01, compute_vega
from style import COLORS, apply_theme

apply_theme()

STRIKE = 0.03
SIGMA = 0.20
NOTIONAL = 1_000_000.0
MATURITY = 5.0


def main():
    curve = ZeroCurve(SWAP_RATES)

    zero_5y = curve.zero_rate(5.0)
    fwd_4_425 = curve.forward_rate(4.0, 4.25)

    total_price, caplet_details = cap_price(
        curve, STRIKE, SIGMA, NOTIONAL, MATURITY
    )

    dv01 = compute_dv01(SWAP_RATES, STRIKE, SIGMA, NOTIONAL, MATURITY)
    vega = compute_vega(curve, STRIKE, SIGMA, NOTIONAL, MATURITY)

    print(f"Zero rate 5y: {zero_5y * 100:.2f}%")
    print(f"Forward rate 4y->4.25y: {fwd_4_425 * 100:.2f}%")
    print(f"Cap price: ${total_price:,.0f}")
    print(f"DV01: ${dv01:,.0f}")
    print(f"Vega (per 1% vol): ${vega:,.0f}")

    plot_curves(curve)
    plot_caplets(caplet_details)
    plot_vega_sensitivity(curve)

    plt.show()


def plot_curves(curve: ZeroCurve):
    """Plot zero rate and instantaneous forward rate vs maturity."""
    maturities = np.linspace(0.25, 10, 200)
    zero_rates = [curve.zero_rate(t) * 100 for t in maturities]
    fwd_rates = [curve.instantaneous_forward(t) * 100 for t in maturities]

    plt.figure(figsize=(8, 5))
    plt.plot(maturities, zero_rates, label="Zero rate r(T)",
              color=COLORS["blue"], linewidth=2)
    plt.plot(maturities, fwd_rates, label="Instantaneous forward f(T)",
              color=COLORS["orange"], linewidth=2)
    plt.scatter(
        curve.pillar_times,
        curve.pillar_zero_rates * 100,
        color=COLORS["white"],
        edgecolors=COLORS["green"],
        linewidths=1.2,
        zorder=5,
        label="Bootstrapped pillars",
    )
    plt.xlabel("Maturity (years)")
    plt.ylabel("Rate (%)")
    plt.title("SOFR Zero Curve and Instantaneous Forward Curve",
              color=COLORS["white"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Graphiques/Curve.png", dpi=150)


def plot_caplets(caplet_details):
    """Bar chart of caplet prices by tenor."""
    T1s = [d[0] for d in caplet_details]
    prices = [d[3] for d in caplet_details]

    plt.figure(figsize=(8, 5))
    plt.bar(T1s, prices, width=0.2, color=COLORS["yellow"],
            edgecolor=COLORS["panel"])
    plt.xlabel("Caplet fixing date T1 (years)")
    plt.ylabel("Caplet price ($)")
    plt.title("Caplet Prices by Tenor (5y Cap, K=3%, quarterly)",
              color=COLORS["white"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Graphiques/Caplets.png", dpi=150)


def plot_vega_sensitivity(curve: ZeroCurve):
    """Cap price as a function of flat Black vol, from 10% to 40%."""
    sigmas = np.linspace(0.10, 0.40, 31)
    prices = [
        cap_price(curve, STRIKE, sigma, NOTIONAL, MATURITY)[0] for sigma in sigmas
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas * 100, prices, color=COLORS["pink"], linewidth=2)
    plt.axvline(SIGMA * 100, color=COLORS["yellow"], linestyle="--",
                linewidth=1, label=f"Base sigma = {SIGMA:.0%}")
    plt.xlabel("Flat Black vol sigma (%)")
    plt.ylabel("Cap price ($)")
    plt.title("Cap Price vs Volatility", color=COLORS["white"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Graphiques/Vega.png", dpi=150)


if __name__ == "__main__":
    main()
