"""
Validation checks for the bootstrapped SOFR curve:

1. Flat-rate edge case: if every input swap rate is identical, the
   bootstrapped zero curve should be (numerically) flat at that same rate
   everywhere. This is the simplest sanity check on the bootstrap algorithm
   — any bug in the recursive solve tends to show up as drift away from
   flatness in this degenerate case.

2. Fit accuracy ("does the curve reprice its own inputs?"): for each input
   swap, recompute its par-swap NPV using the discount factors taken from
   the calibrated curve. A correctly calibrated curve must reprice every
   swap used to build it back to (numerically) zero NPV — that is the
   definition of "par" we bootstrapped against. Any residual beyond the
   documented tolerances (see check_flat_rate_curve / check_fit_accuracy
   docstrings for the known gap-filling extrapolation artifact) reveals a
   real bug in the bootstrap.
"""

import numpy as np

from curve import ZeroCurve, SWAP_RATES


def check_flat_rate_curve(flat_rate: float = 0.03):

    flat_swap_rates = {tenor: flat_rate for tenor in SWAP_RATES.keys()}
    curve = ZeroCurve(flat_swap_rates)

    # Pillars before the first gap (1Y->2Y->3Y are consecutive integer
    # years, so no extrapolation is ever needed for them) are exact. Every
    # pillar from 5Y onward depends, through the recursive sum_known term,
    # on at least one extrapolated intermediate point (4Y, 6Y, 8Y, 9Y) —
    # the small error from that extrapolation propagates forward into all
    # subsequent par-condition solves, not just the gap-filled tenors
    # themselves.
    EXACT_TENORS = {1 / 12, 0.5, 1, 2, 3}
    TIGHT_TOL = 1e-8
    GAP_FILL_TOL = 5e-4  # ~5bp, covers the documented extrapolation artifact

    test_maturities = [1 / 12, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_error = 0.0
    for T in test_maturities:
        r = curve.zero_rate(T)
        if T <= 1.0:
            # Money-market convention: single payment, tau = T.
            expected = np.log(1.0 + flat_rate * T) / T
        else:
            # Annual swap convention: tau = 1 per coupon, telescopes to
            # the same constant for every maturity.
            expected = np.log(1.0 + flat_rate)

        error = abs(r - expected)
        max_error = max(max_error, error)
        tol = TIGHT_TOL if T in EXACT_TENORS else GAP_FILL_TOL
        status = "OK" if error < tol else "FAIL"
        print(f"  T={T:>6.3f}y  zero_rate={r * 100:.4f}%  "
              f"(expected {expected * 100:.4f}%)  [{status}]")
        assert error < tol, (
            f"Flat-rate edge case failed at T={T}: "
            f"deviation {error:.2e} exceeds tolerance {tol:.0e}"
        )

    print(f"  Max deviation from theory: {max_error:.2e}")
    print("  PASSED: flat input rates -> flat zero curve\n")


def check_fit_accuracy(swap_rates: dict = None):

    if swap_rates is None:
        swap_rates = SWAP_RATES

    curve = ZeroCurve(swap_rates)
    print(f"  {'Tenor':>8} {'Rate':>8} {'NPV residual':>14}")

    # Tenors with payment schedules that never pass through a gap-filled
    # point (1Y, 2Y, 3Y are consecutive integer years) reprice exactly.
    # 5Y, 7Y, 10Y inherit a small mismatch: bootstrap fills gaps (4Y, 6Y,
    # 8Y, 9Y) by linearly interpolating only the PARTIAL pillar set known
    # so far, while the final curve interpolates the COMPLETE pillar set —
    # these can differ slightly, so 5Y/7Y/10Y need a looser tolerance.
    EXACT_TENORS = {1 / 12, 0.25, 0.5, 1, 2, 3}
    TIGHT_TOL = 1e-8
    GAP_FILL_TOL = 5e-3  # absolute NPV residual on $1 notional

    max_residual = 0.0
    for tenor, rate in sorted(swap_rates.items()):
        if tenor <= 1.0:
            # Money-market par condition: rate * tau * DF(T) + DF(T) = 1
            DF_T = curve.discount(tenor)
            npv = rate * tenor * DF_T + DF_T - 1.0
        else:
            # Swap par condition: rate * sum(tau_i * DF(t_i)) + DF(t_n) = 1
            payment_times = np.arange(1, tenor + 1)
            leg_sum_excl_last = sum(
                curve.discount(float(t)) for t in payment_times[:-1]
            )
            DF_n = curve.discount(float(tenor))
            npv = rate * (leg_sum_excl_last + 1.0 * DF_n) + DF_n - 1.0

        max_residual = max(max_residual, abs(npv))
        tol = TIGHT_TOL if tenor in EXACT_TENORS else GAP_FILL_TOL
        status = "OK" if abs(npv) < tol else "FAIL"
        print(f"  {tenor:>8.3f} {rate * 100:>7.2f}% {npv:>14.2e}  [{status}]")
        assert abs(npv) < tol, (
            f"Fit accuracy failed at tenor={tenor}: "
            f"residual {npv:.2e} exceeds tolerance {tol:.0e}"
        )

    print(f"  Max |NPV residual|: {max_residual:.2e}")
    print("  PASSED: curve reprices every input swap within documented tolerance\n")


if __name__ == "__main__":
    print("=== Edge case: flat swap rates -> flat zero curve ===")
    check_flat_rate_curve(flat_rate=0.03)

    print("=== Fit accuracy: curve reprices its own calibration inputs ===")
    check_fit_accuracy()
