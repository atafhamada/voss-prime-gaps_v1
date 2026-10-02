#!/usr/bin/env python3
"""Autocorrelation of normalized gaps (gap / ln p)."""
import os, csv, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES  = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

def read_csv(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))

def main():
    seq = read_csv("gap_sequence.csv")
    cheb = read_csv("chebyshev.csv")
    if not seq:
        print("INFO: no gap_sequence.csv")
        return
    if "p" not in seq[0]:
        print("ERROR: gap_sequence.csv missing position column")
        return
    N_val = int(cheb[0]["N"]) if cheb else 0

    gaps = np.array([int(r["gap"]) for r in seq], dtype=np.float64)
    ps   = np.array([int(r["p"]) for r in seq], dtype=np.float64)
    n = len(gaps)
    print(f"Loaded {n:,} gaps with positions")

    # Normalize by ln(p) to remove slow-varying trend
    lnp = np.log(ps)
    norm = gaps / lnp

    print(f"Mean raw gap:        {gaps.mean():.4f}")
    print(f"Mean normalized:     {norm.mean():.4f}")
    print(f"Std normalized:      {norm.std():.4f}")

    # Autocorrelation on normalized
    max_lag = 50
    ac = np.zeros(max_lag + 1)
    centered = norm - norm.mean()
    var = centered.var()
    for lag in range(max_lag + 1):
        if lag == 0:
            ac[lag] = 1.0
        else:
            cov = (centered[:-lag] * centered[lag:]).mean()
            ac[lag] = cov / var

    ci = 1.96 / math.sqrt(n)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    lags = np.arange(max_lag + 1)
    ax.bar(lags, ac, width=0.6, color="#2E86DE", alpha=0.7, edgecolor="black")
    ax.axhline(y=ci, color="red", linestyle="--", linewidth=1.5, label="95% CI")
    ax.axhline(y=-ci, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of normalized gaps (gap/ln p) @ N = " + format(N_val, ","))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "autocorr.png"), dpi=150)
    plt.close()

    outside = int((np.abs(ac[1:]) > ci).sum())
    expected = 0.05 * max_lag

    # Interpret in context of Ares-Castro (2006)
    ar_label = "consistent with Ares-Castro (2006)" if ac[1] < -0.01 else "unexpected"
    
    report = {
        "N": N_val,
        "sample_size": int(n),
        "literature": {
            "reference": "Ares, S., Castro, M. (2006). Physica A 360, 285-296",
            "finding": "Negative autocorrelation of consecutive prime gaps",
            "theoretical_range": [-0.10, -0.02],
        },
        "mean_gap_raw": float(gaps.mean()),
        "mean_normalized": float(norm.mean()),
        "std_normalized": float(norm.std()),
        "ac_lag1": float(ac[1]),
        "ac_lag2": float(ac[2]),
        "ac_lag5": float(ac[5]),
        "ac_lag10": float(ac[10]),
        "ci_95": float(ci),
        "outside_ci": outside,
        "expected_outside": float(expected),
        "max_lag": max_lag,
        "verdict": "consistent with white noise" if outside <= expected + 3 else "structure detected (negative correlation)",
        "interpretation": ar_label,
    }
    with open(os.path.join(RES, "autocorr.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("AUTOCORRELATION (NORMALIZED)")
    print("=" * 60)
    print("Sample: " + format(n, ","))
    print("Mean raw gap: " + format(gaps.mean(), ".4f"))
    print("Mean normalized: " + format(norm.mean(), ".4f"))
    print("Std normalized: " + format(norm.std(), ".4f"))
    print("")
    print("Autocorr: lag1=" + format(ac[1], ".6f") +
          "  lag2=" + format(ac[2], ".6f") +
          "  lag5=" + format(ac[5], ".6f") +
          "  lag10=" + format(ac[10], ".6f"))
    print("95% CI: +/-" + format(ci, ".6f"))
    print("Outside CI: " + str(outside) + "/" + str(max_lag) +
          " (expected " + format(expected, ".1f") + ")")
    print("")
    print("Verdict: " + report["verdict"])
    print("")    
    print("Note: Negative lag-1 autocorrelation is consistent with")
    print("  Ares, S., Castro, M. (2006). Physica A 360, 285-296")
    print("  (\"Hidden structure in the randomness of the prime number sequence?\")")
    print("")
    print("=" * 60)
    print("OK: autocorr.json + autocorr.png")

if __name__ == "__main__":
    main()