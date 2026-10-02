#!/usr/bin/env python3
"""VOSS Kolmogorov-Smirnov Test: Poisson vs GUE."""
import os, csv, json, math
import numpy as np
from scipy import stats
from scipy.special import erf
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

def cdf_poisson(s):
    return 1 - np.exp(-s)

def cdf_gue(s):
    s = np.asarray(s, dtype=float)
    return erf(2*s/np.sqrt(np.pi)) - (4*s/np.pi)*np.exp(-4*s**2/np.pi)

def main():
    rows = read_csv("gap_histogram.csv")
    cheb = read_csv("chebyshev.csv")
    if not rows:
        print("INFO: no gap_histogram.csv")
        json.dump({"status": "empty"}, open(os.path.join(RES, "ks_report.json"), "w"), indent=2)
        return
    N_val = int(cheb[0]["N"]) if cheb else 0

    gaps   = np.array([int(r["gap"]) for r in rows], dtype=float)
    counts = np.array([int(r["count"]) for r in rows], dtype=float)
    total  = counts.sum()
    mean_gap = (gaps * counts).sum() / total
    s = gaps / mean_gap

    # Empirical CDF
    cumcounts = np.cumsum(counts)
    ecdf = cumcounts / total

    # Theoretical CDFs
    cdf_p = cdf_poisson(s)
    cdf_g = cdf_gue(s)

    # KS statistic = max |ECDF - CDF|
    ks_p = float(np.max(np.abs(ecdf - cdf_p)))
    ks_g = float(np.max(np.abs(ecdf - cdf_g)))

    # p-value via asymptotic formula
    # D_critical = (c(alpha)) / sqrt(n)  ~ 1.36/sqrt(n) for 95%
    def ks_pvalue(D, n):
        # Kolmogorov distribution approximation
        en = math.sqrt(n)
        lam = (en + 0.12 + 0.11/en) * D
        # Sum series
        p = 0.0
        for k in range(1, 101):
            p += (-1)**(k-1) * math.exp(-2 * k**2 * lam**2)
        p = 2 * p
        return max(0.0, min(1.0, p))

    p_p = ks_pvalue(ks_p, total)
    p_g = ks_pvalue(ks_g, total)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, ecdf, "o-", color="#2E86DE", markersize=4, linewidth=1.5, label="Empirical CDF")
    s_sm = np.linspace(0.01, 4.0, 400)
    ax.plot(s_sm, cdf_poisson(s_sm), "--", color="#EE5A24", linewidth=2.5, label="Poisson")
    ax.plot(s_sm, cdf_gue(s_sm), "-", color="#27AE60", linewidth=2.5, label="GUE")
    ax.set_xlabel("s = gap / <gap>")
    ax.set_ylabel("CDF")
    ax.set_title("Kolmogorov-Smirnov Test - N = " + format(N_val, ","))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "ks_test.png"), dpi=150)
    plt.close()

    report = {
        "N": N_val,
        "total_gaps": int(total),
        "mean_gap": float(mean_gap),
        "ks_statistic_poisson": ks_p,
        "ks_statistic_gue": ks_g,
        "p_value_poisson": p_p,
        "p_value_gue": p_g,
        "verdict": "Poisson" if ks_p < ks_g else "GUE",
        "ratio_gue_over_poisson": float(ks_g / ks_p) if ks_p > 0 else float("inf"),
    }
    with open(os.path.join(RES, "ks_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("KOLMOGOROV-SMIRNOV TEST")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("Total gaps: " + format(int(total), ","))
    print("Mean gap: " + format(mean_gap, ".4f"))
    print("")
    print("Model     KS stat      p-value")
    print("-" * 60)
    print("Poisson   " + format(ks_p, ">.6e") + "   " + format(p_p, ">.4e"))
    print("GUE       " + format(ks_g, ">.6e") + "   " + format(p_g, ">.4e"))
    print("")
    print("Ratio (GUE / Poisson): " + format(ks_g/ks_p, ".4f"))
    print("Verdict: " + report["verdict"] + " is closer to the empirical distribution")
    
    # ---- Bootstrap robustness check ----
    print("")
    print("BOOTSTRAP ROBUSTNESS (10 resamples):")
    rng = np.random.default_rng(42)
    ratios_boot = []
    for b in range(10):
        idx = rng.choice(len(s), size=len(s), replace=True)
        s_boot = s[idx]
        # Recompute ECDF
        ecdf_boot = np.arange(1, len(s_boot)+1) / len(s_boot)
        s_sorted = np.sort(s_boot)
        cdf_p_b = 1 - np.exp(-s_sorted)
        cdf_g_b = cdf_gue(s_sorted)
        d_p = float(np.max(np.abs(ecdf_boot - cdf_p_b)))
        d_g = float(np.max(np.abs(ecdf_boot - cdf_g_b)))
        ratios_boot.append(d_g / d_p if d_p > 0 else float('inf'))
    ratios_boot = [r for r in ratios_boot if np.isfinite(r)]
    print(f"  Mean ratio: {np.mean(ratios_boot):.4f}")
    print(f"  Std ratio:  {np.std(ratios_boot):.4f}")
    print(f"  Min ratio:  {np.min(ratios_boot):.4f}")
    print(f"  All > 1:    {all(r > 1 for r in ratios_boot)}")
    report["bootstrap_ratios"] = ratios_boot
    report["bootstrap_stable"] = bool(all(r > 1 for r in ratios_boot))
    print("=" * 60)
    print("OK: ks_report.json + ks_test.png")

if __name__ == "__main__":
    main()