#!/usr/bin/env python3
"""VOSS Chi-square: Poisson vs GUE"""
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
os.makedirs(FIGS, exist_ok=True)

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
        print("ERROR: no gap_histogram.csv"); return
    N_val = int(cheb[0]["N"]) if cheb else 0
    gaps   = np.array([int(r["gap"]) for r in rows], dtype=float)
    counts = np.array([int(r["count"]) for r in rows], dtype=float)
    total  = counts.sum()
    mean_gap = (gaps * counts).sum() / total

    bins = np.array([0, 6.5, 12.5, 20.5, 30.5, 50.5, 100.5, 200.5, 500.5, np.inf])
    n_bins = len(bins) - 1
    # plot_bins: finite version for plotting
    plot_bins = bins.copy()
    plot_bins[-1] = plot_bins[-2] * 2

    observed = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (gaps >= bins[i]) & (gaps < bins[i+1])
        observed[i] = counts[mask].sum()

    exp_poisson = (np.exp(-bins[:-1]/mean_gap) - np.exp(-bins[1:]/mean_gap)) * total

    s_bins = bins / mean_gap
    exp_gue = (cdf_gue(np.minimum(s_bins[1:], 10)) - cdf_gue(np.minimum(s_bins[:-1], 10))) * total

    obs_sum = observed.sum()
    exp_poisson = exp_poisson * obs_sum / exp_poisson.sum()
    exp_gue     = exp_gue     * obs_sum / exp_gue.sum()
    # Filter zero-expected bins for chi-square

    mask = (exp_poisson > 0) & (exp_gue > 0)
    obs_f   = observed[mask]
    exp_p_f = exp_poisson[mask]
    exp_g_f = exp_gue[mask]
    # Renormalize after masking to match observed sum
    obs_sum_f = obs_f.sum()
    exp_p_f = exp_p_f * obs_sum_f / exp_p_f.sum()
    exp_g_f = exp_g_f * obs_sum_f / exp_g_f.sum()
    chi2_p, p_p = stats.chisquare(obs_f, exp_p_f)
    chi2_g, p_g = stats.chisquare(obs_f, exp_g_f)

    # Drop last (inf) bin for plotting only
    # ----- Plotting (finite bins only) -----
    s_obs_all = (plot_bins[:-1] + plot_bins[1:]) / 2 / mean_gap
    widths_all = (plot_bins[1:] - plot_bins[:-1]) / mean_gap * 0.85
    densities_all = observed / observed.sum() / (plot_bins[1:] - plot_bins[:-1]) * mean_gap

    # Drop last (capped-inf) bin from plot
    s_obs_plot    = s_obs_all[:-1]
    widths_plot   = widths_all[:-1]
    densities_plot = densities_all[:-1]

    s_sm = np.linspace(0.01, 3.0, 400)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(s_obs_plot, densities_plot, width=widths_plot, alpha=0.6,
           color="#2E86DE", edgecolor="black", label="Observed (VOSS)")
    ax.plot(s_sm, np.exp(-s_sm), "--", color="#EE5A24", linewidth=2.5,
            label="Poisson (exponential)")
    gue_pdf = (32 / np.pi**2) * s_sm**2 * np.exp(-4 * s_sm**2 / np.pi)
    ax.plot(s_sm, gue_pdf, "-", color="#27AE60", linewidth=2.5,
            label="GUE (Wigner surmise)")
    ax.set_xlabel("s = gap / <gap>")
    ax.set_ylabel("Probability density")
    ax.set_title("Prime gap distribution at N = " + format(N_val, ","))
    ax.legend()
    ax.set_xlim(0, 3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "chi_square_plot.png"), dpi=150)
    plt.close()

    report = {
        "N": N_val,
        "total_gaps": int(total),
        "mean_gap": float(mean_gap),
        "bins": int(mask.sum()),
        "dof": int(mask.sum()) - 1,
        "chi2_poisson": float(chi2_p),
        "chi2_gue": float(chi2_g),
        "p_value_poisson": float(p_p),
        "p_value_gue": float(p_g),
        "ratio_gue_over_poisson": float(chi2_g / chi2_p),
        "verdict": "Poisson" if chi2_p < chi2_g else "GUE",
    }
    with open(os.path.join(RES, "chi_square_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("CHI-SQUARE GOODNESS-OF-FIT TEST")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("Total gaps: " + format(int(total), ","))
    print("Mean gap: " + format(mean_gap, ".4f"))
    print("Bins: " + str(n_bins) + " | DOF: " + str(n_bins-1))
    print()
    print("Model       chi2           reduced       p-value")
    print("-" * 60)
    print("Poisson     " + format(chi2_p, ">12.4e") + "   " + format(chi2_p/(n_bins-1), ">10.4e") + "   " + format(p_p, ">10.4e"))
    print("GUE         " + format(chi2_g, ">12.4e") + "   " + format(chi2_g/(n_bins-1), ">10.4e") + "   " + format(p_g, ">10.4e"))
    print()
    print("Ratio chi2_GUE / chi2_Poisson = " + format(chi2_g/chi2_p, ".4e"))
    print("Verdict: " + report["verdict"] + " is the better fit")
    print("=" * 60)
    print("OK: chi_square_report.json + chi_square_plot.png")

if __name__ == "__main__":
    main()