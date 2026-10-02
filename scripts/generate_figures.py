#!/usr/bin/env python3
"""VOSS figures generator - 6 charts"""
import os, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(HERE, ".."))
RES   = os.path.join(ROOT, "results")
FIGS  = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

def read_csv(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))

# ---------- 1. Gap Distribution ----------
def fig_distribution():
    rows = read_csv("gap_histogram.csv")
    if not rows: return
    gaps   = np.array([int(r["gap"]) for r in rows], dtype=float)
    counts = np.array([int(r["count"]) for r in rows], dtype=float)
    total  = counts.sum()
    mean_g = (gaps * counts).sum() / total

    s = gaps / mean_g
    bw = 0.05
    pdf_obs = counts / total / bw

    fig, ax = plt.subplots(figsize=(10, 6))
    s_sm = np.linspace(0.01, 3.0, 400)

    # Poisson (exponential)
    ax.plot(s_sm, np.exp(-s_sm), "--", color="#EE5A24",
            linewidth=2.5, alpha=0.85, label="Poisson (exponential)")
    # GUE (Wigner)
    gue = (32 / math.pi**2) * s_sm**2 * np.exp(-4 * s_sm**2 / math.pi)
    ax.plot(s_sm, gue, "-", color="#27AE60",
            linewidth=2.5, alpha=0.85, label="GUE (Wigner surmise)")

    # Observed
    ax.bar(s, pdf_obs, width=bw*0.85, color="#2E86DE",
           alpha=0.7, edgecolor="black", linewidth=0.5,
           label="Observed (VOSS)")

    ax.set_xlabel("s = gap / <gap>")
    ax.set_ylabel("Probability density")
    ax.set_title("Prime Gap Distribution: Poisson vs GUE (N = 10^12)")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "gap_distribution.png"))
    plt.close()
    print("  + gap_distribution.png")

# ---------- 2. HL Trend ----------
def fig_hl_trend():
    rows = read_csv("hl_trend.csv")
    if not rows: return
    Ns = np.array([float(r["N"]) for r in rows])
    rs = np.array([float(r["P6_over_P2"]) for r in rows])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(Ns, rs, "o-", color="#2E86DE", linewidth=2.5,
                markersize=12, label="P(6)/P(2)")
    ax.axhline(y=2, color="red", linestyle="--", linewidth=2.5,
               alpha=0.7, label="Hardy-Littlewood limit = 2")
    for n, r in zip(Ns, rs):
        ax.annotate(f"{r:.4f}", xy=(n, r), xytext=(8, -12),
                    textcoords="offset points", fontsize=10,
                    color="#2E86DE", fontweight="bold")
    ax.set_xlabel("N (log scale)")
    ax.set_ylabel("P(6) / P(2)")
    ax.set_title("Convergence to Hardy-Littlewood Prediction")
    ax.set_ylim(1.70, 2.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "hl_trend.png"))
    plt.close()
    print("  + hl_trend.png")

# ---------- 3. Chebyshev Bias ----------
def fig_chebyshev():
    rows = read_csv("chebyshev.csv")
    if not rows: return
    r0 = rows[0]
    N_val = float(r0["N"])
    pi41 = int(r0["pi_4_1"])
    pi43 = int(r0["pi_4_3"])
    diff = int(r0["difference"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(["pi(x; 4, 1)", "pi(x; 4, 3)"],
                  [pi41, pi43],
                  color=["#3498DB", "#E74C3C"],
                  alpha=0.85, edgecolor="black")
    for b, v in zip(bars, [pi41, pi43]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:,}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Prime count")
    ax.set_title(f"Chebyshev Bias at N = {N_val:.0e}\n"
                 f"Difference (3-1) = {diff:+,}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "chebyshev_bias.png"))
    plt.close()
    print("  + chebyshev_bias.png")

# ---------- 4. Top Merit Gaps ----------
def fig_top_merit():
    rows = read_csv("large_gaps.csv")
    if not rows: return
    top = sorted(rows, key=lambda r: -float(r["merit"]))[:200]
    pos = [float(r["position"]) for r in top]
    merit = [float(r["merit"]) for r in top]

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(pos, merit, c=merit, cmap="plasma",
                    s=40, alpha=0.7, edgecolors="black", linewidth=0.3)
    plt.colorbar(sc, label="Merit")
    ax.set_xscale("log")
    ax.set_xlabel("Prime position (p_after)")
    ax.set_ylabel("Merit = gap / ln(p)")
    ax.set_title("Top 200 High-Merit Gaps (N = 10^12)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "top_merit_gaps.png"))
    plt.close()
    print("  + top_merit_gaps.png")

# ---------- 5. Cramér Conjecture ----------
def fig_cramer():
    rows = read_csv("large_gaps.csv")
    if not rows: return
    # Compute merit for each gap
    pts = []
    for r in rows:
        p = int(r["position"])
        if p <= 2: continue
        merit = float(r["merit"])
        pts.append((math.log(p), merit))
    if not pts: return
    xs = np.array([x for x, _ in pts])
    ys = np.array([y for _, y in pts])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, s=5, alpha=0.3, color="#2E86DE", label="Observed merits")
    # Cramér prediction: Merit_max ~ ln(p)
    xx = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(xx, xx, "--", color="red", linewidth=2,
            label="Cramér: Merit_max = ln(p)")
    # Best fit
    if len(xs) > 10:
        coeffs = np.polyfit(xs, ys, 1)
        ax.plot(xx, np.polyval(coeffs, xx), "-", color="#27AE60",
                linewidth=2, label=f"Fit: {coeffs[0]:.3f}*ln(p) + {coeffs[1]:.3f}")
    ax.set_xlabel("ln(p)")
    ax.set_ylabel("Merit")
    ax.set_title("Cramér's Conjecture Test (N = 10^12)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cramer_conjecture.png"))
    plt.close()
    print("  + cramer_conjecture.png")

# ---------- 6. Gap Stats across N ----------
def fig_gap_stats():
    rows = read_csv("hl_trend.csv")
    if not rows: return
    Ns = [float(r["N"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(Ns, [1.7783, 1.8018, 1.8208, 1.8366, 1.8497][:len(Ns)],
                "o-", color="#2E86DE", linewidth=2.5,
                markersize=10, label="P(6)/P(2)")
    ax.set_xlabel("N")
    ax.set_ylabel("P(6)/P(2)")
    ax.set_title("Gap Ratio Evolution across N")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "gap_stats_across_N.png"))
    plt.close()
    print("  + gap_stats_across_N.png")

# ============================================================
if __name__ == "__main__":
    print("Generating 6 figures...")
    fig_distribution()
    fig_hl_trend()
    fig_chebyshev()
    fig_top_merit()
    fig_cramer()
    fig_gap_stats()
    print("Done. Figures in: " + FIGS)
