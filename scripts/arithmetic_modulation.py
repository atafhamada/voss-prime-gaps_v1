#!/usr/bin/env python3
"""VOSS Arithmetic Modulation - complete version using per-residue gap histograms."""
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
    # Primary source: per-residue gap histograms
    ghist = read_csv("modq_gap_hist.csv")
    cheb  = read_csv("chebyshev.csv")
    N_val = int(cheb[0]["N"]) if cheb else 0

    if not ghist:
        print("INFO: modq_gap_hist.csv not found or empty")
        # write empty report
        json.dump({"N": N_val, "status": "empty"},
                  open(os.path.join(RES, "arithmetic_modulation.json"), "w"), indent=2)
        return

    # Organize: data[q][a][g] = count
    data = {}
    for r in ghist:
        q = int(r["q"]); a = int(r["residue"]); g = int(r["gap"])
        c = int(r["count"])
        data.setdefault(q, {}).setdefault(a, {})[g] = c

    results = []
    for q in sorted(data.keys()):
        residues = data[q]
        n_res = len(residues)
        if n_res < 2:
            continue

        # For each residue, compute normalized gap distribution (over gaps 1..99)
        P = {}
        for a, gh in residues.items():
            total = sum(gh.values())
            if total == 0: continue
            P[a] = {g: c / total for g, c in gh.items()}

        # q-conditioned distribution: mean over residues
        all_gaps = set()
        for a in P:
            all_gaps.update(P[a].keys())
        P_q = {}
        for g in all_gaps:
            P_q[g] = sum(P[a].get(g, 0.0) for a in P) / len(P)

        # beta(q) = mean L1 deviation between P(g|a) and P(g|q)
        # weighted by 100 for readability
        deviations = []
        for a in P:
            L1 = sum(abs(P[a].get(g, 0) - P_q.get(g, 0)) for g in all_gaps)
            deviations.append(L1)
        beta = float(np.mean(deviations))

        # Also compute Jensen-Shannon style divergence
        # And a simple max-deviation
        max_dev = max(deviations) if deviations else 0

        # Compute top gap for each residue
        top_gaps = {}
        for a, gh in residues.items():
            if gh:
                top_g = max(gh.items(), key=lambda x: x[1])
                top_gaps[a] = {"gap": top_g[0], "count": top_g[1]}

        results.append({
            "q": q,
            "n_residues": len(P),
            "beta": beta,
            "max_dev": float(max_dev),
            "mean_L1": beta,
            "total_events": sum(sum(gh.values()) for gh in residues.values()),
            "top_gaps": top_gaps,
        })

    # Fit beta vs log10(q)
    if len(results) >= 2:
        qs = np.array([r["q"] for r in results], dtype=float)
        betas = np.array([r["beta"] for r in results])
        log_qs = np.log10(qs)
        coeffs = np.polyfit(log_qs, betas, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
    else:
        slope, intercept = 0.0, 0.0

    # Theory
    theory_slope = 0.577
    theory_intercept = -0.601

    # Plot 1: gap distributions per residue for each q
    n_qs = len(results)
    fig, axes = plt.subplots(1, n_qs, figsize=(6*n_qs, 5), squeeze=False)
    for idx, r in enumerate(results):
        q = r["q"]
        ax = axes[0][idx]
        residues = data[q]
        gaps_plot = list(range(2, 60, 2))
        for a in sorted(residues.keys()):
            gh = residues[a]
            total = sum(gh.values())
            ys = [gh.get(g, 0) / total * 100 if total else 0 for g in gaps_plot]
            ax.plot(gaps_plot, ys, "o-", markersize=3, linewidth=1,
                    label="a=" + str(a), alpha=0.7)
        ax.set_xlabel("gap")
        ax.set_ylabel("P(gap | a, q) %")
        ax.set_title("q = " + str(q))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "arithmetic_modulation.png"), dpi=150)
    plt.close()

    # Report
    report = {
        "N": N_val,
        "status": "complete",
        "results": results,
        "fit_slope": slope,
        "fit_intercept": intercept,
        "theory_slope": theory_slope,
        "theory_intercept": theory_intercept,
        "slope_ratio": slope / theory_slope if theory_slope else 0,
        "note": "beta = mean L1 deviation of P(gap|residue) from P(gap|q)",
    }
    with open(os.path.join(RES, "arithmetic_modulation.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Print
    print("=" * 60)
    print("ARITHMETIC MODULATION - COMPLETE")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("")
    print("q   residues   events          beta         max_dev")
    print("-" * 60)
    for r in results:
        print(format(r["q"], ">3") + "   " +
              format(r["n_residues"], ">8") + "   " +
              format(r["total_events"], ">12,") + "   " +
              format(r["beta"], ">11.6f") + "   " +
              format(r["max_dev"], ">11.6f"))
    print("")
    print("Fit: beta(q) = " + format(slope, ".6f") + " * log10(q) + " + format(intercept, ".6f"))
    print("Theory: beta(q) = 0.577 * log10(q) - 0.601")
    print("")
    print("Slope ratio (obs/theory):     " + format(slope / theory_slope, ".4f"))
    print("Intercept ratio (obs/theory): " + format(intercept / theory_intercept, ".4f"))
    print("=" * 60)
    print("OK: arithmetic_modulation.json + arithmetic_modulation.png")

if __name__ == "__main__":
    main()