#!/usr/bin/env python3
"""VOSS Cramér Conjecture test: Merit_max vs ln(p)"""
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
    rows = read_csv("large_gaps.csv")
    cheb = read_csv("chebyshev.csv")
    if not rows:
        print("INFO: large_gaps.csv is empty (no gaps >= 500 or merit >= 10)")
        print("      This is expected for small N.")
        # Write empty JSON report for consistency
        import json as _json
        _empty = {"N": 0, "status": "empty", "note": "No data at this N"}
        _out = os.path.join(RES, os.path.basename(__file__).replace(".py", ".json"))
        with open(_out, "w") as _f:
            _json.dump(_empty, _f, indent=2)
        return
    N_val = int(cheb[0]["N"]) if cheb else 0

    # Extract (ln p, merit) pairs
    pts = []
    for r in rows:
        p = int(r["position"])
        merit = float(r["merit"])
        if p > 2:
            pts.append((math.log(p), merit, p, int(r["gap"])))
    if not pts:
        print("ERROR: no valid points"); return

    xs = np.array([x for x, _, _, _ in pts])
    ys = np.array([y for _, y, _, _ in pts])

    # Find highest merit point per ln(p) bin
    n_bins = 30
    bin_edges = np.linspace(xs.min(), xs.max(), n_bins + 1)
    max_per_bin = []
    for i in range(n_bins):
        mask = (xs >= bin_edges[i]) & (xs < bin_edges[i+1])
        if mask.sum() > 0:
            idx = np.where(mask)[0][np.argmax(ys[mask])]
            max_per_bin.append((xs[idx], ys[idx]))
    if not max_per_bin:
        print("ERROR: no bin maxes"); return

    xm = np.array([x for x, _ in max_per_bin])
    ym = np.array([y for _, y in max_per_bin])

    # Fit line to upper envelope with error estimation
    coeffs = np.polyfit(xm, ym, 1)
    slope, intercept = coeffs[0], coeffs[1]
    
    # Residuals and standard error
    residuals = ym - (slope * xm + intercept)
    n_pts = len(xm)
    if n_pts > 2:
        sse = np.sum(residuals ** 2)
        mse = sse / (n_pts - 2)
        cov_matrix = mse * np.linalg.inv(np.array([[np.sum(xm**2), np.sum(xm)],
                                                    [np.sum(xm), n_pts]]))
        se_slope = float(np.sqrt(cov_matrix[0, 0]))
        se_intercept = float(np.sqrt(cov_matrix[1, 1]))
        # 95% CI
        t_crit = 1.96  # approximate for large n
        slope_ci = (float(slope - t_crit*se_slope), float(slope + t_crit*se_slope))
        intercept_ci = (float(intercept - t_crit*se_intercept), float(intercept + t_crit*se_intercept))
        r_squared = float(1 - sse / np.sum((ym - np.mean(ym))**2))
    else:
        se_slope = se_intercept = 0
        slope_ci = intercept_ci = (0, 0)
        r_squared = 0

    # Cramér prediction line: y = x
    # Observed max merit at N
    max_merit = float(ys.max())
    max_idx = int(np.argmax(ys))
    max_point = pts[max_idx]

    # Cramér prediction
    ln_N = math.log(N_val) if N_val > 0 else 0
    cramer_predict = ln_N

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(xs, ys, s=4, alpha=0.25, color="#2E86DE", label="Observed merits")
    ax.plot(xm, ym, "o-", color="#27AE60", markersize=5, linewidth=1.5,
            label="Upper envelope (bin max)")
    xx = np.linspace(xs.min(), xs.max(), 200)
    ax.plot(xx, xx, "--", color="red", linewidth=2, alpha=0.8,
            label="Cramer: Merit_max = ln(p)")
    ax.plot(xx, np.polyval(coeffs, xx), "-", color="#8E44AD",
            linewidth=1.8, alpha=0.85,
            label="Fit: " + format(slope, ".3f") + "*ln(p) + " + format(intercept, ".3f"))
    ax.axvline(x=ln_N, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("ln(p)")
    ax.set_ylabel("Merit = gap / ln(p)")
    ax.set_title("Cramer Conjecture test at N = " + format(N_val, ","))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cramer_conjecture.png"), dpi=150)
    plt.close()

    # Report
    report = {
        "N": N_val,
        "n_points": len(pts),
        "max_merit_observed": max_merit,
        "max_merit_position": max_point[2],
        "max_merit_gap": max_point[3],
        "cramer_predicted_max": cramer_predict,
        "ratio_observed_over_cramer": max_merit / cramer_predict if cramer_predict > 0 else 0,
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "fit_slope_se": float(se_slope),
        "fit_intercept_se": float(se_intercept),
        "fit_slope_95ci": list(slope_ci),
        "fit_intercept_95ci": list(intercept_ci),
        "r_squared": float(r_squared),
        "n_bins": n_bins,
        "verdict": "consistent with Cramer" if max_merit <= cramer_predict else "EXCEEDS Cramer",
    }
    with open(os.path.join(RES, "cramer_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("CRAMER CONJECTURE TEST")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("Points analyzed: " + format(len(pts), ","))
    print("")
    print("Max merit observed:     " + format(max_merit, ".4f"))
    print("  at position p = " + format(max_point[2], ","))
    print("  with gap = " + str(max_point[3]))
    print("Cramer predicts max:    " + format(cramer_predict, ".4f"))
    print("Ratio (observed/Cramer): " + format(max_merit/cramer_predict, ".4f"))
    print("")
    print("Upper envelope fit: y = " + format(slope, ".4f") + " * x + " + format(intercept, ".4f"))
    print("  95% CI slope:      [" + format(slope_ci[0], ".4f") + ", " + format(slope_ci[1], ".4f") + "]")
    print("  95% CI intercept:  [" + format(intercept_ci[0], ".4f") + ", " + format(intercept_ci[1], ".4f") + "]")
    print("  R^2:                " + format(r_squared, ".4f"))
    print("")
    print("Verdict: " + report["verdict"])
    print("=" * 60)
    print("OK: cramer_report.json + cramer_conjecture.png")

if __name__ == "__main__":
    main()