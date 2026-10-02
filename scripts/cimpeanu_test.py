#!/usr/bin/env python3
"""VOSS Cimpeanu Law Test - RECORD GAPS ONLY"""
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

def cimpeanu_R(ln_p):
    return 0.556745 + 0.006321 * ln_p

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

    # Sort by position
    all_pts = sorted([(int(r["position"]), int(r["gap"]), float(r["merit"]))
                      for r in rows], key=lambda x: x[0])

    # Filter record gaps only
    records = []
    max_so_far = 0
    for p, g, m in all_pts:
        if g > max_so_far:
            records.append((p, g, m))
            max_so_far = g

    print("Total gaps: " + format(len(all_pts), ","))
    print("Record gaps: " + str(len(records)))
    print("")

    # Compute R for each record
    pts = []
    for p, g, m in records:
        ln_p = math.log(p)
        R = g / (ln_p ** 2)
        pred = cimpeanu_R(ln_p)
        pts.append((ln_p, R, pred, g, p))

    if not pts:
        print("ERROR: no record points"); return

    ln_ps = np.array([x for x, _, _, _, _ in pts])
    Rs    = np.array([y for _, y, _, _, _ in pts])
    preds = np.array([pr for _, _, pr, _, _ in pts])

    # Fit line: R = a + b*ln(p)
    coeffs = np.polyfit(ln_ps, Rs, 1)
    fit_a = coeffs[1]
    fit_b = coeffs[0]

    # Compare with Cimpeanu
    ratio_a = fit_a / 0.556745
    ratio_b = fit_b / 0.006321

    # Max observed
    max_R = Rs.max()
    max_R_idx = np.argmax(Rs)
    max_pred_at_max = preds[max_R_idx]
    max_ratio = max_R / max_pred_at_max

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ln_ps, Rs, "o-", color="#27AE60", markersize=8,
            linewidth=1.8, label="Record gaps (observed)")
    xx = np.linspace(ln_ps.min(), ln_ps.max(), 200)
    ax.plot(xx, cimpeanu_R(xx), "--", color="red", linewidth=2.5,
            label="Cimpeanu: 0.5567 + 0.0063*ln p")
    ax.plot(xx, fit_a + fit_b * xx, ":", color="#8E44AD", linewidth=2,
            label="Our fit: " + format(fit_a, ".4f") + " + " + format(fit_b, ".5f") + "*ln p")
    ax.set_xlabel("ln(p)")
    ax.set_ylabel("R = gap / ln^2(p)")
    ax.set_title("Cimpeanu Law - Record Gaps Only - N = " + format(N_val, ","))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cimpeanu_law.png"), dpi=150)
    plt.close()

    # Report
    report = {
        "N": N_val,
        "n_record_gaps": len(records),
        "fit_a": float(fit_a),
        "fit_b": float(fit_b),
        "cimpeanu_a": 0.556745,
        "cimpeanu_b": 0.006321,
        "ratio_a": float(ratio_a),
        "ratio_b": float(ratio_b),
        "R_max_observed": float(max_R),
        "R_max_predicted": float(max_pred_at_max),
        "max_ratio": float(max_ratio),
        "records": [{"p": p, "gap": g, "R": R} for _, R, _, g, p in pts],
    }
    with open(os.path.join(RES, "cimpeanu_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("EMPIRICAL SCALING FIT - Record Gaps")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("Record gaps: " + str(len(records)))
    print("")
    print("Reference (Cimpeanu 2026): R = 0.556745 + 0.006321*ln(p)")
    print("Our fit:          R = " + format(fit_a, ".6f") + " + " + format(fit_b, ".6f") + "*ln(p)")
    print("")
    print("Ratio (our_a / Cimpeanu_a): " + format(ratio_a, ".4f"))
    print("Ratio (our_b / Cimpeanu_b): " + format(ratio_b, ".4f"))
    print("")
    print("Max observed R: " + format(max_R, ".4f"))
    print("Max predicted:  " + format(max_pred_at_max, ".4f"))
    print("Ratio:          " + format(max_ratio, ".4f"))
    print("")
    if 0.9 <= ratio_a <= 1.1 and 0.5 <= ratio_b <= 2.0:
        print("Verdict: CONSISTENT with Cimpeanu")
    else:
        print("Verdict: DEVIATES from Cimpeanu")
    print("=" * 60)
    print("OK: cimpeanu_report.json + cimpeanu_law.png")

if __name__ == "__main__":
    main()