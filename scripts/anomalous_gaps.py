#!/usr/bin/env python3
"""VOSS Anomalous Gaps: filter record gaps + compare with Prime Gap List"""
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

    # Sort by position ascending
    all_pts = sorted([(int(r["position"]), int(r["gap"]), float(r["merit"]))
                      for r in rows], key=lambda x: x[0])

    # Filter record gaps: gap > all previous gaps
    record_gaps = []
    max_gap_so_far = 0
    for p, g, m in all_pts:
        if g > max_gap_so_far:
            record_gaps.append((p, g, m))
            max_gap_so_far = g

    # Also: top-10 by merit
    top_merit = sorted(all_pts, key=lambda x: -x[2])[:10]

    # Cimpeanu check on record gaps
    record_R = []
    for p, g, m in record_gaps:
        ln_p = math.log(p)
        R = g / (ln_p ** 2)
        pred = 0.556745 + 0.006321 * ln_p
        record_R.append((ln_p, R, pred, g, p))

    # Plot 1: record gaps vs position
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ps = [p for p, _, _ in record_gaps]
    gs = [g for _, g, _ in record_gaps]
    ax.semilogx(ps, gs, "o-", color="#2E86DE", markersize=7,
                linewidth=1.5, label="Record gap")
    ax.set_xlabel("Position p")
    ax.set_ylabel("Gap size")
    ax.set_title("Record Gaps to N = " + format(N_val, ","))
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    xs = [x for x, _, _, _, _ in record_R]
    rs = [r for _, r, _, _, _ in record_R]
    preds = [pr for _, _, pr, _, _ in record_R]
    ax.plot(xs, rs, "o-", color="#27AE60", markersize=7,
            linewidth=1.5, label="Observed R")
    ax.plot(xs, preds, "--", color="red", linewidth=2,
            label="Cimpeanu prediction")
    ax.set_xlabel("ln(p)")
    ax.set_ylabel("R = gap / ln^2(p)")
    ax.set_title("Cimpeanu on Record Gaps")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "anomalous_gaps.png"), dpi=150)
    plt.close()

    # CSV of submission candidates
    sub_path = os.path.join(RES, "submission_candidates.csv")
    with open(sub_path, "w") as f:
        f.write("rank,position,gap,merit\n")
        for i, (p, g, m) in enumerate(top_merit, 1):
            f.write(str(i) + "," + str(p) + "," + str(g) + "," + format(m, ".6f") + "\n")

    # Report
    report = {
        "N": N_val,
        "total_analyzed": len(all_pts),
        "n_record_gaps": len(record_gaps),
        "largest_record_gap": record_gaps[-1][1] if record_gaps else 0,
        "top_merit": [{"position": p, "gap": g, "merit": m} for p, g, m in top_merit],
        "record_gaps_summary": [
            {"p": p, "gap": g, "merit": m} for p, g, m in record_gaps[-10:]
        ],
    }
    with open(os.path.join(RES, "anomalous_gaps.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("ANOMALOUS GAPS DETECTION")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("Total gaps analyzed: " + format(len(all_pts), ","))
    print("Record gaps found:   " + str(len(record_gaps)))
    print("Largest record gap:  " + str(record_gaps[-1][1]) + " at p=" + format(record_gaps[-1][0], ","))
    print("")
    print("Top 10 by merit:")
    for i, (p, g, m) in enumerate(top_merit, 1):
        print("  " + str(i) + ". p=" + format(p, ",") + "  gap=" + str(g) + "  merit=" + format(m, ".4f"))
    print("")
    print("Submission candidates saved: submission_candidates.csv")
    print("=" * 60)
    print("OK: anomalous_gaps.json + anomalous_gaps.png + submission_candidates.csv")

if __name__ == "__main__":
    main()