#!/usr/bin/env python3
"""VOSS Jumping Champions mapping."""
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

# Known primorials (possible champions)
PRIMORIALS = {2:2, 6:6, 30:30, 210:210, 2310:2310, 30030:30030}

def is_primorial(g):
    return g in PRIMORIALS

def main():
    rows = read_csv("gap_histogram.csv")
    cheb = read_csv("chebyshev.csv")
    if not rows:
        print("ERROR: no gap_histogram.csv"); return
    N_val = int(cheb[0]["N"]) if cheb else 0

    gaps   = np.array([int(r["gap"]) for r in rows])
    counts = np.array([int(r["count"]) for r in rows], dtype=float)
    total  = counts.sum()

    # Sort by count
    order = np.argsort(-counts)
    top20 = [(int(gaps[i]), int(counts[i]), counts[i]/total) for i in order[:20]]

    # Identify champion
    champion = top20[0]
    champion_gap = champion[0]
    champion_share = champion[2]

    # Runner-ups
    runner_ups = top20[1:5]

    # Is champion a primorial?
    champ_is_prim = is_primorial(champion_gap)

    # Enumerate potential transitions
    # theoretical: 2 -> 6 -> 30 -> 210 -> 2310 (each primorial)
    theory_seq = [2, 6, 30, 210, 2310, 30030]
    # If champion is primorial, find its index
    theory_idx = None
    for i, p in enumerate(theory_seq):
        if champion_gap == p:
            theory_idx = i
            break

    # Compute top-20 share
    top20_share = sum(c[2] for c in top20)

    # Plot: bar chart of top 20
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [str(g) for g, _, _ in top20]
    shares = [s*100 for _, _, s in top20]
    colors = ["#E74C3C" if is_primorial(g) else "#3498DB" for g, _, _ in top20]
    bars = ax.bar(labels, shares, color=colors, alpha=0.85, edgecolor="black")
    for b, s in zip(bars, shares):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                format(s, ".2f") + "%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Gap size")
    ax.set_ylabel("Share of all gaps (%)")
    ax.set_title("Top 20 Gaps by Frequency at N = " + format(N_val, ","))
    ax.grid(alpha=0.3, axis="y")
    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#E74C3C", label="Primorial (candidate champion)"),
              Patch(facecolor="#3498DB", label="Other")]
    ax.legend(handles=legend)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "jumping_champions.png"), dpi=150)
    plt.close()

    # Report
    report = {
        "N": N_val,
        "total_gaps": int(total),
        "champion_gap": champion_gap,
        "champion_share_pct": champion_share * 100,
        "champion_is_primorial": champ_is_prim,
        "theory_seq": theory_seq,
        "theory_idx": theory_idx,
        "top20": [{"gap": g, "count": c, "share_pct": s*100} for g, c, s in top20],
        "top20_share_pct": top20_share * 100,
        "runner_ups": [{"gap": g, "share_pct": s*100} for g, _, s in runner_ups],
    }
    with open(os.path.join(RES, "jumping_champions.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save histogram snapshot for future multi-N analysis
    snap_dir = os.path.join(RES, "hist_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    snap_path = os.path.join(snap_dir, "hist_" + str(N_val) + ".csv")
    with open(snap_path, "w") as f:
        f.write("gap,count\n")
        for g, c in zip(gaps, counts):
            f.write(str(int(g)) + "," + str(int(c)) + "\n")

    print("=" * 60)
    print("JUMPING CHAMPIONS MAPPING")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("Total gaps: " + format(int(total), ","))
    print("")
    print("Champion gap:      " + str(champion_gap))
    print("  share:           " + format(champion_share*100, ".4f") + "%")
    print("  is primorial?    " + ("YES" if champ_is_prim else "NO"))
    if theory_idx is not None:
        print("  sequence index:  " + str(theory_idx) + " in [2, 6, 30, 210, 2310, 30030]")
    print("")
    print("Top 5 gaps:")
    for i, (g, c, s) in enumerate(top20[:5], 1):
        tag = " [primorial]" if is_primorial(g) else ""
        print("  " + str(i) + ". gap=" + str(g) + "  share=" + format(s*100, ".4f") + "%" + tag)
    print("")
    print("Top 20 combined: " + format(top20_share*100, ".4f") + "%")
    print("")
    print("Snapshot saved: " + snap_path)
    print("=" * 60)
    print("OK: jumping_champions.json + jumping_champions.png")

if __name__ == "__main__":
    main()