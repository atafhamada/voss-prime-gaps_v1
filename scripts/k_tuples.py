#!/usr/bin/env python3
"""VOSS k-tuples: twins/cousins/sexy + triplets/quadruplets from real sequence."""
import os, csv, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES  = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

C2 = 0.6601618158468696

def read_csv(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))

def main():
    hist = read_csv("gap_histogram.csv")
    seq  = read_csv("gap_sequence.csv")
    cheb = read_csv("chebyshev.csv")
    if not hist:
        print("INFO: no gap_histogram.csv")
        json.dump({"status": "empty"}, open(os.path.join(RES, "k_tuples.json"), "w"), indent=2)
        return
    N_val = int(cheb[0]["N"]) if cheb else 0

    # Counts from histogram (exact for twins/cousins/sexy)
    gap_counts = {int(r["gap"]): int(r["count"]) for r in hist}
    twins   = gap_counts.get(2, 0)
    cousins = gap_counts.get(4, 0)
    sexy    = gap_counts.get(6, 0)
    octuplets = gap_counts.get(8, 0)

    # Triplets/Quadruplets from real sequence
    triplet_246 = 0  # (p, p+2, p+6): gaps = 2, 4
    triplet_426 = 0  # (p, p+4, p+6): gaps = 4, 2
    quadruplet   = 0 # (p, p+2, p+6, p+8): gaps = 2, 4, 2

    if seq and "gap" in seq[0]:
        gaps = np.array([int(r["gap"]) for r in seq], dtype=np.int32)
        n = len(gaps)
        print(f"Scanning {n:,} real consecutive gaps for patterns...")
        for i in range(n - 2):
            if gaps[i] == 2 and gaps[i+1] == 4:
                triplet_246 += 1
            elif gaps[i] == 4 and gaps[i+1] == 2:
                triplet_426 += 1
        for i in range(n - 3):
            if gaps[i] == 2 and gaps[i+1] == 4 and gaps[i+2] == 2:
                quadruplet += 1
    else:
        print("INFO: no gap_sequence.csv - triplets/quadruplets not computed")

    # Two ranges: full N and sample range
    N_full = float(N_val) if N_val > 0 else 1
    ln_N_full = math.log(N_full)
    
    # Sample range (for triplets/quadruplets)
    if seq and "p" in seq[0]:
        sample_first_p = int(seq[0]["p"])
        sample_last_p = int(seq[-1]["p"])
        N_sample = float(sample_last_p) - float(sample_first_p)
    else:
        N_sample = N_full
        sample_first_p = 0
        sample_last_p = int(N_full)
    ln_N_sample = math.log(N_sample) if N_sample > 1 else 1
    print(f"Sample range: {sample_first_p:,} to {sample_last_p:,} (span: {N_sample:,.0f})")
    print(f"Full range: N = {N_val:,}")
    # Full-N predictions (twins, sexy)
    expected_twins = 2 * C2 * N_full / (ln_N_full ** 2)
    expected_sexy  = 2 * expected_twins
    
    # Sample-range predictions (triplets, quadruplets)
    expected_twins_sample = 2 * C2 * N_sample / (ln_N_sample ** 2)

    # Singlet constants for triplets (Hardy-Littlewood)
    # C_246 = 2.858... (pattern 2-4)
    # C_426 = 2.858... (pattern 4-2)
    # We use published values
    # Hardy-Littlewood: density per GAP, not per position
    #   # gaps in [0, X] ~ X / ln(X)
    #   # triplets(2,4) in [0, X] ~ C_246 * X / ln(X)^3
    #   => triplets per gap ~ C_246 / ln(X)^2
    #   # quadruplets(2,4,2) in [0, X] ~ C_242 * X / ln(X)^4
    #   => quadruplets per gap ~ C_242 / ln(X)^3
    ln_p_mid = math.log((sample_first_p + sample_last_p) / 2)
    n_sample_gaps = len(seq)  # actual number of gaps sampled
    
    C_246 = 2.8582485964
    expected_triplet = C_246 * n_sample_gaps / (ln_p_mid ** 2)
    C_242 = 4.15118086
    expected_quad = C_242 * n_sample_gaps / (ln_p_mid ** 3)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    names = ["Twins\n(2)", "Cousins\n(4)", "Sexy\n(6)", "Oct.\n(8)",
             "Trip 2-4", "Trip 4-2", "Quad 2-4-2"]
    values = [twins, cousins, sexy, octuplets, triplet_246, triplet_426, quadruplet]
    colors = ["#2E86DE", "#27AE60", "#EE5A24", "#8E44AD", "#F39C12", "#E67E22", "#1ABC9C"]
    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                format(v, ","), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("k-tuples counts @ N = " + format(N_val, ","))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    obs = [twins, sexy, triplet_246, quadruplet]
    exp = [expected_twins, expected_sexy, expected_triplet, expected_quad]
    x = np.arange(4)
    w = 0.35
    ax.bar(x - w/2, obs, w, label="Observed", color="#2E86DE")
    ax.bar(x + w/2, exp, w, label="HL prediction", color="#EE5A24")
    ax.set_xticks(x)
    ax.set_xticklabels(["Twins", "Sexy", "Trip 2-4", "Quad"])
    ax.set_ylabel("Count (log)")
    ax.set_yscale("log")
    ax.set_title("Observed vs Hardy-Littlewood")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "k_tuples.png"), dpi=150)
    plt.close()

    report = {
        "N": N_val,
        "twins": twins,
        "cousins": cousins,
        "sexy": sexy,
        "octuplets": octuplets,
        "triplets_246": triplet_246,
        "triplets_426": triplet_426,
        "quadruplets": quadruplet,
        "expected_twins_HL": float(expected_twins),
        "expected_sexy_HL": float(expected_sexy),
        "expected_triplet_HL": float(expected_triplet),
        "expected_quad_HL": float(expected_quad),
        "ratio_twins": float(twins / expected_twins) if expected_twins else 0,
        "ratio_sexy": float(sexy / expected_sexy) if expected_sexy else 0,
        "ratio_triplet": float(triplet_246 / expected_triplet) if expected_triplet else 0,
        "ratio_quad": float(quadruplet / expected_quad) if expected_quad else 0,
        "status": "exact (from real sequence)" if seq else "partial",
    }
    with open(os.path.join(RES, "k_tuples.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("k-TUPLES COUNTS")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("")
    print("Type                Count              Expected (HL)   Ratio")
    print("-" * 60)
    print("Twins (p,p+2)       " + format(twins, ">15,") + "  " +
          format(int(expected_twins), ">15,") + "  " +
          format(twins/expected_twins if expected_twins else 0, ">7.4f") + "  [full N]")
    print("Cousins (p,p+4)     " + format(cousins, ">15,"))
    print("Sexy (p,p+6)        " + format(sexy, ">15,") + "  " +
          format(int(expected_sexy), ">15,") + "  " +
          format(sexy/expected_sexy if expected_sexy else 0, ">7.4f") + "  [full N]")
    print("Octuplets (p,p+8)   " + format(octuplets, ">15,"))
    print("Triplets 2-4        " + format(triplet_246, ">15,") + "  " +
          format(int(expected_triplet), ">15,") + "  " +
          format(triplet_246/expected_triplet if expected_triplet else 0, ">7.4f") + "  [sample]")
    print("Triplets 4-2        " + format(triplet_426, ">15,"))
    print("Quadruplets 2-4-2   " + format(quadruplet, ">15,") + "  " +
          format(int(expected_quad), ">15,") + "  " +
          format(quadruplet/expected_quad if expected_quad else 0, ">7.4f") + "  [sample]")
    print("")
    print("Status: " + report["status"])
    print("=" * 60)
    print("OK: k_tuples.json + k_tuples.png")

if __name__ == "__main__":
    main()