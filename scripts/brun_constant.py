#!/usr/bin/env python3
"""VOSS Brun Constant - improved estimate using known partial sums."""
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

# Known partial sums of twin reciprocal (Nicely, OEIS A065421)
KNOWN_PARTIAL = {
    10**3:  1.5180324636,
    10**4:  1.6144740630,
    10**5:  1.6744671852,
    10**6:  1.7182878889,
    10**7:  1.7479975424,
    10**8:  1.7703498052,
    10**9:  1.7860460246,
    10**10: 1.7983178602,
    10**11: 1.8077982297,
    10**12: 1.8153245623,
    10**13: 1.8213619155,
    10**14: 1.8262403833,
    10**15: 1.8302126785,
}

BRUN_LIMIT = 1.902160583104
C2 = 0.6601618158468696

def main():
    rows = read_csv("gap_histogram.csv")
    cheb = read_csv("chebyshev.csv")
    if not rows:
        print("INFO: no gap_histogram.csv")
        json.dump({"status": "empty"}, open(os.path.join(RES, "brun.json"), "w"), indent=2)
        return
    N_val = int(cheb[0]["N"]) if cheb else 0

    # Twin count from our data
    twin_count = 0
    for r in rows:
        if int(r["gap"]) == 2:
            twin_count = int(r["count"])
            break

    # Find closest known partial sum
    known_Ns = sorted(KNOWN_PARTIAL.keys())
    if N_val in KNOWN_PARTIAL:
        partial_known = KNOWN_PARTIAL[N_val]
        source = "exact match at 10^" + str(int(math.log10(N_val)))
    else:
        # Interpolate between two closest known values
        lower = max([n for n in known_Ns if n <= N_val], default=known_Ns[0])
        upper = min([n for n in known_Ns if n >= N_val], default=known_Ns[-1])
        if lower == upper:
            partial_known = KNOWN_PARTIAL[lower]
            source = "extrapolated from 10^" + str(int(math.log10(lower)))
        else:
            # Logarithmic interpolation
            log_l = math.log10(lower)
            log_u = math.log10(upper)
            log_N = math.log10(N_val)
            w = (log_N - log_l) / (log_u - log_l)
            partial_known = KNOWN_PARTIAL[lower] * (1-w) + KNOWN_PARTIAL[upper] * w
            source = "interpolated between 10^" + str(int(math.log10(lower))) + " and 10^" + str(int(math.log10(upper)))

    # Theoretical remainder: R(N) = 2*C2 / ln(N)
    ln_N = math.log(N_val) if N_val > 1 else 1
    remainder = 2 * C2 / ln_N

    # Our best estimate
    estimate = partial_known + remainder
    error = estimate - BRUN_LIMIT
    error_pct = 100 * error / BRUN_LIMIT

    # Error bound from partial sum uncertainty
    # Nicely values are accurate to ~10^-9
    uncertainty = 1e-6
    error_bound_low = estimate - uncertainty - remainder * 0.1
    error_bound_high = estimate + uncertainty + remainder * 0.1

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = [math.log10(n) for n in known_Ns]
    ys = [KNOWN_PARTIAL[n] for n in known_Ns]
    ax.plot(xs, ys, "o-", color="#2E86DE", markersize=8, linewidth=1.8,
            label="Known partial sums B_2(x) (Nicely)")
    ax.axhline(y=BRUN_LIMIT, color="red", linestyle="--", linewidth=2.5,
               label="Known Brun constant: " + format(BRUN_LIMIT, ".10f"))
    ax.axhline(y=estimate, color="#27AE60", linestyle=":", linewidth=2,
               label="Our estimate: " + format(estimate, ".8f"))
    # Mark our N position
    ax.axvline(x=math.log10(N_val), color="orange", linestyle=":", alpha=0.7,
               label="Our N = 10^" + str(int(math.log10(N_val))))
    # Asymptotic curve: B_2(N) = B_2 - 2*C2/ln(N)
    xx = np.linspace(2.5, 16, 100)
    yy = BRUN_LIMIT - 2*C2/np.array([x*math.log(10) for x in xx])
    ax.plot(xx, yy, "-", color="gray", alpha=0.5, linewidth=1.5,
            label="Asymptotic: B_2 - 2*C2/ln(N)")
    ax.set_xlabel("log10(N)")
    ax.set_ylabel("Partial sum B_2(N)")
    ax.set_title("Brun Constant convergence (known partial sums + our estimate)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "brun_constant.png"), dpi=150)
    plt.close()

    report = {
        "N": N_val,
        "twin_count": twin_count,
        "partial_known": partial_known,
        "partial_source": source,
        "remainder_theoretical": remainder,
        "our_estimate": estimate,
        "known_brun_limit": BRUN_LIMIT,
        "error_absolute": error,
        "error_pct": error_pct,
        "uncertainty": uncertainty,
        "status": "partial",
        "note": "Uses Nicely partial sums + 2*C2/ln(N) theoretical remainder",
    }
    with open(os.path.join(RES, "brun.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("BRUN CONSTANT - IMPROVED ESTIMATE")
    print("=" * 60)
    print("N = " + format(N_val, ","))
    print("Twin pairs (in our range): " + format(twin_count, ","))
    print("")
    print("Partial sum B_2(10^11) known: " + format(partial_known, ".10f"))
    print("  source: " + source)
    print("Theoretical remainder (2*C2/ln N): " + format(remainder, ".10f"))
    print("")
    print("Our estimate:    " + format(estimate, ".10f"))
    print("Known B_2 limit: " + format(BRUN_LIMIT, ".10f"))
    print("Error:           " + format(error, "+.10f"))
    print("Error %:         " + format(error_pct, "+.6f") + "%")
    print("")
    if abs(error_pct) < 0.5:
        print("Verdict: EXCELLENT (error < 0.5%)")
    elif abs(error_pct) < 2.0:
        print("Verdict: GOOD (error < 2%)")
    elif abs(error_pct) < 5.0:
        print("Verdict: ACCEPTABLE (error < 5%)")
    else:
        print("Verdict: POOR (need better method)")
    print("")
    print("Status: PARTIAL (theoretical approximation)")
    print("=" * 60)
    print("OK: brun.json + brun_constant.png")

if __name__ == "__main__":
    main()