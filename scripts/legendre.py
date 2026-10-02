#!/usr/bin/env python3
"""VOSS Legendre Conjecture test (optimized: sieve-based)."""
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

def sieve(limit):
    """Sieve of Eratosthenes up to limit."""
    s = bytearray([1]) * (limit + 1)
    s[0] = s[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            s[i*i::i] = bytearray(len(s[i*i::i]))
    return s

def main():
    rows = read_csv("gap_histogram.csv")
    cheb = read_csv("chebyshev.csv")
    if not rows:
        print("INFO: no gap_histogram.csv")
        json.dump({"status": "empty"}, open(os.path.join(RES, "legendre.json"), "w"), indent=2)
        return
    N_val = int(cheb[0]["N"]) if cheb else 0

    n_max = 10000
    limit = (n_max + 1) * (n_max + 1)
    print(f"Sieve up to {limit:,}...")
    t = __import__("time").time()
    is_prime_sieve = sieve(limit)
    print(f"  done in {__import__("time").time()-t:.2f}s")

    failures = []
    prime_counts = []
    gap_to_next = []

    for n in range(1, n_max + 1):
        lo = n * n
        hi = (n + 1) * (n + 1)
        found = 0
        first = -1
        for p in range(lo + 1, hi + 1):
            if is_prime_sieve[p]:
                if found == 0:
                    first = p
                found += 1
        prime_counts.append(found)
        if found == 0:
            failures.append(n)
        else:
            gap_to_next.append(first - lo)

    prime_counts = np.array(prime_counts)
    gap_to_next = np.array(gap_to_next)
    n_pass = n_max - len(failures)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(np.arange(1, n_max + 1), prime_counts, ".", color="#2E86DE", markersize=2, alpha=0.5)
    ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5, label="Legendre threshold")
    ax.set_xlabel("n")
    ax.set_ylabel("# primes in (n^2, (n+1)^2]")
    ax.set_title("Legendre Conjecture test (n up to " + format(n_max, ",") + ")")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(gap_to_next, bins=50, color="#27AE60", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Distance to first prime after n^2")
    ax.set_ylabel("Frequency")
    ax.set_title("First-prime distance distribution")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "legendre.png"), dpi=150)
    plt.close()

    report = {
        "N": N_val,
        "n_max": n_max,
        "n_pass": n_pass,
        "n_fail": len(failures),
        "success_rate": 100.0 * n_pass / n_max,
        "failures": failures[:20],
        "min_prime_count": int(prime_counts.min()),
        "max_prime_count": int(prime_counts.max()),
        "mean_prime_count": float(prime_counts.mean()),
        "max_gap_to_first_prime": int(gap_to_next.max()) if len(gap_to_next) else 0,
        "verdict": "PASS" if len(failures) == 0 else "FAIL",
    }
    with open(os.path.join(RES, "legendre.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("LEGENDRE CONJECTURE TEST")
    print("=" * 60)
    print("N (main) = " + format(N_val, ","))
    print("n_max    = " + format(n_max, ","))
    print("")
    print("Pass: " + format(n_pass, ",") + "/" + format(n_max, ","))
    print("Fail: " + format(len(failures), ","))
    print("Rate: " + format(100.0*n_pass/n_max, ".4f") + "%")
    print("")
    print("Prime count per interval:")
    print("  min = " + str(int(prime_counts.min())))
    print("  max = " + str(int(prime_counts.max())))
    print("  mean = " + format(float(prime_counts.mean()), ".4f"))
    print("")
    print("Max gap to first prime: " + str(int(gap_to_next.max())))
    print("Verdict: " + report["verdict"])
    print("=" * 60)
    print("OK: legendre.json + legendre.png")

if __name__ == "__main__":
    main()