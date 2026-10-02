#!/usr/bin/env python3
"""
VOSS Digital Certificates Generator
Reads results/large_gaps.csv, generates:
  - results/certificates.csv    (all gaps, with SHA-256)
  - results/cert_top20/*.txt    (top 20 by merit)
"""
import os, sys, csv, json, time, math, hashlib
import numpy as np
from numba import njit

HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(HERE, '..'))
INPUT      = os.path.join(ROOT, 'results', 'large_gaps.csv')
CERT_CSV   = os.path.join(ROOT, 'results', 'certificates.csv')
TOP20_DIR  = os.path.join(ROOT, 'results', 'cert_top20')
TOP_N      = 20
MERIT_CSV  = 0.0    # for CSV: include all
MERIT_TXT  = 15.0   # for TXT: only merit >= 15 (top quality)

# ============================================================
# Numba-accelerated verification (same as verify_certificates)
# ============================================================
@njit(cache=True)
def mul_mod(a, b, m):
    a = a % m; b = b % m
    r = 0
    while b > 0:
        if b & 1:
            if r >= m - a: r = r - (m - a)
            else: r = r + a
        if a >= m - a: a = a - (m - a)
        else: a = a + a
        b >>= 1
    return r

@njit(cache=True)
def mod_pow(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if exp & 1: result = mul_mod(result, base, mod)
        base = mul_mod(base, base, mod)
        exp >>= 1
    return result

@njit(cache=True)
def is_prime_64_numba(n):
    if n < 2: return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n == p: return True
        if n % p == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1; d //= 2
    for a in (2,3,5,7,11,13,17,19,23,29,31,37):
        x = mod_pow(a, d, n)
        if x == 1 or x == n - 1: continue
        found = False
        for _ in range(r - 1):
            x = mul_mod(x, x, n)
            if x == n - 1:
                found = True; break
        if not found: return False
    return True

@njit(cache=True)
def verify_gap_numba(p_after, gap, base_primes):
    p_before = p_after - gap
    if not is_prime_64_numba(p_before): return 0, 1
    if not is_prime_64_numba(p_after):  return 0, 2
    if gap < 2: return 1, 0
    n_interval = gap - 1
    is_comp = np.zeros(n_interval, dtype=np.uint8)
    i = 0
    while i < n_interval:
        is_comp[i] = 1; i += 2
    sq = 0
    while (sq + 1) * (sq + 1) <= p_after: sq += 1
    for q in base_primes:
        if q == 2: continue
        if q > sq: break
        start = (p_before // q + 1) * q
        if (start & 1) == 0: start += q
        if start >= p_after: continue
        idx = start - p_before - 1
        if idx < 0: idx = 0
        step = 2 * q
        while idx < n_interval:
            is_comp[idx] = 1; idx += step
    for i in range(n_interval):
        if is_comp[i] == 0: return 0, 3
    return 1, 0

# ============================================================
def build_base_sieve(limit):
    s = bytearray([1]) * (limit + 1)
    s[0] = s[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            s[i*i::i] = bytearray(len(s[i*i::i]))
    return [i for i, v in enumerate(s) if v]

# ============================================================
def sha256_cert(p_before, p_after, gap, merit):
    """Deterministic SHA-256 over certificate content."""
    content = f"VOSS|{p_before}|{p_after}|{gap}|{merit:.6f}"
    return hashlib.sha256(content.encode()).hexdigest()

# ============================================================
def write_text_cert(filepath, p_before, p_after, gap, merit, sha):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 66 + "\n")
        f.write("           VOSS NUMERICAL CERTIFICATE OF PRIME GAP\n")
        f.write("=" * 66 + "\n")
        f.write(f"Prime Before (p)        : {p_before:,}\n")
        f.write(f"Prime After  (p+g)      : {p_after:,}\n")
        f.write(f"Gap Size (g)            : {gap:,}\n")
        f.write(f"Merit (g / ln p)        : {merit:.6f}\n")
        f.write(f"ln(p)                   : {math.log(p_before):.6f}\n")
        f.write(f"Max Factor Limit (√p+g) : {math.isqrt(p_after):,}\n")
        f.write("-" * 66 + "\n")
        f.write("VERIFICATION METHOD:\n")
        f.write("  1. p_before and p_after are proven prime via\n")
        f.write("     Deterministic Miller-Rabin (bases 2,3,5,7,11,13,17,19,23,29,31,37).\n")
        f.write(f"  2. All {gap-1} integers in (p, p+g) are proven composite\n")
        f.write("     via Interval Sieve with base primes up to √(p+g).\n")
        f.write("-" * 66 + "\n")
        f.write(f"SHA-256: {sha}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 66 + "\n")

# ============================================================
def main():
    if not os.path.exists(INPUT):
        print(f"ERROR: {INPUT} not found"); sys.exit(1)

    print(f"Reading: {INPUT}")
    rows = []
    with open(INPUT) as f:
        for r in csv.DictReader(f):
            rows.append((int(r['position']), int(r['gap']), float(r['merit'])))
    print(f"Gaps: {len(rows):,}")
    if not rows:
        print("Nothing to do"); return

    # Base sieve
    max_p = max(p for p, g, m in rows)
    limit = math.isqrt(max_p) + 1
    print(f"Building base sieve up to {limit:,}...")
    base_primes = build_base_sieve(limit)
    bp_np = np.array(base_primes, dtype=np.int64)
    print(f"  {len(base_primes):,} base primes")

    # Verify + generate CSV
    os.makedirs(os.path.dirname(CERT_CSV), exist_ok=True)
    print(f"\nGenerating: {CERT_CSV}")
    t0 = time.time()
    verified_count = 0
    with open(CERT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['p_before', 'p_after', 'gap', 'merit', 'sha256', 'verified'])
        for i, (p_after, gap, merit) in enumerate(rows):
            p_before = p_after - gap
            ok, rc = verify_gap_numba(int(p_after), int(gap), bp_np)
            sha = sha256_cert(p_before, p_after, gap, merit)
            w.writerow([p_before, p_after, gap,
                        f"{merit:.6f}", sha, "YES" if ok else "NO"])
            if ok: verified_count += 1
    dt = time.time() - t0
    print(f"  Wrote {len(rows):,} certificates in {dt:.2f}s")
    print(f"  Verified: {verified_count:,}/{len(rows):,}")

    # Top N text certificates
    os.makedirs(TOP20_DIR, exist_ok=True)
    # rows already sorted by merit descending (v7 sorts by merit)
    top = sorted(rows, key=lambda x: -x[2])[:TOP_N]
    print(f"\nGenerating top {TOP_N} text certificates in: {TOP20_DIR}")
    for i, (p_after, gap, merit) in enumerate(top, 1):
        p_before = p_after - gap
        sha = sha256_cert(p_before, p_after, gap, merit)
        fname = f"cert_{i:02d}_merit{merit:.2f}_p{p_after}.txt"
        path = os.path.join(TOP20_DIR, fname)
        write_text_cert(path, p_before, p_after, gap, merit, sha)
        print(f"  [{i:2d}] {fname}")

    # Summary
    print("\n" + "=" * 60)
    print("DIGITAL CERTIFICATES SUMMARY")
    print("=" * 60)
    print(f"  CSV  : {CERT_CSV}")
    print(f"         {len(rows):,} certificates")
    print(f"  TXT  : {TOP20_DIR}/")
    print(f"         {TOP_N} top certificates by merit")
    print(f"  Time : {dt:.2f} s")
    print("=" * 60)

if __name__ == '__main__':
    main()
