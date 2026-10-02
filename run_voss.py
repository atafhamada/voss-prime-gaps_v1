#!/usr/bin/env python3
# ============================================================
# VOSS One-Click Runner
# ============================================================
# Usage:
#   1. Edit N_VALUE below (default: 10^12)
#   2. Run: python3 run_voss.py
# ============================================================

# ============================================================
# CONFIGURATION — change this only
# ============================================================
N_VALUE = 1000000000000   # 10^12
SEG_NUM = 30000000000     # 3x10^10 (recommended for A100-40GB)

# ============================================================
# Pipeline configuration
# ============================================================
SKIP_VERIFY       = False   # set True to skip Miller-Rabin verification
SKIP_CERTIFICATES = False   # set True to skip certificate generation
SKIP_HTML         = False   # set True to skip HTML report
SKIP_FIGURES      = False   # set True to skip figures

# ============================================================
import os, sys, subprocess, time, importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def section(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70, flush=True)

def check_deps():
    section("[0/8] Checking dependencies")
    needed = ["numpy", "pandas", "scipy", "matplotlib", "numba"]
    missing = []
    for pkg in needed:
        try:
            importlib.import_module(pkg)
            print(f"  OK {pkg}")
        except ImportError:
            print(f"  MISSING: {pkg} - installing...")
            missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("  Installed: " + ", ".join(missing))
    print("  All dependencies ready.")

def run_script(rel_path, label):
    section(label)
    path = os.path.join(HERE, rel_path)
    if not os.path.exists(path):
        print("  SKIP: not found: " + rel_path)
        return 0
    t0 = time.time()
    # Use shell exec for binaries, python for scripts
    if path.endswith(".py"):
        cmd = [sys.executable, path]
    else:
        # C++ binary: pass default args
        cmd = [path,
               "results/large_gaps.csv",
               "results/verification_details.csv",
               "results/verification_cache.csv"]
    proc = subprocess.Popen(cmd,
                            cwd=HERE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            bufsize=0)
    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            sys.stdout.write(ch.decode("utf-8", errors="replace"))
            sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()
    proc.wait()
    dt = time.time() - t0
    print(f"\n  [{label}] finished in {dt:.1f}s (rc={proc.returncode})")
    return proc.returncode

def update_master():
    """Update voss_master.py with the chosen N and SEG_NUM."""
    import re
    master = os.path.join(HERE, "src", "voss_master.py")
    if not os.path.exists(master):
        print("ERROR: src/voss_master.py not found")
        return False
    src = open(master).read()
    src = re.sub(r"(N_VALUE\s*=\s*)\d+", r"\g<1>" + str(N_VALUE), src, count=1)
    src = re.sub(r"(SEG_NUM\s*=\s*)\d+", r"\g<1>" + str(SEG_NUM), src, count=1)
    open(master, "w").write(src)
    print(f"  Updated voss_master.py: N={N_VALUE:,}  SEG_NUM={SEG_NUM:,}")
    return True

def summary():
    section("[8/8] Pipeline Summary")
    res = os.path.join(HERE, "results")
    figs = os.path.join(HERE, "figures")
    print("  Generated artifacts:")
    items = [
        ("results/gap_histogram.csv",      "gap histogram"),
        ("results/chebyshev.csv",          "Chebyshev bias"),
        ("results/hl_trend.csv",           "P(6)/P(2) trend"),
        ("results/large_gaps.csv",         "high-merit gaps"),
        ("results/verification_report.json","verification report"),
        ("results/certificates.csv",       "digital certificates"),
        ("results/verification_report.html","HTML report"),
        ("figures/gap_distribution.png",   "figure: distribution"),
        ("figures/hl_trend.png",           "figure: HL trend"),
        ("figures/chebyshev_bias.png",     "figure: Chebyshev"),
        ("figures/top_merit_gaps.png",     "figure: top merit"),
        ("figures/cramer_conjecture.png",  "figure: Cramer"),
        ("figures/gap_stats_across_N.png", "figure: stats across N"),
    ]
    for rel, desc in items:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            sz = os.path.getsize(p)
            print(f"    OK  {rel:45s}  {sz:>10,} bytes  ({desc})")
        else:
            print(f"    --  {rel:45s}  (not generated)")

def main():
    global SKIP_VERIFY, SKIP_CERTIFICATES, SKIP_HTML, SKIP_FIGURES
    t_start = time.time()
    print("=" * 70)
    print("  VOSS One-Click Pipeline")
    print(f"  N = {N_VALUE:,}")
    print(f"  SEG_NUM = {SEG_NUM:,}")
    print(f"  NUM_SEG = {(N_VALUE + SEG_NUM - 1) // SEG_NUM}")
    print("=" * 70, flush=True)

    check_deps()

    section("[1/8] Configuring VOSS")
    if not update_master():
        sys.exit(1)

    if run_script("src/voss_master.py", "[2/8] VOSS main sieve") != 0:
        print("\nVOSS failed. Stopping pipeline.")
        sys.exit(1)

    # Check if large_gaps.csv has data
    lg_path = os.path.join(HERE, "results", "large_gaps.csv")
    lg_has_data = False
    if os.path.exists(lg_path):
        with open(lg_path) as _f:
            n_data = sum(1 for _ in _f) - 1  # minus header
        lg_has_data = (n_data > 0)
        print(f"\n  large_gaps.csv: {n_data} entries")
    else:
        print(f"\n  large_gaps.csv: not found (small N?)")
    
    if not lg_has_data:
        print("  → Skipping verification/certificates (no gaps >= 500)")
        SKIP_VERIFY = True
        SKIP_CERTIFICATES = True
    
    if not SKIP_VERIFY:
        # Build C++ verifier if not exists
        cpp_src = os.path.join(HERE, "src", "verify_cpp.cpp")
        cpp_bin = os.path.join(HERE, "bin", "verify_cpp")
        if os.path.exists(cpp_src):
            if not os.path.exists(cpp_bin) or \
               os.path.getmtime(cpp_src) > os.path.getmtime(cpp_bin):
                section("[3a/8] Building C++ verifier")
                r = subprocess.run(
                    ["g++", "-O3", "-march=native", "-fopenmp",
                     "-o", cpp_bin, cpp_src],
                    capture_output=True, text=True)
                if r.returncode != 0:
                    print("C++ build failed, falling back to Python verifier")
                    run_script("scripts/verify_certificates.py", "[3/8] Verification")
                else:
                    print("  OK: " + cpp_bin)
                    run_script("bin/verify_cpp", "[3/8] Verification (C++)")
            else:
                run_script("bin/verify_cpp", "[3/8] Verification (C++, cached)")
        else:
            run_script("scripts/verify_certificates.py", "[3/8] Verification")

    if not SKIP_CERTIFICATES:
        run_script("scripts/generate_certificates.py", "[4/8] Certificates")

    if not SKIP_HTML:
        run_script("scripts/generate_html_report.py", "[5/8] HTML report")

    if not SKIP_FIGURES:
        run_script("scripts/generate_figures.py", "[6/8] Figures")

    # ============ Analyses (Phase 4) ============
    section("[7/8] Mathematical analyses")

    analyses = [
        ("scripts/chi_square_test.py",       "Chi-square (Poisson vs GUE)"),
        ("scripts/cramer_test.py",           "Cramer conjecture"),
        ("scripts/jumping_champions.py",     "Jumping champions"),
        ("scripts/cimpeanu_test.py",         "Cimpeanu law"),
        ("scripts/anomalous_gaps.py",        "Anomalous gaps"),
        ("scripts/arithmetic_modulation.py", "Arithmetic modulation"),
    ]
    for script, label in analyses:
        run_script(script, "   " + label)

    summary()

    total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  Pipeline complete. Total time: {total:.1f}s ({total/60:.2f} min)")
    print("=" * 70)

if __name__ == "__main__":
    main()
