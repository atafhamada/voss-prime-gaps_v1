#!/usr/bin/env python3
# ============================================================
# VOSS Master — Orchestrator
# ============================================================
# يقرأ src/voss_v7_live.cu، يعدل N و SEG_NUM و MAX_POS،
# يبني إلى bin/، ويشغل مع بث حي + التحقق من OEIS.
# ============================================================
import os, sys, subprocess, json, math, datetime, urllib.request, re

HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(HERE, '..'))
CUDA_TEMPL = os.path.join(HERE, 'voss_v7_live.cu')
CUDA_BUILD = os.path.join(ROOT, 'bin', 'voss_v7_live.cu')
BIN_FILE   = os.path.join(ROOT, 'bin', 'voss_v7_live')
DB_FILE    = os.path.join(ROOT, 'data', 'verification_db.json')
RESULTS    = os.path.join(ROOT, 'results')
OEIS_URL   = 'https://oeis.org/A006880/b006880.txt'

# ============================================================
# ⚡ الإعدادات — غيّر هنا فقط
# ============================================================
N_VALUE = 100000000000    # 10^12
SEG_NUM = 30000000000      # 3x10^10 (مناسب لـ A100-40GB)

# ============================================================
def ensure_db():
    if os.path.exists(DB_FILE):
        db = json.load(open(DB_FILE))
        if len(db.get('pi_of_10_pow_n', {})) >= 10:
            print('DB OK: ' + DB_FILE)
            return db
    print('Downloading ' + OEIS_URL)
    req = urllib.request.Request(OEIS_URL,
                                 headers={'User-Agent': 'VOSS/1.0'})
    raw = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
    pi_map = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        p = line.split()
        if len(p) == 2:
            try: pi_map[str(int(p[0]))] = int(p[1])
            except ValueError: pass
    db = {'source': 'OEIS A006880', 'url': OEIS_URL,
          'downloaded': datetime.datetime.now().isoformat(timespec='seconds'),
          'pi_of_10_pow_n': pi_map}
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    json.dump(db, open(DB_FILE, 'w'), indent=2)
    return db

def detect_gpu():
    gpu = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
        capture_output=True, text=True).stdout.strip()
    mem_gb = float(gpu.split(',')[1].strip().split()[0]) / 1024.0
    arch = 'sm_80' if mem_gb >= 35 else 'sm_75'
    return gpu, mem_gb, arch

def patch_cuda(src, n, seg, max_pos):
    """يستبدل N و SEG_NUM و MAX_POS في كود CUDA"""
    # N: يدعم كلا النمطين (const int64_t N = ...LL أو static const int64_t N = ...;)
    src = re.sub(r'(const\s+int64_t\s+N\s*=\s*)\d+(LL)',
                 rf'\g<1>{n}\g<2>', src)
    src = re.sub(r'(static\s+const\s+int64_t\s+N\s*=\s*)\d+(;)',
                 rf'\g<1>{n}\g<2>', src)
    # SEG_NUM
    src = re.sub(r'(const\s+int64_t\s+SEG_NUM\s*=\s*)\d+(LL)',
                 rf'\g<1>{seg}\g<2>', src)
    src = re.sub(r'(static\s+const\s+int64_t\s+SEG_NUM\s*=\s*)\d+(;)',
                 rf'\g<1>{seg}\g<2>', src)
    # MAX_POS
    src = re.sub(r'(static\s+const\s+int64_t\s+MAX_POS\s*=\s*)\d+(;)',
                 rf'\g<1>{max_pos}\g<2>', src)
    src = re.sub(r'(const\s+int64_t\s+MAX_POS\s*=\s*)\d+(;)',
                 rf'\g<1>{max_pos}\g<2>', src)
    # fallback: إذا لم يجد MAX_POS، استبدل القيمة القديمة 500000000
    if 'static const int64_t MAX_POS' not in src and \
       'const int64_t MAX_POS' not in src:
        src = src.replace('500000000', str(max_pos), 1)
    return src

# ============================================================
def main():
    print("=" * 60)
    print("VOSS Master")
    print("=" * 60)
    print(f"ROOT  = {ROOT}")
    print(f"Templ = {CUDA_TEMPL}")
    print()

    if not os.path.exists(CUDA_TEMPL):
        print(f"ERROR: {CUDA_TEMPL} not found")
        print("Make sure src/voss_v7_live.cu exists")
        sys.exit(1)

    # قاعدة التحقق
    db = ensure_db()
    n_log10 = round(math.log10(N_VALUE))
    is_pow10 = abs(10**n_log10 - N_VALUE) / N_VALUE < 1e-9
    expected_pi = db['pi_of_10_pow_n'].get(str(n_log10)) if is_pow10 else None
    if expected_pi:
        print(f"pi(10^{n_log10}) from OEIS = {expected_pi:,}")
    print()

    # GPU
    gpu, mem_gb, arch = detect_gpu()
    print(f"GPU: {gpu}  arch={arch}")

    # MAX_POS المناسب
    max_primes_in_seg = int(1.3 * SEG_NUM / math.log(N_VALUE))
    max_pos_needed = max(max_primes_in_seg, 100_000_000)
    if mem_gb >= 70:   pos_budget_gb = 34.0
    elif mem_gb >= 35: pos_budget_gb = 20.0
    else:              pos_budget_gb = 8.0
    max_pos_avail = int(pos_budget_gb * 1e9 / 8)
    final_max_pos = min(max_pos_needed, max_pos_avail)

    print(f"N={N_VALUE:,}  SEG_NUM={SEG_NUM:,}  NUM_SEG={math.ceil(N_VALUE/SEG_NUM)}")
    print(f"MAX_POS={final_max_pos:,}  ({final_max_pos*8/1e9:.2f} GB)")
    print()

    # قراءة + patch + كتابة
    src = open(CUDA_TEMPL, 'r', encoding='utf-8').read()
    src = patch_cuda(src, N_VALUE, SEG_NUM, final_max_pos)
    os.makedirs(os.path.dirname(CUDA_BUILD), exist_ok=True)
    open(CUDA_BUILD, 'w', encoding='utf-8').write(src)
    print(f"Written: {CUDA_BUILD}")

    # البناء
    print(f"Building ({arch})...")
    r = subprocess.run(
        ['nvcc', '-O3', f'-arch={arch}', CUDA_BUILD, '-o', BIN_FILE],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED:\n", r.stderr)
        sys.exit(1)
    print(f"Built: {BIN_FILE}")
    print("=" * 60)

    # التشغيل
    os.makedirs(RESULTS, exist_ok=True)
    print(f"Running in: {RESULTS}\n")
    proc = subprocess.Popen([BIN_FILE], cwd=RESULTS,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0)
    captured = []
    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch: break
            t = ch.decode('utf-8', errors='replace')
            captured.append(t)
            sys.stdout.write(t); sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()
    proc.wait()
    full = ''.join(captured)

    # التحقق التلقائي
    print("\n" + "=" * 60)
    print("=== AUTO-VERIFICATION vs OEIS A006880 ===")
    print("=" * 60)
    m = re.search(r'Total primes:\s*(\d+)', full)
    actual = int(m.group(1)) if m else None
    if actual and expected_pi:
        print(f"OEIS pi(10^{n_log10}) = {expected_pi:>25,}")
        print(f"VOSS pi(10^{n_log10}) = {actual:>25,}")
        d = actual - expected_pi
        print(f"Diff = {d:+,}")
        print("PASS" if d == 0 else f"FAIL ({d:,})")
    else:
        print("Verification skipped (not a power of 10, or parse failed).")

if __name__ == '__main__':
    main()
