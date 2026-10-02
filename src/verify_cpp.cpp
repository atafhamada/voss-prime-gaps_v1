
// ============================================================
// VOSS C++ Verifier with persistent cache
// ============================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <utility>
#include <omp.h>
#include <atomic>
#include <thread>

using u64  = uint64_t;
using u128 = __uint128_t;

static inline u64 mul_mod(u64 a, u64 b, u64 m) { return (u64)((u128)a * b % m); }

static inline u64 mod_pow(u64 base, u64 exp, u64 mod) {
    u64 result = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mul_mod(result, base, mod);
        base = mul_mod(base, base, mod);
        exp >>= 1;
    }
    return result;
}

static const u64 BASES[12] = {2,3,5,7,11,13,17,19,23,29,31,37};

static bool is_prime_u64(u64 n) {
    if (n < 2) return false;
    for (int i = 0; i < 12; i++) {
        if (n == BASES[i]) return true;
        if (n % BASES[i] == 0) return false;
    }
    u64 d = n - 1; int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    for (int i = 0; i < 12; i++) {
        u64 x = mod_pow(BASES[i], d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = mul_mod(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

static bool verify_gap(u64 p_after, int gap, const std::vector<uint32_t>& bp) {
    u64 p_before = p_after - (u64)gap;
    if (!is_prime_u64(p_before)) return false;
    if (!is_prime_u64(p_after))  return false;
    if (gap < 2) return true;

    int n_int = gap - 1;
    uint8_t stack_buf[1024];
    std::vector<uint8_t> heap_buf;
    uint8_t* is_comp;
    if (n_int <= 1024) is_comp = stack_buf;
    else { heap_buf.assign(n_int, 0); is_comp = heap_buf.data(); }
    std::memset(is_comp, 0, n_int);

    for (int i = 0; i < n_int; i += 2) is_comp[i] = 1;

    u64 sq = (u64)sqrtl((long double)p_after);
    while ((sq + 1) * (sq + 1) <= p_after) sq++;
    while (sq * sq > p_after) sq--;

    for (size_t k = 0; k < bp.size(); k++) {
        u64 q = bp[k];
        if (q == 2) continue;
        if (q > sq) break;
        u64 start = (p_before / q + 1) * q;
        if ((start & 1) == 0) start += q;
        if (start >= p_after) continue;
        int64_t idx = (int64_t)(start - p_before - 1);
        if (idx < 0) idx = 0;
        int step = (int)(2 * q);
        for (int i = (int)idx; i < n_int; i += step) is_comp[i] = 1;
    }
    for (int i = 0; i < n_int; i++) if (!is_comp[i]) return false;
    return true;
}

static std::vector<uint32_t> build_base_sieve(u64 limit) {
    std::vector<uint8_t> s(limit + 1, 1);
    s[0] = s[1] = 0;
    for (u64 i = 2; i * i <= limit; i++)
        if (s[i]) for (u64 j = i * i; j <= limit; j += i) s[j] = 0;
    std::vector<uint32_t> primes;
    for (u64 i = 2; i <= limit; i++) if (s[i]) primes.push_back((uint32_t)i);
    return primes;
}

// ---------- pair hash ----------
struct PairHash {
    size_t operator()(const std::pair<u64,int>& p) const {
        return std::hash<u64>()(p.first) ^ (std::hash<int>()(p.second) * 0x9e3779b97f4a7c15ULL);
    }
};

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* input      = (argc > 1) ? argv[1] : "results/large_gaps.csv";
    const char* output     = (argc > 2) ? argv[2] : "results/verification_details.csv";
    const char* cache_path = (argc > 3) ? argv[3] : "results/verification_cache.csv";

    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- Read gaps ----
    std::ifstream fin(input);
    if (!fin) { fprintf(stderr, "cannot open %s\n", input); return 1; }

    std::vector<std::pair<u64,int>> gaps;
    std::vector<double> merits;
    std::string line;
    std::getline(fin, line);
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        size_t c1 = line.find(',');
        if (c1 == std::string::npos) continue;
        size_t c2 = line.find(',', c1 + 1);
        u64 p = std::stoull(line.substr(0, c1));
        int g = std::stoi(line.substr(c1 + 1, c2 - c1 - 1));
        double m = (c2 != std::string::npos) ? std::stod(line.substr(c2 + 1)) : 0.0;
        gaps.push_back({p, g});
        merits.push_back(m);
    }
    fin.close();
    int n = (int)gaps.size();
    printf("Read %d gaps from %s\n", n, input);
    if (n == 0) return 1;

    // ---- Load cache ----
    std::unordered_set<std::pair<u64,int>, PairHash> cache;
    {
        std::ifstream cf(cache_path);
        if (cf) {
            std::string cl;
            std::getline(cf, cl);  // header
            while (std::getline(cf, cl)) {
                if (cl.empty()) continue;
                size_t c = cl.find(',');
                if (c == std::string::npos) continue;
                u64 p = std::stoull(cl.substr(0, c));
                int g = std::stoi(cl.substr(c+1));
                cache.insert({p, g});
            }
            printf("Loaded %zu cached verifications from %s\n", cache.size(), cache_path);
        } else {
            printf("No cache yet — will create: %s\n", cache_path);
        }
    }

    // ---- Identify new gaps (not in cache) ----
    std::vector<int> to_verify;
    std::vector<int> cached_ok(n, 0);
    int cached_count = 0;
    for (int i = 0; i < n; i++) {
        if (cache.count(gaps[i])) {
            cached_ok[i] = 1;
            cached_count++;
        } else {
            to_verify.push_back(i);
        }
    }
    printf("Cached: %d, To verify: %zu\n", cached_count, to_verify.size());

    // ---- Base sieve ----
    u64 max_p = 0;
    for (auto& g : gaps) if (g.first > max_p) max_p = g.first;
    u64 limit = (u64)sqrtl((long double)max_p) + 1;
    printf("Base sieve up to %llu...\n", (unsigned long long)limit);
    auto bp = build_base_sieve(limit);
    printf("  %zu base primes\n", bp.size());

    // ---- Parallel verification of NEW gaps ----
    int nthreads = omp_get_max_threads();
    printf("Verifying using %d threads...\n", nthreads);

    std::vector<uint8_t> ok(n, 0);
    for (int i = 0; i < n; i++) if (cached_ok[i]) ok[i] = 1;

    // Long-running progress: check from a monitor thread
    std::atomic<size_t> progress(0);
    std::thread monitor([&]() {
        auto t_start = std::chrono::high_resolution_clock::now();
        while (progress.load() < to_verify.size()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            size_t p = progress.load();
            if (p >= to_verify.size()) break;
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double rate = (elapsed > 0.1) ? (double)p / elapsed : 0;
            double eta = (rate > 0) ? (double)(to_verify.size() - p) / rate : 0;
            printf("  [%zu/%zu] %.1f%%  elapsed=%.0fs  ETA=%.0fs\n",
                   p, to_verify.size(), 100.0*p/to_verify.size(), elapsed, eta);
            fflush(stdout);
        }
    });

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t k = 0; k < to_verify.size(); k++) {
        int i = to_verify[k];
        ok[i] = verify_gap(gaps[i].first, gaps[i].second, bp) ? 1 : 0;
        progress.fetch_add(1, std::memory_order_relaxed);
    }

    monitor.join();

    // ---- Count ----
    long long verified = 0, failed = 0;
    for (int i = 0; i < n; i++) {
        if (ok[i]) verified++; else failed++;
    }

    // ---- Append new verified to cache ----
    {
        std::ifstream check(cache_path);
        bool exists = check.good();
        check.close();

        std::ofstream cf(cache_path, std::ios::app);
        if (!exists) cf << "p_after,gap\n";
        size_t appended = 0;
        for (size_t k = 0; k < to_verify.size(); k++) {
            int i = to_verify[k];
            if (ok[i]) {
                cf << gaps[i].first << "," << gaps[i].second << "\n";
                appended++;
            }
        }
        printf("Appended %zu new verifications to cache\n", appended);
    }

    // ---- Write details CSV ----
    std::ofstream fout(output);
    fout << "position,gap,merit,verified\n";
    char buf[160];
    for (int i = 0; i < n; i++) {
        snprintf(buf, sizeof(buf), "%llu,%d,%.6f,%d",
                 (unsigned long long)gaps[i].first, gaps[i].second,
                 merits[i], (int)ok[i]);
        fout << buf << "\n";
    }
    fout.close();

    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    printf("\n");
    printf("============================================================\n");
    printf("C++ VERIFIER RESULT (cached)\n");
    printf("============================================================\n");
    printf("Total gaps:    %d\n", n);
    printf("Cached:        %d  (%.1f%%)\n", cached_count, 100.0 * cached_count / n);
    printf("Newly verify:  %zu\n", to_verify.size());
    printf("Verified:      %lld\n", verified);
    printf("Failed:        %lld\n", failed);
    printf("Threads:       %d\n", nthreads);
    printf("Duration:      %.2f s\n", dt);
    printf("Details:       %s\n", output);
    printf("Cache:         %s\n", cache_path);
    printf("============================================================\n");
    return (failed > 0) ? 1 : 0;
}
