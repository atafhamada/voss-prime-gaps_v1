
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA err %s:%d: %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)

#define R_N 480
#define WHEEL 2310
#define MAX_GAP 100000
#define SHARED_HIST_SIZE 10000
#define HIST_BLOCK 256
#define LARGE_GAP_THRESHOLD 500
#define MAX_LARGE_GAPS 2000000

__constant__ int R2310_DEV[R_N] = { 1,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,169,173,179,181,191,193,197,199,211,221,223,227,229,233,239,241,247,251,257,263,269,271,277,281,283,289,293,299,307,311,313,317,323,331,337,347,349,353,359,361,367,373,377,379,383,389,391,397,401,403,409,419,421,431,433,437,439,443,449,457,461,463,467,479,481,487,491,493,499,503,509,521,523,527,529,533,541,547,551,557,559,563,569,571,577,587,589,593,599,601,607,611,613,617,619,629,631,641,643,647,653,659,661,667,673,677,683,689,691,697,701,703,709,713,719,727,731,733,739,743,751,757,761,767,769,773,779,787,793,797,799,809,811,817,821,823,827,829,839,841,851,853,857,859,863,871,877,881,883,887,893,899,901,907,911,919,923,929,937,941,943,947,949,953,961,967,971,977,983,989,991,997,1003,1007,1009,1013,1019,1021,1027,1031,1033,1037,1039,1049,1051,1061,1063,1069,1073,1079,1081,1087,1091,1093,1097,1103,1109,1117,1121,1123,1129,1139,1147,1151,1153,1157,1159,1163,1171,1181,1187,1189,1193,1201,1207,1213,1217,1219,1223,1229,1231,1237,1241,1247,1249,1259,1261,1271,1273,1277,1279,1283,1289,1291,1297,1301,1303,1307,1313,1319,1321,1327,1333,1339,1343,1349,1357,1361,1363,1367,1369,1373,1381,1387,1391,1399,1403,1409,1411,1417,1423,1427,1429,1433,1439,1447,1451,1453,1457,1459,1469,1471,1481,1483,1487,1489,1493,1499,1501,1511,1513,1517,1523,1531,1537,1541,1543,1549,1553,1559,1567,1571,1577,1579,1583,1591,1597,1601,1607,1609,1613,1619,1621,1627,1633,1637,1643,1649,1651,1657,1663,1667,1669,1679,1681,1691,1693,1697,1699,1703,1709,1711,1717,1721,1723,1733,1739,1741,1747,1751,1753,1759,1763,1769,1777,1781,1783,1787,1789,1801,1807,1811,1817,1819,1823,1829,1831,1843,1847,1849,1853,1861,1867,1871,1873,1877,1879,1889,1891,1901,1907,1909,1913,1919,1921,1927,1931,1933,1937,1943,1949,1951,1957,1961,1963,1973,1979,1987,1993,1997,1999,2003,2011,2017,2021,2027,2029,2033,2039,2041,2047,2053,2059,2063,2069,2071,2077,2081,2083,2087,2089,2099,2111,2113,2117,2119,2129,2131,2137,2141,2143,2147,2153,2159,2161,2171,2173,2179,2183,2197,2201,2203,2207,2209,2213,2221,2227,2231,2237,2239,2243,2249,2251,2257,2263,2267,2269,2273,2279,2281,2287,2291,2293,2297,2309 };
static const int R2310_HOST[R_N] = {1,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,169,173,179,181,191,193,197,199,211,221,223,227,229,233,239,241,247,251,257,263,269,271,277,281,283,289,293,299,307,311,313,317,323,331,337,347,349,353,359,361,367,373,377,379,383,389,391,397,401,403,409,419,421,431,433,437,439,443,449,457,461,463,467,479,481,487,491,493,499,503,509,521,523,527,529,533,541,547,551,557,559,563,569,571,577,587,589,593,599,601,607,611,613,617,619,629,631,641,643,647,653,659,661,667,673,677,683,689,691,697,701,703,709,713,719,727,731,733,739,743,751,757,761,767,769,773,779,787,793,797,799,809,811,817,821,823,827,829,839,841,851,853,857,859,863,871,877,881,883,887,893,899,901,907,911,919,923,929,937,941,943,947,949,953,961,967,971,977,983,989,991,997,1003,1007,1009,1013,1019,1021,1027,1031,1033,1037,1039,1049,1051,1061,1063,1069,1073,1079,1081,1087,1091,1093,1097,1103,1109,1117,1121,1123,1129,1139,1147,1151,1153,1157,1159,1163,1171,1181,1187,1189,1193,1201,1207,1213,1217,1219,1223,1229,1231,1237,1241,1247,1249,1259,1261,1271,1273,1277,1279,1283,1289,1291,1297,1301,1303,1307,1313,1319,1321,1327,1333,1339,1343,1349,1357,1361,1363,1367,1369,1373,1381,1387,1391,1399,1403,1409,1411,1417,1423,1427,1429,1433,1439,1447,1451,1453,1457,1459,1469,1471,1481,1483,1487,1489,1493,1499,1501,1511,1513,1517,1523,1531,1537,1541,1543,1549,1553,1559,1567,1571,1577,1579,1583,1591,1597,1601,1607,1609,1613,1619,1621,1627,1633,1637,1643,1649,1651,1657,1663,1667,1669,1679,1681,1691,1693,1697,1699,1703,1709,1711,1717,1721,1723,1733,1739,1741,1747,1751,1753,1759,1763,1769,1777,1781,1783,1787,1789,1801,1807,1811,1817,1819,1823,1829,1831,1843,1847,1849,1853,1861,1867,1871,1873,1877,1879,1889,1891,1901,1907,1909,1913,1919,1921,1927,1931,1933,1937,1943,1949,1951,1957,1961,1963,1973,1979,1987,1993,1997,1999,2003,2011,2017,2021,2027,2029,2033,2039,2041,2047,2053,2059,2063,2069,2071,2077,2081,2083,2087,2089,2099,2111,2113,2117,2119,2129,2131,2137,2141,2143,2147,2153,2159,2161,2171,2173,2179,2183,2197,2201,2203,2207,2209,2213,2221,2227,2231,2237,2239,2243,2249,2251,2257,2263,2267,2269,2273,2279,2281,2287,2291,2293,2297,2309};


struct LargeGap { uint64_t position; int32_t gap; int32_t _pad; };

static const int64_t N       = 100000000000;
static const int64_t SEG_NUM = 30030000000;
static const int64_t SEG_BITS = 6240000000;
static const int64_t MAX_POS = 1541311116;

static void gen_base_primes(int limit, uint32_t** out, int* cnt) {
    bool* s = (bool*)malloc(limit+1);
    for (int i=0;i<=limit;i++) s[i]=true;
    s[0]=s[1]=false;
    for (int i=2;(long long)i*i<=limit;i++) if (s[i])
        for (int j=i*i;j<=limit;j+=i) s[j]=false;
    int c=0; for (int i=13;i<=limit;i++) if (s[i]) c++;
    uint32_t* a=(uint32_t*)malloc(c*sizeof(uint32_t));
    int k=0; for (int i=13;i<=limit;i++) if (s[i]) a[k++]=i;
    free(s); *out=a; *cnt=c;
}

static int64_t modinv(int64_t a, int64_t m) {
    int64_t t=0, nt=1, r=m, nr=a%m;
    while (nr!=0) { int64_t q=r/nr, tmp=nt; nt=t-q*nt; t=tmp;
        tmp=nr; nr=r-q*nr; r=tmp; }
    if (t<0) t+=m; return t;
}

// ============================================================
// Sieve kernel: 480 threads, one per residue
// ============================================================
__global__ void sieve_w2310_kernel(
    uint32_t* __restrict__ bits,
    uint32_t p,
    int64_t inv2310,
    int64_t k_base,
    int64_t min_k,
    int64_t seg_bits, int64_t chunk_steps)
{
    int r_idx = threadIdx.x;
    int r = R2310_DEV[r_idx];
    int64_t avail = seg_bits - 1 - r_idx;
    if (avail < 0) return;
    int64_t k_mod = ((int64_t)((-r % (int)p + (int)p) % (int)p) * inv2310) % (int64_t)p;
    int64_t delta = ((k_mod - min_k) % (int64_t)p + (int64_t)p) % (int64_t)p;
    int64_t k_start = min_k + delta;
    if (k_start == 0 && (int64_t)r == (int64_t)p) k_start += p;
    int64_t k_max = k_base + avail / R_N;
    int64_t total = (k_max - k_start) / (int64_t)p + 1;
    if (total <= 0) return;
    int64_t sb = (int64_t)blockIdx.x * chunk_steps;
    int64_t se = sb + chunk_steps;
    if (se > total) se = total;
    for (int64_t s = sb; s < se; s++) {
        int64_t k = k_start + s * (int64_t)p;
        int64_t lk = k - k_base;
        int64_t bit_idx = lk * R_N + r_idx;
        if (bit_idx >= seg_bits) break;
        uint32_t w = (uint32_t)(bit_idx >> 5);
        uint32_t b = (uint32_t)(bit_idx & 31);
        atomicAnd(&bits[w], ~(1u << b));
    }
}

// ============================================================
// Extract: bit index → 32-bit word positions
// ============================================================
__global__ void extract_w2310_kernel(
    const uint32_t* __restrict__ bits, int64_t nwords, int64_t seg_bits,
    uint64_t* __restrict__ positions, uint64_t* __restrict__ gcount)
{
    int64_t widx = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (widx >= nwords) return;
    uint32_t word = bits[widx]; if (!word) return;
    int64_t base_idx = widx * 32;
    int valid = 0; uint32_t w = word;
    while (w) { int bit = __ffs(w)-1; w &= w-1;
                if (base_idx+bit < seg_bits) valid++; }
    if (!valid) return;
    uint64_t base = atomicAdd((unsigned long long*)gcount, (unsigned long long)valid);
    w = word;
    while (w) { int bit = __ffs(w)-1; w &= w-1;
                int64_t idx = base_idx+bit;
                if (idx < seg_bits) positions[base++] = (uint64_t)idx; }
}

// ============================================================
// Helper: bit_idx → value
// ============================================================
__device__ __forceinline__ uint64_t bit_to_value(int64_t idx, int64_t k_base) {
    int64_t k = idx / R_N;
    int r = R2310_DEV[idx % R_N];
    return 2310ULL * (k + k_base) + (uint64_t)r;
}

__global__ void mod4_w2310_kernel(const uint64_t* __restrict__ pos,
    uint64_t n, uint64_t k_base,
    unsigned long long* __restrict__ c1, unsigned long long* __restrict__ c3)
{
    uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t v = bit_to_value((int64_t)pos[i], k_base);
    uint32_t r = (uint32_t)(v & 3);
    if (r == 1) atomicAdd(c1, 1ULL);
    else if (r == 3) atomicAdd(c3, 1ULL);
}

__global__ void gaps_w2310_kernel(const uint64_t* __restrict__ pos,
    uint64_t n, uint64_t k_base, int64_t* __restrict__ hist,
    LargeGap* __restrict__ lg, unsigned int* __restrict__ lgc,
    int lg_thr, int lg_max)
{
    __shared__ int lh[SHARED_HIST_SIZE];
    int tid = threadIdx.x;
    for (int i = tid; i < SHARED_HIST_SIZE; i += HIST_BLOCK) lh[i] = 0;
    __syncthreads();
    uint64_t i = (uint64_t)blockIdx.x*HIST_BLOCK + tid;
    if (i > 0 && i < n) {
        uint64_t v1 = bit_to_value((int64_t)pos[i],   k_base);
        uint64_t v0 = bit_to_value((int64_t)pos[i-1], k_base);
        uint64_t d = v1 - v0;
        if (d > 0) {
            if (d < (uint64_t)SHARED_HIST_SIZE) atomicAdd(&lh[d], 1);
            else if (d < (uint64_t)MAX_GAP) atomicAdd((unsigned long long*)&hist[d], 1ULL);
            if (d >= (uint64_t)lg_thr && d < (uint64_t)MAX_GAP) {
                unsigned s = atomicAdd(lgc, 1u);
                if (s < (unsigned)lg_max) { lg[s].position=v1; lg[s].gap=(int32_t)d; }
            }
        }
    }
    __syncthreads();
    for (int j = tid; j < SHARED_HIST_SIZE; j += HIST_BLOCK)
        if (lh[j] > 0) atomicAdd((unsigned long long*)&hist[j], (unsigned long long)lh[j]);
}

static void print_progress(int64_t seg, int64_t nseg, uint64_t tp, double el) {
    int pc = (int)((seg*100)/nseg);
    double rate = (el>0.1) ? ((double)seg/el) : 0;
    double eta = (rate>0) ? ((double)(nseg-seg)/rate) : 0;
    printf("\r[%3d%%] seg %lld/%lld | primes: %-13llu | elapsed: %5.1f s | ETA: %5.1f s   ",
           pc, (long long)seg, (long long)nseg, (unsigned long long)tp, el, eta);
    fflush(stdout);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    int64_t SEG_K = SEG_NUM / 2310;
    int64_t NUM_SEG = (N + SEG_NUM - 1) / SEG_NUM;

    printf("=== VOSS v8 Wheel-2310 ===\n");
    printf("N=%lld  SEG_NUM=%lld  SEG_BITS=%lld  NUM_SEG=%lld\n",
           (long long)N, (long long)SEG_NUM, (long long)SEG_BITS, (long long)NUM_SEG);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.1f GB)\n\n", prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem/1e9);

    int limit = (int)sqrt((double)N) + 1;
    uint32_t* base_h; int base_count;
    gen_base_primes(limit, &base_h, &base_count);
    printf("Base primes (>=13) up to %d : %d\n\n", limit, base_count);

    int64_t seg_words = (SEG_BITS + 31) / 32;
    size_t bits_bytes = seg_words * sizeof(uint32_t);
    size_t pos_bytes  = (size_t)MAX_POS * sizeof(uint64_t);
    size_t lg_bytes   = (size_t)MAX_LARGE_GAPS * sizeof(LargeGap);

    printf("Bit array: %.2f GB, Positions: %.2f GB\n\n",
           bits_bytes/1e9, pos_bytes/1e9);

    uint32_t *bits_d, *bp_d;
    uint64_t *pos_d, *cnt_d;
    int64_t *gaps_d; LargeGap *lg_d; unsigned int *lgc_d;
    unsigned long long *m1_d, *m3_d;

    CUDA_CHECK(cudaMalloc(&bits_d, bits_bytes));
    CUDA_CHECK(cudaMalloc(&bp_d, base_count*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&pos_d, pos_bytes));
    CUDA_CHECK(cudaMalloc(&cnt_d, 8));
    CUDA_CHECK(cudaMalloc(&gaps_d, MAX_GAP*8));
    CUDA_CHECK(cudaMalloc(&lg_d, lg_bytes));
    CUDA_CHECK(cudaMalloc(&lgc_d, 4));
    CUDA_CHECK(cudaMalloc(&m1_d, 8));
    CUDA_CHECK(cudaMalloc(&m3_d, 8));
    CUDA_CHECK(cudaMemcpy(bp_d, base_h, base_count*4, cudaMemcpyHostToDevice));

    // precompute inv2310 for all base primes
    int64_t* inv_h = (int64_t*)malloc(base_count*8);
    for (int i = 0; i < base_count; i++)
        inv_h[i] = modinv(2310 % base_h[i], base_h[i]);
    int64_t* inv_d;
    CUDA_CHECK(cudaMalloc(&inv_d, base_count*8));
    CUDA_CHECK(cudaMemcpy(inv_d, inv_h, base_count*8, cudaMemcpyHostToDevice));
    /* inv_h kept alive until end */

    int64_t* gaps_total = (int64_t*)calloc(MAX_GAP, 8);
    uint64_t last_prime = 13, total_primes = 5;  // 2,3,5,7,11
    unsigned long long c1t = 0, c3t = 0;
    double t_sieve=0, t_extract=0, t_sort=0, t_gaps=0, t_mod4=0;

    const int BLOCK = 256;
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int64_t seg = 0; seg < NUM_SEG; seg++) {
        int64_t seg_high = (seg+1)*SEG_NUM;
        if (seg_high > N) seg_high = N;
        int64_t k_base = seg * SEG_K;
        int64_t seg_bits = SEG_BITS;
        if (seg == NUM_SEG-1) {
            int64_t r_num = N - seg*SEG_NUM;
            int64_t r_k = r_num / 2310;
            int64_t r_r = r_num % 2310;
            seg_bits = r_k * 480;
            for (int i = 0; i < 480; i++) if (R2310_HOST[i] <= r_r) seg_bits++;
        }
        int64_t cur_words = (seg_bits + 31) / 32;

        auto t_seg = std::chrono::high_resolution_clock::now();

        std::vector<uint32_t> lp; std::vector<int64_t> li, lmin, lkb;
        for (int i = 0; i < base_count; i++) {
            uint32_t p = base_h[i];
            int64_t p_sq = (int64_t)p*p;
            if (p_sq > seg_high) break;
            int64_t min_k = p_sq / 2310;
            if (min_k < k_base) min_k = k_base;
            lp.push_back(p); li.push_back(i); lmin.push_back(min_k); lkb.push_back(k_base);
        }

        if (seg == 0) {
            sieve_w2310_kernel<<<1, 480, 0, stream>>>(bits_d, 13, 1, 0, 0, 0, 1);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaGraph_t gr; cudaGraphExec_t ge;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (size_t i = 0; i < lp.size(); i++) {
            {
                uint32_t p = lp[i];
                int64_t min_k = lmin[i];
                int64_t max_steps_p = 0;
                for (int r_idx = 0; r_idx < 480; r_idx++) {
                    int r = R2310_HOST[r_idx];
                    int64_t avail = seg_bits - 1 - r_idx;
                    if (avail < 0) continue;
                    int64_t k_mod = ((int64_t)((-r % (int)p + (int)p) % (int)p) * inv_h[i]) % (int64_t)p;
                    int64_t delta = ((k_mod - min_k) % (int64_t)p + (int64_t)p) % (int64_t)p;
                    int64_t k_start = min_k + delta;
                    if (k_start == 0 && (int64_t)r == (int64_t)p) k_start += p;
                    int64_t k_max = k_base + avail / R_N;
                    int64_t total = (k_max - k_start) / (int64_t)p + 1;
                    if (total > max_steps_p) max_steps_p = total;
                }
                if (max_steps_p <= 0) continue;
                int64_t grid_x = max_steps_p;
                int64_t chunk = (max_steps_p + grid_x - 1) / grid_x;
                sieve_w2310_kernel<<<(int)grid_x, 480, 0, stream>>>(
                    bits_d, p, inv_h[i], lkb[i], min_k, seg_bits, chunk);
            }
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &gr));
        CUDA_CHECK(cudaGraphInstantiate(&ge, gr, NULL, NULL, 0));

        CUDA_CHECK(cudaMemset(bits_d, 0xFF, cur_words*4));
        if (seg == 0) {
            // clear bit 0 in word 0 (value 1 is not prime)
            uint32_t fw = 0xFFFFFFFE;
            CUDA_CHECK(cudaMemcpy(bits_d, &fw, 4, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaGraphLaunch(ge, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        t_sieve += std::chrono::duration<double,std::milli>(t1-t_seg).count();
        cudaGraphExecDestroy(ge); cudaGraphDestroy(gr);

        CUDA_CHECK(cudaMemset(cnt_d, 0, 8));
        int eg = (int)((cur_words + BLOCK - 1)/BLOCK);
        extract_w2310_kernel<<<eg, BLOCK>>>(bits_d, cur_words, seg_bits, pos_d, cnt_d);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        t_extract += std::chrono::duration<double,std::milli>(t2-t1).count();

        uint64_t npos; CUDA_CHECK(cudaMemcpy(&npos, cnt_d, 8, cudaMemcpyDeviceToHost));
        thrust::device_ptr<uint64_t> ptr(pos_d);
        thrust::sort(ptr, ptr+npos);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t3 = std::chrono::high_resolution_clock::now();
        t_sort += std::chrono::duration<double,std::milli>(t3-t2).count();

        uint64_t fi=0, li2=0;
        if (npos>0) {
            CUDA_CHECK(cudaMemcpy(&fi, pos_d, 8, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&li2, pos_d+npos-1, 8, cudaMemcpyDeviceToHost));
        }
        // value of first/last in segment
        int64_t fk = fi / 480; int fr = R2310_HOST[fi % 480];
        int64_t lk = li2 / 480; int lr = R2310_HOST[li2 % 480];
        uint64_t fval = 2310ULL*(k_base+fk) + fr;
        uint64_t lval = 2310ULL*(k_base+lk) + lr;

        if (npos>0) {
            uint64_t d0 = fval - last_prime;
            if (d0>0 && d0<MAX_GAP) gaps_total[d0]++;
        }
        if (npos>0) {
            int mg = (int)((npos+BLOCK-1)/BLOCK);
            mod4_w2310_kernel<<<mg, BLOCK>>>(pos_d, npos, k_base, m1_d, m3_d);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        t_mod4 += std::chrono::duration<double,std::milli>(t4-t3).count();

        int gg = (int)((npos+HIST_BLOCK-1)/HIST_BLOCK);
        gaps_w2310_kernel<<<gg, HIST_BLOCK>>>(pos_d, npos, k_base, gaps_d,
            lg_d, lgc_d, LARGE_GAP_THRESHOLD, MAX_LARGE_GAPS);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t5 = std::chrono::high_resolution_clock::now();
        t_gaps += std::chrono::duration<double,std::milli>(t5-t4).count();

        int64_t* gs = (int64_t*)malloc(MAX_GAP*8);
        CUDA_CHECK(cudaMemcpy(gs, gaps_d, MAX_GAP*8, cudaMemcpyDeviceToHost));
        for (int i=1;i<MAX_GAP;i++) gaps_total[i]+=gs[i];
        free(gs);
        CUDA_CHECK(cudaMemset(gaps_d, 0, MAX_GAP*8));

        unsigned long long a1, a3;
        CUDA_CHECK(cudaMemcpy(&a1, m1_d, 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&a3, m3_d, 8, cudaMemcpyDeviceToHost));
        c1t += a1; c3t += a3;
        CUDA_CHECK(cudaMemset(m1_d, 0, 8)); CUDA_CHECK(cudaMemset(m3_d, 0, 8));

        if (npos>0) { last_prime = lval; total_primes += npos; }

        auto tn = std::chrono::high_resolution_clock::now();
        double el = std::chrono::duration<double>(tn-t0).count();
        print_progress(seg+1, NUM_SEG, total_primes, el);
    }
    printf("\n\n");

    auto te = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(te-t0).count();

    gaps_total[1] += 1;  // 2→3
    gaps_total[2] += 1;  // 3→5
    gaps_total[2] += 1;  // 5→7
    gaps_total[4] += 1;  // 7→11
    gaps_total[2] += 1;  // 11→13

    printf("=============================================================\n");
    printf("TOTAL TIME: %.2f s\n", total);
    printf("Total primes: %llu\n", (unsigned long long)total_primes);
    printf("=============================================================\n");

    printf("Sieve:    %8.2f s (%6.2f%%)\n", t_sieve/1000, 100*t_sieve/(total*1000));
    printf("Extract:  %8.2f s (%6.2f%%)\n", t_extract/1000, 100*t_extract/(total*1000));
    printf("Sort:     %8.2f s (%6.2f%%)\n", t_sort/1000, 100*t_sort/(total*1000));
    printf("Mod-4:    %8.2f s (%6.2f%%)\n", t_mod4/1000, 100*t_mod4/(total*1000));
    printf("Gaps:     %8.2f s (%6.2f%%)\n", t_gaps/1000, 100*t_gaps/(total*1000));
    double oth = total - (t_sieve+t_extract+t_sort+t_mod4+t_gaps)/1000;
    printf("Other:    %8.2f s (%6.2f%%)\n\n", oth, 100*oth/total);

    printf("Gap 1: %lld  Gap 2: %lld  Gap 4: %lld  Gap 6: %lld  Gap 30: %lld\n\n",
        (long long)gaps_total[1], (long long)gaps_total[2], (long long)gaps_total[4],
        (long long)gaps_total[6], (long long)gaps_total[30]);

    unsigned long long p41=c1t+1, p43=c3t+3;
    int64_t tgc=0, wsum=0;
    for (int g=1;g<MAX_GAP;g++) { tgc+=gaps_total[g]; wsum+=(int64_t)g*gaps_total[g]; }
    printf("Chebyshev diff = %lld\n", (long long)p43-(long long)p41);
    printf("Check1 (count): %s\n", (tgc==(int64_t)(total_primes-1))?"PASS":"FAIL");
    printf("Check2 (span):  %s\n", (wsum==(int64_t)(last_prime-2))?"PASS":"FAIL");
    printf("Check3 (cheb):  %s\n", ((p41+p43+1)==total_primes)?"PASS":"FAIL");

    FILE* fp = fopen("gap_histogram.csv", "w");
    if (fp) { fprintf(fp,"gap,count\n");
        for (int g=1;g<MAX_GAP;g++) if (gaps_total[g]>0)
            fprintf(fp,"%d,%lld\n",g,(long long)gaps_total[g]);
        fclose(fp); }

    cudaFree(bits_d); cudaFree(bp_d); cudaFree(pos_d); cudaFree(cnt_d);
    cudaFree(gaps_d); cudaFree(lg_d); cudaFree(lgc_d);
    cudaFree(m1_d); cudaFree(m3_d); cudaFree(inv_d);
    free(base_h); free(gaps_total); free(inv_h);
    return 0;
}
