
// ============================================================
// VOSS-Wheel30 v7 LIVE
// ============================================================
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
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); } } while (0)

#define MAX_GAP 100000
#define SHARED_HIST_SIZE 10000
#define MODQ_MAX_GAP 100
#define MODQ_NUM_Q6 2
#define MODQ_NUM_Q30 8
#define MODQ_NUM_TOTAL 10

__constant__ int W30_TO_Q6_IDX[8]  = {0, 0, 1, 0, 1, 0, 1, 1};
__constant__ int W30_TO_Q30_IDX[8] = {0, 1, 2, 3, 4, 5, 6, 7};
#define HIST_BLOCK 256
#define LARGE_GAP_THRESHOLD 500
#define MAX_LARGE_GAPS 5000000

__constant__ int W30_DEV[8] = {1, 7, 11, 13, 17, 19, 23, 29};
static const int W30[8] = {1, 7, 11, 13, 17, 19, 23, 29};

struct LargeGap { uint64_t position; int32_t gap; float merit; };

#define CKPT_MAGIC  0x564F53534F4D4ELL

struct CheckpointState {
    int64_t magic;
    int64_t N;
    int64_t SEG_NUM;
    int64_t next_segment;
    uint64_t last_prime;
    uint64_t total_primes;
    unsigned long long class1_total;
    unsigned long long class3_total;
    int64_t gaps_total[MAX_GAP];
};

static bool load_checkpoint(const char* path, int64_t N_expected,
                            int64_t SEG_NUM_expected,
                            CheckpointState* out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    size_t n = fread(out, sizeof(CheckpointState), 1, f);
    fclose(f);
    if (n != 1) return false;
    if (out->magic != CKPT_MAGIC) return false;
    if (out->N != N_expected) return false;
    if (out->SEG_NUM != SEG_NUM_expected) return false;
    return true;
}

static void save_checkpoint(const char* path,
                            int64_t N_, int64_t SEG_NUM_, int64_t next_seg,
                            uint64_t last_prime_, uint64_t total_primes_,
                            unsigned long long c1_, unsigned long long c3_,
                            const int64_t* gaps_total_) {
    CheckpointState s;
    s.magic = CKPT_MAGIC;
    s.N = N_;
    s.SEG_NUM = SEG_NUM_;
    s.next_segment = next_seg;
    s.last_prime = last_prime_;
    s.total_primes = total_primes_;
    s.class1_total = c1_;
    s.class3_total = c3_;
    memcpy(s.gaps_total, gaps_total_, MAX_GAP * sizeof(int64_t));
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(&s, sizeof(CheckpointState), 1, f);
    fclose(f);
}


// MAX_POS and N are injected by Python — see below
static const int64_t MAX_POS = 1411457066;
static const int64_t N       = 1000000000000;

static void generate_base_primes(int limit, uint32_t** out, int* count) {
    bool* sieve = (bool*)malloc(limit + 1);
    for (int i = 0; i <= limit; i++) sieve[i] = true;
    sieve[0] = sieve[1] = false;
    for (int i = 2; (long long)i*i <= limit; i++)
        if (sieve[i]) for (int j = i*i; j <= limit; j += i) sieve[j] = false;
    int cnt = 0;
    for (int i = 7; i <= limit; i++) if (sieve[i]) cnt++;
    uint32_t* arr = (uint32_t*)malloc(cnt * sizeof(uint32_t));
    int idx = 0;
    for (int i = 7; i <= limit; i++) if (sieve[i]) arr[idx++] = i;
    free(sieve); *out = arr; *count = cnt;
}

static int64_t modinv(int64_t a, int64_t m) {
    int64_t t = 0, newt = 1, r = m, newr = a % m;
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = newt; newt = t - q*newt; t = tmp;
        tmp = newr; newr = r - q*newr; r = tmp;
    }
    if (t < 0) t += m; return t;
}

__global__ void sieve_w30_seg_kernel(uint32_t* __restrict__ bits, uint32_t p,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t seg_bits)
{
    int64_t tid = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    int o = (int)(tid & 7); int64_t k = tid >> 3;
    int64_t start;
    switch (o) {
        case 0: start = s0; break; case 1: start = s1; break;
        case 2: start = s2; break; case 3: start = s3; break;
        case 4: start = s4; break; case 5: start = s5; break;
        case 6: start = s6; break; default: start = s7; break;
    }
    if (start < 0) return;
    int64_t idx = start + k * 8 * (int64_t)p;
    if (idx >= seg_bits) return;
    uint32_t w = (uint32_t)(idx >> 5);
    uint32_t i = (uint32_t)(idx & 31);
    atomicAnd(&bits[w], ~(1u << i));
}

__global__ void extract_w30_seg_kernel(const uint32_t* __restrict__ bits,
    int64_t num_words, int64_t seg_bits,
    uint64_t* __restrict__ positions, uint64_t* __restrict__ global_count)
{
    int64_t widx = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (widx >= num_words) return;
    uint32_t word = bits[widx]; if (word == 0) return;
    int64_t base_idx = widx * 32; int valid = 0; uint32_t w = word;
    while (w) { int bit = __ffs(w)-1; w &= w-1;
                if (base_idx+bit < seg_bits) valid++; }
    if (valid == 0) return;
    uint64_t base = atomicAdd((unsigned long long*)global_count,
                              (unsigned long long)valid);
    w = word;
    while (w) { int bit = __ffs(w)-1; w &= w-1;
                int64_t idx = base_idx+bit;
                if (idx < seg_bits) positions[base++] = (uint64_t)idx; }
}

__global__ void mod4_count_kernel(const uint64_t* __restrict__ positions,
    uint64_t n, uint64_t k_base,
    unsigned long long* __restrict__ cnt1,
    unsigned long long* __restrict__ cnt3)
{
    uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t idx = positions[i];
    uint64_t val = 30ULL*(k_base + (idx>>3)) + (uint64_t)W30_DEV[idx & 7];
    uint32_t r = (uint32_t)(val & 3);
    if (r == 1) atomicAdd(cnt1, 1ULL);
    else if (r == 3) atomicAdd(cnt3, 1ULL);
}

#define NUM_Q 2
#define MAX_Q_SLOTS 40

__constant__ int Q_VALUES[NUM_Q] = {6, 30};
__constant__ int Q_OFFSETS[NUM_Q] = {0, 6};

// Precomputed lookup tables: W30 residue index -> residue mod q
__constant__ int W30_TO_Q6[8]  = {1, 1, 5, 1, 5, 1, 5, 5};
__constant__ int W30_TO_Q8[8]  = {1, 7, 3, 5, 1, 3, 7, 5};
__constant__ int W30_TO_Q12[8] = {1, 7, 11, 1, 5, 7, 11, 5};
__constant__ int W30_TO_Q24[8] = {1, 7, 11, 13, 17, 19, 23, 5};
__constant__ int W30_TO_Q30[8] = {1, 7, 11, 13, 17, 19, 23, 29};

__global__ void modq_count_kernel(
    const uint64_t* __restrict__ positions,
    uint64_t n,
    uint64_t k_base,
    unsigned long long* __restrict__ counters)
{
    // Shared-memory private counters (offset 0 for q=6, offset 6 for q=30)
    __shared__ unsigned int sh[MAX_Q_SLOTS];
    int tid = threadIdx.x;
    for (int i = tid; i < MAX_Q_SLOTS; i += blockDim.x) sh[i] = 0;
    __syncthreads();
    
    uint64_t i = (uint64_t)blockIdx.x*blockDim.x + tid;
    if (i < n) {
        int w = (int)(positions[i] & 7);
        atomicAdd(&sh[0 + W30_TO_Q6[w]], 1u);
        atomicAdd(&sh[6 + W30_TO_Q30[w]], 1u);
    }
    __syncthreads();
    
    // Merge ALL shared slots to global
    for (int i = tid; i < MAX_Q_SLOTS; i += blockDim.x) {
        if (sh[i] > 0) atomicAdd(&counters[i], (unsigned long long)sh[i]);
    }
}

__global__ void gaps_w30_seg_kernel(
    const uint64_t* __restrict__ positions,
    uint64_t n, uint64_t k_base,
    int64_t* __restrict__ global_hist,
    int64_t* __restrict__ global_modq_hist,
    LargeGap* __restrict__ large_gaps,
    unsigned int* __restrict__ large_gap_count,
    int large_gap_threshold, int max_large_gaps)
{
    __shared__ int local_hist[SHARED_HIST_SIZE];
    __shared__ int modq_hist[MODQ_NUM_TOTAL][MODQ_MAX_GAP];
    int tid = threadIdx.x;
    for (int i = tid; i < SHARED_HIST_SIZE; i += HIST_BLOCK) local_hist[i] = 0;
    for (int j = tid; j < MODQ_NUM_TOTAL * MODQ_MAX_GAP; j += HIST_BLOCK)
        ((int*)modq_hist)[j] = 0;
    __syncthreads();

    uint64_t i = (uint64_t)blockIdx.x*HIST_BLOCK + tid;
    if (i > 0 && i < n) {
        uint64_t idx1 = positions[i], idx0 = positions[i-1];
        uint64_t v1 = 30ULL*(k_base+(idx1>>3)) + (uint64_t)W30_DEV[idx1&7];
        uint64_t v0 = 30ULL*(k_base+(idx0>>3)) + (uint64_t)W30_DEV[idx0&7];
        uint64_t diff = v1 - v0;
        if (diff > 0) {
            if (diff < (uint64_t)SHARED_HIST_SIZE) atomicAdd(&local_hist[diff], 1);
            else if (diff < (uint64_t)MAX_GAP) atomicAdd((unsigned long long*)&global_hist[diff], 1ULL);

            // Per-residue gap histogram (classify by residue of v0)
            if (diff < (uint64_t)MODQ_MAX_GAP) {
                int w0 = (int)(idx0 & 7);
                atomicAdd(&modq_hist[W30_TO_Q6_IDX[w0]][diff], 1);
                atomicAdd(&modq_hist[MODQ_NUM_Q6 + W30_TO_Q30_IDX[w0]][diff], 1);
            }

            double merit_d = 0.0;
            if (diff > 100) merit_d = (double)diff / log((double)v0);
            if (diff >= (uint64_t)large_gap_threshold || merit_d >= 10.0) {
                if (diff < (uint64_t)MAX_GAP) {
                    unsigned int slot = atomicAdd(large_gap_count, 1u);
                    if (slot < (unsigned int)max_large_gaps) {
                        large_gaps[slot].position = v1;
                        large_gaps[slot].gap = (int32_t)diff;
                        large_gaps[slot].merit = (float)merit_d;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int j = tid; j < SHARED_HIST_SIZE; j += HIST_BLOCK) {
        int v = local_hist[j];
        if (v > 0) atomicAdd((unsigned long long*)&global_hist[j], (unsigned long long)v);
    }
    for (int j = tid; j < MODQ_NUM_TOTAL * MODQ_MAX_GAP; j += HIST_BLOCK) {
        int v = ((int*)modq_hist)[j];
        if (v > 0) atomicAdd((unsigned long long*)&global_modq_hist[j], (unsigned long long)v);
    }
}


struct LaunchInfo { uint32_t p; int64_t s[8]; int grid_x; };

static void print_progress(int64_t seg, int64_t num_seg, uint64_t total_primes,
                           double elapsed_s, uint64_t last_prime)
{
    int percent = (int)((seg * 100) / num_seg);
    double rate = (elapsed_s > 0.1) ? ((double)seg / elapsed_s) : 0.0;
    double eta_s = (rate > 0.0) ? ((double)(num_seg - seg) / rate) : 0.0;
    char eta_str[32], el_str[32];
    if (eta_s < 60.0) snprintf(eta_str, 32, "%5.1f s", eta_s);
    else if (eta_s < 3600.0) snprintf(eta_str, 32, "%5.1f m", eta_s/60.0);
    else snprintf(eta_str, 32, "%5.1f h", eta_s/3600.0);
    if (elapsed_s < 60.0) snprintf(el_str, 32, "%5.1f s", elapsed_s);
    else if (elapsed_s < 3600.0) snprintf(el_str, 32, "%5.1f m", elapsed_s/60.0);
    else snprintf(el_str, 32, "%5.1f h", elapsed_s/3600.0);
    fprintf(stdout, "\r[%3d%%] seg %lld/%lld | primes: %-13llu | last p: %-16llu | elapsed: %s | ETA: %s   ",
            percent, (long long)seg, (long long)num_seg,
            (unsigned long long)total_primes, (unsigned long long)last_prime,
            el_str, eta_str);
    fflush(stdout);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int64_t SEG_NUM = 30000000000LL;
    const int64_t SEG_K   = SEG_NUM / 30;
    const int64_t SEG_BITS = SEG_K * 8;
    const int64_t NUM_SEG = (N + SEG_NUM - 1) / SEG_NUM;

    printf("=============================================================\n");
    printf("  VOSS-Wheel30 v7 LIVE\n");
    printf("=============================================================\n");
    printf("N         = %lld\n", (long long)N);
    printf("NUM_SEG   = %lld\n", (long long)NUM_SEG);
    printf("MAX_POS   = %lld (%.2f GB)\n", (long long)MAX_POS,
           (double)MAX_POS*8.0/1e9);
    printf("=============================================================\n\n");

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.1f GB)\n\n", prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem/1e9);

    int limit = (int)sqrt((double)N) + 1;
    uint32_t* base_h; int base_count;
    generate_base_primes(limit, &base_h, &base_count);
    printf("Base primes (>=7) up to sqrt(N)=%d : %d\n\n", limit, base_count);

    int64_t seg_words = (SEG_BITS + 31) / 32;
    size_t bits_bytes = seg_words * sizeof(uint32_t);
    size_t pos_bytes  = (size_t)MAX_POS * sizeof(uint64_t);
    size_t lg_bytes   = (size_t)MAX_LARGE_GAPS * sizeof(LargeGap);
    size_t total_alloc = bits_bytes + pos_bytes + lg_bytes;

    printf("Bit array:      %.2f GB\n", bits_bytes/1e9);
    printf("Positions:      %.2f GB\n", pos_bytes/1e9);
    printf("Large-gap:      %.2f GB\n", lg_bytes/1e9);
    printf("Peak:           %.2f GB / %.2f GB\n\n",
           total_alloc/1e9, prop.totalGlobalMem/1e9);

    uint32_t *bits_d; uint64_t *positions_d, *count_d;
    int64_t *gaps_d; LargeGap *large_gaps_d; unsigned int *lg_count_d;
    unsigned long long *m1_d, *m3_d;

    CUDA_CHECK(cudaMalloc(&bits_d, bits_bytes));
    CUDA_CHECK(cudaMalloc(&positions_d, pos_bytes));
    CUDA_CHECK(cudaMalloc(&count_d, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&gaps_d, MAX_GAP * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&large_gaps_d, lg_bytes));
    CUDA_CHECK(cudaMalloc(&lg_count_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&m1_d, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&m3_d, sizeof(unsigned long long)));
    int64_t* modq_gap_hist_d;
    CUDA_CHECK(cudaMalloc(&modq_gap_hist_d, MODQ_NUM_TOTAL * MODQ_MAX_GAP * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(modq_gap_hist_d, 0, MODQ_NUM_TOTAL * MODQ_MAX_GAP * sizeof(int64_t)));
    unsigned long long* modq_d;
    CUDA_CHECK(cudaMalloc(&modq_d, MAX_Q_SLOTS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(modq_d, 0, MAX_Q_SLOTS * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemset(count_d, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(gaps_d, 0, MAX_GAP*sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(lg_count_d, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(m1_d, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(m3_d, 0, sizeof(unsigned long long)));

    int64_t* gaps_total = (int64_t*)calloc(MAX_GAP, sizeof(int64_t));
    uint64_t last_prime = 5, total_primes = 3;
    unsigned long long c1_total = 0, c3_total = 0;
    int64_t start_seg = 0;
    {
        CheckpointState ckpt;
        if (load_checkpoint("checkpoint.bin", N, SEG_NUM, &ckpt)) {
            start_seg = ckpt.next_segment;
            last_prime = ckpt.last_prime;
            total_primes = ckpt.total_primes;
            c1_total = ckpt.class1_total;
            c3_total = ckpt.class3_total;
            memcpy(gaps_total, ckpt.gaps_total, MAX_GAP * sizeof(int64_t));
            printf(">>> RESUMING FROM SEGMENT %lld / %lld <<<\n",
                   (long long)start_seg, (long long)NUM_SEG);
        } else {
            printf(">>> STARTING FRESH <<<\n");
        }
    }
    double t_sieve=0, t_extract=0, t_sort=0, t_gaps=0, t_mod4=0;

    const int BLOCK = 256;
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    auto t_all_start = std::chrono::high_resolution_clock::now();

    for (int64_t seg = start_seg; seg < NUM_SEG; seg++) {
        int64_t seg_low = seg*SEG_NUM + 1;
        int64_t seg_high = (seg+1)*SEG_NUM;
        if (seg_high > N) seg_high = N;
        int64_t k_base = seg * SEG_K;
        int64_t seg_bits = SEG_BITS;
        if (seg == NUM_SEG-1) {
            int64_t r_num = N - seg*SEG_NUM;
            int64_t r_k = r_num/30, r_r = r_num%30;
            seg_bits = r_k*8;
            for (int i = 0; i < 8; i++) if (W30[i] <= r_r) seg_bits++;
        }
        int64_t cur_words = (seg_bits + 31) / 32;
        auto t_seg = std::chrono::high_resolution_clock::now();

        std::vector<LaunchInfo> launches;
        for (int i = 0; i < base_count; i++) {
            uint32_t p = base_h[i];
            int64_t p_sq = (int64_t)p*p;
            if (p_sq > seg_high) break;
            int64_t inv30 = modinv(30 % p, (int64_t)p);
            LaunchInfo L; L.p = p; int64_t max_steps = 0;
            for (int o = 0; o < 8; o++) {
                int r = W30[o];
                int64_t k0 = ((int64_t)((-r%(int)p+(int)p)%(int)p)*inv30) % (int64_t)p;
                int64_t n = 30LL*k0 + r;
                int64_t lb = (p_sq > seg_low) ? p_sq : seg_low;
                if (n < lb) {
                    int64_t step = 30LL*(int64_t)p;
                    int64_t d = lb - n;
                    n += ((d + step - 1)/step)*step;
                }
                if (n > seg_high) { L.s[o] = -1; continue; }
                int64_t k_val = (n-r)/30;
                int64_t li = (k_val - k_base)*8 + o;
                if (li < 0 || li >= seg_bits) { L.s[o] = -1; continue; }
                L.s[o] = li;
                int64_t steps = (seg_bits-1-li)/(8*(int64_t)p) + 1;
                if (steps > max_steps) max_steps = steps;
            }
            if (max_steps <= 0) continue;
            int64_t thr = max_steps*8;
            L.grid_x = (int)((thr + BLOCK - 1)/BLOCK);
            launches.push_back(L);
        }

        if (seg == 0) {
            sieve_w30_seg_kernel<<<1,32,0,stream>>>(
                bits_d, 7, -1,-1,-1,-1,-1,-1,-1,-1, seg_bits);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaGraph_t graph; cudaGraphExec_t graphExec;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (auto& L : launches) {
            sieve_w30_seg_kernel<<<L.grid_x, BLOCK, 0, stream>>>(
                bits_d, L.p, L.s[0],L.s[1],L.s[2],L.s[3],
                L.s[4],L.s[5],L.s[6],L.s[7], seg_bits);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

        CUDA_CHECK(cudaMemset(bits_d, 0xFF, cur_words*sizeof(uint32_t)));
        if (seg == 0) {
            uint32_t fw = 0xFFFFFFFE;
            CUDA_CHECK(cudaMemcpy(bits_d, &fw, 4, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        t_sieve += std::chrono::duration<double,std::milli>(t1-t_seg).count();

        CUDA_CHECK(cudaMemset(count_d, 0, sizeof(uint64_t)));
        int eg = (int)((cur_words + BLOCK - 1)/BLOCK);
        extract_w30_seg_kernel<<<eg, BLOCK>>>(bits_d, cur_words, seg_bits,
                                              positions_d, count_d);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        t_extract += std::chrono::duration<double,std::milli>(t2-t1).count();

        uint64_t n_pos;
        CUDA_CHECK(cudaMemcpy(&n_pos, count_d, 8, cudaMemcpyDeviceToHost));
        if (n_pos > (uint64_t)MAX_POS) {
            fprintf(stderr, "\nFATAL: n_pos=%llu > MAX_POS=%lld\n",
                    (unsigned long long)n_pos, (long long)MAX_POS);
            exit(1);
        }

        thrust::device_ptr<uint64_t> ptr(positions_d);
        thrust::sort(ptr, ptr + n_pos);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t3 = std::chrono::high_resolution_clock::now();
        t_sort += std::chrono::duration<double,std::milli>(t3-t2).count();

        uint64_t f_idx = 0, l_idx = 0;
        if (n_pos > 0) {
            CUDA_CHECK(cudaMemcpy(&f_idx, positions_d, 8, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&l_idx, positions_d+n_pos-1, 8, cudaMemcpyDeviceToHost));
        }
        uint64_t f_val = 30ULL*(k_base+(f_idx>>3)) + W30[f_idx&7];
        uint64_t l_val = 30ULL*(k_base+(l_idx>>3)) + W30[l_idx&7];

        if (n_pos > 0) {
            uint64_t d0 = f_val - last_prime;
            if (d0 > 0 && d0 < MAX_GAP) gaps_total[d0]++;
        }

        if (n_pos > 0) {
            int mg = (int)((n_pos + BLOCK - 1)/BLOCK);
            mod4_count_kernel<<<mg, BLOCK>>>(positions_d, n_pos, k_base, m1_d, m3_d);
            CUDA_CHECK(cudaDeviceSynchronize());
            modq_count_kernel<<<mg, BLOCK>>>(positions_d, n_pos, k_base, modq_d);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        t_mod4 += std::chrono::duration<double,std::milli>(t4-t3).count();

        int gg = (int)((n_pos + HIST_BLOCK - 1)/HIST_BLOCK);
        gaps_w30_seg_kernel<<<gg, HIST_BLOCK>>>(positions_d, n_pos, k_base,
            gaps_d, modq_gap_hist_d, large_gaps_d, lg_count_d, LARGE_GAP_THRESHOLD, MAX_LARGE_GAPS);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t5 = std::chrono::high_resolution_clock::now();
        t_gaps += std::chrono::duration<double,std::milli>(t5-t4).count();

        int64_t* gs = (int64_t*)malloc(MAX_GAP*sizeof(int64_t));
        CUDA_CHECK(cudaMemcpy(gs, gaps_d, MAX_GAP*sizeof(int64_t), cudaMemcpyDeviceToHost));
        for (int i = 1; i < MAX_GAP; i++) gaps_total[i] += gs[i];
        free(gs);
        CUDA_CHECK(cudaMemset(gaps_d, 0, MAX_GAP*sizeof(int64_t)));

        unsigned long long a1, a3;
        CUDA_CHECK(cudaMemcpy(&a1, m1_d, sizeof(a1), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&a3, m3_d, sizeof(a3), cudaMemcpyDeviceToHost));
        c1_total += a1; c3_total += a3;
        CUDA_CHECK(cudaMemset(m1_d, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(m3_d, 0, sizeof(unsigned long long)));

        if (n_pos > 0) { last_prime = l_val; total_primes += n_pos; }

        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);

        if (seg == 0 || seg == NUM_SEG - 1 || (seg + 1) % 5 == 0) {
            save_checkpoint("checkpoint.bin", N, SEG_NUM, seg + 1,
                            last_prime, total_primes,
                            c1_total, c3_total, gaps_total);
            // Stop-file check (for testing clean exit)
            FILE* stop = fopen("stop.txt", "r");
            if (stop) {
                fclose(stop);
                remove("stop.txt");
                printf("\n>>> STOP-FILE detected at seg %lld — exiting cleanly <<<\n",
                       (long long)seg);
                // Cleanup (v7 variable names)
                cudaStreamDestroy(stream);
                cudaFree(bits_d); cudaFree(positions_d); cudaFree(count_d);
                cudaFree(gaps_d); cudaFree(large_gaps_d); cudaFree(lg_count_d);
                cudaFree(m1_d); cudaFree(m3_d);
    cudaFree(modq_gap_hist_d);
                free(base_h); free(gaps_total);
                return 0;
            }
        }

        auto tn = std::chrono::high_resolution_clock::now();
        double el = std::chrono::duration<double>(tn - t_all_start).count();
        print_progress(seg+1, NUM_SEG, total_primes, el, last_prime);
    }
    printf("\n\n");

    remove("checkpoint.bin");

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_all_start).count();

    gaps_total[1] += 1;
    gaps_total[2] += 1;

    printf("=============================================================\n");
    printf("TOTAL TIME: %.2f s (%.2f min)\n", total_s, total_s/60.0);
    printf("Total primes: %llu\n", (unsigned long long)total_primes);
    printf("=============================================================\n\n");

    double accounted = (t_sieve+t_extract+t_sort+t_gaps+t_mod4)/1000.0;
    double other = total_s - accounted;
    printf("=== Timing Breakdown ===\n");
    printf("Sieve:     %10.2f s (%6.2f%%)\n", t_sieve/1000.0, 100*t_sieve/(total_s*1000));
    printf("Extract:   %10.2f s (%6.2f%%)\n", t_extract/1000.0, 100*t_extract/(total_s*1000));
    printf("Sort:      %10.2f s (%6.2f%%)\n", t_sort/1000.0, 100*t_sort/(total_s*1000));
    printf("Mod-4:     %10.2f s (%6.2f%%)\n", t_mod4/1000.0, 100*t_mod4/(total_s*1000));
    printf("Gaps:      %10.2f s (%6.2f%%)\n", t_gaps/1000.0, 100*t_gaps/(total_s*1000));
    printf("Other:     %10.2f s (%6.2f%%)\n\n", other, 100*other/total_s);

    int64_t tgc = 0, wsum = 0;
    for (int g = 1; g < MAX_GAP; g++) { tgc += gaps_total[g]; wsum += (int64_t)g*gaps_total[g]; }
    double mg = (double)wsum/(double)tgc;
    double vs=0, ss=0, ks=0;
    for (int g = 1; g < MAX_GAP; g++) {
        if (gaps_total[g]==0) continue;
        double d = (double)g - mg;
        vs += (double)gaps_total[g]*d*d;
        ss += (double)gaps_total[g]*d*d*d;
        ks += (double)gaps_total[g]*d*d*d*d;
    }
    double var = vs/(double)tgc, sd = sqrt(var);
    double sk = ss/(double)tgc/(var*sd);
    double ku = ks/(double)tgc/(var*var) - 3.0;

    printf("=== Gap Statistics ===\n");
    printf("Total gaps:  %lld\n", (long long)tgc);
    printf("Mean gap:    %.4f\n", mg);
    printf("Std dev:     %.4f\n", sd);
    printf("Skewness:    %.4f\n", sk);
    printf("Kurtosis:    %.4f\n\n", ku);

    printf("=== Gap Counts ===\n");
    printf("Gap 1:  %lld\n", (long long)gaps_total[1]);
    printf("Gap 2:  %lld\n", (long long)gaps_total[2]);
    printf("Gap 4:  %lld\n", (long long)gaps_total[4]);
    printf("Gap 6:  %lld\n", (long long)gaps_total[6]);
    printf("Gap 12: %lld\n", (long long)gaps_total[12]);
    printf("Gap 30: %lld\n\n", (long long)gaps_total[30]);

    unsigned long long p41 = c1_total + 1, p43 = c3_total + 1;
    long long diff = (long long)p43 - (long long)p41;
    printf("=== Chebyshev Bias ===\n");
    printf("pi(x;4,1) = %llu\n", p41);
    printf("pi(x;4,3) = %llu\n", p43);
    printf("Diff      = %lld\n\n", diff);

    double p6p2 = (double)gaps_total[6]/(double)gaps_total[2];
    printf("=== Gap Repulsion ===\n");
    printf("P(6)/P(2) = %.6f\n\n", p6p2);

    printf("=== Self-Consistency Checks ===\n");
    printf("[Check 1] sum(N(g)) = pi(N) - 1\n");
    printf("  sum(N(g))  = %lld\n", (long long)tgc);
    printf("  pi(N) - 1  = %lld\n", (long long)(total_primes-1));
    printf("  -> %s\n", (tgc==(int64_t)(total_primes-1))?"PASS":"FAIL");
    printf("[Check 2] sum(g*N(g)) = p_last - 2\n");
    printf("  sum(g*N(g)) = %lld\n", (long long)wsum);
    printf("  p_last - 2  = %lld\n", (long long)(last_prime-2));
    printf("  -> %s\n", (wsum==(int64_t)(last_prime-2))?"PASS":"FAIL");
    printf("[Check 3] pi(4,1) + pi(4,3) + 1 = pi(N)\n");
    printf("  pi41+pi43+1 = %llu\n", p41+p43+1);
    printf("  pi(N)       = %llu\n", (unsigned long long)total_primes);
    printf("  -> %s\n\n", ((p41+p43+1)==total_primes)?"PASS":"FAIL");

    FILE* fp = fopen("gap_histogram.csv", "w");
    if (fp) { fprintf(fp, "gap,count\n");
        for (int g=1; g<MAX_GAP; g++)
            if (gaps_total[g]>0) fprintf(fp, "%d,%lld\n", g, (long long)gaps_total[g]);
        fclose(fp); printf("OK gap_histogram.csv\n"); }

    fp = fopen("hl_trend.csv", "w");
    if (fp) {
        fprintf(fp, "N,P6_over_P2\n");
        static const long long HL_NS[] = {
            1000000000LL, 10000000000LL, 100000000000LL,
            1000000000000LL, 10000000000000LL
        };
        static const double HL_RS[] = {
            1.778300, 1.801800, 1.820800, 1.836600, 1.849700
        };
        int already = 0;
        for (int i = 0; i < 5; i++) {
            fprintf(fp, "%lld,%.6f\n", HL_NS[i], HL_RS[i]);
            if (N == HL_NS[i]) already = 1;
        }
        if (!already) {
            fprintf(fp, "%lld,%.6f\n", (long long)N, p6p2);
        }
        fclose(fp);
        printf("OK hl_trend.csv (hybrid)\n");
    }

        fp = fopen("modq_counts.csv", "w");
    if (fp) {
        unsigned long long* modq_h = (unsigned long long*)malloc(MAX_Q_SLOTS * sizeof(unsigned long long));
        CUDA_CHECK(cudaMemcpy(modq_h, modq_d, MAX_Q_SLOTS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        static const int QV[2] = {6, 30};
        static const int QO[2] = {0, 6};
        fprintf(fp, "q,a,count\n");
        for (int qi = 0; qi < 2; qi++) {
            int q = QV[qi];
            int off = QO[qi];
            for (int a = 0; a < q; a++)
                fprintf(fp, "%d,%d,%llu\n", q, a, modq_h[off + a]);
        }
        fclose(fp);
        free(modq_h);
        printf("OK modq_counts.csv\n");
    }

    // Per-residue gap histograms (Arithmetic Modulation complete)
    fp = fopen("modq_gap_hist.csv", "w");
    if (fp) {
        int64_t* mqh = (int64_t*)malloc(MODQ_NUM_TOTAL * MODQ_MAX_GAP * sizeof(int64_t));
        CUDA_CHECK(cudaMemcpy(mqh, modq_gap_hist_d, MODQ_NUM_TOTAL * MODQ_MAX_GAP * sizeof(int64_t), cudaMemcpyDeviceToHost));
        static const int QV[2] = {6, 30};
        static const int Q_OFF[2] = {0, MODQ_NUM_Q6};
        static const int Q_NRES[2] = {MODQ_NUM_Q6, MODQ_NUM_Q30};
        static const int Q_RES[2][8] = {
            {1, 5, 0, 0, 0, 0, 0, 0},
            {1, 7, 11, 13, 17, 19, 23, 29}
        };
        fprintf(fp, "q,residue,gap,count\n");
        for (int qi = 0; qi < 2; qi++) {
            for (int ri = 0; ri < Q_NRES[qi]; ri++) {
                for (int g = 1; g < MODQ_MAX_GAP; g++) {
                    int64_t c = mqh[(Q_OFF[qi] + ri) * MODQ_MAX_GAP + g];
                    if (c > 0)
                        fprintf(fp, "%d,%d,%d,%lld\n", QV[qi], Q_RES[qi][ri], g, (long long)c);
                }
            }
        }
        fclose(fp);
        free(mqh);
        printf("OK modq_gap_hist.csv\n");
    }

    fp = fopen("chebyshev.csv", "w");
    if (fp) { fprintf(fp, "N,pi_4_1,pi_4_3,difference\n");
        fprintf(fp, "%lld,%llu,%llu,%lld\n", (long long)N, p41, p43, diff);
        fclose(fp); printf("OK chebyshev.csv\n"); }

    unsigned int lgc;
    CUDA_CHECK(cudaMemcpy(&lgc, lg_count_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("\n=== Large Gaps (>= %d) ===\nCount: %u\n", LARGE_GAP_THRESHOLD, lgc);
        {
        unsigned int tr = (lgc < (unsigned)MAX_LARGE_GAPS) ? lgc : (unsigned)MAX_LARGE_GAPS;
        LargeGap* lgh = (LargeGap*)malloc((tr > 0 ? tr : 1) * sizeof(LargeGap));
        if (tr > 0) {
            CUDA_CHECK(cudaMemcpy(lgh, large_gaps_d, tr*sizeof(LargeGap), cudaMemcpyDeviceToHost));
            std::sort(lgh, lgh+tr, [](const LargeGap&a, const LargeGap&b){return a.merit>b.merit;});
        }
        fp = fopen("large_gaps.csv", "w");
        if (fp) {
            fprintf(fp, "position,gap,merit\n");
            for (unsigned int i = 0; i < tr; i++)
                fprintf(fp, "%llu,%d,%.6f\n", (unsigned long long)lgh[i].position, lgh[i].gap, lgh[i].merit);
            fclose(fp);
            printf("OK large_gaps.csv (%u entries)\n", tr);
            for (unsigned int i = 0; i < tr && i < 10; i++)
                printf("  gap=%4d merit=%6.2f at p=%llu\n", lgh[i].gap, lgh[i].merit,
                       (unsigned long long)lgh[i].position);
        }
        free(lgh);
    }

    cudaStreamDestroy(stream);
    cudaFree(bits_d); cudaFree(positions_d); cudaFree(count_d);
    cudaFree(gaps_d); cudaFree(large_gaps_d); cudaFree(lg_count_d);
    cudaFree(m1_d); cudaFree(m3_d);
    cudaFree(modq_d);
    free(base_h); free(gaps_total);
    printf("\nDone.\n");
    return 0;
}
