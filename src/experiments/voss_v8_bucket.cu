
// ============================================================
// VOSS v8 - Bucket Concept (Prototype)
// ============================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
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
#define HIST_BLOCK 256
#define LARGE_GAP_THRESHOLD 500
#define MAX_LARGE_GAPS 2000000
#define BUCKET_WORDS 8192
#define BUCKET_BITS  (BUCKET_WORDS * 32)

__constant__ int W30_DEV[8] = {1, 7, 11, 13, 17, 19, 23, 29};
static const int W30[8] = {1, 7, 11, 13, 17, 19, 23, 29};

struct LargeGap { uint64_t position; int32_t gap; int32_t _pad; };

static const int64_t N       = 100000000000;
static const int64_t SEG_NUM = 30000000000;
static const int64_t MAX_POS = 1539771344;

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

__device__ __host__ __forceinline__ static int64_t modinv(int64_t a, int64_t m) {
    int64_t t = 0, newt = 1, r = m, newr = a % m;
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = newt; newt = t - q*newt; t = tmp;
        tmp = newr; newr = r - q*newr; r = tmp;
    }
    if (t < 0) t += m; return t;
}

// ============================================================
// NEW: Bucket-based sieve kernel
// ============================================================
__global__ void sieve_buckets_kernel(
    uint32_t* __restrict__ bits,
    int64_t num_buckets,
    int64_t seg_k_base,
    int64_t seg_bits,
    const uint32_t* __restrict__ base_primes,
    const int64_t* __restrict__ inv30_arr,
    int base_count)
{
    __shared__ uint32_t local_bits[BUCKET_WORDS];
    
    for (int64_t b = blockIdx.x; b < num_buckets; b += gridDim.x) {
        int64_t bucket_first_bit = b * BUCKET_BITS;
        if (bucket_first_bit >= seg_bits) break;
        int64_t bucket_last_bit = bucket_first_bit + BUCKET_BITS;
        if (bucket_last_bit > seg_bits) bucket_last_bit = seg_bits;
        int64_t bucket_words = (bucket_last_bit - bucket_first_bit + 31) / 32;
        
        // Load bucket into shared memory
        const uint32_t* gsrc = bits + b * BUCKET_WORDS;
        for (int i = threadIdx.x; i < bucket_words; i += blockDim.x)
            local_bits[i] = gsrc[i];
        __syncthreads();
        
        // Compute k-range and bucket_high
        int64_t k_start = seg_k_base + bucket_first_bit / 8;
        int64_t k_end   = seg_k_base + (bucket_last_bit - 1) / 8 + 1;
        int64_t bucket_high = 30LL * (k_end - 1) + 29;
        
        // Sieve
        for (int i = threadIdx.x; i < base_count; i += blockDim.x) {
            uint32_t p = base_primes[i];
            int64_t p_sq = (int64_t)p * p;
            if (p_sq > bucket_high) continue;
            
            int64_t inv30 = inv30_arr[i];
            
            for (int o = 0; o < 8; o++) {
                int r = W30_DEV[o];
                int64_t k0 = ((int64_t)((-r % (int)p + (int)p) % (int)p) * inv30) % (int64_t)p;
                
                // First k >= k_start with k ≡ k0 (mod p)
                int64_t delta = ((k0 - k_start) % (int64_t)p + (int64_t)p) % (int64_t)p;
                int64_t k = k_start + delta;
                
                // Ensure 30*k + r >= p^2
                while (k < k_end && 30LL*k + r < p_sq) k += p;
                
                // Sieve
                for (; k < k_end; k += p) {
                    int64_t bit = (k - seg_k_base) * 8 + o;
                    if (bit < bucket_first_bit || bit >= bucket_last_bit) continue;
                    int64_t local_bit = bit - bucket_first_bit;
                    int64_t word = local_bit >> 5;
                    int bit_in_word = (int)(local_bit & 31);
                    atomicAnd(&local_bits[word], ~(1u << bit_in_word));
                }
            }
        }
        __syncthreads();
        
        // Write bucket back
        uint32_t* gdst = bits + b * BUCKET_WORDS;
        for (int i = threadIdx.x; i < bucket_words; i += blockDim.x)
            gdst[i] = local_bits[i];
        __syncthreads();
    }
}

// ============================================================
// Other kernels (same as v7)
// ============================================================
__global__ void extract_w30_seg_kernel(
    const uint32_t* __restrict__ bits, int64_t num_words, int64_t seg_bits,
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

__global__ void gaps_w30_seg_kernel(const uint64_t* __restrict__ positions,
    uint64_t n, uint64_t k_base, int64_t* __restrict__ global_hist,
    LargeGap* __restrict__ large_gaps, unsigned int* __restrict__ large_gap_count,
    int large_gap_threshold, int max_large_gaps)
{
    __shared__ int local_hist[SHARED_HIST_SIZE];
    int tid = threadIdx.x;
    for (int i = tid; i < SHARED_HIST_SIZE; i += HIST_BLOCK) local_hist[i] = 0;
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
            if (diff >= (uint64_t)large_gap_threshold && diff < (uint64_t)MAX_GAP) {
                unsigned int slot = atomicAdd(large_gap_count, 1u);
                if (slot < (unsigned int)max_large_gaps) {
                    large_gaps[slot].position = v1;
                    large_gaps[slot].gap = (int32_t)diff;
                }
            }
        }
    }
    __syncthreads();
    for (int j = tid; j < SHARED_HIST_SIZE; j += HIST_BLOCK) {
        int v = local_hist[j];
        if (v > 0) atomicAdd((unsigned long long*)&global_hist[j], (unsigned long long)v);
    }
}

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
    fprintf(stdout, "\r[%3d%%] seg %lld/%lld | primes: %-13llu | elapsed: %s | ETA: %s   ",
            percent, (long long)seg, (long long)num_seg,
            (unsigned long long)total_primes, el_str, eta_str);
    fflush(stdout);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int64_t SEG_K = SEG_NUM / 30;
    const int64_t SEG_BITS = SEG_K * 8;
    const int64_t NUM_SEG = (N + SEG_NUM - 1) / SEG_NUM;

    printf("=== VOSS v8 BUCKET (Prototype) ===\n");
    printf("N=%lld  SEG_NUM=%lld  NUM_SEG=%lld\n",
           (long long)N, (long long)SEG_NUM, (long long)NUM_SEG);
    printf("BUCKET_WORDS=%d (shared=%.1f KB)\n\n",
           BUCKET_WORDS, (double)BUCKET_WORDS * 4 / 1024);

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
    printf("Bit array: %.2f GB, Positions: %.2f GB\n\n",
           bits_bytes/1e9, pos_bytes/1e9);

    uint32_t *bits_d, *base_primes_d;
    uint64_t *positions_d, *count_d;
    int64_t *gaps_d; LargeGap *large_gaps_d; unsigned int *lg_count_d;
    unsigned long long *m1_d, *m3_d;

    CUDA_CHECK(cudaMalloc(&bits_d, bits_bytes));
    CUDA_CHECK(cudaMalloc(&base_primes_d, base_count * sizeof(uint32_t)));
    int64_t* inv30_h = (int64_t*)malloc(base_count * sizeof(int64_t));
    for (int i = 0; i < base_count; i++)
        inv30_h[i] = modinv(30 % base_h[i], (int64_t)base_h[i]);
    int64_t* inv30_d;
    CUDA_CHECK(cudaMalloc(&inv30_d, base_count * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(inv30_d, inv30_h, base_count*sizeof(int64_t), cudaMemcpyHostToDevice));
    free(inv30_h);
    CUDA_CHECK(cudaMalloc(&positions_d, pos_bytes));
    CUDA_CHECK(cudaMalloc(&count_d, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&gaps_d, MAX_GAP * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&large_gaps_d, lg_bytes));
    CUDA_CHECK(cudaMalloc(&lg_count_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&m1_d, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&m3_d, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(base_primes_d, base_h,
                          base_count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    int64_t* gaps_total = (int64_t*)calloc(MAX_GAP, sizeof(int64_t));
    uint64_t last_prime = 5, total_primes = 3;
    unsigned long long c1_total = 0, c3_total = 0;
    double t_sieve=0, t_extract=0, t_sort=0, t_gaps=0, t_mod4=0;

    const int BLOCK = 256;
    auto t_all_start = std::chrono::high_resolution_clock::now();

    for (int64_t seg = 0; seg < NUM_SEG; seg++) {
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
        int64_t num_buckets = (seg_bits + BUCKET_BITS - 1) / BUCKET_BITS;

        auto t_seg = std::chrono::high_resolution_clock::now();

        // Init bits
        CUDA_CHECK(cudaMemset(bits_d, 0xFF, cur_words*sizeof(uint32_t)));
        if (seg == 0) {
            uint32_t fw = 0xFFFFFFFE;
            CUDA_CHECK(cudaMemcpy(bits_d, &fw, 4, cudaMemcpyHostToDevice));
        }

        // Launch bucket sieve
        int grid = (num_buckets < 432) ? (int)num_buckets : 432;
        sieve_buckets_kernel<<<grid, BLOCK>>>(
            bits_d, num_buckets, k_base, seg_bits, base_primes_d, inv30_d, base_count);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        t_sieve += std::chrono::duration<double,std::milli>(t1-t_seg).count();

        // Extract
        CUDA_CHECK(cudaMemset(count_d, 0, sizeof(uint64_t)));
        int eg = (int)((cur_words + BLOCK - 1)/BLOCK);
        extract_w30_seg_kernel<<<eg, BLOCK>>>(bits_d, cur_words, seg_bits,
                                              positions_d, count_d);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        t_extract += std::chrono::duration<double,std::milli>(t2-t1).count();

        uint64_t n_pos;
        CUDA_CHECK(cudaMemcpy(&n_pos, count_d, 8, cudaMemcpyDeviceToHost));

        thrust::device_ptr<uint64_t> ptr(positions_d);
        thrust::sort(ptr, ptr + n_pos);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t3 = std::chrono::high_resolution_clock::now();
        t_sort += std::chrono::duration<double,std::milli>(t3-t2).count();

        uint64_t f_idx=0, l_idx=0;
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
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        t_mod4 += std::chrono::duration<double,std::milli>(t4-t3).count();

        int gg = (int)((n_pos + HIST_BLOCK - 1)/HIST_BLOCK);
        gaps_w30_seg_kernel<<<gg, HIST_BLOCK>>>(positions_d, n_pos, k_base,
            gaps_d, large_gaps_d, lg_count_d, LARGE_GAP_THRESHOLD, MAX_LARGE_GAPS);
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

        auto tn = std::chrono::high_resolution_clock::now();
        double el = std::chrono::duration<double>(tn - t_all_start).count();
        print_progress(seg+1, NUM_SEG, total_primes, el, last_prime);
    }
    printf("\n\n");

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_all_start).count();

    gaps_total[1] += 1;
    gaps_total[2] += 1;

    printf("=============================================================\n");
    printf("TOTAL TIME: %.2f s (%.2f min)\n", total_s, total_s/60.0);
    printf("Total primes: %llu\n", (unsigned long long)total_primes);
    printf("=============================================================\n\n");

    printf("=== Timing Breakdown ===\n");
    printf("Sieve:     %10.2f s (%6.2f%%)\n", t_sieve/1000.0, 100*t_sieve/(total_s*1000));
    printf("Extract:   %10.2f s (%6.2f%%)\n", t_extract/1000.0, 100*t_extract/(total_s*1000));
    printf("Sort:      %10.2f s (%6.2f%%)\n", t_sort/1000.0, 100*t_sort/(total_s*1000));
    printf("Mod-4:     %10.2f s (%6.2f%%)\n", t_mod4/1000.0, 100*t_mod4/(total_s*1000));
    printf("Gaps:      %10.2f s (%6.2f%%)\n", t_gaps/1000.0, 100*t_gaps/(total_s*1000));
    double other = total_s - (t_sieve+t_extract+t_sort+t_mod4+t_gaps)/1000.0;
    printf("Other:     %10.2f s (%6.2f%%)\n\n", other, 100*other/total_s);

    int64_t tgc=0, wsum=0;
    for (int g=1; g<MAX_GAP; g++) { tgc += gaps_total[g]; wsum += (int64_t)g*gaps_total[g]; }

    printf("=== Gap Counts ===\n");
    printf("Gap 1:  %lld\n", (long long)gaps_total[1]);
    printf("Gap 2:  %lld\n", (long long)gaps_total[2]);
    printf("Gap 4:  %lld\n", (long long)gaps_total[4]);
    printf("Gap 6:  %lld\n", (long long)gaps_total[6]);
    printf("Gap 30: %lld\n\n", (long long)gaps_total[30]);

    unsigned long long p41 = c1_total+1, p43 = c3_total+1;
    printf("=== Chebyshev ===\n");
    printf("pi41=%llu  pi43=%llu  diff=%lld\n\n",
           p41, p43, (long long)p43 - (long long)p41);

    printf("=== Self-Consistency Checks ===\n");
    printf("[Check 1] sum(N(g))=pi(N)-1: %s\n",
           (tgc==(int64_t)(total_primes-1))?"PASS":"FAIL");
    printf("[Check 2] sum(g*N(g))=p_last-2: %s\n",
           (wsum==(int64_t)(last_prime-2))?"PASS":"FAIL");
    printf("[Check 3] pi41+pi43+1=pi(N): %s\n\n",
           ((p41+p43+1)==total_primes)?"PASS":"FAIL");

    FILE* fp = fopen("gap_histogram.csv", "w");
    if (fp) { fprintf(fp, "gap,count\n");
        for (int g=1; g<MAX_GAP; g++)
            if (gaps_total[g]>0) fprintf(fp, "%d,%lld\n", g, (long long)gaps_total[g]);
        fclose(fp); }

    cudaFree(bits_d); cudaFree(base_primes_d); cudaFree(inv30_d);
    cudaFree(positions_d); cudaFree(count_d);
    cudaFree(gaps_d); cudaFree(large_gaps_d); cudaFree(lg_count_d);
    cudaFree(m1_d); cudaFree(m3_d);
    free(base_h); free(gaps_total);
    return 0;
}
