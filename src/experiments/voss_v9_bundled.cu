
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

#define MAX_GAP 100000
#define SHARED_HIST_SIZE 10000
#define HIST_BLOCK 256
#define LARGE_GAP_THRESHOLD 500
#define MAX_LARGE_GAPS 2000000

__constant__ int W30_DEV[8] = {1, 7, 11, 13, 17, 19, 23, 29};
static const int W30[8] = {1, 7, 11, 13, 17, 19, 23, 29};

struct LargeGap { uint64_t position; int32_t gap; int32_t _pad; };

static const int64_t N       = 100000000000;
static const int64_t SEG_NUM = 30000000000;
static const int64_t MAX_POS = 1550000000;
static const int     BUNDLE  = 1024;

static void gen_base_primes(int limit, uint32_t** out, int* cnt) {
    bool* s = (bool*)malloc(limit+1);
    for (int i=0;i<=limit;i++) s[i]=true;
    s[0]=s[1]=false;
    for (int i=2;(long long)i*i<=limit;i++) if (s[i])
        for (int j=i*i;j<=limit;j+=i) s[j]=false;
    int c=0; for (int i=7;i<=limit;i++) if (s[i]) c++;
    uint32_t* a=(uint32_t*)malloc(c*sizeof(uint32_t));
    int k=0; for (int i=7;i<=limit;i++) if (s[i]) a[k++]=i;
    free(s); *out=a; *cnt=c;
}

static int64_t modinv(int64_t a, int64_t m) {
    int64_t t=0, nt=1, r=m, nr=a%m;
    while (nr!=0) { int64_t q=r/nr, tmp=nt; nt=t-q*nt; t=tmp;
        tmp=nr; nr=r-q*nr; r=tmp; }
    if (t<0) t+=m; return t;
}

// ============================================================
// Sieve kernel (bundled): blockIdx.y = prime index in group
// Layout: eo[{s0..s7, p, inv, max_steps, grid_x}] per bundled prime
// ============================================================
struct PrimeMeta {
    int64_t s[8];
    uint32_t p;
    int32_t _pad;
    int64_t inv30;
    int64_t max_steps;
};

__global__ void sieve_bundled_kernel(
    uint32_t* __restrict__ bits,
    const PrimeMeta* __restrict__ metas,
    int meta_offset,
    int meta_count,
    int64_t seg_bits)
{
    int pi = meta_offset + blockIdx.y;
    if (pi >= meta_offset + meta_count) return;
    const PrimeMeta& M = metas[pi];
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int o = (int)(tid & 7);
    int64_t k = tid >> 3;
    int64_t start = M.s[o];
    if (start < 0) return;
    int64_t idx = start + k * 8 * (int64_t)M.p;
    if (idx >= seg_bits) return;
    uint32_t w = (uint32_t)(idx >> 5);
    uint32_t b = (uint32_t)(idx & 31);
    atomicAnd(&bits[w], ~(1u << b));
}

// ============================================================
// Extract kernel
// ============================================================
__global__ void extract_w30_seg_kernel(
    const uint32_t* __restrict__ bits, int64_t num_words, int64_t seg_bits,
    uint64_t* __restrict__ positions, uint64_t* __restrict__ gcount)
{
    int64_t widx = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (widx >= num_words) return;
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

__global__ void mod4_count_kernel(const uint64_t* __restrict__ positions,
    uint64_t n, uint64_t k_base,
    unsigned long long* __restrict__ cnt1, unsigned long long* __restrict__ cnt3)
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
    int lg_thr, int lg_max)
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
        uint64_t d = v1 - v0;
        if (d > 0) {
            if (d < (uint64_t)SHARED_HIST_SIZE) atomicAdd(&local_hist[d], 1);
            else if (d < (uint64_t)MAX_GAP) atomicAdd((unsigned long long*)&global_hist[d], 1ULL);
            if (d >= (uint64_t)lg_thr && d < (uint64_t)MAX_GAP) {
                unsigned s = atomicAdd(large_gap_count, 1u);
                if (s < (unsigned)lg_max) { large_gaps[s].position=v1; large_gaps[s].gap=(int32_t)d; }
            }
        }
    }
    __syncthreads();
    for (int j = tid; j < SHARED_HIST_SIZE; j += HIST_BLOCK)
        if (local_hist[j] > 0) atomicAdd((unsigned long long*)&global_hist[j], (unsigned long long)local_hist[j]);
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
    const int64_t SEG_K = SEG_NUM / 30;
    const int64_t SEG_BITS = SEG_K * 8;
    const int64_t NUM_SEG = (N + SEG_NUM - 1) / SEG_NUM;

    printf("=== VOSS v9 BUNDLED ===\n");
    printf("N=%lld SEG_NUM=%lld NUM_SEG=%lld BUNDLE=%d\n\n",
           (long long)N, (long long)SEG_NUM, (long long)NUM_SEG, BUNDLE);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.1f GB)\n\n", prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem/1e9);

    int limit = (int)sqrt((double)N) + 1;
    uint32_t* base_h; int base_count;
    gen_base_primes(limit, &base_h, &base_count);
    printf("Base primes (>=7) up to %d : %d\n\n", limit, base_count);

    int64_t seg_words = (SEG_BITS + 31) / 32;
    size_t bits_bytes = seg_words * sizeof(uint32_t);
    size_t pos_bytes  = (size_t)MAX_POS * sizeof(uint64_t);
    size_t lg_bytes   = (size_t)MAX_LARGE_GAPS * sizeof(LargeGap);

    printf("Bit array: %.2f GB, Positions: %.2f GB\n\n",
           bits_bytes/1e9, pos_bytes/1e9);

    // Allocate
    uint32_t *bits_d;
    uint64_t *pos_d, *cnt_d;
    int64_t *gaps_d; LargeGap *lg_d; unsigned int *lgc_d;
    unsigned long long *m1_d, *m3_d;
    PrimeMeta* metas_d;

    CUDA_CHECK(cudaMalloc(&bits_d, bits_bytes));
    CUDA_CHECK(cudaMalloc(&pos_d, pos_bytes));
    CUDA_CHECK(cudaMalloc(&cnt_d, 8));
    CUDA_CHECK(cudaMalloc(&gaps_d, MAX_GAP*8));
    CUDA_CHECK(cudaMalloc(&lg_d, lg_bytes));
    CUDA_CHECK(cudaMalloc(&lgc_d, 4));
    CUDA_CHECK(cudaMalloc(&m1_d, 8));
    CUDA_CHECK(cudaMalloc(&m3_d, 8));
    CUDA_CHECK(cudaMalloc(&metas_d, base_count * sizeof(PrimeMeta)));

    int64_t* gaps_total = (int64_t*)calloc(MAX_GAP, 8);
    uint64_t last_prime = 5, total_primes = 3;
    unsigned long long c1t = 0, c3t = 0;
    double t_sieve=0, t_extract=0, t_sort=0, t_gaps=0, t_mod4=0;

    const int BLOCK = 256;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto t0 = std::chrono::high_resolution_clock::now();

    // Host buffer for metas (updated per segment)
    std::vector<PrimeMeta> metas_h(base_count);

    for (int64_t seg = 0; seg < NUM_SEG; seg++) {
        int64_t seg_high = (seg+1)*SEG_NUM; if (seg_high > N) seg_high = N;
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

        // Build metas for this segment
        std::vector<int> active_indices;  // which base primes are active
        for (int i = 0; i < base_count; i++) {
            uint32_t p = base_h[i];
            int64_t p_sq = (int64_t)p*p;
            if (p_sq > seg_high) break;
            int64_t inv30 = modinv(30 % p, (int64_t)p);
            PrimeMeta M; M.p = p; M._pad = 0; M.inv30 = inv30;
            int64_t max_steps = 0;
            for (int o = 0; o < 8; o++) {
                int r = W30[o];
                int64_t k0 = ((int64_t)((-r % (int)p + (int)p) % (int)p) * inv30) % (int64_t)p;
                int64_t n = 30LL * k0 + r;
                int64_t lb = (p_sq > seg*SEG_NUM + 1) ? p_sq : (seg*SEG_NUM + 1);
                if (n < lb) {
                    int64_t step = 30LL*(int64_t)p;
                    int64_t diff = lb - n;
                    n += ((diff + step - 1)/step)*step;
                }
                if (n > seg_high) { M.s[o] = -1; continue; }
                int64_t k_val = (n-r)/30;
                int64_t li = (k_val - k_base)*8 + o;
                if (li < 0 || li >= seg_bits) { M.s[o] = -1; continue; }
                M.s[o] = li;
                int64_t steps = (seg_bits-1-li)/(8*(int64_t)p) + 1;
                if (steps > max_steps) max_steps = steps;
            }
            if (max_steps <= 0) continue;
            M.max_steps = max_steps;
            metas_h[active_indices.size()] = M;
            active_indices.push_back(i);
        }

        int active_count = (int)active_indices.size();

        // Upload metas
        if (active_count > 0) {
            CUDA_CHECK(cudaMemcpy(metas_d, metas_h.data(),
                                  active_count * sizeof(PrimeMeta),
                                  cudaMemcpyHostToDevice));
        }

        // Warmup
        if (seg == 0 && active_count > 0) {
            sieve_bundled_kernel<<<dim3(1,1), 32, 0, stream>>>(bits_d, metas_d, 0, 1, seg_bits);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Init bits
        CUDA_CHECK(cudaMemset(bits_d, 0xFF, cur_words*4));
        if (seg == 0) { uint32_t fw = 0xFFFFFFFE;
            CUDA_CHECK(cudaMemcpy(bits_d, &fw, 4, cudaMemcpyHostToDevice)); }

        // Build graph
        cudaGraph_t gr; cudaGraphExec_t ge;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        for (int offset = 0; offset < active_count; offset += BUNDLE) {
            int cnt = active_count - offset;
            if (cnt > BUNDLE) cnt = BUNDLE;
            // Find max grid_x in this bundle
            int64_t max_grid = 0;
            for (int j = 0; j < cnt; j++) {
                const PrimeMeta& M = metas_h[offset + j];
                int64_t thr = M.max_steps * 8;
                int64_t g = (thr + BLOCK - 1) / BLOCK;
                if (g > max_grid) max_grid = g;
            }
            dim3 grid((unsigned)max_grid, (unsigned)cnt);
            sieve_bundled_kernel<<<grid, BLOCK, 0, stream>>>(
                bits_d, metas_d, offset, cnt, seg_bits);
        }

        CUDA_CHECK(cudaStreamEndCapture(stream, &gr));
        CUDA_CHECK(cudaGraphInstantiate(&ge, gr, NULL, NULL, 0));
        CUDA_CHECK(cudaGraphLaunch(ge, stream));
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        t_sieve += std::chrono::duration<double,std::milli>(t1-t_seg).count();
        cudaGraphExecDestroy(ge); cudaGraphDestroy(gr);

        // Extract
        CUDA_CHECK(cudaMemset(cnt_d, 0, 8));
        int eg = (int)((cur_words + BLOCK - 1)/BLOCK);
        extract_w30_seg_kernel<<<eg, BLOCK>>>(bits_d, cur_words, seg_bits, pos_d, cnt_d);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        t_extract += std::chrono::duration<double,std::milli>(t2-t1).count();

        uint64_t npos; CUDA_CHECK(cudaMemcpy(&npos, cnt_d, 8, cudaMemcpyDeviceToHost));
        thrust::device_ptr<uint64_t> ptr(pos_d);
        thrust::sort(ptr, ptr+npos);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t3 = std::chrono::high_resolution_clock::now();
        t_sort += std::chrono::duration<double,std::milli>(t3-t2).count();

        uint64_t fi=0, li=0;
        if (npos>0) {
            CUDA_CHECK(cudaMemcpy(&fi, pos_d, 8, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&li, pos_d+npos-1, 8, cudaMemcpyDeviceToHost));
        }
        uint64_t fv = 30ULL*(k_base+(fi>>3)) + W30[fi&7];
        uint64_t lv = 30ULL*(k_base+(li>>3)) + W30[li&7];

        if (npos>0) { uint64_t d0 = fv - last_prime;
            if (d0>0 && d0<MAX_GAP) gaps_total[d0]++; }

        if (npos>0) {
            int mg = (int)((npos+BLOCK-1)/BLOCK);
            mod4_count_kernel<<<mg, BLOCK>>>(pos_d, npos, k_base, m1_d, m3_d);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        t_mod4 += std::chrono::duration<double,std::milli>(t4-t3).count();

        int gg = (int)((npos+HIST_BLOCK-1)/HIST_BLOCK);
        gaps_w30_seg_kernel<<<gg, HIST_BLOCK>>>(pos_d, npos, k_base, gaps_d,
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

        if (npos>0) { last_prime = lv; total_primes += npos; }

        auto tn = std::chrono::high_resolution_clock::now();
        double el = std::chrono::duration<double>(tn-t0).count();
        print_progress(seg+1, NUM_SEG, total_primes, el);
    }
    printf("\n\n");

    auto te = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(te-t0).count();

    gaps_total[1] += 1; gaps_total[2] += 1;

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

    unsigned long long p41=c1t+1, p43=c3t+1;
    int64_t tgc=0, wsum=0;
    for (int g=1;g<MAX_GAP;g++) { tgc+=gaps_total[g]; wsum+=(int64_t)g*gaps_total[g]; }
    printf("Chebyshev diff = %lld\n", (long long)p43-(long long)p41);
    printf("Check1: %s\n", (tgc==(int64_t)(total_primes-1))?"PASS":"FAIL");
    printf("Check2: %s\n", (wsum==(int64_t)(last_prime-2))?"PASS":"FAIL");
    printf("Check3: %s\n", ((p41+p43+1)==total_primes)?"PASS":"FAIL");

    FILE* fp = fopen("gap_histogram.csv", "w");
    if (fp) { fprintf(fp,"gap,count\n");
        for (int g=1;g<MAX_GAP;g++) if (gaps_total[g]>0)
            fprintf(fp,"%d,%lld\n",g,(long long)gaps_total[g]);
        fclose(fp); }

    cudaStreamDestroy(stream);
    cudaFree(bits_d); cudaFree(pos_d); cudaFree(cnt_d);
    cudaFree(gaps_d); cudaFree(lg_d); cudaFree(lgc_d);
    cudaFree(m1_d); cudaFree(m3_d); cudaFree(metas_d);
    free(base_h); free(gaps_total);
    return 0;
}
