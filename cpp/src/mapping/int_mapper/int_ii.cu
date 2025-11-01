#include <cuda_runtime.h>
#include <iostream>

// Size definitions
constexpr unsigned int BM_06 = 64;
constexpr unsigned int BN_06 = 32;
constexpr unsigned int BK_06 = 16;
constexpr unsigned int WM_06 = 16;
constexpr unsigned int WN_06 = 16;
constexpr unsigned int TM_06 = 4;
constexpr unsigned int TN_06 = 2;

constexpr unsigned int I_BIT = 8;

constexpr unsigned int SPLIT_SIZE = 3;

// ADC constants
constexpr float ALPHA = 1.0f;
constexpr float MIN_CUR = -800.0f;
constexpr float MAX_CUR = 800.0f; // N * (LRS-HRS)
constexpr float STEP_SIZE = 0.00152588f; // 2 * max_cur / (2^(RESOLUTION)-1)

// Not used. But these are the values for both used mappings
constexpr float LRS = 30.0f;
constexpr float HRS = 5.0f;

constexpr unsigned int XBAR_N = 32;

constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }

// Performs conversion of SYMADC
__device__ __forceinline__ float analog_digital_conversion(float current) {
    float clip = fminf(fmaxf(current, ALPHA * MIN_CUR), ALPHA * MAX_CUR);
    float adc_res = round(clip / STEP_SIZE) * STEP_SIZE;
    return adc_res;
}

/*
    This MMM kernel also includes the ADC functionality required for the analog
   mapping. It is essentially a modified warp-tiled GEMM kernel without alpha
   and beta scalars.
*/
__global__ void mmm_kernel(int M, int N, int K, const float *mat_A_p,
                           const float *mat_A_m, const int32_t *mat_B,
                           int32_t *res, int32_t *sum_w_) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    unsigned int wx = threadIdx.x / (WN_06 / TN_06);
    unsigned int wy = threadIdx.y / (WM_06 / TM_06);

    unsigned int tx = (threadIdx.x % (WN_06 / TN_06));
    unsigned int ty = (threadIdx.y % (WM_06 / TM_06));

    // Pointers to the block's top-left (position in res)
    const int res_tile_offs = N * BM_06 * by + BN_06 * bx;
    // Offset to row=0 and col=bx in mat_B
    int mat_B_tile_offs = BN_06 * bx;
    // Offset to row=by and col=0 in mat_A
    int mat_A_tile_offs = SPLIT_SIZE * BM_06 * K * by;

    // Shared-memory buffers for mat_A and mat_B tiles
    __shared__ float mat_As_p[SPLIT_SIZE * BM_06 * BK_06];
    __shared__ float mat_As_m[SPLIT_SIZE * BM_06 * BK_06];
    __shared__ int32_t mat_Bs[BK_06 * BN_06];

    // k = {0, BK_06, 2*BK_06, ...}
    for (int k = 0; k < K; k += BK_06) {
        // Each thread loads SPLIT_SIZE * TM_06 values into mat_As_p and mat_As_m
        for (int tm = 0; tm < SPLIT_SIZE * TM_06; ++tm) {
            bool notExceedingK = (k + (WN_06 / TN_06) * wx + tx < K); // WN/TN is amount of y-threads inside warp. Correct
            bool notExceedingM =
                (SPLIT_SIZE * (BM_06 * by + WM_06 * wy + TM_06 * ty) + tm <
                 SPLIT_SIZE *
                     M); // Also correct, just sum of hierarchy along y-axis

            int dest = BK_06 * (SPLIT_SIZE * (WM_06 * wy + TM_06 * ty) + tm) +
                       (WN_06 / TN_06) * wx + tx;
            int src = mat_A_tile_offs +
                      K * (SPLIT_SIZE * (WM_06 * wy + TM_06 * ty) + tm) +
                      (WN_06 / TN_06) * wx + tx;

            if (notExceedingK && notExceedingM) {
                mat_As_p[dest] =
                    mat_A_p[src];
                mat_As_m[dest] =
                    mat_A_m[src];
            } else {
                mat_As_m[dest] = 0.0f;
                mat_As_p[dest] = 0.0f;
            }
        }
        mat_A_tile_offs += BK_06;

        // Each thread loads TN_06 values into mat_Bs
        for (int tn = 0; tn < TN_06; ++tn) {
            if ((BN_06 * bx + WN_06 * wx + TN_06 * tx + tn < N) &&
                (k + (WM_06 / TM_06) * wy + ty < K)) {
                mat_Bs[BN_06 * ((WM_06 / TM_06) * wy + ty) + WN_06 * wx +
                       TN_06 * tx + tn] =
                    mat_B[mat_B_tile_offs + N * ((WM_06 / TM_06) * wy + ty) +
                          WN_06 * wx + TN_06 * tx + tn] +
                    (1
                     << (I_BIT -
                         1)); // Include shifting to pos range (+ 2^(B-1)) here
            } else {
                mat_Bs[BN_06 * ((WM_06 / TM_06) * wy + ty) + WN_06 * wx +
                       TN_06 * tx + tn] = 0.0f;
            }
        }
        mat_B_tile_offs += N * BK_06;
        __syncthreads();

        // Temporary results of the TM_06xTN_06 mini-GEMM within a thread (for
        // each bit)
        float tmp_bits[TM_06 * SPLIT_SIZE][TN_06 * (I_BIT+1)] = {0.0f};
        int32_t tmp[TM_06][TN_06] = {0};

        float i_step_size_[] = {
            25, 6.25,
            3.125}; // Each step size depends on bits that are stored per cell.
                    // This vector stores sizes for all different cells.
        int32_t shift_[] = {7, 4, 0};

        // *******************************************************************
        // ***** This part will be discussed in "docs/01_register_blocking.md"
        for (int tm = 0; tm < SPLIT_SIZE * TM_06; ++tm) {
            for (int tn = 0; tn < TN_06; ++tn) {
                for (int bk = 0; bk < BK_06; ++bk) {

                    for (size_t i_bit = 0; i_bit < I_BIT+1; ++i_bit) {
                        int32_t b_val = mat_Bs[BN_06 * bk + WN_06 * wx + TN_06 * tx + tn];
                        int32_t b_bit =
                            (b_val >>
                             i_bit) &
                            1;
                        int A_idx = BK_06 * (SPLIT_SIZE * (WM_06 * wy + TM_06 * ty) + tm) +
                                     bk;
                        float diff =
                            mat_As_p[A_idx] -
                            mat_As_m[A_idx];

                        tmp_bits[tm][tn * I_BIT + i_bit] += diff * b_bit;
                    }
                }
                // ADC handling
                for (size_t i_bit = 0; i_bit < I_BIT + 1; ++i_bit) {
                    int32_t cast = static_cast<int32_t>(
                        round(analog_digital_conversion(
                                    tmp_bits[tm][tn * I_BIT + i_bit]) /
                                i_step_size_[tm % SPLIT_SIZE] * std::pow(2, shift_[tm % SPLIT_SIZE]) *
                                std::pow(2, i_bit)));
                    // Watchout to only write 3 elements in tmp
                    tmp[tm/SPLIT_SIZE][tn] += cast;
                }
            }
        }
        // *******************************************************************

        // Each thread copies its part of the block to res
        for (int tm = 0; tm < TM_06; ++tm) {
            for (int tn = 0; tn < TN_06; ++tn) {

                // Subtract compile time constant
                tmp[tm][tn] -= ((sum_w_)[BM_06 * by + WM_06 * wy + TM_06 * ty + tm] << (I_BIT - 1));

                bool condition1 = (bx * BN_06 + WN_06 * wx + TN_06 * tx + tn < N); // correct
                bool condition2 = (by * BM_06 + WM_06 * wy + TM_06 * ty + tm < M); // correct
                if (condition1 &&
                    condition2) {
                    // Again, plain copying to C matrix
                    const unsigned int res_elem_addr =
                        res_tile_offs + N * (WM_06 * wy + TM_06 * ty + tm) +
                        WN_06 * wx + TN_06 * tx + tn;
                    res[res_elem_addr] += tmp[tm][tn];
                }
            }
        }
        __syncthreads();
    }
}

// This is a regular C++ function you can call from outside
extern "C" void a_mmm_launch(int32_t *res, const float *mat_A_p,
                             const float *mat_A_m, const int32_t *mat_B, int m,
                             int k, int n, int32_t *sum_w_) {
    float *d_a_p, *d_a_m;
    int32_t *d_b, *d_c, *d_sum_w_;

    cudaMalloc(&d_a_p, m * k * SPLIT_SIZE * sizeof(float));
    cudaMalloc(&d_a_m, m * k * SPLIT_SIZE * sizeof(float));
    cudaMalloc(&d_b, n * k * sizeof(int32_t));
    cudaMalloc(&d_c, m * n * sizeof(int32_t));
    cudaMalloc(&d_sum_w_, 32 * sizeof(int32_t));

    std::cout << "m: " << m << ", k: " << k << std::endl;
    std::cout << "Cu: mat_A_p[64]: " << mat_A_p[2 * 3] << std::endl;

    cudaMemcpy(d_a_p, mat_A_p, m * k * SPLIT_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_m, mat_A_m, m * k * SPLIT_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, mat_B, n * k * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, res, m * n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum_w_, sum_w_, 32 * sizeof(int32_t), cudaMemcpyHostToDevice);

    dim3 gridDim_06(CEIL_DIV(n, BN_06), CEIL_DIV(m, BM_06), 1);
    std::cout << "Blockcount: " << gridDim_06.x * gridDim_06.y * gridDim_06.z << std::endl;
    dim3 blockDim_06(BN_06 / TN_06, BM_06 / TM_06, 1);
    std::cout << "Threadcount: " << blockDim_06.x * blockDim_06.y * blockDim_06.z << std::endl;
    mmm_kernel<<<gridDim_06, blockDim_06>>>(m, n, k, d_a_p, d_a_m, d_b, d_c, d_sum_w_);

    cudaMemcpy(res, d_c, m * n * sizeof(int32_t),
               cudaMemcpyDeviceToHost); // Memcpy call waits for kernel
                                        // execution to finish
    float res2[m][n] = {0.0f};

    std::cout << "m: " << m << ", n: " << n << std::endl;
    std::cout << "Res : " << std::endl;
    for(int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << ", " << std::to_string(res[i*n+j]);
        }
        std::cout << std::endl;
    }

    cudaFree(d_a_p);
    cudaFree(d_a_m);
    cudaFree(d_b);
    cudaFree(d_c);
}
