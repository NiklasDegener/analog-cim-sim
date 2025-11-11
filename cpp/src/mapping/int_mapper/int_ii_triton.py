import math
import torch
import triton
import triton.language as tl
from itertools import product
import numpy as np
from triton.testing import do_bench

@triton.jit
def analog_digital_conversion(current: torch.Tensor):
    clipped = tl.minimum(tl.maximum(current, -800), 800)
    # Somehow triton does not want to inline a separated round function    
    res = (tl.where(clipped/0.00152588 >= 0, tl.floor(clipped/0.00152588 + 0.5), tl.ceil(clipped/0.00152588 - 0.5))) * 0.00152588
    return clipped

@triton.jit
def matmul_kernel(dbg_ptr, a_low_ptr, a_mid_ptr, a_high_ptr, b_ptr, c_ptr, M, N, K, XBAR_M: tl.constexpr, XBAR_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, i_step_size: tl.constexpr, shift: tl.constexpr, powers_ptr, sum_w):
    """
    C = A x B
    A: (M, K) float32
    B: (K, N) float32
    C: (M, N) float32
    """
    # pid: program ID
    # Corresponds to blockIdx.x in CUDA (axis=0)
    # Get the maximum number of program IDs: tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)

    # Maximum number of program IDs along M and N axes
    # tl.num_programs(axis=0) = num_pid_m * num_pid_n
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # No swizzle here
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Initial BLOCK_SIZE_M × BLOCK_SIZE_N tile that this program instance will compute
    acc_low0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_low1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_low2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_low3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_low4 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_low5 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_low6 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_low7 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_mid0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_mid1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_mid2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_mid3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_mid4 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_mid5 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_mid6 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_mid7 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_high0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_high1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_high2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_high3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_high4 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_high5 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_high6 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_high7 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Do xbar tiling
    for x in range(0, tl.cdiv(K, XBAR_N)):
    #for x in range(0, 1):
        xbar_id_k = x * XBAR_N
        
        # load A tile
        a_block_offset = pid_m * BLOCK_SIZE_M * K # y-offset due to block position
        a_xbar_offset = xbar_id_k + a_block_offset # Offset a_ptr later on
        # load B tile
        b_block_offset = pid_n * BLOCK_SIZE_N
        b_xbar_offset = b_ptr + x * XBAR_M * N + b_block_offset
        
        # This assumes BLOCK_SIZE_K <= XBAR_N
        for k in range(0, tl.cdiv(tl.minimum(XBAR_N, K), BLOCK_SIZE_K)):
            # load A-tile
            a_k_offset = a_xbar_offset + k * BLOCK_SIZE_K
            # load all 3 splits for A
            a_xbar_ptrs = a_k_offset + (tl.arange(0, BLOCK_SIZE_M)[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]) # may be problematic. N is not a dimension of mat_A
            
            mask_a_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] < M
            mask_a_k = (xbar_id_k + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] < K
            a_low_k = tl.load(a_low_ptr + a_xbar_ptrs, mask=mask_a_m & mask_a_k)
            a_mid_k = tl.load(a_mid_ptr + a_xbar_ptrs, mask=mask_a_m & mask_a_k)
            a_high_k = tl.load(a_high_ptr + a_xbar_ptrs, mask=mask_a_m & mask_a_k)
            
            # load B-tile
            b_k_offset = b_xbar_offset + k * BLOCK_SIZE_K * N
            b_xbar_ptrs = b_k_offset + (tl.arange(0, BLOCK_SIZE_K)[:, None] * N + tl.arange(0, BLOCK_SIZE_N))
            
            mask_b_k = (x * XBAR_M + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[:, None] < K
            mask_b_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
            b_k = tl.load(b_xbar_ptrs, mask=mask_b_k & mask_b_n)

            # Shift b to positive range
            b_k += 2**7
            
            # i_bit results have to be seperated for ADC handling to be able to recombine them correctly
            '''
            for i_bit in range(0, 8):
                #accumulator += torch.matmul(a_k, ((b_k >> i_bit) & 1)) # Doesn't work due to size mismatch
                # Expand b_k to a_k shape
                # OR do it manually:
                accumulator_low[i_bit] += tl.dot(a_low_k, ((b_k >> i_bit) & 1).to(tl.float32))
                accumulator_mid[i_bit] += tl.dot(a_mid_k, ((b_k >> i_bit) & 1).to(tl.float32))
                accumulator_high[i_bit] += tl.dot(a_high_k.to(tl.float32), ((b_k >> i_bit) & 1).to(tl.float32))
            '''
            
            # Manually unroll the i_bit loop because accesses to local memory are hard to realize
            acc_low0 += tl.dot(a_low_k, (b_k & 1).to(tl.float32))
            #accumulator = tl.dot(a_low_k, ((b_k >> 0) & 1).to(tl.float32))
            #accumulator = tl.broadcast_to(a_low_k, (64, 64)) #(64,16)
            '''
            offs_cm = pid_m * 16 + tl.arange(0, 16)
            offs_cn = pid_n * 64 + tl.arange(0, 64)
            c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, b_k, mask=c_mask)
            '''
            
            acc_mid0 += tl.dot(a_mid_k, (b_k & 1).to(tl.float32))
            acc_high0 += tl.dot(a_high_k.to(tl.float32), (b_k & 1).to(tl.float32))
            acc_low1 += tl.dot(a_low_k, ((b_k >> 1) & 1).to(tl.float32))
            acc_mid1 += tl.dot(a_mid_k, ((b_k >> 1) & 1).to(tl.float32))
            acc_high1 += tl.dot(a_high_k.to(tl.float32), ((b_k >> 1) & 1).to(tl.float32))
            acc_low2 += tl.dot(a_low_k, ((b_k >> 2) & 1).to(tl.float32))
            acc_mid2 += tl.dot(a_mid_k, ((b_k >> 2) & 1).to(tl.float32))
            acc_high2 += tl.dot(a_high_k.to(tl.float32), ((b_k >> 2) & 1).to(tl.float32))
            acc_low3 += tl.dot(a_low_k, ((b_k >> 3) & 1).to(tl.float32))
            acc_mid3 += tl.dot(a_mid_k, ((b_k >> 3) & 1).to(tl.float32))
            acc_high3 += tl.dot(a_high_k.to(tl.float32), ((b_k >> 3) & 1).to(tl.float32))
            acc_low4 += tl.dot(a_low_k, ((b_k >> 4) & 1).to(tl.float32))
            acc_mid4 += tl.dot(a_mid_k, ((b_k >> 4) & 1).to(tl.float32))
            acc_high4 += tl.dot(a_high_k.to(tl.float32), ((b_k >> 4) & 1).to(tl.float32))
            acc_low5 += tl.dot(a_low_k, ((b_k >> 5) & 1).to(tl.float32))
            acc_mid5 += tl.dot(a_mid_k, ((b_k >> 5) & 1).to(tl.float32))
            acc_high5 += tl.dot(a_high_k.to(tl.float32), ((b_k >> 5) & 1).to(tl.float32))
            acc_low6 += tl.dot(a_low_k, ((b_k >> 6) & 1).to(tl.float32))
            acc_mid6 += tl.dot(a_mid_k, ((b_k >> 6) & 1).to(tl.float32))
            acc_high6 += tl.dot(a_high_k.to(tl.float32), ((b_k >> 6) & 1).to(tl.float32))
            acc_low7 += tl.dot(a_low_k, ((b_k >> 7) & 1).to(tl.float32))
            acc_mid7 += tl.dot(a_mid_k, ((b_k >> 7) & 1).to(tl.float32))
            acc_high7 += tl.dot(a_high_k.to(tl.float32), ((b_k >> 7) & 1).to(tl.float32))

        # Do ADC handling for each XBAR (also unroll this manually)
        accumulator += analog_digital_conversion(acc_high0) / i_step_size[0] * 2**shift[0] * 1
        accumulator += analog_digital_conversion(acc_mid0) / i_step_size[1] * 2**shift[1] * 1
        accumulator += analog_digital_conversion(acc_low0) / i_step_size[2] * 2**shift[2] * 1
        accumulator += analog_digital_conversion(acc_high1) / i_step_size[0] * 2**shift[0] * 2
        accumulator += analog_digital_conversion(acc_mid1) / i_step_size[1] * 2**shift[1] * 2
        accumulator += analog_digital_conversion(acc_low1) / i_step_size[2] * 2**shift[2] * 2
        accumulator += analog_digital_conversion(acc_high2) / i_step_size[0] * 2**shift[0] * 4
        accumulator += analog_digital_conversion(acc_mid2) / i_step_size[1] * 2**shift[1] * 4
        accumulator += analog_digital_conversion(acc_low2) / i_step_size[2] * 2**shift[2] * 4
        accumulator += analog_digital_conversion(acc_high3) / i_step_size[0] * 2**shift[0] * 8
        accumulator += analog_digital_conversion(acc_mid3) / i_step_size[1] * 2**shift[1] * 8
        accumulator += analog_digital_conversion(acc_low3) / i_step_size[2] * 2**shift[2] * 8
        accumulator += analog_digital_conversion(acc_high4) / i_step_size[0] * 2**shift[0] * 16
        accumulator += analog_digital_conversion(acc_mid4) / i_step_size[1] * 2**shift[1] * 16
        accumulator += analog_digital_conversion(acc_low4) / i_step_size[2] * 2**shift[2] * 16
        accumulator += analog_digital_conversion(acc_high5) / i_step_size[0] * 2**shift[0] * 32
        accumulator += analog_digital_conversion(acc_mid5) / i_step_size[1] * 2**shift[1] * 32
        accumulator += analog_digital_conversion(acc_low5) / i_step_size[2] * 2**shift[2] * 32
        accumulator += analog_digital_conversion(acc_high6) / i_step_size[0] * 2**shift[0] * 64
        accumulator += analog_digital_conversion(acc_mid6) / i_step_size[1] * 2**shift[1] * 64
        accumulator += analog_digital_conversion(acc_low6) / i_step_size[2] * 2**shift[2] * 64
        accumulator += analog_digital_conversion(acc_high7) / i_step_size[0] * 2**shift[0] * 128
        accumulator += analog_digital_conversion(acc_mid7) / i_step_size[1] * 2**shift[1] * 128
        accumulator += analog_digital_conversion(acc_low7) / i_step_size[2] * 2**shift[2] * 128
        
        # Reset accumulators
        acc_low0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_low1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_low2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_low3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_low4 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_low5 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_low6 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_low7 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_mid0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_mid1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_mid2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_mid3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_mid4 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_mid5 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_mid6 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_mid7 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_high0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_high1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_high2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_high3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_high4 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_high5 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_high6 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_high7 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


    sums = tl.load(sum_w + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    sums = tl.broadcast_to(tl.expand_dims(sums, 1), (BLOCK_SIZE_M, BLOCK_SIZE_N))

    accumulator -= sums * 2**7

    
    c = accumulator

    # Row and column indices of the C-tile
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # BLOCK_SIZE_M × BLOCK_SIZE_N matrix with pointers to the C-tile elements
    c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    # Mask to avoid out-of-bounds accesses
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def to_analog(tensor: torch.Tensor):
    '''
        Generates the i_m/p difference tensor for a [4,3,0] split
    '''
    
    # Modulo defined differently on negative values, so do it on abs and restore signs afterwards
    tensor_abs = torch.abs(tensor)
    tensor_sign = torch.sign(tensor)
    tensor_low = tensor_abs % pow(2, 4) * 3.125 * tensor_sign
    tensor_abs >>= 4
    tensor_mid = tensor_abs % pow(2, 3) * 6.25 * tensor_sign
    tensor_abs >>= 3
    tensor_high = tensor_abs * 25 * tensor_sign
    
    # Rather return 3 matrices, interleaved one is hard to handle in triton
    '''
    res = torch.empty((tensor.shape[0]*3, tensor.shape[1]), device=tensor.device, dtype=torch.float32)
    for m in range(0, res.shape[0]):
        for n in range(0, res.shape[1]):
            if m % 3 == 0:
                res[m][n] = tensor_high[m//3][n]
            if m % 3 == 1:
                res[m][n] = tensor_mid[m//3][n]
            if m % 3 == 2:
                res[m][n] = tensor_low[m//3][n]
    # Don't add HRS, is removed through subtraction
    '''
    
    return tensor_low, tensor_mid, tensor_high

def gen_sum_w(tensor: torch.Tensor):
    '''
        Generates the sum_w_ vector (sums over n in matrix A)
    '''
    return tensor.sum(dim=1)

def check_and_launch_matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    dbg = torch.zeros((M, N), device=a.device, dtype=torch.float32)

    a_low, a_mid, a_high = to_analog(a)
    sum_w_ = gen_sum_w(a)
    i_step_size = (25, 6.25, 3.125)
    shift = (7, 4, 0)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128, 256], device=a.device)

    # Number of blocks = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)
    # Number of threads per block: num_warps * 32
    # Each thread computes ≈ (BLOCK_SIZE_M * BLOCK_SIZE_N) / (num_warps * 32) elements of C
    # Define a 1-D grid of blocks
    grid = lambda META: ( \
        triton.cdiv(M, META['BLOCK_SIZE_M']) * \
        triton.cdiv(N, META['BLOCK_SIZE_N']), )
    time_ms = do_bench(lambda: matmul_kernel[grid](dbg, a_low, a_mid, a_high, b, c, M, N, K, XBAR_M=32, XBAR_N=32, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=16, i_step_size=i_step_size, shift=shift, powers_ptr=powers, sum_w=sum_w_))
    print("Kernel runtime: " + str(time_ms))
    int_tensor = c.cpu().to(int)
    np.savetxt("triton.txt", int_tensor.numpy(), fmt='%d')
    
    
    
    return c

if __name__ == "__main__":
    
    m_matrix = 256 # Needs to be at least 256 so there is a valid swizzle config
    k_matrix = 256
    n_matrix = 256
    np.random.seed(42)
    mat_A = torch.randint(-127, 127, (m_matrix, k_matrix), device='cuda').float()
    print(mat_A.cpu()[0:64][0:64].to(torch.int32))
    mat_B = torch.randint(-127, 127, (k_matrix, n_matrix), device='cuda').float()
    
    mat_C = check_and_launch_matmul(mat_A, mat_B)
    
    #print(np.dot(mat_A.cpu(), mat_B.cpu()))
    
    np.savetxt("triton_goal.txt", np.dot(mat_A.cpu(), mat_B.cpu()), fmt='%d')
    np.testing.assert_array_equal(mat_C.cpu(), np.dot(mat_A.cpu(), mat_B.cpu()))
    