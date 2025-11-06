import math
import torch
import triton
import triton.language as tl
from itertools import product
import numpy as np

@triton.jit
def matmul_kernel(dbg_ptr, a_ptr, b_ptr, c_ptr, M, N, K, XBAR_M: tl.constexpr, XBAR_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
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

    # Row indices in tile (pid_m, pid_n) in A and C
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  #Is the modulo correct? Overflow would access low elements
    # Column indices in tile (pid_m, pid_n) in B and C
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # Relative indices in K dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # BLOCK_SIZE_M × BLOCK_SIZE_K matrix with pointers to the A-tile elements
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])  #[:, None] converts to column-vector
    # BLOCK_SIZE_K × BLOCK_SIZE_N matrix with pointers to the B-tile elements
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    # Initial BLOCK_SIZE_M × BLOCK_SIZE_N tile that this program instance will compute
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Do xbar tiling
    for x in range(0, tl.cdiv(K, XBAR_N)):
    #for x in range(0, 1):
        xbar_id_k = x * XBAR_N
        
        # load A tile
        a_block_offset = pid_m * BLOCK_SIZE_M * M
        a_xbar_offset = a_ptr + xbar_id_k + a_block_offset
        # load B tile
        b_block_offset = pid_n * BLOCK_SIZE_N
        b_xbar_offset = b_ptr + x * XBAR_M * N + b_block_offset
        
        # assume BLOCK_SIZE_K <= XBAR_N
        for k in range(0, tl.cdiv(XBAR_N, BLOCK_SIZE_K)):
            # load A-tile
            a_k_offset = a_xbar_offset + k * BLOCK_SIZE_K
            a_xbar_ptrs = a_k_offset + (tl.arange(0, BLOCK_SIZE_M)[:, None] * N + tl.arange(0, BLOCK_SIZE_K))
            a_test = tl.load(a_xbar_ptrs)
            
            # load B-tile
            b_k_offset = b_xbar_offset + k * BLOCK_SIZE_K * N
            b_xbar_ptrs = b_k_offset + (tl.arange(0, BLOCK_SIZE_K)[:, None] * N + tl.arange(0, BLOCK_SIZE_N))
            b_test = tl.load(b_xbar_ptrs)

            prod = tl.dot(a_test, b_test)
            accumulator += prod

    c = accumulator

    # Row and column indices of the C-tile
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # BLOCK_SIZE_M × BLOCK_SIZE_N matrix with pointers to the C-tile elements
    c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    # Mask to avoid out-of-bounds accesses
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def check_and_launch_matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    dbg = torch.zeros((M, N), device=a.device, dtype=torch.float32)

    #np.savetxt("triton_goal.txt", a.cpu()[0:64][0:64].numpy(), fmt='%d')

    # Number of blocks = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)
    # Number of threads per block: num_warps * 32
    # Each thread computes ≈ (BLOCK_SIZE_M * BLOCK_SIZE_N) / (num_warps * 32) elements of C
    # Define a 1-D grid of blocks
    grid = lambda META: ( \
        triton.cdiv(M, META['BLOCK_SIZE_M']) * \
        triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](dbg, a, b, c, M, N, K, XBAR_M=32, XBAR_N=32, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=16)
    print(c.cpu()[0:64][0:64].to(torch.int32))
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
    