# Auto-generated submission.py (2025-11-14T16:02:25Z)
# Combines:
#  - gemv/reference.py (verbatim)
#  - gemv/custom_kernel.cu (embedded as string)
#  - gemv/custom_kernel.py (adapted to use embedded CUDA source and in-file reference symbols)

# ===== reference.py =====
import torch
from task import input_t, output_t
from utils import make_match_reference

# Scaling factor vector size
sf_vec_size = 16


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data

    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape

    # Call torch._scaled_mm to compute the GEMV result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, 0, l_idx] = res[:, 0]
    return c_ref


def generate_input(
    m: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled GEMV.

    Args:
        m: Number of rows in matrix A
        k: Number of columns in A (and length of vector b)
        l: Batch size
        seed: Random seed for reproducibility

    Returns:
        Tuple of (a, b, scale_a, scale_b, c) where:
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [1, k, l] - Input vector in torch.float4e2m1fn_x2 data type
            scale_a: [m, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b: [1, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_a_permuted: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b_permuted: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, 1, l] - Output vector in torch.float16 data type
    """
    torch.manual_seed(seed)

    # GEMV N dimension is always 1
    n = 1
    # Scaling factor needs to pad the N size to 128
    n_padded_128 = 128

    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        0, 4, (l, m, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    # Pad b tensor's N dimension to 128 to call torch._scaled_mm for nvfp4 dot product computation
    b_ref = torch.randint(
        0, 4, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    # Helper function to prepare the scale factor tensors for both reference
    # kernel and customize kernel. The customized data layout can be found in:
    # https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    def create_scale_factor_tensors(l, mn, sf_k):
        # Create the reference scale factor tensor (mn, sf_k, l) on CPU.
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        # Init with uint8 tensor, then convert to float8_e4m3fn
        ref_f8_random_int = torch.randint(
            0, 3, ref_shape, dtype=torch.int8, device="cuda"
        )
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        # permute to match ref_permute_order
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,  # batch size
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        # Reorder scale factor tensor to (32, 4, rest_m, 4, rest_k, l) layout
        # Which is needed by the CuTe customized kernel
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        # Generate a random int8 tensor, then convert to float8_e4m3fn
        rand_int_tensor = torch.randint(
            0, 3, mma_shape, dtype=torch.int8, device="cuda"
        )
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        # Permute according to mma_permute_order
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(
            *mma_permute_order
        )

        # GPU-side vectorized reordering (replaces slow CPU nested loops)
        # Create index grids for all dimensions
        i_idx = torch.arange(mn, device="cuda")
        j_idx = torch.arange(sf_k, device="cuda")
        b_idx = torch.arange(l, device="cuda")

        # Create meshgrid for all combinations of (i, j, b)
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing="ij")

        # Calculate target indices in vectorized manner
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k

        # Perform the reordering with advanced indexing (all on GPU)
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = (
            ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]
        )

        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)

    sfa_ref = sfa_ref_cpu.to("cuda")
    sfb_ref = sfb_ref_cpu.to("cuda")

    return (a_ref, b_ref, sfa_ref, sfb_ref, sfa_permuted, sfb_permuted, c_ref)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)

# ===== custom_kernel.cu (embedded) =====
custom_kernel_cuda_source = r"""
// NVFP4 GEMV with CTA-level B/SFB staging (B200 tuned)
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.hpp>
#include <cuda_runtime.h>
#include <torch/extension.h>

using at::Tensor;

static inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

__forceinline__ static __device__ __half fp8e4m3_to_half(unsigned char x) {
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)x, __NV_E4M3);
    return *reinterpret_cast<__half*>(&h);
}

__device__ __align__(4) unsigned int g_fp4x2_lut[256];
__forceinline__ static __device__ __half2 fp4x2e2m1_to_half2_lut(unsigned char x, const unsigned int* __restrict__ lut) {
    __half2 h2; reinterpret_cast<unsigned int&>(h2) = lut[x]; return h2;
}
__global__ void init_fp4x2_lut_kernel() {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 256u) {
        __half2_raw h2r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)idx, __NV_E2M1);
        g_fp4x2_lut[idx] = h2r.x;
    }
}

template <int WARPS_PER_BLOCK, int ROWS_PER_WARP, int MIN_BLOCKS>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, MIN_BLOCKS)
void nvfp4_gemv_kernel_sfa2d_warp_rows(
    const unsigned char* __restrict__ A_l_m_k2_base,
    int64_t a_sL, int64_t a_sM, int64_t a_sK2,
    const unsigned char* __restrict__ B_l_k2_n_base,
    int64_t b_sL, int64_t b_sK2,
    const unsigned char* __restrict__ SFA2_base,
    int64_t sfa2_sM, int64_t sfa2_sJ, int64_t sfa2_sL,
    const unsigned char* __restrict__ SFB2_base,
    int64_t sfb2_sJ, int64_t sfb2_sL,
    __half* __restrict__ out_ml,
    int64_t out_sM, int64_t out_sL,
    int64_t M, int64_t L,
    int64_t sf_k) {

    extern __shared__ unsigned int shmem32[];
    // Layout of shared segment:
    // [0..255]   : LUT (uint32_t per entry)
    // [next .. ] : B decoded as __half2[8] per j  (32 bytes per j)
    // [next .. ] : SFB converted to __half (2 bytes per j)
    unsigned int* fp4x2_lut = shmem32;  // 256 entries (1KB)
    if (threadIdx.x < 256) shmem32[threadIdx.x] = g_fp4x2_lut[threadIdx.x];
    __syncthreads();

    const int li = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m_base = blockIdx.x * (WARPS_PER_BLOCK * ROWS_PER_WARP) + warp_id * ROWS_PER_WARP;
    if (li >= L || m_base >= M) return;

    const unsigned char* A_base = A_l_m_k2_base + (size_t)li * (size_t)a_sL;
    const unsigned char* B_base = B_l_k2_n_base + (size_t)li * (size_t)b_sL;
    const unsigned char* SFA_row_base[ROWS_PER_WARP];
    bool valid_row[ROWS_PER_WARP];
    const unsigned char* a_row_ptr[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int m = m_base + r;
        valid_row[r] = (m < M);
        if (valid_row[r]) {
            SFA_row_base[r] = SFA2_base + (size_t)m * (size_t)sfa2_sM + (size_t)li * (size_t)sfa2_sL;
            a_row_ptr[r] = A_base + (size_t)m * (size_t)a_sM;
        } else {
            SFA_row_base[r] = nullptr;
            a_row_ptr[r] = nullptr;
        }
        acc[r] = 0.0f;
    }
    const unsigned char* SFB_l_base = SFB2_base + (size_t)li * (size_t)sfb2_sL;
    __half* out_base = out_ml + (size_t)li * (size_t)out_sL;
    const unsigned full_mask = 0xFFFFFFFFu;

    // Stage B (packed fp4x2 bytes) and SFB (fp8) into shared memory for this CTA.
    // Every block cooperatively loads all j in [0, sf_k).
    // Using 64-bit vectorized loads for B and 8-bit loads for SFB, then convert SFB to __half in registers.
    // Shared pointers (compute after LUT region):
    // Place B array aligned to 16 bytes after LUT (256*4 bytes).
    const size_t lut_bytes = 256u * sizeof(unsigned int);
    unsigned char* smem_bytes = reinterpret_cast<unsigned char*>(shmem32);
    unsigned long long* smem_B = reinterpret_cast<unsigned long long*>(smem_bytes + lut_bytes);
    __half* smem_SFBh = reinterpret_cast<__half*>(smem_B + (size_t)sf_k);

    // Cooperative load: distribute j across all threads in the block
    for (int64_t j = threadIdx.x; j < sf_k; j += blockDim.x) {
        // Load B 8B packet for group j
        const int64_t k2_base = j << 3;
        unsigned long long b_pack =
            __ldg(reinterpret_cast<const unsigned long long*>(B_base + (size_t)k2_base * (size_t)b_sK2));
        smem_B[j] = b_pack;
        // Load and convert SFB[j] to __half
        unsigned char sfb_u8 = __ldg(reinterpret_cast<const unsigned char*>(SFB_l_base + (size_t)j * (size_t)sfb2_sJ));
        __half sfb_h = fp8e4m3_to_half(sfb_u8);
        smem_SFBh[j] = sfb_h;
    }
    __syncthreads();

    // Each warp now iterates j using staged B/SFB from shared memory.
    for (int64_t j = lane; j < sf_k; j += 64) {
        const int64_t j0 = j;
        const int64_t j1 = j + 32;

        // j0
        const int64_t k2_base0 = j0 << 3;
        const uint64_t b_pack0 = smem_B[j0];
        const __half sb0 = smem_SFBh[j0];

        __half scale0_h[ROWS_PER_WARP];
        uint64_t a_pack0[ROWS_PER_WARP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if (valid_row[r]) {
                const unsigned char* sfa_ptr = SFA_row_base[r] + (size_t)j0 * (size_t)sfa2_sJ;
                unsigned char sa0_u8 = __ldg(reinterpret_cast<const unsigned char*>(sfa_ptr));
                __half sa0_h = fp8e4m3_to_half(sa0_u8);
                scale0_h[r] = __hmul(sa0_h, sb0);
                a_pack0[r] = __ldg(reinterpret_cast<const unsigned long long*>(a_row_ptr[r] + (size_t)k2_base0 * (size_t)a_sK2));
            } else {
                scale0_h[r] = __float2half(0.0f);
                a_pack0[r] = 0ull;
            }
        }

        __half2 accA[ROWS_PER_WARP];
        __half2 accB[ROWS_PER_WARP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) { accA[r] = __float2half2_rn(0.0f); accB[r] = __float2half2_rn(0.0f); }
        for (int t = 0; t < 8; t += 2) {
            unsigned char b_byte = (unsigned char)((b_pack0 >> (t * 8)) & 0xFFu);
            __half2 b_h2 = fp4x2e2m1_to_half2_lut(b_byte, fp4x2_lut);
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (!valid_row[r]) continue;
                unsigned char a_byte = (unsigned char)((a_pack0[r] >> (t * 8)) & 0xFFu);
                __half2 a_h2 = fp4x2e2m1_to_half2_lut(a_byte, fp4x2_lut);
                accA[r] = __hfma2(a_h2, b_h2, accA[r]);
            }
            b_byte = (unsigned char)((b_pack0 >> ((t + 1) * 8)) & 0xFFu);
            b_h2 = fp4x2e2m1_to_half2_lut(b_byte, fp4x2_lut);
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (!valid_row[r]) continue;
                unsigned char a_byte = (unsigned char)((a_pack0[r] >> ((t + 1) * 8)) & 0xFFu);
                __half2 a_h2 = fp4x2e2m1_to_half2_lut(a_byte, fp4x2_lut);
                accB[r] = __hfma2(a_h2, b_h2, accB[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if (!valid_row[r]) continue;
            const __half2 s2 = __hadd2(accA[r], accB[r]);
            const __half s = __hadd(__low2half(s2), __high2half(s2));
            acc[r] += __half2float(__hmul(s, scale0_h[r]));
        }

        // j1
        if (j1 < sf_k) {
            const int64_t k2_base1 = j1 << 3;
            const uint64_t b_pack1 = smem_B[j1];
            const __half sb1 = smem_SFBh[j1];

            __half scale1_h[ROWS_PER_WARP];
            uint64_t a_pack1[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (valid_row[r]) {
                    const unsigned char* sfa_ptr = SFA_row_base[r] + (size_t)j1 * (size_t)sfa2_sJ;
                    unsigned char sa1_u8 = __ldg(reinterpret_cast<const unsigned char*>(sfa_ptr));
                    __half sa1_h = fp8e4m3_to_half(sa1_u8);
                    scale1_h[r] = __hmul(sa1_h, sb1);
                    a_pack1[r] = __ldg(reinterpret_cast<const unsigned long long*>(a_row_ptr[r] + (size_t)k2_base1 * (size_t)a_sK2));
                } else {
                    scale1_h[r] = __float2half(0.0f);
                    a_pack1[r] = 0ull;
                }
            }

            __half2 accA1[ROWS_PER_WARP];
            __half2 accB1[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) { accA1[r] = __float2half2_rn(0.0f); accB1[r] = __float2half2_rn(0.0f); }
            for (int t = 0; t < 8; t += 2) {
                unsigned char b_byte = (unsigned char)((b_pack1 >> (t * 8)) & 0xFFu);
                __half2 b_h2 = fp4x2e2m1_to_half2_lut(b_byte, fp4x2_lut);
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    if (!valid_row[r]) continue;
                    unsigned char a_byte = (unsigned char)((a_pack1[r] >> (t * 8)) & 0xFFu);
                    __half2 a_h2 = fp4x2e2m1_to_half2_lut(a_byte, fp4x2_lut);
                    accA1[r] = __hfma2(a_h2, b_h2, accA1[r]);
                }
                b_byte = (unsigned char)((b_pack1 >> ((t + 1) * 8)) & 0xFFu);
                b_h2 = fp4x2e2m1_to_half2_lut(b_byte, fp4x2_lut);
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    if (!valid_row[r]) continue;
                    unsigned char a_byte = (unsigned char)((a_pack1[r] >> ((t + 1) * 8)) & 0xFFu);
                    __half2 a_h2 = fp4x2e2m1_to_half2_lut(a_byte, fp4x2_lut);
                    accB1[r] = __hfma2(a_h2, b_h2, accB1[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (!valid_row[r]) continue;
                const __half2 s2 = __hadd2(accA1[r], accB1[r]);
                const __half s = __hadd(__low2half(s2), __high2half(s2));
                acc[r] += __half2float(__hmul(s, scale1_h[r]));
            }
        }
    }

    float sum[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) sum[r] = acc[r];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) sum[r] += __shfl_down_sync(full_mask, sum[r], offset);
    }
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int m = m_base + r;
            if (m < M) out_base[m * out_sM] = __float2half_rn(sum[r]);
        }
    }
}

// Host entry point (known-good launcher)
torch::Tensor custom_gemv_cuda(torch::Tensor a,
    torch::Tensor b,  // expects [Npad, K/2, L]
    torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda() && c.device().is_cuda(), "a, b, c must be CUDA tensors");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && c.dim() == 3, "Expected a,b,c to be 3D tensors");

    const int64_t M = a.size(0);
    const int64_t K2 = a.size(1);  // K packed as K/2 columns (fp4x2)
    const int64_t L = a.size(2);

    Tensor a_lmk2 = a.permute({2, 0, 1});
    Tensor b_lk2n = b.permute({2, 1, 0});
    Tensor out_ml = c.select(1, 0);
    auto out_strides = out_ml.strides();

    const int64_t sf_k = ceil_div(K2, (int64_t)8);  // K/16
    auto a_strides = a_lmk2.strides();
    auto b_strides = b_lk2n.strides();

    static bool lut_initialized = false;
    if (!lut_initialized) { auto stream = at::cuda::getCurrentCUDAStream(); init_fp4x2_lut_kernel<<<1, 256, 0, stream>>>(); lut_initialized = true; }
    const size_t lut_bytes = 256u * sizeof(unsigned int);

    TORCH_CHECK(sfa.dim() == 3 && sfb.dim() == 3, "sfa/sfb must be 3D dense tensors [M,sf_k,L] and [Npad,sf_k,L]");
    auto sfa_s = sfa.strides();
    auto sfb_s = sfb.strides();

    if (L >= 8) {
        // For larger batch L, increase per-CTA parallelism to reuse staged B/SFB
        // Use a moderate WPB to balance occupancy and L2 pressure.
        constexpr int WPB = 4;           // warps per block
        constexpr int RPW = 4;           // rows per warp
        dim3 block(WPB * 32);
        dim3 grid((unsigned)((M + WPB * RPW - 1) / (WPB * RPW)), (unsigned)L);
        size_t shmem_bytes = 256u * sizeof(unsigned int)                     // LUT
                           + (size_t)ceil_div((int64_t)K2, (int64_t)8) * 8  // B staged (uint64 per j)
                           + (size_t)ceil_div((int64_t)K2, (int64_t)8) * sizeof(__half); // SFB staged (__half per j)
        nvfp4_gemv_kernel_sfa2d_warp_rows<WPB, RPW, 1><<<grid, block, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const unsigned char*>(a_lmk2.data_ptr()), a_strides[0], a_strides[1], a_strides[2],
            reinterpret_cast<const unsigned char*>(b_lk2n.data_ptr()), b_strides[0], b_strides[1],
            reinterpret_cast<const unsigned char*>(sfa.data_ptr()), sfa_s[0], sfa_s[1], sfa_s[2],
            reinterpret_cast<const unsigned char*>(sfb.data_ptr()), sfb_s[1], sfb_s[2],
            reinterpret_cast<__half*>(out_ml.data_ptr()), out_strides[0], out_strides[1],
            M, L, sf_k);
    } else {
        // For small L, keep moderate CTA size for occupancy
        constexpr int WPB = 4;
        constexpr int RPW = 4;
        dim3 block(WPB * 32);
        dim3 grid((unsigned)((M + WPB * RPW - 1) / (WPB * RPW)), (unsigned)L);
        size_t shmem_bytes = 256u * sizeof(unsigned int)
                           + (size_t)ceil_div((int64_t)K2, (int64_t)8) * 8
                           + (size_t)ceil_div((int64_t)K2, (int64_t)8) * sizeof(__half);
        nvfp4_gemv_kernel_sfa2d_warp_rows<WPB, RPW, 2><<<grid, block, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const unsigned char*>(a_lmk2.data_ptr()), a_strides[0], a_strides[1], a_strides[2],
            reinterpret_cast<const unsigned char*>(b_lk2n.data_ptr()), b_strides[0], b_strides[1],
            reinterpret_cast<const unsigned char*>(sfa.data_ptr()), sfa_s[0], sfa_s[1], sfa_s[2],
            reinterpret_cast<const unsigned char*>(sfb.data_ptr()), sfb_s[1], sfb_s[2],
            reinterpret_cast<__half*>(out_ml.data_ptr()), out_strides[0], out_strides[1],
            M, L, sf_k);
    }
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_gemv_cuda", &custom_gemv_cuda, "NVFP4 GEMV custom kernel (raw CUDA)");
}
"""

# ===== custom_kernel.py (adapted) =====
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import os


# read custom_kernel.cu
root_dir = Path(__file__).parent

_cc = "".join(map(str, torch.cuda.get_device_capability()))
_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-Xptxas=-dlcm=ca",
    # Allow the compiler to choose register count for better ILP/latency hiding
    "-std=c++17",
    f"-gencode=arch=compute_{_cc},code=sm_{_cc}",
    f"-gencode=arch=compute_{_cc},code=compute_{_cc}",
]

custom_kernel_module = load_inline(
    name="custom_kernel_nvfp4_gemv",
    cpp_sources="""
#include <torch/extension.h>
torch::Tensor custom_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
""",
    cuda_sources=custom_kernel_cuda_source,
    extra_cuda_cflags=_cuda_cflags,
)
torch.cuda.empty_cache()


def custom_kernel(data: input_t) -> output_t:
    """Run the NVFP4 block-scaled GEMV kernel using dense 3D scale tensors.

    Expects:
      - sfa: [M, K/16, L] in torch.float8_e4m3fn
      - sfb: [Npad, K/16, L] in torch.float8_e4m3fn
    """
    assert len(data) == 7
    a, b, sfa, sfb, _sfa_perm, _sfb_perm, c = data

    custom_kernel_module.custom_gemv_cuda(a, b, sfa, sfb, c)
    return c


if __name__ == "__main__":

    def bench_custom(data_tuple, warmup=10, iters=50):
        for _ in range(warmup):
            _ = custom_kernel(data_tuple)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = custom_kernel(data_tuple)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))  # milliseconds
        return times_ms

    list_params = [
        (7168, 16384, 1),
        (4096, 7168, 8),
        (7168, 2048, 4),
    ]
    for params in list_params:
        M, K, L = params
        data = generate_input(M, K, L, seed=0)
        # Optional correctness check (set DO_CHECK=1 to enable)
        if os.environ.get("DO_CHECK", "0") == "1":
            out = custom_kernel(data)
            out = out.clone()
            results = check_implementation(data, out)
            print("Check implementation:", results)

        # End-to-end timing of custom_kernel (includes GPU scale prep + CUDA kernel)
        times = bench_custom(data, warmup=10, iters=50)
        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        print(
            f"E2E custom_kernel M={M} K={K} L={L}: avg {avg_ms:.3f} ms, min {min_ms:.3f} ms over {len(times)} runs (10 warmups)"
        )
