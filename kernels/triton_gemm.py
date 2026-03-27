import torch
import triton
import triton.language as tl

from tiny_gemm.quantization.packed_int4 import pack_int4_signed, quantize_per_tensor_int4

_INT4_GEMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 1, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
        num_warps=2,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
        num_warps=2,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
        num_warps=2,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 2, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 2},
        num_warps=2,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 4, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 2},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=2,
    ),
]

@triton.autotune(configs=_INT4_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def kernel_gemm_packed_int4(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    group_id = pid // grid_n
    group_id = group_id * GROUP_SIZE_M + (pid % GROUP_SIZE_M)
    pid_n = pid % grid_n
    pid_m = group_id

    if pid_m >= grid_m:
        return

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_blocks = (K + BLOCK_K - 1) // BLOCK_K

    for kb in range(k_blocks):
        k_offset = kb * (BLOCK_K // 2)

        a_row_mask = offs_m[:, None] < M
        a_col_mask = (k_offset + offs_k[None, :]) < (K // 2)
        a_mask = a_row_mask & a_col_mask

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (
            (k_offset + offs_k[None, :]) * stride_ak
        )
        a_packed = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.uint8)

        b_row_mask = (k_offset + offs_k[:, None]) < (K // 2)
        b_col_mask = offs_n[None, :] < N
        b_mask = b_row_mask & b_col_mask

        b_ptrs = B_ptr + ((k_offset + offs_k[:, None]) * stride_bk) + (
            offs_n[None, :] * stride_bn
        )
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.uint8)

        a_lo = a_packed & 0x0F
        a_hi = (a_packed >> 4) & 0x0F
        b_lo = b_packed & 0x0F
        b_hi = (b_packed >> 4) & 0x0F

        a_lo = tl.where(a_lo >= 8, a_lo - 16, a_lo).to(tl.float32)
        a_hi = tl.where(a_hi >= 8, a_hi - 16, a_hi).to(tl.float32)
        b_lo = tl.where(b_lo >= 8, b_lo - 16, b_lo).to(tl.float32)
        b_hi = tl.where(b_hi >= 8, b_hi - 16, b_hi).to(tl.float32)

        acc += tl.dot(a_lo, b_lo) + tl.dot(a_hi, b_hi)

    c_row_mask = offs_m[:, None] < M
    c_col_mask = offs_n[None, :] < N
    c_mask = c_row_mask & c_col_mask

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def kernel_gemm_packed_int4_static(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    group_id = pid // grid_n
    group_id = group_id * GROUP_SIZE_M + (pid % GROUP_SIZE_M)
    pid_n = pid % grid_n
    pid_m = group_id

    if pid_m >= grid_m:
        return

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_blocks = (K + BLOCK_K - 1) // BLOCK_K

    for kb in range(k_blocks):
        k_offset = kb * (BLOCK_K // 2)

        a_row_mask = offs_m[:, None] < M
        a_col_mask = (k_offset + offs_k[None, :]) < (K // 2)
        a_mask = a_row_mask & a_col_mask

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (
            (k_offset + offs_k[None, :]) * stride_ak
        )
        a_packed = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.uint8)

        b_row_mask = (k_offset + offs_k[:, None]) < (K // 2)
        b_col_mask = offs_n[None, :] < N
        b_mask = b_row_mask & b_col_mask

        b_ptrs = B_ptr + ((k_offset + offs_k[:, None]) * stride_bk) + (
            offs_n[None, :] * stride_bn
        )
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.uint8)

        a_lo = a_packed & 0x0F
        a_hi = (a_packed >> 4) & 0x0F
        b_lo = b_packed & 0x0F
        b_hi = (b_packed >> 4) & 0x0F

        a_lo = tl.where(a_lo >= 8, a_lo - 16, a_lo).to(tl.float32)
        a_hi = tl.where(a_hi >= 8, a_hi - 16, a_hi).to(tl.float32)
        b_lo = tl.where(b_lo >= 8, b_lo - 16, b_lo).to(tl.float32)
        b_hi = tl.where(b_hi >= 8, b_hi - 16, b_hi).to(tl.float32)

        acc += tl.dot(a_lo, b_lo) + tl.dot(a_hi, b_hi)

    c_row_mask = offs_m[:, None] < M
    c_col_mask = offs_n[None, :] < N
    c_mask = c_row_mask & c_col_mask

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=c_mask)


def get_best_config(
    m: int,
    n: int,
    k: int,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
):
    return kernel_gemm_packed_int4.cache.get((m, n, k, a_dtype, b_dtype, c_dtype))


def _bucket_m(m: int) -> str | None:
    if m <= 1:
        return "m1"
    if m <= 4:
        return "small"
    if m <= 8:
        return "medium"
    return None


def _shape_family(n: int, k: int) -> str:
    if n > k:
        return "ffn_up"
    if n == k:
        return "q_proj"
    if n <= 1280:
        return "kv_proj"
    return "ffn_down"


# BEGIN STATIC CONFIGS
_STATIC_CONFIGS: dict[tuple[str, str], dict[str, int]] = {
    ("m1", "ffn_down"): {
        "BLOCK_K": 128,
        "BLOCK_M": 8,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("m1", "ffn_up"): {
        "BLOCK_K": 128,
        "BLOCK_M": 4,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 2,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("m1", "kv_proj"): {
        "BLOCK_K": 256,
        "BLOCK_M": 1,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 1,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("m1", "q_proj"): {
        "BLOCK_K": 128,
        "BLOCK_M": 8,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("medium", "ffn_down"): {
        "BLOCK_K": 128,
        "BLOCK_M": 8,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("medium", "ffn_up"): {
        "BLOCK_K": 256,
        "BLOCK_M": 8,
        "BLOCK_N": 128,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 8,
    },
    ("medium", "kv_proj"): {
        "BLOCK_K": 128,
        "BLOCK_M": 8,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("medium", "q_proj"): {
        "BLOCK_K": 128,
        "BLOCK_M": 8,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("small", "ffn_down"): {
        "BLOCK_K": 128,
        "BLOCK_M": 8,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("small", "ffn_up"): {
        "BLOCK_K": 256,
        "BLOCK_M": 8,
        "BLOCK_N": 128,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 8,
    },
    ("small", "kv_proj"): {
        "BLOCK_K": 256,
        "BLOCK_M": 1,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 1,
        "num_stages": 4,
        "num_warps": 4,
    },
    ("small", "q_proj"): {
        "BLOCK_K": 128,
        "BLOCK_M": 8,
        "BLOCK_N": 64,
        "GROUP_SIZE_M": 4,
        "num_stages": 4,
        "num_warps": 4,
    },
}
# END STATIC CONFIGS


def triton_gemm_packed_int4(
    A_packed: torch.Tensor,
    B_packed: torch.Tensor,
    K: int,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    BLOCK_K: int | None = None,
    GROUP_SIZE_M: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    use_static_config: bool = True,
) -> torch.Tensor:
    """GEMM with packed signed INT4 (2 values per byte).

    A_packed: [M, K//2] uint8
    B_packed: [K//2, N] uint8
    """
    assert A_packed.is_cuda and B_packed.is_cuda, "All tensors must be on GPU"
    assert A_packed.dtype == torch.uint8 and B_packed.dtype == torch.uint8
    assert K % 2 == 0, "K must be even for packed INT4"

    M, Kp = A_packed.shape
    Kb, N = B_packed.shape
    assert Kp == K // 2 and Kb == K // 2

    C = torch.empty((M, N), device=A_packed.device, dtype=torch.float32)

    stride_am, stride_ak = A_packed.stride()
    stride_bk, stride_bn = B_packed.stride()
    stride_cm, stride_cn = C.stride()

    grid_size_fn = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
        * (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
        // meta["GROUP_SIZE_M"],
    )

    meta = {}
    if BLOCK_M is not None:
        meta["BLOCK_M"] = BLOCK_M
    if BLOCK_N is not None:
        meta["BLOCK_N"] = BLOCK_N
    if BLOCK_K is not None:
        meta["BLOCK_K"] = BLOCK_K
    if GROUP_SIZE_M is not None:
        meta["GROUP_SIZE_M"] = GROUP_SIZE_M
    if num_warps is not None:
        meta["num_warps"] = num_warps
    if num_stages is not None:
        meta["num_stages"] = num_stages
    if use_static_config and not meta:
        m_bucket = _bucket_m(M)
        family = _shape_family(N, K)
        static_meta = _STATIC_CONFIGS.get((m_bucket, family)) if m_bucket else None
        if static_meta:
            meta = dict(static_meta)

    if use_static_config and meta:
        kernel_gemm_packed_int4_static[grid_size_fn](
            A_packed,
            B_packed,
            C,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            **meta,
        )
    else:
        kernel_gemm_packed_int4[grid_size_fn](
            A_packed,
            B_packed,
            C,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            **meta,
        )

    return C



###############################################################################
# Example usage / basic test
###############################################################################
if __name__ == "__main__":
    # Problem size
    M, N, K = 256, 256, 256
    
    # Create random input
    A_fp = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B_fp = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Quantize to signed INT4 and pack
    A_q, A_scale = quantize_per_tensor_int4(A_fp)
    B_q, B_scale = quantize_per_tensor_int4(B_fp)
    A_packed = pack_int4_signed(A_q, axis=1)
    B_packed = pack_int4_signed(B_q, axis=0)

    # Ground truth (dequantized)
    C_ref = (A_q.float() * A_scale) @ (B_q.float() * B_scale)

    # Run our Triton kernel
    C = triton_gemm_packed_int4(
        A_packed, B_packed, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=8
    )

    # Compare
    max_abs_diff = (C - C_ref).abs().max().item()
    print(f"Max abs diff: {max_abs_diff:.6f}")
    if max_abs_diff < 1e-2:
        print("Success: Triton kernel and cuBLAS match within tolerance!")
    else:
        print("Warning: Triton kernel and cuBLAS differ. Tune or debug further.")

    # ------------------------------------------------------------------------
    # Benchmark the kernel vs. cuBLAS with torch.matmul
    # ------------------------------------------------------------------------
    import time

    def benchmark_op(op, warmup=3, rep=10):
        # Warmup
        for _ in range(warmup):
            op()
            torch.cuda.synchronize()
        # Timing
        start = time.time()
        for _ in range(rep):
            op()
        torch.cuda.synchronize()
        end = time.time()
        return (end - start) / rep

    # Triton kernel
    def triton_run():
        triton_gemm_packed_int4(A_packed, B_packed, K)

    # cuBLAS kernel
    def cublas_run():
        torch.matmul(A_q.float() * A_scale, B_q.float() * B_scale)

    triton_time = benchmark_op(triton_run)
    cublas_time = benchmark_op(cublas_run)
    print(f"Triton GEMM time:  {triton_time*1e3:.3f} ms")
    print(f"cuBLAS GEMM time:  {cublas_time*1e3:.3f} ms")