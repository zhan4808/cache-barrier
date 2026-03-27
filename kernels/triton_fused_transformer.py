import math
import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    # Pointers to matrices
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Matrix dimensions
    B, H, N, D,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Meta-parameters
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    ATTENTION_DROPOUT_RATE: tl.constexpr
):
    """
    Fused self-attention kernel for small batch transformer inference.
    
    Computes: Softmax(Q·K^T/sqrt(d_k))·V
    
    Q, K, V: (batch_size, num_heads, seq_len, head_dim)
    Output: (batch_size, num_heads, seq_len, head_dim)
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_d = tl.cdiv(D, BLOCK_D)
    
    # Batch and head index
    batch_id = pid // (H * num_pid_n)
    head_id = (pid % (H * num_pid_n)) // num_pid_n
    n_id = (pid % num_pid_n) * BLOCK_N
    
    # Initialize offsets
    offs_n = n_id + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Scale factor for Q·K^T
    scale = SCALE
    
    # Load Q block for current batch, head, sequence position
    q_ptrs = Q_ptr + batch_id * stride_qb + head_id * stride_qh + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize accumulator for attention scores
    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    # Loop over sequence length for K and V
    for k_block_idx in range(0, tl.cdiv(N, BLOCK_N)):
        k_start = k_block_idx * BLOCK_N
        offs_k = k_start + tl.arange(0, BLOCK_N)
        
        # Load K block for current batch, head, sequence
        k_ptrs = K_ptr + batch_id * stride_kb + head_id * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_k[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Load V block for current batch, head, sequence
        v_ptrs = V_ptr + batch_id * stride_vb + head_id * stride_vh + offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_k[:, None] < N) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Compute QK^T
        scores = tl.zeros([BLOCK_N, BLOCK_N], dtype=tl.float32)
        for d_idx in range(tl.cdiv(D, BLOCK_D)):
            d_start = d_idx * BLOCK_D
            offs_d_inner = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d_inner < D
            
            # Load Q and K sub-blocks
            q_sub = tl.load(Q_ptr + batch_id * stride_qb + head_id * stride_qh + 
                           offs_n[:, None] * stride_qn + offs_d_inner[None, :] * stride_qd,
                           mask=(offs_n[:, None] < N) & (offs_d_inner[None, :] < D), other=0.0)
            
            k_sub = tl.load(K_ptr + batch_id * stride_kb + head_id * stride_kh + 
                           offs_k[:, None] * stride_kn + offs_d_inner[None, :] * stride_kd,
                           mask=(offs_k[:, None] < N) & (offs_d_inner[None, :] < D), other=0.0)
            
            # Update QK^T scores
            scores += tl.dot(q_sub, tl.trans(k_sub))
        
        # Apply scale and mask for causal attention
        scores = scores * scale
        causal_mask = offs_n[:, None] >= offs_k[None, :]  # Causal masking
        scores = tl.where(causal_mask, scores, float("-inf"))
        
        # Apply softmax to get attention weights
        max_scores = tl.max(scores, axis=1)
        scores = scores - max_scores[:, None]
        exp_scores = tl.exp(scores)
        denom = tl.sum(exp_scores, axis=1)
        attn_weights = exp_scores / denom[:, None]
        
        # Apply attention dropout if specified
        if ATTENTION_DROPOUT_RATE > 0.0:
            dropout_mask = tl.rand(attn_weights.shape) > ATTENTION_DROPOUT_RATE
            attn_weights = tl.where(dropout_mask, attn_weights / (1.0 - ATTENTION_DROPOUT_RATE), 0.0)
        
        # Compute attention output
        for d_idx in range(tl.cdiv(D, BLOCK_D)):
            d_start = d_idx * BLOCK_D
            offs_d_inner = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d_inner < D
            
            v_sub = tl.load(V_ptr + batch_id * stride_vb + head_id * stride_vh + 
                           offs_k[:, None] * stride_vn + offs_d_inner[None, :] * stride_vd,
                           mask=(offs_k[:, None] < N) & (offs_d_inner[None, :] < D), other=0.0)
            v_sub = v_sub.to(tl.float32)
            
            acc_sub = tl.dot(attn_weights, v_sub)
            
            # Accumulate result
            acc_mask = (offs_n[:, None] < N) & (offs_d_inner[None, :] < D)
            acc_ptrs = Out_ptr + batch_id * stride_ob + head_id * stride_oh + offs_n[:, None] * stride_on + offs_d_inner[None, :] * stride_od
            tl.store(acc_ptrs, acc_sub, mask=acc_mask)


@triton.jit
def fused_ffn_kernel(
    # Pointers to matrices
    X_ptr, W1_ptr, B1_ptr, W2_ptr, B2_ptr, Out_ptr,
    # Matrix dimensions
    B, N, D_in, D_hidden,
    # Strides
    stride_xb, stride_xn, stride_xd,
    stride_w1i, stride_w1o,
    stride_w2i, stride_w2o,
    stride_ob, stride_on, stride_od,
    # Meta-parameters
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    D_HIDDEN: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """
    Fused feed-forward network for small batch transformer inference.
    
    Computes: GELU(X·W1 + B1)·W2 + B2
    
    X: (batch_size, seq_len, d_model)
    W1: (d_model, d_hidden)
    B1: (d_hidden)
    W2: (d_hidden, d_model)
    B2: (d_model)
    Output: (batch_size, seq_len, d_model)
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Batch and sequence position
    batch_id = pid // num_pid_n
    n_id = (pid % num_pid_n) * BLOCK_N
    
    # Initialize offsets
    offs_n = n_id + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    
    # Compute second linear layer block-by-block.
    # Note: to keep compile-time shapes, we compute hidden blocks on the fly.
    for d_out_block in range(0, tl.cdiv(D_in, BLOCK_D)):
        d_out_start = d_out_block * BLOCK_D
        offs_d_out = d_out_start + tl.arange(0, BLOCK_D)
        d_out_mask = offs_d_out < D_in
        
        out_partial = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        
        for d_hidden_block in range(0, tl.cdiv(D_hidden, BLOCK_D)):
            d_hidden_start = d_hidden_block * BLOCK_D
            offs_d_hidden = d_hidden_start + tl.arange(0, BLOCK_D)
            d_hidden_mask = offs_d_hidden < D_hidden
            
            # Accumulate hidden slice for this block
            hidden_slice = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
            for d_in_block in range(0, tl.cdiv(D_in, BLOCK_D)):
                d_in_start = d_in_block * BLOCK_D
                offs_d_in = d_in_start + tl.arange(0, BLOCK_D)
                d_in_mask = offs_d_in < D_in
                
                x_ptrs = X_ptr + batch_id * stride_xb + offs_n[:, None] * stride_xn + offs_d_in[None, :] * stride_xd
                x_mask = n_mask[:, None] & d_in_mask[None, :]
                x = tl.load(x_ptrs, mask=x_mask, other=0.0)
                
                w1_ptrs = W1_ptr + offs_d_in[:, None] * stride_w1i + offs_d_hidden[None, :] * stride_w1o
                w1_mask = d_in_mask[:, None] & d_hidden_mask[None, :]
                w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)
                
                partial_hidden = tl.dot(x, w1)
                hidden_mask = n_mask[:, None] & d_hidden_mask[None, :]
                hidden_slice = hidden_slice + tl.where(hidden_mask, partial_hidden, 0.0)
            
            b1 = tl.load(B1_ptr + offs_d_hidden, mask=d_hidden_mask, other=0.0)
            hidden_mask = n_mask[:, None] & d_hidden_mask[None, :]
            hidden_slice = tl.where(hidden_mask, hidden_slice + b1[None, :], hidden_slice)
            
            if ACTIVATION == 0:  # GELU (sigmoid approximation)
                hidden_slice = hidden_slice * tl.sigmoid(1.702 * hidden_slice)
            elif ACTIVATION == 1:  # ReLU
                hidden_slice = tl.maximum(hidden_slice, 0.0)
            elif ACTIVATION == 2:  # SiLU/Swish
                hidden_slice = hidden_slice * tl.sigmoid(hidden_slice)
            
            # Load W2 block
            w2_ptrs = W2_ptr + offs_d_hidden[:, None] * stride_w2i + offs_d_out[None, :] * stride_w2o
            w2_mask = d_hidden_mask[:, None] & d_out_mask[None, :]
            w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)
            w2 = w2.to(tl.float32)
            
            # Update output
            out_partial += tl.dot(hidden_slice, w2)
        
        # Load bias and add
        b2 = tl.load(B2_ptr + offs_d_out, mask=d_out_mask, other=0.0)
        out_mask = n_mask[:, None] & d_out_mask[None, :]
        out_partial = tl.where(out_mask, out_partial + b2[None, :], out_partial)
        
        # Store output
        out_ptrs = Out_ptr + batch_id * stride_ob + offs_n[:, None] * stride_on + offs_d_out[None, :] * stride_od
        tl.store(out_ptrs, out_partial, mask=out_mask)


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    dropout_p: float = 0.0,
    BLOCK_N: int = 64,
    BLOCK_D: int = 64
):
    """
    Optimized fused attention for small batch transformer inference.
    
    Args:
        q: Query tensor (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor (batch_size, num_heads, seq_len, head_dim)
        causal: Whether to apply causal masking
        dropout_p: Attention dropout probability
        
    Returns:
        Output tensor (batch_size, num_heads, seq_len, head_dim)
    """
    # Check shapes
    assert q.ndim == 4, "Query tensor must be 4D (batch_size, num_heads, seq_len, head_dim)"
    assert k.shape == q.shape, "Key tensor must have same shape as query"
    assert v.shape == q.shape, "Value tensor must have same shape as query"
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Ensure tensors are on GPU and have correct data type
    assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be on GPU"
    if q.dtype != torch.float16:
        q = q.half()
        k = k.half()
        v = v.half()
    
    # Create output tensor
    output = torch.zeros_like(q)
    
    # Calculate grid size
    grid = (batch_size * num_heads * triton.cdiv(seq_len, BLOCK_N),)
    
    # Launch kernel
    fused_attention_kernel[grid](
        q, k, v, output,
        batch_size, num_heads, seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        SCALE=1.0 / math.sqrt(head_dim),
        ATTENTION_DROPOUT_RATE=dropout_p,
        num_warps=8
    )
    
    return output


def fused_ffn(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    activation: str = "gelu",
    BLOCK_N: int = 64,
    BLOCK_D: int = 64
):
    """
    Optimized fused feed-forward network for small batch transformer inference.
    
    Args:
        x: Input tensor (batch_size, seq_len, d_model)
        w1: First weight matrix (d_model, d_hidden)
        b1: First bias vector (d_hidden)
        w2: Second weight matrix (d_hidden, d_model)
        b2: Second bias vector (d_model)
        activation: Activation function, one of ["gelu", "relu", "silu"]
        
    Returns:
        Output tensor (batch_size, seq_len, d_model)
    """
    # Check shapes
    assert x.ndim == 3, "Input tensor must be 3D (batch_size, seq_len, d_model)"
    d_model = x.shape[2]
    d_hidden = w1.shape[1]
    
    assert w1.shape == (d_model, d_hidden), "w1 shape should be (d_model, d_hidden)"
    assert b1.shape == (d_hidden,), "b1 shape should be (d_hidden,)"
    assert w2.shape == (d_hidden, d_model), "w2 shape should be (d_hidden, d_model)"
    assert b2.shape == (d_model,), "b2 shape should be (d_model,)"
    
    # Ensure tensors are on GPU and have correct data type
    assert x.is_cuda and w1.is_cuda and b1.is_cuda and w2.is_cuda and b2.is_cuda, "All tensors must be on GPU"
    if x.dtype != torch.float16:
        x = x.half()
        w1 = w1.half()
        b1 = b1.half()
        w2 = w2.half()
        b2 = b2.half()
    
    # Map activation string to integer
    activation_map = {"gelu": 0, "relu": 1, "silu": 2}
    activation_int = activation_map.get(activation.lower(), 0)
    
    # Get dimensions
    batch_size, seq_len, d_model = x.shape
    
    # Create output tensor
    output = torch.zeros_like(x)
    
    # Calculate grid size
    grid = (batch_size * triton.cdiv(seq_len, BLOCK_N),)
    
    # Launch kernel
    fused_ffn_kernel[grid](
        x, w1, b1, w2, b2, output,
        batch_size, seq_len, d_model, d_hidden,
        x.stride(0), x.stride(1), x.stride(2),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        D_HIDDEN=d_hidden,
        ACTIVATION=activation_int,
        num_warps=8
    )
    
    return output


if __name__ == "__main__":
    # Test parameters
    batch_size = 1
    num_heads = 8
    seq_len = 128
    head_dim = 64
    d_model = head_dim * num_heads
    d_hidden = d_model * 4  # Common ratio in transformers
    
    # Create sample data for testing
    torch.manual_seed(42)
    device = torch.device('cuda')
    
    # Attention test
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # PyTorch reference implementation
    def attention_ref(q, k, v, causal=True):
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            scores.masked_fill_(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    # FFN test
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    w1 = torch.randn(d_model, d_hidden, device=device, dtype=torch.float16)
    b1 = torch.randn(d_hidden, device=device, dtype=torch.float16)
    w2 = torch.randn(d_hidden, d_model, device=device, dtype=torch.float16)
    b2 = torch.randn(d_model, device=device, dtype=torch.float16)
    
    # PyTorch reference implementation
    def ffn_ref(x, w1, b1, w2, b2):
        hidden = torch.matmul(x, w1) + b1
        hidden = torch.nn.functional.gelu(hidden)
        return torch.matmul(hidden, w2) + b2
    
    # Run Triton implementations
    try:
        print("Testing fused attention kernel...")
        attn_out = fused_attention(q, k, v)
        
        print("Testing fused FFN kernel...")
        ffn_out = fused_ffn(x, w1, b1, w2, b2)
        
        # Calculate reference outputs
        with torch.no_grad():
            attn_ref_out = attention_ref(q, k, v)
            ffn_ref_out = ffn_ref(x, w1, b1, w2, b2)
        
        # Compare results
        attn_max_diff = (attn_out - attn_ref_out).abs().max().item()
        ffn_max_diff = (ffn_out - ffn_ref_out).abs().max().item()
        
        print(f"Attention max abs diff: {attn_max_diff:.6f}")
        print(f"FFN max abs diff: {ffn_max_diff:.6f}")
        
        # Benchmark
        import time
        
        def benchmark_op(op, warmup=3, rep=10):
            # Warmup
            for _ in range(warmup):
                op()
                torch.cuda.synchronize()
            # Timing
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(rep):
                op()
                torch.cuda.synchronize()
            end = time.time()
            return (end - start) / rep
        
        # Triton kernels
        triton_attn_time = benchmark_op(lambda: fused_attention(q, k, v))
        triton_ffn_time = benchmark_op(lambda: fused_ffn(x, w1, b1, w2, b2))
        
        # PyTorch reference
        torch_attn_time = benchmark_op(lambda: attention_ref(q, k, v))
        torch_ffn_time = benchmark_op(lambda: ffn_ref(x, w1, b1, w2, b2))
        
        print(f"\nBenchmark results for batch_size={batch_size}, seq_len={seq_len}:")
        print(f"Attention - Triton: {triton_attn_time*1e3:.3f} ms, PyTorch: {torch_attn_time*1e3:.3f} ms, Speedup: {torch_attn_time/triton_attn_time:.2f}x")
        print(f"FFN - Triton: {triton_ffn_time*1e3:.3f} ms, PyTorch: {torch_ffn_time*1e3:.3f} ms, Speedup: {torch_ffn_time/triton_ffn_time:.2f}x")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}") 