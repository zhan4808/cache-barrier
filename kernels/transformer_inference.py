import torch
import torch.nn as nn
import math
import time

import tiny_gemm.ops  # registers torch.library ops
from triton_fused_transformer import fused_attention, fused_ffn


def _fused_attention_dispatch(q, k, v, causal, dropout_p):
    return torch.ops.tiny_gemm.fused_attention(q, k, v, causal, float(dropout_p))


def _fused_ffn_dispatch(x, w1, b1, w2, b2, activation):
    activation_map = {"gelu": 0, "relu": 1, "silu": 2}
    activation_int = activation_map.get(activation, 0)
    return torch.ops.tiny_gemm.fused_ffn(x, w1, b1, w2, b2, activation_int)

class FusedTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Parameters for attention
        self.wq = nn.Parameter(torch.empty(d_model, d_model))
        self.wk = nn.Parameter(torch.empty(d_model, d_model))
        self.wv = nn.Parameter(torch.empty(d_model, d_model))
        self.wo = nn.Parameter(torch.empty(d_model, d_model))
        
        # Parameters for feed-forward
        self.w1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.w2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        
        # Layer norms
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = dropout
        
        # Activation function
        self.activation = activation
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.wq)
        nn.init.xavier_uniform_(self.wk)
        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wo)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
    
    def forward(self, x, use_triton=True, causal=True, use_dispatch=False):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # Pre-layernorm
        residual = x
        x = self.attn_norm(x)
        
        if use_triton:
            # Project to Q, K, V
            q = torch.matmul(x, self.wq).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.matmul(x, self.wk).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = torch.matmul(x, self.wv).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Fused attention
            if use_dispatch:
                attn_output = _fused_attention_dispatch(
                    q, k, v, causal=causal, dropout_p=self.dropout
                )
            else:
                attn_output = fused_attention(
                    q, k, v, causal=causal, dropout_p=self.dropout
                )
            
            # Project back
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
            attn_output = torch.matmul(attn_output, self.wo)
            
            # Residual connection
            x = residual + attn_output
            
            # Pre-layernorm for FFN
            residual = x
            x = self.ffn_norm(x)
            
            # Fused FFN
            if use_dispatch:
                x = _fused_ffn_dispatch(
                    x,
                    self.w1,
                    self.b1,
                    self.w2,
                    self.b2,
                    activation=self.activation,
                )
            else:
                x = fused_ffn(
                    x,
                    self.w1,
                    self.b1,
                    self.w2,
                    self.b2,
                    activation=self.activation,
                )
            
            # Residual connection
            x = residual + x
        
        else:
            # Standard PyTorch implementation
            # Attention
            q = torch.matmul(x, self.wq).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.matmul(x, self.wk).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = torch.matmul(x, self.wv).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
                scores.masked_fill_(mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)
            if self.dropout > 0:
                attn_weights = torch.dropout(attn_weights, p=self.dropout, train=self.training)
            attn_output = torch.matmul(attn_weights, v)
            
            # Project back
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
            attn_output = torch.matmul(attn_output, self.wo)
            
            # Residual connection
            x = residual + attn_output
            
            # Pre-layernorm for FFN
            residual = x
            x = self.ffn_norm(x)
            
            # Feed-forward network
            hidden = torch.matmul(x, self.w1) + self.b1
            if self.activation == "gelu":
                hidden = torch.nn.functional.gelu(hidden)
            elif self.activation == "relu":
                hidden = torch.nn.functional.relu(hidden)
            elif self.activation == "silu":
                hidden = torch.nn.functional.silu(hidden)
            
            x = torch.matmul(hidden, self.w2) + self.b2
            if self.dropout > 0:
                x = torch.dropout(x, p=self.dropout, train=self.training)
            
            # Residual connection
            x = residual + x
        
        return x


class FusedTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len=2048, 
                 dropout=0.1, activation="gelu"):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embedding
        self.embedding = nn.Embedding(32000, d_model)  # Typical vocab size for transformers
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            FusedTransformerLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, 32000, bias=False)
        
        # Initialize positional encoding
        self._init_pos_encoding(max_seq_len, d_model)
    
    def _init_pos_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data = pe.unsqueeze(0)
    
    def forward(self, input_ids, use_triton=True, use_dispatch=False):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.embedding(input_ids)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, use_triton=use_triton, use_dispatch=use_dispatch)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        return logits


def inference_benchmark(model, input_ids, warmup=3, reps=10):
    device = input_ids.device
    
    # Warmup runs
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids, use_triton=False)
        torch.cuda.synchronize()
    
    # PyTorch baseline
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps):
        with torch.no_grad():
            out_pytorch = model(input_ids, use_triton=False)
        torch.cuda.synchronize()
    end = time.time()
    pytorch_time = (end - start) / reps
    
    # Triton implementation
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps):
        with torch.no_grad():
            out_triton = model(input_ids, use_triton=True)
        torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) / reps
    
    # Verify correctness
    max_diff = (out_pytorch - out_triton).abs().max().item()
    
    return {
        "pytorch_time": pytorch_time * 1000,  # ms
        "triton_time": triton_time * 1000,    # ms
        "speedup": pytorch_time / triton_time,
        "max_diff": max_diff
    }


if __name__ == "__main__":
    # Model parameters (small transformer for testing)
    num_layers = 4
    d_model = 512
    num_heads = 8
    d_ff = 2048
    batch_size = 1
    seq_lengths = [32, 64, 128, 256, 512, 1024]
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusedTransformer(num_layers, d_model, num_heads, d_ff).to(device).half()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer model with {num_layers} layers, {d_model} dimensions, {num_heads} heads")
    print(f"Total parameters: {num_params/1e6:.2f}M")
    
    # Benchmark with different sequence lengths
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nBenchmarking with sequence length {seq_len}...")
        
        # Random input
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        
        # Run benchmark
        result = inference_benchmark(model, input_ids)
        result["seq_len"] = seq_len
        results.append(result)
        
        print(f"PyTorch: {result['pytorch_time']:.2f} ms")
        print(f"Triton:  {result['triton_time']:.2f} ms")
        print(f"Speedup: {result['speedup']:.2f}x")
        print(f"Max diff: {result['max_diff']:.6f}")
    
    # Print summary
    print("\nSummary of results:")
    print(f"{'Seq Len':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['seq_len']:<10} {r['pytorch_time']:<15.2f} {r['triton_time']:<15.2f} {r['speedup']:<10.2f}x") 