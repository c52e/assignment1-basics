import torch
import math
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(torch.nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma2 = 2.0 / (in_features + out_features)
        sigma = sigma2**0.5
        torch.nn.init.trunc_normal_(self.weight, 0, sigma, -3.0 * sigma, 3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot('out in, ... in -> ... out', self.weight, x)

class Embedding(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((vocab_size, d_model), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, 0, 1.0, -3.0, 3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        #return self.weight[token_ids]
        return einx.get_at('[vocab_size] d_model, ... -> ... d_model', self.weight, token_ids)

class RMSNorm(torch.nn.Module):
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device = None,
        dtype = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (einx.mean('... [d_model]', x * x) + self.eps) ** 0.5
        result = einx.multiply('... d_model, d_model', einx.divide('... d_model, ...', x, rms), self.weight)
        return result.to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu_w1_x = w1_x / (1.0 + torch.exp(-w1_x))
        w3_x = self.w3(x)
        result = self.w2(silu_w1_x * w3_x)
        return result

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        item_k = torch.arange(0, d_k, 2, device=device).to(torch.float32) / d_k
        inv_freq = 1.0 / torch.exp(item_k * math.log(theta))
        pos = torch.arange(0, max_seq_len, device=device)
        theta_i_k = einx.multiply('i, k -> i k', pos, inv_freq)
        cos = torch.cos(theta_i_k)
        sin = torch.sin(theta_i_k)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        def get_cos_sin(param):
            res = param[token_positions]
            res = einx.rearrange('... k->... (k 2)', res)
            res = res.unsqueeze(1) 
            return res
        cos = get_cos_sin(self.cos)
        sin = get_cos_sin(self.sin)
        x_pairs = einx.rearrange('... (d d2) -> ... d d2', x, d2=2)
        x_pairs_interleaved_minus = torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1)
        x_interleaved_minus = einx.rearrange('... d d2 -> ... (d d2)', x_pairs_interleaved_minus, d2=2)
        result = x * cos + x_interleaved_minus * sin
        return result
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value = torch.max(x, dim=dim, keepdim=True).values
    x_normed = x - max_value
    exp_value = torch.exp(x_normed)
    res = exp_value / torch.sum(exp_value, dim=dim, keepdim=True)
    return res

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... seq_len d_k"],
    K: Float[Tensor, " ... seq_len d_k"],
    V: Float[Tensor, " ... seq_len d_v"],
    mask: Bool[Tensor, " ... seq_len seq_len"] | None = None,
) -> Float[Tensor, " ... seq_len d_v"]:
    d_k = Q.shape[-1]
    qk = einx.dot('... queries d_k, ... keys d_k -> ... queries keys', Q, K)
    qk_normed = qk / math.sqrt(float(d_k))
    if mask is not None:
        qk_normed = qk_normed.masked_fill(~mask, float('-inf'))
    softmax_qk = softmax(qk_normed, -1)
    res = einx.dot('... queries seq_len, ... seq_len d_v -> ... queries d_v', softmax_qk, V)
    return res

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_k = d_model // num_heads * num_heads

        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_k, device=device, dtype=dtype)
        self.output_proj = Linear(d_k, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).to(torch.bool)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = einx.rearrange('... seq_len (h d) -> ... h seq_len d', Q, h=self.num_heads)
        K = einx.rearrange('... seq_len (h d) -> ... h seq_len d', K, h=self.num_heads)
        V = einx.rearrange('... seq_len (h d) -> ... h seq_len d', V, h=self.num_heads)

        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        att = scaled_dot_product_attention(Q, K, V, mask)
        att = einx.rearrange('... h seq_len d -> ... seq_len (h d)', att)
        res = self.output_proj(att)
        return res
    
class TransformerBlock(torch.nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device,dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        if token_positions is None:
            seq_len = x.shape[-2]
            batch_size = x.shape[0]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        att = x + self.attn(self.ln1(x), token_positions)
        return att + self.ffn(self.ln2(att))

class TransformerLM(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        #x = softmax(x, -1)
        return x

if __name__ == '__main__':
    pass
