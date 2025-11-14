"""
Group 6
Group Memebers:
- Yajing Zhang
- Huoyuan Tan
- Yinghao Chen
- Yiming Li
"""

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""

    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
        intermediate_size=None,
        num_attention_heads=None,
        num_hidden_layers=None,
        rope_theta=None,
        hidden_act="silu",
        max_position_embeddings=None,
        rms_norm_eps=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers


class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""

    def __init__(self, config):
        super().__init__()
        assert config.hidden_act == "silu"
        # TODO: initalize components here
        # SwiGLU needs three linear layers: W1, W2, W3
        # W1 and W2 map from hidden_size to intermediate_size
        # W3 maps from intermediate_size back to hidden_size
        # All layers have no bias (bias=False)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        # SwiGLU: SiLU(xW1) * (xW2) then multiply by W3
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""

    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        # TODO: initalize weights here
        # Manual implementation of RMSNorm
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        # RMSNorm formula: x / sqrt(mean(x^2) + eps) * weight
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""

    def __init__(self, config):
        super().__init__()
        # TODO: set up W_q, W_k, W_v, W_o here
        # TODO: set up normalizers here
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Linear projections for query, key, value, and output
        # All without bias (bias=False)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # RMSNorm normalizers for query and key
        self.q_norm = A2RMSNorm(config)
        self.k_norm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        # Get batch size and sequence length
        b, m, d = hidden_states.shape

        # Step 1: Compute query, key, value representations
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Apply normalizers after query and key
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Step 2: Reshape to separate attention heads
        # Reshape from (b, m, d) to (b, m, num_heads, head_dim) then transpose to (b, num_heads, m, head_dim)
        n_h = self.num_heads
        d_h = self.head_dim
        q = q.view(b, m, n_h, d_h).transpose(1, 2)
        k = k.view(b, m, n_h, d_h).transpose(1, 2)
        v = v.view(b, m, n_h, d_h).transpose(1, 2)

        # Step 3: Apply RoPE rotations to query and key
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)

        # Step 4: Compute scaled dot-product attention with causal masking
        # Use PyTorch's scaled_dot_product_attention with is_causal=True
        attn_out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Step 5: Combine results from individual attention heads
        # Transpose back and reshape: (b, num_heads, m, head_dim) -> (b, m, d)
        attn_out = attn_out.transpose(1, 2).reshape(b, m, d)

        # Step 6: Apply output projection
        output = self.o_proj(attn_out)

        return output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""

    def __init__(self, config):
        super().__init__()
        # Set up attention, MLP, and normalizers here.
        # Following the diagram: pre-norm before attention and MLP, and residual connections
        self.attn = A2Attention(config)
        self.attn_norm = A2RMSNorm(config)

        self.mlp = A2MLP(config)
        self.mlp_norm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        # Attention block
        normed_states = self.attn_norm(hidden_states)
        attn_output = self.attn(normed_states, rope_rotations)
        hidden_states = hidden_states + attn_output  # Residual connection

        # MLP block
        normed_states = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(normed_states)
        hidden_states = hidden_states + mlp_output  # Residual connection

        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""

    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer decoder layers
        layers = []
        for _ in range(config.num_hidden_layers):
            layers.append(A2DecoderLayer(config))
        self.layers = nn.ModuleList(layers)

        # Final RMSNorm (post-transformer) and lm_head (linear unembedding)
        self.final_norm = A2RMSNorm(config)
        # unembedding
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # This line should be called after you have set up all components.
        self.post_init()

    def forward(self, input_ids):
        rope_rotations = self.rotary_emb(
            input_ids
        )  # pass this to all the transformer decoder layers

        # Call embedding
        hidden_states = self.wte(input_ids)

        # Pass through all decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_rotations)

        # Final normalization and project to logits
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


#### RoPE implementation (copied and simplified from HuggingFace). ####


def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert q.shape == k.shape
    assert len(q.shape) == 4
    cos, sin = rope_rotations
    assert q.shape[2] == cos.shape[1]
    assert q.shape[3] == cos.shape[2]
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin
