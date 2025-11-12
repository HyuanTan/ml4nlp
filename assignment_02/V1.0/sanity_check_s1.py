"""
Sanity check tests for Stage 1 components:
- MLP layer (SwiGLU)
- RMSNorm normalization layer
- Multi-head attention (MHA)
"""

import torch
from a2_s1 import A2ModelConfig, A2MLP, A2RMSNorm, A2Attention, A2RotaryEmbedding


def test_mlp():
    """Test MLP layer: input and output should have the same shape."""
    print("Testing MLP layer...")

    # Create config
    config = A2ModelConfig(hidden_size=128, intermediate_size=256, hidden_act="silu")

    # Create MLP layer
    mlp = A2MLP(config)

    # Create test input: (batch_size, seq_len, hidden_size)
    batch_size = 2
    seq_len = 10
    test_input = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass
    output = mlp(test_input)

    # Check shape
    assert output.shape == test_input.shape, (
        f"Output shape {output.shape} != input shape {test_input.shape}"
    )
    print(
        f"  ✓ MLP test passed! Input shape: {test_input.shape}, Output shape: {output.shape}"
    )


def test_rmsnorm():
    """Test RMSNorm layer: input and output should have the same shape."""
    print("\nTesting RMSNorm layer...")

    # Create config
    config = A2ModelConfig(hidden_size=128, rms_norm_eps=1e-6)

    # Create RMSNorm layer
    norm = A2RMSNorm(config)

    # Create test input: (batch_size, seq_len, hidden_size)
    batch_size = 2
    seq_len = 10
    test_input = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass
    output = norm(test_input)

    # Check shape
    assert output.shape == test_input.shape, (
        f"Output shape {output.shape} != input shape {test_input.shape}"
    )
    print(
        f"  ✓ RMSNorm test passed! Input shape: {test_input.shape}, Output shape: {output.shape}"
    )


def test_attention_step1():
    """Test MHA layer step 1: check that it doesn't crash and shapes are correct after initial steps."""
    print("\nTesting MHA layer (Step 1: basic forward pass)...")

    # Create config
    config = A2ModelConfig(
        hidden_size=128, num_attention_heads=4, rms_norm_eps=1e-6, rope_theta=10000.0
    )

    # Create attention layer
    attention = A2Attention(config)

    # Create rotary embedding
    rotary_emb = A2RotaryEmbedding(config)

    # Create test input: (batch_size, seq_len, hidden_size)
    batch_size = 2
    seq_len = 10
    test_input = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create dummy input_ids for RoPE (we need integers for the rotary embedding)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # Get RoPE rotations
    rope_rotations = rotary_emb(input_ids)

    # Forward pass
    try:
        output = attention(test_input, rope_rotations)

        # Check shape: output should have same shape as input
        assert output.shape == test_input.shape, (
            f"Output shape {output.shape} != input shape {test_input.shape}"
        )
        print(
            f"  ✓ MHA test passed! Input shape: {test_input.shape}, Output shape: {output.shape}"
        )
    except Exception as e:
        print(f"  ✗ MHA test failed with error: {e}")
        raise


def test_attention_multiple_shapes():
    """Test MHA layer with different input shapes."""
    print("\nTesting MHA layer with different shapes...")

    # Create config
    config = A2ModelConfig(
        hidden_size=256, num_attention_heads=8, rms_norm_eps=1e-6, rope_theta=10000.0
    )

    # Create attention layer
    attention = A2Attention(config)
    rotary_emb = A2RotaryEmbedding(config)

    # Test with different batch sizes and sequence lengths
    test_cases = [
        (1, 5),  # Single batch, short sequence
        (4, 20),  # Multiple batches, longer sequence
        (2, 1),  # Multiple batches, single token
    ]

    for batch_size, seq_len in test_cases:
        test_input = torch.randn(batch_size, seq_len, config.hidden_size)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        rope_rotations = rotary_emb(input_ids)

        output = attention(test_input, rope_rotations)
        assert output.shape == test_input.shape, (
            f"Shape mismatch for batch={batch_size}, seq_len={seq_len}"
        )
        print(f"  ✓ Test passed for batch_size={batch_size}, seq_len={seq_len}")


if __name__ == "__main__":
    print("=" * 60)
    print("Sanity Check Tests for Stage 1 Components")
    print("=" * 60)

    try:
        test_mlp()
        test_rmsnorm()
        test_attention_step1()
        test_attention_multiple_shapes()

        print("\n" + "=" * 60)
        print("All sanity checks passed! ✓")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Sanity check failed: {e}")
        print("=" * 60)
        raise
