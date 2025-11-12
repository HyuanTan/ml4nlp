# Assignment 2 - Step 1 Implementation

This document explains what was implemented for Stage 1 of Assignment 2.

## What Was Done

### 1. MLP Layer (SwiGLU)

The MLP layer uses the SwiGLU architecture. It has three linear layers:

- `gate_proj`: Maps from `hidden_size` to `intermediate_size` (no bias)
- `up_proj`: Maps from `hidden_size` to `intermediate_size` (no bias)
- `down_proj`: Maps from `intermediate_size` back to `hidden_size` (no bias)

The forward pass works like this:

1. Apply SiLU activation to the gate projection
2. Multiply the activated gate with the up projection (element-wise)
3. Apply the down projection to get the final output

The input and output have the same shape: `(batch_size, sequence_length, hidden_size)`.

### 2. RMSNorm Normalization Layer

The RMSNorm layer normalizes the input using root mean square normalization. It:

- Computes the variance of the input
- Normalizes by dividing by the square root of variance plus a small epsilon value
- Scales the result using a learnable weight parameter

The input and output have the same shape.

### 3. Multi-Head Attention (MHA) Layer

The multi-head attention layer implements the attention mechanism with the following steps:

1. Compute query (Q), key (K), and value (V) projections from the input
2. Apply RMSNorm to Q and K
3. Reshape Q, K, V to separate attention heads
4. Apply RoPE (Rotary Position Embedding) rotations to Q and K
5. Compute scaled dot-product attention with causal masking
6. Combine the results from all attention heads
7. Apply the output projection

All linear layers (Q, K, V, O projections) have no bias terms. The input and output have the same shape.

## Files

- `a2_s1.py`: Contains the implementation of all three components
- `sanity_check_s1.py`: Contains test code to verify that the components work correctly
- `sanity_check_s1_output.txt`: Contains test output

## Testing

Run the sanity check file to test the components (in the step 1, there is no need to use GPU, so we can run the code directly in the terminal when you access to Minerva):

```bash
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
python sanity_check_s1.py
```

The sanity checks verify that:

- MLP layer maintains input/output shapes
- RMSNorm layer maintains input/output shapes
- MHA layer maintains input/output shapes and doesn't crash
- All components work with different input shapes
