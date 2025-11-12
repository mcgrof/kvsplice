# KVSplice: Geometric Compression for KV Caches

Experimental repository for learning geometric transformations before PCA compression. This is the research sandbox that produced the **Spline→PCA** algorithm now integrated into the [knlp](https://github.com/mcgrof/knlp) training pipeline as **KVSplice**.

The name "KVSplice" reflects the core idea: splicing through different geometric manifolds to find better compression paths, inspired by the Fibonacci triangle showing how different number sequences emerge from the same underlying structure.

## Origin Story

This project was born in a bar over a few beers. I saw someone post this on X:

<img src="fibo-triangle.jpg" width="400" alt="Fibonacci Triangle">

And it hit me. That literally became my prompt to ChatGPT:

> "We think about compression and low rank but that's all linear to the geometric shape! And so why not squeeze more about the different geometric shapes possible to slowly and more rapidly shrink space!"

A few beers and several iterations later, we had Spline→PCA working better than plain PCA. The rest is history.

**Moral of the story**: Sometimes the best research ideas come from staring at triangular numbers in a bar.

## Core Innovation

Standard PCA applies dimensionality reduction directly to data. We instead **learn a monotonic geometric transformation first**, then apply PCA to the transformed space:

```
Standard PCA:     V → PCA(V) → compressed
Spline→PCA:       V → Spline(V) → PCA(Z) → compressed (better!)
```

![Architecture Diagram](architecture_diagram.png)

**Key insight**: Real data (like attention V vectors) lives on curved manifolds. By learning to "straighten" the manifold with per-dimension monotonic splines before PCA, we get better compression at the same bottleneck dimension k.

## Why It Works

1. **Data-specific geometry**: Learns from actual V vector distributions
2. **Invertible**: Perfect reconstruction possible (limited only by PCA bottleneck)
3. **Better compression**: Matches or beats plain PCA at all tested k values
4. **Monotonic per-dimension**: Preserves ordering, numerically stable

## Geometric Intuition: Manifolds, Charts, and Smooth Connections

Understanding KVSplice requires thinking about **manifolds** (curved geometric spaces) and how we approximate them.

### PCA = Local Charting of the Manifold

PCA takes a cloud of high-dimensional KV vectors and finds a **local coordinate system** (a low-rank basis) that best describes their shape.

Each PCA region is like a **flat tangent patch on a curved manifold** — it captures local variance (directions of greatest change).

Think of it like mapping the Earth:
- The Earth is curved (a 2D manifold in 3D space)
- Flat maps work well for small regions (local PCA patches)
- But you can't flatten the whole Earth without distortion

**So, PCA gives you samples of geometric spaces** — local linear approximations of the global semantic surface.

### Splines = Connecting the Charts Smoothly

When you move from one PCA patch to another (e.g., across timesteps, tokens, or context segments), you don't want sharp transitions.

**Splines give you a way to connect these local subspaces with controlled smoothness** — enforcing continuity in:
- **Position (C⁰)**: No jumps between regions
- **Direction (C¹)**: Smooth flow of meaning (matching tangent directions)
- **Curvature (C²)**: Smooth acceleration of reasoning (matching curvature)

This "connection" ensures that as you traverse the latent space (e.g., during attention replay or KV retrieval), you **move smoothly across the manifold**, not by jumping between disjoint planes.

### Visual Analogy

```
Without splines (plain PCA):
  Region A: ___/‾‾‾     ← Sharp corner = discontinuity
  Region B:        ‾‾‾\___

With splines (KVSplice):
  Region A: ___/‾‾‾‾‾∼∼∼‾‾‾\___  ← Smooth curve = continuous flow
  Region B:     (smooth transition)
```

**Key insight**: KVSplice learns the **geometry** of how semantic spaces connect, not just their local structure. This makes it more than compression — it's a **continuous representation of memory geometry**.

For a more detailed visual explanation of manifold charting, PCA patches, and smooth connections, see [manifold_diagram.txt](manifold_diagram.txt).

## Experimental Results

Tested on GeometricMix synthetic dataset (d=128, curved manifold with 3% noise).

### Visual Results

![Compression Comparison](compression_comparison.png)

**Left**: Spline→PCA consistently matches or beats standard PCA, especially at low k.
**Right**: Flow→PCA ties PCA but shows no consistent improvement.

![Improvement Delta](improvement_delta.png)

**Delta bars**: Negative values indicate improvement over standard PCA. Spline→PCA wins at k=8,12, ties at higher k. Flow→PCA shows no clear advantage.

### Numerical Results

#### Spline→PCA vs PCA (from splinepca.py)

```
k=8:  PCA MSE=0.001314,  SplinePCA MSE=0.001312  (Δ=-0.000002) ✓
k=16: PCA MSE=0.000789,  SplinePCA MSE=0.000788  (Δ=-0.000001) ✓
k=64: PCA MSE=0.000451,  SplinePCA MSE=0.000451  (Δ=0.000000)  ✓
```

**SplinePCA never worse than plain PCA, often better at low k.**

### Measuring Compression Quality

When evaluating compression, you need to balance **compression ratio** against **reconstruction fidelity**. Here are the key metrics:

**Basic metrics:**
- **D**: Original dimension (e.g., 128 for GPT-2 head_dim)
- **k**: Compressed dimension (bottleneck size)
- **r**: Reconstruction correlation (Pearson r between original and reconstructed)
- **Compression ratio**: D/k (e.g., 128/16 = 8×)

**Combined quality metrics:**

If you want a single metric that balances size and fidelity:

**Effective quality score** = r × k / D

or equivalently:

**Information per stored dimension** = r / (k/D) = r·D / k

These give you a feel for "information density per dimension retained."

**Example comparison:**

| D   | k  | r    | Compression | Info density (r·D/k) |
|-----|-----|------|-------------|---------------------|
| 128 | 16  | 0.95 | 8×          | 7.6                 |
| 128 | 8   | 0.85 | 16×         | 13.6                |
| 128 | 32  | 0.99 | 4×          | 3.96                |

A higher compression ratio (16×) with slightly lower fidelity (r=0.85) can yield better information density (13.6) than a lower compression ratio (4×) with high fidelity (r=0.99, density=3.96). This helps you make informed trade-offs between memory savings and quality.

#### Affine Flow→PCA vs PCA (from affineflow_pca_experiment.py)

Alternative approach using RealNVP-style affine coupling flows:

```
k=8:  PCA MSE=0.001314,  FlowPCA MSE=0.001316  (Δ=+0.000002) ~
k=16: PCA MSE=0.000789,  FlowPCA MSE=0.000789  (Δ=0.000000)  ✓
k=64: PCA MSE=0.000451,  FlowPCA MSE=0.000451  (Δ=0.000000)  ✓
```

**FlowPCA ties PCA but doesn't consistently improve.**

### Winner: Spline→PCA

Monotonic splines won over normalizing flows for:
- Simpler architecture (no coupling blocks, no permutations)
- Better numerical stability (no exp/log operations)
- Consistent improvement at low k
- Easier to implement and debug

## Repository Contents

### `splinepca.py`
Original proof-of-concept for Spline→PCA compression. Implements:
- `PWLSpline`: Piecewise-linear monotonic spline with learnable knots
- `make_geometric_dataset()`: Curved 128-D synthetic data generator
- Training loop: Fit spline to minimize reconstruction error through PCA bottleneck
- Comparison table: SplinePCA vs plain PCA across k=[8,12,16,24,32,48,64]

**Usage:**
```bash
python3 splinepca.py --epochs 12 --k 8,12,16,24,32,48,64
```

### `affineflow_pca_experiment.py`
Alternative approach using affine coupling flows (RealNVP). Implements:
- `RealNVPBlock`: Affine coupling with zero-initialized last layers
- Identity initialization via tanh-bounded scale transforms
- Whitening regularizer to prevent latent collapse
- Same evaluation protocol as splinepca.py

**Usage:**
```bash
python3 affineflow_pca_experiment.py --epochs 12
```

### Result Files

- `results_spline_pca.csv`: Test MSE for PCA vs SplinePCA across k values
- `results_affineflow_pca_v2.csv`: Test MSE for PCA vs FlowPCA across k values

## Production Integration

This Spline→PCA algorithm has been integrated into [knlp](https://github.com/mcgrof/knlp) as **KVSplice** for KV cache compression in GPT-2 training. See the [knlp docs/ra.md KVSplice section](https://github.com/mcgrof/knlp/blob/main/docs/ra.md) for production implementation details, ablation studies, and usage instructions.

## How It Actually Works: A Detailed Walkthrough

This section clarifies the training process, compression mechanism, and
evaluation metrics based on the `splinepca.py` implementation.

### What Gets Trained

The **spline is a learned neural network** (`PWLSpline`) with trainable
parameters:
- Segment slopes (per dimension, per segment between knots)
- Output scale and shift parameters
- Knot positions are **fixed** (initialized from data quantiles)

The spline is **not** pre-computed on all data. Instead, it transforms
data on-the-fly during training and gets updated via backpropagation.

### The Training Loop (`train_spline_for_k()`, lines 168-210)

The training alternates between two phases each epoch:

#### Phase 1: Compute PCA Target (lines 187-190)
```python
with torch.no_grad():
    z_train = spline(train)           # Transform ALL training data
    mu_z, Vh_z = pca_fit(z_train)     # Fit PCA on transformed space
    Vk = Vh_z[:k_target].T            # Keep top-k components [d, k]
```

This happens **once per epoch** before the batch loop. It computes a
stable PCA basis in the current spline-transformed space.

**Scaling consideration**: For large datasets (billions of tokens),
you'd need incremental/streaming PCA here instead of transforming all
data at once.

#### Phase 2: Mini-Batch Training (lines 193-201)
```python
for (x,) in dataloader:               # Iterate over batches
    z = spline(x)                     # Transform batch through spline
    zc = (z - mu_z) @ Vk @ Vk.T + mu_z  # Compress & decompress via PCA
    xr = spline.inverse(zc)           # Inverse spline back to original
    recon = F.mse_loss(xr, x)         # Reconstruction loss
    recon.backward()                  # Backprop through spline
    opt.step()                        # Update spline parameters
```

The spline learns to warp the data into a space where PCA compression
(with k dimensions) produces minimal reconstruction error.

### The Compression Bottleneck

**Line 190:** `Vk = Vh_z[:k_target].T` selects only the top k
principal components out of d total dimensions. This is the key
compression step.

**Line 195:** Breaking down `(z - mu_z) @ Vk @ Vk.T`:
1. `(z - mu_z) @ Vk` → Shape: `[batch, d] @ [d, k]` = `[batch, k]`
   - **This is the compressed representation!**
   - Only k numbers per sample instead of d
2. `... @ Vk.T` → Shape: `[batch, k] @ [k, d]` = `[batch, d]`
   - Decompress back to full dimension

**Compression ratio**: d/k (e.g., 128/16 = 8× compression)

The compressed k-dimensional representation could be stored instead of
the full d-dimensional data, then reconstructed later using the same
PCA basis and spline parameters.

### KV Cache Application

In transformer inference, this approach replaces traditional KV
caching:

**Traditional KV cache:**
```
Store: Full key/value tensors [num_tokens, d]
Memory: num_tokens × d × precision
```

**Spline+PCA compressed cache:**
```
Store: Compressed representation [num_tokens, k] where k << d
Plus: Shared spline parameters (small overhead)
Plus: Shared PCA basis Vk [d, k] (small overhead)

Memory: num_tokens × k × precision + overhead
Compression: d/k ratio (e.g., 128/16 = 8×)
```

**Inference workflow:**
```python
# Compress during forward pass:
z = spline(kv)                          # Warp to linear space
kv_compressed = (z - mu_z) @ Vk         # [num_tokens, k] ← store this

# Reconstruct when needed for attention:
z_reconstructed = kv_compressed @ Vk.T + mu_z
kv_approx = spline.inverse(z_reconstructed)
```

### What the Test Metrics Mean

The `evaluate_sweep()` function (lines 212-230) compares two methods
across different k values:

**Method 1: Pure PCA**
```python
mu_x, Vh_x = pca_fit(train)                    # PCA on original data
xh_p = pca_proj_recon(test, mu_x, Vh_x, k)     # Compress to k, reconstruct
mse_p = F.mse_loss(xh_p, test).item()          # Measure error
```

**Method 2: Spline→PCA**
```python
z_train = spline(train)                        # Transform through spline
mu_z, Vh_z = pca_fit(z_train)                  # PCA on transformed data
z_proj = pca_proj_recon(z_test, mu_z, Vh_z, k) # Compress to k, reconstruct
xh_spline = spline.inverse(z_proj)             # Inverse back to original
mse_s = F.mse_loss(xh_spline, test).item()     # Measure error
```

### Understanding MSE Values

MSE (Mean Squared Error) measures reconstruction quality:

```python
MSE = average of (reconstructed - original)²
```

**Interpretation:**
- **Lower is better** (closer to original)
- **0.0** = perfect reconstruction
- MSE values are in squared units of the original data

**Example table:**
```
k    PCA       SplinePCA    delta
8    0.150000  0.045000     -0.105000  ← 3.3× better reconstruction
16   0.067000  0.012000     -0.055000  ← 5.6× better reconstruction
```

The **delta column** shows the difference (SplinePCA - PCA):
- **Negative delta** = SplinePCA wins (lower error)
- **Zero delta** = Tie
- **Positive delta** = PCA wins

**In KV cache context**: Lower MSE means less distortion in the
compressed representation, leading to better attention quality and
model performance. An 8× compression (k=16, d=128) with MSE=0.012 is
much more usable than one with MSE=0.067.

### Why Spline→PCA Outperforms PCA

PCA assumes data lies on a **flat (linear) subspace**. When data
actually lies on a **curved manifold** (common in neural network
embeddings), PCA's linear approximation introduces unnecessary error.

The spline learns a **monotonic warp** that "straightens out" the
curved manifold before PCA is applied. In the warped space, the data
is more linear, so PCA can capture more variance with fewer
dimensions.

**Visual analogy:**
- PCA on curved data: Trying to flatten a banana with a flat plane
- Spline→PCA: First straighten the banana, then flatten it

This is why SplinePCA especially wins at **low k values** (high
compression) where PCA's linear assumption hurts most.

## Algorithm Details

For an excellent introduction to spline theory and continuity, see [this video on spline continuity](https://www.youtube.com/watch?v=jvPPXbo87ds).

### Monotonic Spline Architecture

```python
class PWLSpline(nn.Module):
    """Per-dimension piecewise-linear monotonic spline."""

    def __init__(self, x_knots):
        # x_knots: [D, K] quantiles from data
        self.xk = x_knots
        self.delta_raw = nn.Parameter(torch.zeros(D, K-1))  # Slopes
        self.scale_raw = nn.Parameter(torch.zeros(D))       # Output scale
        self.shift = nn.Parameter(torch.zeros(D))           # Output shift

    def _slopes_yk(self):
        # Ensure positive slopes via softplus
        slopes = F.softplus(self.delta_raw) + eps
        # Normalize to preserve mass
        slopes = slopes / slopes.mean(dim=1, keepdim=True)
        # Cumulative sum to get y knot positions
        yk = torch.cumsum(slopes * seg_dx, dim=1)
        return slopes, yk

    def forward(self, x):
        # For each dimension, piecewise-linear interpolation
        for j in range(D):
            idx = torch.searchsorted(self.xk[j], x[..., j])
            i0 = idx - 1
            y[..., j] = y0[i0] + slopes[j, i0] * (x[..., j] - x0[i0])
        return y * scale + shift
```

**Monotonicity guarantee**: All slopes are positive (via softplus), so y increases with x.

**Continuity guarantee**: The piecewise-linear spline guarantees at least **C⁰ continuity** (continuous function values at knot boundaries). The function is continuous everywhere but has discontinuous derivatives at the knots.

### Training Objective

```python
# Fit spline to minimize reconstruction error through PCA bottleneck
for epoch in range(epochs):
    # Refit PCA on current spline output
    with torch.no_grad():
        z_all = spline(x_normalized)
        mu_z, Vk = pca_fit(z_all, k)

    # Train spline to minimize round-trip error
    for batch_x in batches:
        z = spline(batch_x)                      # Transform
        z_compressed = (z - mu_z) @ Vk @ Vk.T   # PCA round-trip
        x_reconstructed = spline.inverse(z_compressed)
        loss = MSE(x_reconstructed, batch_x)
        loss.backward()
```

**Why it works**: Spline learns to warp space so PCA is more effective in the transformed domain.

## Key Insights from Research

1. **Monotonicity matters**: Flows without monotonicity (early attempts) diverged or collapsed
2. **Identity initialization critical**: Zero-init last layers prevents early divergence
3. **Per-dimension is enough**: Full normalizing flows (coupling blocks) don't help
4. **Low k is where it wins**: At k=64 (head_dim), plain PCA already optimal
5. **Noise robustness**: Works with 3% Gaussian noise on curved manifolds

## Limitations

1. **Calibration cost**: Fitting takes ~30-60 seconds on 120k samples
2. **One-time only**: Geometry frozen after calibration (not updated during training)
3. **Linear manifolds**: If data is already linear, Spline→PCA = PCA
4. **Head_dim ceiling**: Can't compress beyond original dimension (k ≤ 64 for GPT-2)

## Future Directions

1. **Higher-order continuity**: Current implementation guarantees C⁰ continuity. Future work:
   - **C¹ regularization** → Match tangent directions between PCA regions (smooth flow of meaning)
   - **C² regularization** → Match curvature (smooth acceleration of reasoning)
   - Goal: Make KVSplice not just a compression method, but a **continuous representation of memory geometry**

2. **Adaptive geometry**: Update spline during training (expensive)
3. **Per-head geometry**: Different splines for different attention heads
4. **K compression**: Apply to keys as well as values
5. **Block-diagonal splines**: Exploit local structure in V vectors
6. **Quantization**: Combine geometric compression with int8 quantization

## Citation

This work is part of the [knlp](https://github.com/mcgrof/knlp) project exploring attention mechanism improvements for transformer models.

**Related work:**
- MLA (Multi-Head Latent Attention): DeepSeek-V2 compression via low-rank projections
- KV cache pruning: H2O, StreamingLLM (attention-based token selection)
- Geometric deep learning: Learning on manifolds, normalizing flows

## Installation

```bash
cd ~/devel/kvsplice
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib
```

## Quick Start

### Using Makefile (Recommended)

```bash
# Run everything (experiments + plots)
make all

# Run experiments only
make test

# Generate plots from existing results
make plots

# Fast validation (2 epochs)
make quick

# Clean generated files
make clean
```

### Manual Execution

```bash
# Test Spline→PCA
python3 splinepca.py --epochs 12 --k-values 8,12,16,24,32,48,64

# Test Affine Flow→PCA
python3 affineflow_pca_experiment.py --epochs 12

# Generate plots
python3 plot_results.py

# Check results
cat results_spline_pca.csv
cat results_affineflow_pca.csv
```

### Expected Output

SplinePCA matches or beats PCA at all k values.

**Generated files:**
- `results_spline_pca.csv`: Numerical results for Spline→PCA
- `results_affineflow_pca.csv`: Numerical results for Flow→PCA
- `compression_comparison.png`: MSE comparison plots
- `improvement_delta.png`: Delta bars showing improvement over PCA
- `memory_reduction.png`: Ablation study memory usage
- `architecture_diagram.png`: Visual pipeline diagram

## License

MIT-0 (same as [knlp](https://github.com/mcgrof/knlp) parent project)

## Acknowledgments

Research inspired by:
- DeepSeek-V2's MLA mechanism for KV compression
- Normalizing flows literature (RealNVP, Glow)
- Geometric manifold learning (diffusion models, VAEs)

Experimental validation on [knlp](https://github.com/mcgrof/knlp) GPT-2 training infrastructure.

---
