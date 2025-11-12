#!/usr/bin/env python3
"""
Spline -> PCA vs PCA on a curved synthetic dataset (GeometricMix, d=128).

This version:
- fixes the non-contiguous warning (makes bucketize/searchsorted inputs contiguous)
- optional CUDA and FP16
- prints table and saves CSV
"""

import argparse, csv
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Dataset
# -------------------------

def make_geometric_dataset(n: int = 6000, d: int = 128, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.random((n, 1))
    tri = 0.5 * t * (t + 1.0)
    atoms = np.concatenate([
        np.ones_like(t),
        t, t**2, tri,
        np.sqrt(np.clip(t, 1e-6, 1.0)),
        np.sin(2 * np.pi * t),
        np.cos(2 * np.pi * t),
    ], axis=1)  # [n, 7]

    A = 48
    W_atoms = rng.normal(0, 1, (atoms.shape[1], A))
    mask = rng.random(W_atoms.shape) < 0.6
    W_atoms *= mask
    pre = atoms @ W_atoms  # [n, A]

    pre = np.concatenate([pre, np.tanh(pre), 1 / (1 + np.exp(-pre))], axis=1)  # [n, 3A]
    M = rng.normal(0, 1, (pre.shape[1], d))
    M, _ = np.linalg.qr(M)  # orthonormal-ish
    X = pre @ M  # [n, d]
    X = X / (X.std(axis=0, keepdims=True) + 1e-6)
    X += rng.normal(0, 0.03, X.shape)
    return X.astype(np.float32)

# -------------------------
# PCA helpers
# -------------------------

def pca_fit(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    U, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    return mu, Vh

def pca_proj_recon(x: torch.Tensor, mu: torch.Tensor, Vh: torch.Tensor, k: int) -> torch.Tensor:
    Vk = Vh[:k].T
    z = (x - mu) @ Vk
    xh = z @ Vk.T + mu
    return xh

# -------------------------
# Monotone PWL spline (per-dim, invertible)
# -------------------------

class PWLSpline(nn.Module):
    """
    Per-dimension monotone piecewise-linear spline:
    - Knot positions x_knots[d, K] are fixed (quantiles of train data).
    - Positive slopes per segment (softplus) ensure monotonicity.
    - Final per-dim affine scale/shift (positive scale).
    - Inverse is analytic per segment.
    """
    def __init__(self, x_knots: torch.Tensor):
        super().__init__()
        # ensure contiguous storage once up front
        self.register_buffer("xk", x_knots.contiguous())  # [d, K]
        d, K = self.xk.shape
        self.K = K
        self.delta_raw = nn.Parameter(torch.zeros(d, K - 1))
        self.scale_raw = nn.Parameter(torch.zeros(d))
        self.shift = nn.Parameter(torch.zeros(d))
        self.eps = 1e-4

    def _slopes_and_yk(self):
        xk = self.xk  # [d, K] (contiguous)
        seg_dx = xk[:, 1:] - xk[:, :-1]             # [d, K-1]
        slopes = F.softplus(self.delta_raw) + self.eps
        # normalize average slope to ~1 to stabilize range
        avg_slope = (slopes * seg_dx).sum(dim=1, keepdim=True) / (seg_dx.sum(dim=1, keepdim=True) + 1e-8)
        slopes = slopes / (avg_slope + 1e-8)

        yk = torch.zeros(xk.shape[0], xk.shape[1], device=xk.device, dtype=xk.dtype)
        yk[:, 1:] = torch.cumsum(slopes * seg_dx, dim=1)  # y at each knot
        return slopes, yk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, d = x.shape
        xk = self.xk
        K = self.K
        slopes, yk = self._slopes_and_yk()

        ys = []
        for j in range(d):
            # make both inputs contiguous before bucketize/searchsorted
            xj = x[:, j].contiguous()
            xkj = xk[j].contiguous()
            idx = torch.searchsorted(xkj, xj, right=False)  # equivalent to bucketize on sorted
            idx = torch.clamp(idx, 1, K - 1)
            i0 = idx - 1

            x0 = xkj[i0]
            y0 = yk[j, i0]
            m  = slopes[j, i0]
            yj = y0 + m * (xj - x0)
            ys.append(yj.unsqueeze(1))
        y = torch.cat(ys, dim=1)

        scale = F.softplus(self.scale_raw).unsqueeze(0) + 1e-3
        y = y * scale + self.shift.unsqueeze(0)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        N, d = y.shape
        xk = self.xk
        K = self.K
        slopes, yk = self._slopes_and_yk()

        scale = F.softplus(self.scale_raw).unsqueeze(0) + 1e-3
        y_ = (y - self.shift.unsqueeze(0)) / scale

        xs = []
        for j in range(d):
            yj = y_[:, j].contiguous()
            ykj = yk[j].contiguous()
            idx = torch.searchsorted(ykj, yj, right=False)
            idx = torch.clamp(idx, 1, K - 1)
            i0 = idx - 1

            y0 = ykj[i0]
            m  = slopes[j, i0]
            x0 = xk[j, i0]
            xj = x0 + (yj - y0) / (m + 1e-8)
            xs.append(xj.unsqueeze(1))
        x = torch.cat(xs, dim=1)
        return x

def build_spline_from_data(x: torch.Tensor, K: int = 7) -> PWLSpline:
    qs = torch.linspace(0.0, 1.0, K, device=x.device, dtype=x.dtype)
    x_sorted, _ = torch.sort(x, dim=0)
    idxs = (qs * (x.shape[0] - 1)).long()
    xk = x_sorted[idxs, :].T.contiguous()  # [d, K], contiguous to silence warnings
    return PWLSpline(xk)

# -------------------------
# Train & Eval
# -------------------------

def train_spline_for_k(
    spline: PWLSpline,
    train: torch.Tensor,
    test: torch.Tensor,
    k_target: int = 16,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
) -> None:
    ds = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    opt = torch.optim.AdamW(spline.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        with torch.no_grad():
            z_train = spline(train)
            mu_z, Vh_z = pca_fit(z_train)
            Vk = Vh_z[:k_target].T

        losses = []
        for (x,) in ds:
            z = spline(x)
            zc = (z - mu_z) @ Vk @ Vk.T + mu_z
            xr = spline.inverse(zc)
            recon = F.mse_loss(xr, x)
            opt.zero_grad(set_to_none=True)
            recon.backward()
            opt.step()
            losses.append(float(recon.detach().cpu()))
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            with torch.no_grad():
                zt = spline(test)
                xrt = spline.inverse((zt - mu_z) @ Vk @ Vk.T + mu_z)
                mse_t = F.mse_loss(xrt, test).item()
            print(f"[Spline] epoch {ep:02d}  train_mse={np.mean(losses):.6f}  test_mse={mse_t:.6f}")

def evaluate_sweep(train: torch.Tensor, test: torch.Tensor, spline: PWLSpline, ks: List[int]) -> List[dict]:
    rows = []
    with torch.no_grad():
        mu_x, Vh_x = pca_fit(train)
        z_train = spline(train)
        z_test = spline(test)
        for k in ks:
            xh_p = pca_proj_recon(test, mu_x, Vh_x, k)
            mse_p = F.mse_loss(xh_p, test).item()

            mu_z, Vh_z = pca_fit(z_train)
            z_proj = pca_proj_recon(z_test, mu_z, Vh_z, k)
            xh_spline = spline.inverse(z_proj)
            mse_s = F.mse_loss(xh_spline, test).item()

            rows.append({"k": k, "test_mse_PCA": mse_p, "test_mse_SplinePCA": mse_s})
    return rows

def print_and_save(rows: List[dict], out_csv: str = "results_spline_pca.csv") -> None:
    print("\n=== Test MSE: PCA vs Spline->PCA (lower is better) ===")
    print(f"{'k':>4}  {'PCA':>14}  {'SplinePCA':>14}  {'delta(SplinePCA-PCA)':>24}")
    for r in rows:
        delta = r["test_mse_SplinePCA"] - r["test_mse_PCA"]
        print(f"{r['k']:4d}  {r['test_mse_PCA']:14.6f}  {r['test_mse_SplinePCA']:14.6f}  {delta:24.6f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k","test_mse_PCA","test_mse_SplinePCA"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved CSV -> {out_csv}")

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--k-train", type=int, default=16)
    ap.add_argument("--k-values", type=str, default="8,12,16,24,32,48,64",
                    help="Comma-separated k values to test")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--fp16", action="store_true", help="run model/data in float16 (on CUDA recommended)")
    ap.add_argument("--knots", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(5); np.random.seed(5)

    X = make_geometric_dataset(n=args.n, d=args.d, seed=args.seed)
    X_train, X_test = X[: int(0.75 * len(X))], X[int(0.75 * len(X)) :]

    want_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if want_cuda else "cpu")
    if want_cuda:
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    # use float16 if asked (mainly on CUDA)
    dtype = torch.float16 if (args.fp16 and want_cuda) else torch.float32

    train = torch.tensor(X_train, device=device, dtype=dtype)
    test  = torch.tensor(X_test,  device=device, dtype=dtype)

    # Build spline with quantile knots from train (match dtype/device)
    spline = build_spline_from_data(train, K=args.knots).to(device=device, dtype=dtype)

    # Train spline to be PCA-k friendly
    train_spline_for_k(
        spline=spline,
        train=train,
        test=test,
        k_target=args.k_train,
        epochs=args.epochs,
    )

    # Evaluate across a sweep of k
    ks = [int(k.strip()) for k in args.k_values.split(',')]
    rows = evaluate_sweep(train, test, spline, ks)
    print_and_save(rows, out_csv="results_spline_pca.csv")

if __name__ == "__main__":
    main()
