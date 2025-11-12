#!/usr/bin/env python3
"""
Stable Affine-Coupling Flow -> PCA vs PCA on a curved synthetic dataset.

Key fixes vs v1:
- RealNVP blocks with ZERO-INIT last layers for s/t (identity start).
- Stronger clamp on s, tanh-bounded scale (safe exp).
- Small latent whitening regularizer: mean(z)^2 + (var(z)-1)^2.
- Channel permutations between blocks.
- Lower LR and gradient clipping.

Expected: test MSE Flow->PCA ties/edges PCA at the same k.
"""

import argparse, csv, math
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Dataset (same as spline)
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
    M, _ = np.linalg.qr(M)
    X = pre @ M
    X = X / (X.std(axis=0, keepdims=True) + 1e-6)
    X += rng.normal(0, 0.03, X.shape)
    return X.astype(np.float32)

# -------------------------
# PCA helpers
# -------------------------

def pca_fit(x: torch.Tensor):
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
# RealNVP-like flow (stable)
# -------------------------

class ZeroInitLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128, zero_last: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = ZeroInitLinear(hidden, d_out) if zero_last else nn.Linear(hidden, d_out)
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(m.weight, a=0.2); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)

class AffineCoupling(nn.Module):
    """
    y1 = x1
    y2 = x2 * exp(s(x1)) + t(x1), with s bounded via tanh -> clamp
    Start exactly at identity: s_net and t_net last layers zero-initialized.
    """
    def __init__(self, d: int, hidden: int, mask: torch.Tensor, s_clip: float = 1.0):
        super().__init__()
        self.register_buffer("mask", mask)         # [d] bool
        d_in = int(mask.sum().item())
        d_out = d - d_in
        # ZERO-INIT -> identity at start
        self.s_net = TinyMLP(d_in, d_out, hidden, zero_last=True)
        self.t_net = TinyMLP(d_in, d_out, hidden, zero_last=True)
        self.s_clip = s_clip

    def _split(self, x):
        x1 = x[:, self.mask]
        x2 = x[:, ~self.mask]
        return x1, x2

    def _merge(self, x1, x2, like):
        y = like.clone()
        y[:, self.mask] = x1
        y[:, ~self.mask] = x2
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._split(x)
        s = self.s_net(x1)
        t = self.t_net(x1)
        # tame scale: s in [-s_clip, s_clip]
        s = self.s_clip * torch.tanh(s / self.s_clip)
        y2 = x2 * torch.exp(s) + t
        return self._merge(x1, y2, x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = self._split(y)
        s = self.s_net(y1)
        t = self.t_net(y1)
        s = self.s_clip * torch.tanh(s / self.s_clip)
        x2 = (y2 - t) * torch.exp(-s)
        return self._merge(y1, x2, y)

class Permute(nn.Module):
    """Fixed channel permutation (and its inverse)."""
    def __init__(self, d: int, perm: torch.Tensor):
        super().__init__()
        self.register_buffer("perm", perm)               # [d]
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(d, device=perm.device)
        self.register_buffer("inv", inv)

    def forward(self, x):  return x[:, self.perm]
    def inverse(self, y):  return y[:, self.inv]

class Flow(nn.Module):
    def __init__(self, d: int, blocks: int = 3, hidden: int = 128, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        # alternating masks
        half = d // 2
        m0 = torch.zeros(d, dtype=torch.bool); m0[:half] = True
        m1 = ~m0
        masks = [m0 if i % 2 == 0 else m1 for i in range(blocks)]
        # random permutations between blocks
        perms = []
        for _ in range(blocks):
            p = torch.randperm(d)
            perms.append(p)
        self.layers = nn.ModuleList([])
        for i in range(blocks):
            self.layers.append(AffineCoupling(d, hidden, masks[i]))
            self.layers.append(Permute(d, perms[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y) if hasattr(layer, "inverse") else layer(y)  # Permute has inverse
        return y

# -------------------------
# Train flow to be PCA-k friendly (stable)
# -------------------------

def pca_k_in_latent(flow, x, k):
    with torch.no_grad():
        z = flow(x)
        mu, Vh = pca_fit(z)
        Vk = Vh[:k].T
    return mu, Vk

def train_flow_for_k(
    flow: Flow,
    train: torch.Tensor,
    test: torch.Tensor,
    k_target: int = 16,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lam_whiten: float = 1e-4,   # tiny whitening regularizer
    grad_clip: float = 1.0,
):
    ds = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        mu_z, Vk = pca_k_in_latent(flow, train, k_target)
        losses = []
        for (x,) in ds:
            z = flow(x)
            # PCA-k in latent, then invert
            zc = (z - mu_z) @ Vk @ Vk.T + mu_z
            xr = flow.inverse(zc)
            recon = F.mse_loss(xr, x)

            # latent whitening (keep flow near scale/shift identity)
            z_detach = z.detach()
            m = z_detach.mean(dim=0)
            v = z_detach.var(dim=0, unbiased=False)
            whiten = (m.pow(2).mean() + (v - 1.0).pow(2).mean())

            loss = recon + lam_whiten * whiten
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
            opt.step()
            losses.append(float(recon.detach().cpu()))

        if ep % 4 == 0 or ep == 1 or ep == epochs:
            with torch.no_grad():
                zt = flow(test)
                mu, Vh = pca_fit(zt)
                Vk_t = Vh[:k_target].T
                xrt = flow.inverse((zt - mu) @ Vk_t @ Vk_t.T + mu)
                mse_t = F.mse_loss(xrt, test).item()
            print(f"[Flow] epoch {ep:02d}  train_mse={np.mean(losses):.6f}  test_mse={mse_t:.6f}")

# -------------------------
# Eval sweep
# -------------------------

def evaluate_sweep(train: torch.Tensor, test: torch.Tensor, flow: Flow, ks: List[int]) -> List[dict]:
    rows = []
    with torch.no_grad():
        mu_x, Vh_x = pca_fit(train)
        z_train = flow(train)
        z_test = flow(test)
        for k in ks:
            xh_p = pca_proj_recon(test, mu_x, Vh_x, k)
            mse_p = F.mse_loss(xh_p, test).item()

            mu_z, Vh_z = pca_fit(z_train)
            z_proj = pca_proj_recon(z_test, mu_z, Vh_z, k)
            xh_flow = flow.inverse(z_proj)
            mse_f = F.mse_loss(xh_flow, test).item()

            rows.append({"k": k, "test_mse_PCA": mse_p, "test_mse_FlowPCA": mse_f})
    return rows

def print_and_save(rows: List[dict], out_csv: str = "results_affineflow_pca.csv") -> None:
    print("\n=== Test MSE: PCA vs Flow->PCA (lower is better) ===")
    print(f"{'k':>4}  {'PCA':>14}  {'FlowPCA':>14}  {'delta(FlowPCA-PCA)':>22}")
    for r in rows:
        delta = r["test_mse_FlowPCA"] - r["test_mse_PCA"]
        print(f"{r['k']:4d}  {r['test_mse_PCA']:14.6f}  {r['test_mse_FlowPCA']:14.6f}  {delta:22.6f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k","test_mse_PCA","test_mse_FlowPCA"])
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
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--k-train", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(5); np.random.seed(5)

    X = make_geometric_dataset(n=args.n, d=args.d, seed=args.seed)
    X_train, X_test = X[: int(0.75 * len(X))], X[int(0.75 * len(X)) :]
    want_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if want_cuda else "cpu")
    if want_cuda:
        try: torch.set_float32_matmul_precision("medium")
        except Exception: pass

    dtype = torch.float16 if (args.fp16 and want_cuda) else torch.float32
    train = torch.tensor(X_train, device=device, dtype=dtype)
    test  = torch.tensor(X_test,  device=device, dtype=dtype)

    flow = Flow(d=args.d, blocks=args.blocks, hidden=args.hidden, seed=42).to(device=device, dtype=dtype)

    # Train with stable settings
    train_flow_for_k(
        flow=flow,
        train=train,
        test=test,
        k_target=args.k_train,
        epochs=args.epochs,
        lr=1e-3,
        lam_whiten=1e-4,
        grad_clip=1.0,
    )

    # Evaluate
    ks = [8, 12, 16, 24, 32, 48, 64]
    rows = evaluate_sweep(train, test, flow, ks)
    print_and_save(rows, out_csv="results_affineflow_pca.csv")

if __name__ == "__main__":
    main()
