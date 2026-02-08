from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class VAEConfig:
    seed: int = 42
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3
    hidden: int = 16
    latent: int = 2
    beta: float = 1.0  # KL weight
    device: str = "cpu"


class VAE(nn.Module):
    def __init__(self, hidden: int = 16, latent: int = 2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)

        self.dec = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_score_vae(s: pd.Series, cfg: VAEConfig = VAEConfig()) -> pd.Series:
    """
    Trains a tiny VAE and returns reconstruction error as anomaly score.
    Higher score => more anomalous. Missing values => NaN score.
    """
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    x = s.astype(float).to_numpy()
    mask = ~np.isnan(x)

    scores = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 200:
        return pd.Series(scores, index=s.index)

    X = x[mask].reshape(-1, 1).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model = VAE(hidden=cfg.hidden, latent=cfg.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for _ in range(cfg.epochs):
        for (batch,) in dl:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)

            recon_loss = (recon - batch).pow(2).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + cfg.beta * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

    # score
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        recon, _, _ = model(X_t)
        err = (recon - X_t).pow(2).mean(dim=1).cpu().numpy()

    scores[mask] = err
    return pd.Series(scores, index=s.index)