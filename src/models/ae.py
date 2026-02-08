from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class AEConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    hidden: int = 16
    latent: int = 4
    device: str = "cpu"


class AutoEncoder(nn.Module):
    def __init__(self, hidden: int = 16, latent: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_score_ae(s: pd.Series, cfg: AEConfig = AEConfig()) -> pd.Series:
    """
    Trains a tiny AE on non-missing points and returns reconstruction error as anomaly score.
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

    model = AutoEncoder(hidden=cfg.hidden, latent=cfg.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(cfg.epochs):
        for (batch,) in dl:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # score
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        recon = model(X_t)
        err = (recon - X_t).pow(2).mean(dim=1).cpu().numpy()

    scores[mask] = err
    return pd.Series(scores, index=s.index)