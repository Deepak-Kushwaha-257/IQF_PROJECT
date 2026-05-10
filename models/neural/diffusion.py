"""
models/neural/diffusion.py
==========================
Diffusion model (TimeGrad) — Section 2.3.3.2

This model uses GluonTS's TimeGrad implementation.
It requires additional dependencies:
    pip install gluonts==0.12.9 mxnet==1.9.1

Due to complex MXNet/GluonTS versioning issues, we provide
an adapter that wraps the external TimeGrad implementation.

External repo: github.com/mbohlkeschneider/gluon-ts (timegrad branch)

SETUP INSTRUCTIONS:
  1. pip install gluonts==0.12.9
  2. pip install mxnet==1.9.1  (exact version required)
  3. If MXNet fails: conda install -c anaconda mxnet=1.9.1

The DiffusionModel below is a simplified PyTorch implementation
that captures the key concepts for the project without GluonTS.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from models.base_model import BaseModel

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TimeGradRNN(nn.Module):
    """Simplified TimeGrad noise predictor (GRU-based)."""
    def __init__(self, d, hidden_dim=64, n_diffusion_steps=100):
        super().__init__()
        self.rnn = nn.GRU(d + 1, hidden_dim, batch_first=True)  # +1 for step index
        self.fc  = nn.Linear(hidden_dim, d)
        self.T   = n_diffusion_steps

    def forward(self, x_noisy, step_idx, h=None):
        # x_noisy: (B, q, d), step_idx: (B,)
        step_emb = step_idx.float().unsqueeze(1).unsqueeze(2) / self.T
        step_emb = step_emb.expand(-1, x_noisy.size(1), 1)
        inp = torch.cat([x_noisy, step_emb], dim=-1)
        out, h_new = self.rnn(inp, h)
        noise_pred = self.fc(out)
        return noise_pred, h_new


class DiffusionModel(BaseModel):
    """
    Simplified DDPM-style diffusion model for financial time series.

    Paper uses TimeGrad from GluonTS. This is a PyTorch re-implementation
    of the key concepts: forward noising process + learned denoising.

    T=100 diffusion steps, linear variance schedule β_1=1e-4 to β_T=0.1
    """

    def __init__(
        self,
        p: int = 10, q: int = 10, d: int = 9,
        hidden_dim: int = 64,
        T: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
    ):
        super().__init__("DIFFUSION", p, q, d)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")

        self.T          = T
        self.batch_size = batch_size
        self.epochs     = epochs
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Linear variance schedule (Section 2.3.3.2)
        beta  = torch.linspace(1e-4, 0.1, T)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer = lambda n, t: setattr(self, n, t.to(self.device))
        self.beta      = beta.to(self.device)
        self.alpha     = alpha.to(self.device)
        self.alpha_bar = alpha_bar.to(self.device)

        self.model = TimeGradRNN(d, hidden_dim, T).to(self.device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """Add noise for diffusion step t."""
        ab_t = self.alpha_bar[t].view(-1, 1, 1)
        eps  = torch.randn_like(x0)
        x_t  = torch.sqrt(ab_t) * x0 + torch.sqrt(1 - ab_t) * eps
        return x_t, eps

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "DiffusionModel":
        from torch.utils.data import TensorDataset, DataLoader
        N  = X_tgt_train.shape[0]
        ds = TensorDataset(
            torch.tensor(X_tgt_train, dtype=torch.float32),
            torch.tensor(X_cond_train, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        print(f"Training DIFFUSION: {self.epochs} epochs, T={self.T}")
        self.model.train()

        for epoch in range(self.epochs):
            losses = []
            for x0, cond in loader:
                x0   = x0.to(self.device)
                cond = cond.to(self.device)
                B    = x0.size(0)

                # Random diffusion step for each sample
                t = torch.randint(0, self.T, (B,), device=self.device)
                x_t, eps = self._forward_process(x0, t)

                eps_pred, _ = self.model(x_t, t)
                loss = ((eps - eps_pred) ** 2).mean()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.append(loss.item())

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | Loss={np.mean(losses):.4f}")

        self.is_fitted = True
        return self

    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        """Reverse diffusion: sample from noise → data."""
        self.model.eval()
        paths = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Start from pure noise
                x = torch.randn(1, self.q, self.d, device=self.device)

                # Reverse process
                for t_idx in reversed(range(self.T)):
                    t_tensor = torch.tensor([t_idx], device=self.device)
                    eps_pred, _ = self.model(x, t_tensor)

                    alpha_t = self.alpha[t_idx]
                    ab_t    = self.alpha_bar[t_idx]
                    beta_t  = self.beta[t_idx]

                    # DDPM reverse step
                    x = (1 / torch.sqrt(alpha_t)) * (
                        x - (beta_t / torch.sqrt(1 - ab_t)) * eps_pred
                    )
                    if t_idx > 0:
                        x += torch.sqrt(beta_t) * torch.randn_like(x)

                paths.append(x.squeeze(0).cpu().numpy())

        return np.stack(paths, axis=0)   # (n_samples, q, d)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "diffusion.pt"))

    def load(self, path: str) -> "DiffusionModel":
        self.model.load_state_dict(
            torch.load(os.path.join(path, "diffusion.pt"), map_location=self.device)
        )
        self.is_fitted = True
        return self
