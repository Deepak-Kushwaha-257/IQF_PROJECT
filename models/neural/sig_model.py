"""
models/neural/sig_model.py
==========================
Signature Conditional Wasserstein GAN (SIGCWGAN).

Uses path signatures as the discriminator loss (instead of a learned critic).

Architecture (Section 2.3.2.4):
  1. Compute signature S(X) of real and synthetic paths
  2. Fit signature forecast: L(S_{t-p:t}) → Ŝ_{t+1:t+q}
  3. Generator minimizes ||E[S(X̃)] - Ŝ||²

Requires: pip install signatory --no-binary signatory
External: github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs

FALLBACK: If signatory is unavailable, uses moment-matching as a proxy.
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

try:
    import signatory
    SIGNATORY_AVAILABLE = True
except ImportError:
    SIGNATORY_AVAILABLE = False
    print("WARNING: signatory not installed. SIG model will use moment proxy.")
    print("Install: pip install signatory --no-binary signatory")


class ARFNN(nn.Module):
    """Autoregressive Feed-Forward NN for signature prediction."""
    def __init__(self, in_dim, out_dim, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SIGModel(BaseModel):
    """
    Signature CWGAN. Uses signatory library for path signatures.
    Falls back to moment-matching if signatory is unavailable.
    """

    def __init__(
        self,
        p: int = 10, q: int = 10, d: int = 9,
        noise_dim: int = 30,
        sig_depth: int = 2,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 200,
    ):
        super().__init__("SIG", p, q, d)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")

        self.noise_dim = noise_dim
        self.sig_depth = sig_depth
        self.batch_size = batch_size
        self.epochs     = epochs
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute signature dimensions
        if SIGNATORY_AVAILABLE:
            # Augmented: d + time channel
            aug_d = d + 1
            self.sig_dim_cond = signatory.signature_channels(aug_d, sig_depth)
            self.sig_dim_tgt  = signatory.signature_channels(aug_d, sig_depth)
        else:
            # Proxy: use first 2 moments per tenor
            self.sig_dim_cond = d * 2
            self.sig_dim_tgt  = d * 2

        # Step 1: Signature predictor L: S(cond) → Ŝ(target)
        self.sig_predictor = ARFNN(
            self.sig_dim_cond, self.sig_dim_tgt, hidden_dim
        ).to(self.device)

        # Step 2: Generator G: z + cond → x̃
        self.generator = ARFNN(
            noise_dim + p * d, q * d, hidden_dim
        ).to(self.device)

        self.sig_opt = torch.optim.Adam(self.sig_predictor.parameters(), lr=lr)
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=lr)

    def _compute_sig(self, x: torch.Tensor) -> torch.Tensor:
        """Compute path signature. x: (B, T, d)"""
        if SIGNATORY_AVAILABLE:
            # Add time channel as augmentation
            B, T, d = x.shape
            t_chan = torch.linspace(0, 1, T, device=x.device).view(1, T, 1).expand(B, -1, -1)
            x_aug = torch.cat([x, t_chan], dim=-1)
            return signatory.signature(x_aug, self.sig_depth)
        else:
            # Moment proxy
            return torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "SIGModel":
        from torch.utils.data import TensorDataset, DataLoader

        N = X_cond_train.shape[0]
        cond_t = torch.tensor(X_cond_train, dtype=torch.float32)
        tgt_t  = torch.tensor(X_tgt_train,  dtype=torch.float32)

        ds     = TensorDataset(cond_t, tgt_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        print(f"Training SIG: {self.epochs} epochs (signatory={'yes' if SIGNATORY_AVAILABLE else 'proxy'})")

        # Phase 1: train signature predictor
        print("  Phase 1: training signature predictor...")
        for epoch in range(self.epochs // 2):
            losses = []
            for cond, tgt in loader:
                cond, tgt = cond.to(self.device), tgt.to(self.device)
                sig_cond  = self._compute_sig(cond)
                sig_tgt   = self._compute_sig(tgt)
                sig_pred  = self.sig_predictor(sig_cond)
                loss = ((sig_tgt - sig_pred) ** 2).mean()
                self.sig_opt.zero_grad()
                loss.backward()
                self.sig_opt.step()
                losses.append(loss.item())
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1} | Sig loss={np.mean(losses):.4f}")

        # Phase 2: train generator
        print("  Phase 2: training generator...")
        for epoch in range(self.epochs // 2):
            losses = []
            for cond, tgt in loader:
                cond, tgt  = cond.to(self.device), tgt.to(self.device)
                B = cond.size(0)
                sig_cond   = self._compute_sig(cond)
                sig_target = self.sig_predictor(sig_cond).detach()

                z      = torch.randn(B, self.noise_dim, device=self.device)
                cond_f = cond.view(B, -1)
                x_fake = self.generator(torch.cat([z, cond_f], dim=1))
                x_fake_seq = x_fake.view(B, self.q, self.d)
                sig_fake = self._compute_sig(x_fake_seq)
                sig_fake_mean = sig_fake.mean(dim=0)

                loss = ((sig_fake_mean - sig_target.mean(dim=0)) ** 2).mean()
                self.gen_opt.zero_grad()
                loss.backward()
                self.gen_opt.step()
                losses.append(loss.item())

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1} | Gen loss={np.mean(losses):.6f}")

        self.is_fitted = True
        return self

    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        self.generator.eval()
        cond_t = torch.tensor(condition.reshape(1, -1), dtype=torch.float32).to(self.device)
        cond_r = cond_t.expand(n_samples, -1)
        z = torch.randn(n_samples, self.noise_dim, device=self.device)
        with torch.no_grad():
            out = self.generator(torch.cat([z, cond_r], dim=1))
        return out.cpu().numpy().reshape(n_samples, self.q, self.d)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.sig_predictor.state_dict(), os.path.join(path, "sig_pred.pt"))
        torch.save(self.generator.state_dict(),     os.path.join(path, "generator.pt"))

    def load(self, path: str) -> "SIGModel":
        self.sig_predictor.load_state_dict(
            torch.load(os.path.join(path, "sig_pred.pt"), map_location=self.device)
        )
        self.generator.load_state_dict(
            torch.load(os.path.join(path, "generator.pt"), map_location=self.device)
        )
        self.is_fitted = True
        return self
