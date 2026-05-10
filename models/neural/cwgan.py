"""
models/neural/cwgan.py
======================
Conditional Wasserstein GAN (CWGAN) — best neural network model.

Architecture (Section 2.3.2.3):
  Generator G(z, y) → x̃
    - FC layers, ReLU hidden, Linear output

  Critic D(x, y) → real scalar (NOT sigmoid — Wasserstein critic)
    - FC layers, ReLU hidden, linear output
    - Weight clipping enforces 1-Lipschitz constraint

Loss: Wasserstein distance (Eq. 22-23)
  min_G  max_{D ∈ 1-Lip}  E[D(x,y)] - E[D(x̃,y)]
Framework: PyTorch
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from models.base_model import BaseModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Run: pip install torch")


def _fc_block(in_dim, out_dim, hidden_dim, n_layers, final_activation=None):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    if final_activation == "linear" or final_activation is None:
        pass
    elif final_activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, out_dim, hidden_dim=512, n_layers=4):
        super().__init__()
        self.net = _fc_block(noise_dim + cond_dim, out_dim, hidden_dim, n_layers)

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.net(x)


class Critic(nn.Module):
    """Wasserstein critic — outputs unbounded real scalar."""
    def __init__(self, sample_dim, cond_dim, hidden_dim=512, n_layers=4):
        super().__init__()
        self.net = _fc_block(sample_dim + cond_dim, 1, hidden_dim, n_layers)

    def forward(self, x, cond):
        inp = torch.cat([x, cond], dim=1)
        return self.net(inp)


class CWGAN(BaseModel):

    def __init__(
        self,
        p: int = 10, q: int = 10, d: int = 9,
        noise_dim: int = 30,
        hidden_dim: int = 512,
        n_layers: int = 4,
        clip_value: float = 0.01,
        lr_g: float = 5e-5,
        lr_d: float = 5e-5,
        batch_size: int = 64,
        epochs: int = 300,
        n_critic: int = 5,
    ):
        super().__init__("CWGAN", p, q, d)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")

        self.noise_dim  = noise_dim
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.epochs     = epochs
        self.n_critic   = n_critic

        cond_dim = p * d
        out_dim  = q * d

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CWGAN device: {self.device}")

        self.G = Generator(noise_dim, cond_dim, out_dim, hidden_dim, n_layers).to(self.device)
        self.C = Critic(out_dim, cond_dim, hidden_dim, n_layers).to(self.device)

        self.g_opt = torch.optim.RMSprop(self.G.parameters(), lr=lr_g)
        self.c_opt = torch.optim.RMSprop(self.C.parameters(), lr=lr_d)

        self.cond_dim = cond_dim
        self.out_dim  = out_dim

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "CWGAN":
        N = X_cond_train.shape[0]
        cond_t = torch.tensor(X_cond_train.reshape(N, -1), dtype=torch.float32)
        tgt_t  = torch.tensor(X_tgt_train.reshape(N, -1),  dtype=torch.float32)

        dataset = TensorDataset(tgt_t, cond_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        print(f"Training CWGAN: {self.epochs} epochs, n_critic={self.n_critic}")
        for epoch in range(self.epochs):
            c_losses, g_losses = [], []

            for real_x, cond in loader:
                real_x = real_x.to(self.device)
                cond   = cond.to(self.device)
                bs     = real_x.size(0)

                # ── Train Critic n_critic times ──
                for _ in range(self.n_critic):
                    z = torch.randn(bs, self.noise_dim, device=self.device)
                    fake_x = self.G(z, cond).detach()

                    c_real = self.C(real_x, cond).mean()
                    c_fake = self.C(fake_x, cond).mean()
                    c_loss = c_fake - c_real   # Wasserstein: maximize E[D(real)] - E[D(fake)]

                    self.c_opt.zero_grad()
                    c_loss.backward()
                    self.c_opt.step()

                    # Weight clipping (Eq. 22, 0.01 threshold)
                    for p in self.C.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                    c_losses.append(c_loss.item())

                # ── Train Generator ──
                z = torch.randn(bs, self.noise_dim, device=self.device)
                fake_x = self.G(z, cond)
                g_loss = -self.C(fake_x, cond).mean()

                self.g_opt.zero_grad()
                g_loss.backward()
                self.g_opt.step()
                g_losses.append(g_loss.item())

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | "
                      f"C={np.mean(c_losses):.4f} | G={np.mean(g_losses):.4f}")

        self.is_fitted = True
        return self

    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        self.G.eval()
        cond = torch.tensor(
            np.tile(condition.reshape(1, -1), (n_samples, 1)), dtype=torch.float32
        ).to(self.device)
        z = torch.randn(n_samples, self.noise_dim, device=self.device)
        with torch.no_grad():
            fake = self.G(z, cond).cpu().numpy()
        return fake.reshape(n_samples, self.q, self.d)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.G.state_dict(), os.path.join(path, "generator.pt"))
        torch.save(self.C.state_dict(), os.path.join(path, "critic.pt"))

    def load(self, path: str) -> "CWGAN":
        self.G.load_state_dict(torch.load(os.path.join(path, "generator.pt"),
                                           map_location=self.device))
        self.C.load_state_dict(torch.load(os.path.join(path, "critic.pt"),
                                           map_location=self.device))
        self.is_fitted = True
        return self
