"""自编码器（AE）基线模型

用于异常检测：训练时重建正常时序，测试时通过重建误差检测异常。
"""
import torch
import torch.nn as nn


class AEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class AEDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    """一维时序自编码器"""

    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = AEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = AEDecoder(input_dim, hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, C, W) -> 取单通道 (B, W)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon.unsqueeze(1)  # (B, 1, W)

    def compute_loss(self, x):
        recon = self.forward(x)
        if recon.shape != x.shape:
            recon = recon.view_as(x)
        return torch.nn.functional.mse_loss(recon, x)

    @torch.no_grad()
    def reconstruct(self, x):
        return self.forward(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)
