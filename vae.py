"""变分自编码器（VAE）基线模型

在AE基础上引入KL散度正则化，学习数据的概率分布。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.fc_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def forward(self, x):
        return self.fc_mu(x), self.fc_logvar(x)


class VAEDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    """一维时序变分自编码器"""

    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(input_dim, hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x_flat = x.squeeze(1)
        else:
            x_flat = x
        mu, logvar = self.encoder(x_flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon.unsqueeze(1), mu, logvar

    def compute_loss(self, x):
        recon, mu, logvar = self.forward(x)
        x_target = x if x.dim() == 3 else x.unsqueeze(1)
        if recon.shape != x_target.shape:
            recon = recon.view_as(x_target)

        recon_loss = F.mse_loss(recon, x_target, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kld_loss) / x.shape[0]

    @torch.no_grad()
    def reconstruct(self, x):
        recon, _, _ = self.forward(x)
        return recon

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)
