"""AnoGAN 基线模型

基于GAN的异常检测：判别器+生成器，
异常检测时通过搜索潜空间找到最佳重建，用判别器特征距离+重建误差作为异常分数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.feature(x)
        out = self.classifier(feat)
        return out, feat


class AnoGAN(nn.Module):
    """AnoGAN: GAN-based anomaly detection

    训练阶段：训练GAN拟合正常数据分布
    推理阶段：梯度搜索潜空间，找到最佳重建
    """

    def __init__(self, input_dim, latent_dim=16, hidden_dim=64, device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        self.generator = Generator(latent_dim, input_dim, hidden_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim)

    def compute_loss(self, x, optimizer_d=None, optimizer_g=None):
        """一步训练：交替优化判别器和生成器"""
        B = x.shape[0]
        if x.dim() == 3 and x.shape[1] == 1:
            x_flat = x.squeeze(1)
        else:
            x_flat = x

        # ===== 判别器 =====
        z = torch.randn(B, self.latent_dim, device=self.device)
        fake = self.generator(z)

        real_out, _ = self.discriminator(x_flat)
        fake_out, _ = self.discriminator(fake.detach())

        d_loss = -(torch.mean(real_out) - torch.mean(fake_out))

        if optimizer_d is not None:
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

        # ===== 生成器 =====
        z = torch.randn(B, self.latent_dim, device=self.device)
        fake = self.generator(z)
        fake_out, fake_feat = self.discriminator(fake)
        _, real_feat = self.discriminator(x_flat)

        g_loss = -torch.mean(fake_out)
        feat_loss = F.mse_loss(fake_feat, real_feat.detach())
        g_total = g_loss + 10.0 * feat_loss

        if optimizer_g is not None:
            optimizer_g.zero_grad()
            g_total.backward()
            optimizer_g.step()

        return d_loss.item() + g_total.item()

    def reconstruct(self, x, n_steps=30, lr=0.1):
        """通过梯度搜索潜空间找到最佳重建"""
        if x.dim() == 3 and x.shape[1] == 1:
            x_flat = x.squeeze(1)
        else:
            x_flat = x

        B = x_flat.shape[0]
        z = torch.randn(B, self.latent_dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr)

        with torch.no_grad():
            _, target_feat = self.discriminator(x_flat)
            target_feat = target_feat.detach()

        for _ in range(n_steps):
            optimizer.zero_grad()
            fake = self.generator(z)
            recon_loss = F.mse_loss(fake, x_flat.detach())
            _, fake_feat = self.discriminator(fake)
            feat_loss = F.mse_loss(fake_feat, target_feat)
            loss = recon_loss + 0.1 * feat_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            recon = self.generator(z)
        return recon.unsqueeze(1)

    def save(self, path):
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
