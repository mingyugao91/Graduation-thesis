"""扩散模型（DDPM）核心模块

包含：
- 前向扩散过程（加噪）
- 反向去噪过程（采样）
- 训练目标（去噪损失）
- 条件扩散支持
- 多尺度重建支持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class DiffusionModel(nn.Module):
    """DDPM 扩散模型

    用于时序异常检测：训练时学习正常数据的分布，
    推理时通过重建误差区分异常。
    """

    def __init__(self, unet, n_timesteps=200, beta_start=1e-4, beta_end=0.02,
                 schedule="linear", device="cpu"):
        super().__init__()
        self.unet = unet
        self.n_timesteps = n_timesteps
        self.device = device

        # beta schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _cosine_schedule(n, s=0.008):
        """余弦调度，减少噪声添加的剧烈程度"""
        steps = torch.arange(n + 1)
        f = torch.cos((steps / n + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, 0.0001, 0.999)

    def q_sample(self, x0, t, noise=None):
        """前向过程：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1, 1)
        sqrt_1mab = (1.0 - self.alpha_bars[t]).sqrt().view(-1, 1, 1)

        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    def compute_loss(self, x0, cond=None):
        """计算训练损失：预测噪声的MSE

        Args:
            x0: (B, C, W) 干净时序
            cond: (B, cond_dim) 条件信息（可选）
        Returns:
            loss: 标量
        """
        B = x0.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=self.device)
        x_t, noise = self.q_sample(x0, t)

        pred_noise = self.unet(x_t, t, cond)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t, cond=None):
        """反向过程单步去噪"""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

        pred_noise = self.unet(x_t, t_tensor, cond)

        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        mean = (1.0 / alpha.sqrt()) * (x_t - (beta / (1 - alpha_bar).sqrt()) * pred_noise)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = beta.sqrt()
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def reconstruct(self, x0, cond=None, use_multiscale=False, noise_ratio=0.5,
                    on_step=None):
        """部分扩散重建：对输入加部分噪声后去噪，用于异常检测

        核心思路：不从纯噪声开始，而是对原始输入加噪到 t_start 步，
        再逆向去噪。这样正常数据能被较好还原（低重建误差），
        而异常数据因模型未见过其分布，还原效果差（高重建误差）。

        Args:
            x0: (B, C, W) 原始时序
            cond: 条件信息
            use_multiscale: 是否使用多尺度重建
            noise_ratio: 加噪比例 (0~1)，0.5表示加到一半步数
            on_step: 可选回调，每完成一步去噪时调用 on_step(frac)，
                     frac∈(0,1] 表示当前批次去噪进度（用于进度条）
        Returns:
            recon: (B, C, W) 重建时序
        """
        B, C, W = x0.shape

        if use_multiscale:
            return self._multiscale_reconstruct(x0, cond, noise_ratio, on_step)

        # 部分扩散：对输入加噪到 t_start 步
        t_start = max(1, int(self.n_timesteps * noise_ratio))
        t_tensor = torch.full((B,), t_start - 1, device=self.device, dtype=torch.long)
        x_noisy, _ = self.q_sample(x0, t_tensor)

        # 逆向去噪
        x = x_noisy
        for step, t in enumerate(reversed(range(t_start))):
            x = self.p_sample(x, t, cond)
            if on_step is not None:
                on_step((step + 1) / t_start)
        return x

    @torch.no_grad()
    def _multiscale_reconstruct(self, x0, cond=None, noise_ratio=0.5, on_step=None):
        """多尺度重建：在不同分辨率下分别部分扩散重建，加权融合"""
        B, C, W = x0.shape
        results = []
        has_cond = cond is not None

        def compute_cond(x):
            """根据输入数据计算条件特征"""
            return torch.cat([
                x.mean(dim=2), x.std(dim=2),
                x.max(dim=2).values, x.min(dim=2).values,
            ], dim=1)

        # 各尺度去噪步数一致，累计回调进度
        t_start = max(1, int(self.n_timesteps * noise_ratio))
        n_scales = 1 + (1 if W > 4 else 0) + (1 if W > 8 else 0)
        step_global = {"i": 0}

        def _report():
            if on_step is not None:
                on_step(min(1.0, step_global["i"] / max(1, t_start * n_scales)))

        # 尺度1：原始分辨率
        t_tensor = torch.full((B,), t_start - 1, device=self.device, dtype=torch.long)
        x_noisy, _ = self.q_sample(x0, t_tensor)
        cond_1 = cond if has_cond else None
        for t in reversed(range(t_start)):
            x_noisy = self.p_sample(x_noisy, t, cond_1)
            step_global["i"] += 1
            _report()
        results.append(x_noisy)

        # 尺度2：半分辨率
        if W > 4:
            x_half = F.avg_pool1d(x0, 2)
            t_h = torch.full((B,), t_start - 1, device=self.device, dtype=torch.long)
            x_h_noisy, _ = self.q_sample(x_half, t_h)
            cond_2 = compute_cond(x_half) if has_cond else None
            for t in reversed(range(t_start)):
                x_h_noisy = self.p_sample(x_h_noisy, t, cond_2)
                step_global["i"] += 1
                _report()
            x_half_up = F.interpolate(x_h_noisy, size=W, mode="linear",
                                       align_corners=False)
            results.append(x_half_up)

        # 尺度3：1/4分辨率
        if W > 8:
            x_quarter = F.avg_pool1d(x0, 4)
            x_q_noisy, _ = self.q_sample(x_quarter, t_h)
            cond_3 = compute_cond(x_quarter) if has_cond else None
            for t in reversed(range(t_start)):
                x_q_noisy = self.p_sample(x_q_noisy, t, cond_3)
                step_global["i"] += 1
                _report()
            x_q_up = F.interpolate(x_q_noisy, size=W, mode="linear",
                                   align_corners=False)
            results.append(x_q_up)

        # 加权融合
        weights = [0.5, 0.3, 0.2][:len(results)]
        fused = sum(w * r for w, r in zip(weights, results))
        return fused

    def save(self, path):
        torch.save({
            "unet_state_dict": self.unet.state_dict(),
            "n_timesteps": self.n_timesteps,
            "betas": self.betas.cpu(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(ckpt["unet_state_dict"])
        self.n_timesteps = ckpt["n_timesteps"]
        self.betas = ckpt["betas"].to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
