"""一维 U-Net 去噪主干网络

用于扩散模型的去噪网络，输入为带噪一维时序 + 时间步嵌入。
结构：编码器（下采样）→ 瓶颈层 → 解码器（上采样 + 跳跃连接）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码，将时间步映射为高维嵌入"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) 时间步
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class ResBlock1d(nn.Module):
    """一维残差块，融合时间步嵌入"""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None]  # (B, C, 1) 广播
        h = self.conv2(h)
        return h + self.shortcut(x)


class Downsample1d(nn.Module):
    """下采样：步长为2的卷积"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """上采样：最近邻插值 + 卷积"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet1D(nn.Module):
    """一维 U-Net 去噪网络

    Args:
        in_channels: 输入通道数（单变量时序=1）
        base_channels: 基础通道数
        channel_mults: 各层通道倍数，如 [1, 2, 4]
        num_res_blocks: 每层残差块数量
        time_dim: 时间嵌入维度
        dropout: dropout率
        out_channels: 输出通道数（通常=in_channels）
        cond_dim: 条件信息维度（条件扩散用，None表示无条件）
    """

    def __init__(self, in_channels=1, base_channels=32, channel_mults=(1, 2, 4, 4),
                 num_res_blocks=2, time_dim=128, dropout=0.1, out_channels=None,
                 cond_dim=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.cond_dim = cond_dim

        # 时间嵌入: 始终输出 time_dim 维，条件在 forward 中拼接
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ResBlock 接收的时间嵌入维度 = time_dim + cond_dim (如有条件)
        time_total_dim = time_dim + (cond_dim if cond_dim is not None else 0)

        # 输入投影
        channels = base_channels
        self.init_conv = nn.Conv1d(in_channels, channels, 3, padding=1)

        # 编码器
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        down_channels = [channels]
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock1d(channels, out_ch, time_total_dim, dropout))
                channels = out_ch
                down_channels.append(channels)
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample1d(channels))
                down_channels.append(channels)

        # 瓶颈层
        mid_ch = channels
        self.mid1 = ResBlock1d(mid_ch, mid_ch, time_total_dim, dropout)
        self.mid2 = ResBlock1d(mid_ch, mid_ch, time_total_dim, dropout)

        # 解码器
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for j in range(num_res_blocks + 1):
                skip_ch = down_channels.pop()
                self.ups.append(ResBlock1d(channels + skip_ch, out_ch, time_total_dim, dropout))
                channels = out_ch
            if i > 0:
                self.ups.append(Upsample1d(channels))

        # 输出投影
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t, cond=None):
        """
        Args:
            x: (B, C, W) 带噪时序
            t: (B,) 时间步 (long)
            cond: (B, cond_dim) 条件信息 (可选)
        Returns:
            (B, C, W) 预测的噪声
        """
        # 时间嵌入
        t_emb = self.time_mlp(t.float())
        if cond is not None and self.cond_dim is not None:
            t_emb = torch.cat([t_emb, cond], dim=-1)

        # 输入
        x = self.init_conv(x)
        input_len = x.shape[-1]

        # 编码器（保存skip）
        skips = [x]
        for layer in self.downs:
            if isinstance(layer, ResBlock1d):
                x = layer(x, t_emb)
            else:
                x = layer(x)
            skips.append(x)

        # 瓶颈层
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # 解码器
        for layer in self.ups:
            if isinstance(layer, ResBlock1d):
                skip = skips.pop()
                if x.shape[-1] != skip.shape[-1]:
                    skip = F.interpolate(skip, size=x.shape[-1], mode="linear",
                                         align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
            else:
                x = layer(x)

        # 确保输出长度与输入一致（处理非2的幂的窗口长度）
        if x.shape[-1] != input_len:
            x = F.interpolate(x, size=input_len, mode="linear", align_corners=False)

        return self.out_conv(x)
