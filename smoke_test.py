"""快速冒烟测试：验证模型结构和前向传播正确性

运行方式: python scripts/smoke_test.py
"""
import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models.unet_1d import UNet1D
from src.models.diffusion import DiffusionModel
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.anogan import AnoGAN


def test_unet():
    """测试1D U-Net前向传播"""
    print("测试 UNet1D...")
    B, C, W = 4, 1, 100
    model = UNet1D(in_channels=1, base_channels=16, channel_mults=(1, 2, 4),
                  num_res_blocks=1, time_dim=64)
    x = torch.randn(B, C, W)
    t = torch.randint(0, 100, (B,))

    out = model(x, t)
    assert out.shape == (B, C, W), f"期望 {(B,C,W)}, 得到 {out.shape}"
    print(f"  输入: {x.shape} -> 输出: {out.shape}  [OK]")

    # 带条件
    model_cond = UNet1D(in_channels=1, base_channels=16, channel_mults=(1, 2, 4),
                        num_res_blocks=1, time_dim=64, cond_dim=4)
    cond = torch.randn(B, 4)
    out_cond = model_cond(x, t, cond)
    assert out_cond.shape == (B, C, W)
    print(f"  条件扩散: cond={cond.shape} -> 输出: {out_cond.shape}  [OK]")
    return model


def test_diffusion(unet=None):
    """测试扩散模型前向/反向过程"""
    print("\n测试 DiffusionModel...")
    if unet is None:
        unet = UNet1D(in_channels=1, base_channels=16, channel_mults=(1, 2, 4),
                      num_res_blocks=1, time_dim=64)

    model = DiffusionModel(unet, n_timesteps=50, schedule="cosine")
    B, C, W = 4, 1, 100
    x0 = torch.randn(B, C, W)

    # 前向加噪
    t = torch.tensor([0, 10, 25, 49])
    x_t, noise = model.q_sample(x0, t)
    assert x_t.shape == x0.shape
    print(f"  前向加噪: {x0.shape} -> {x_t.shape}  [OK]")

    # 训练损失
    loss = model.compute_loss(x0)
    assert loss.dim() == 0 and loss.item() > 0
    print(f"  训练损失: {loss.item():.4f}  [OK]")

    # 重建 (少步数快速测试)
    recon = model.reconstruct(x0, use_multiscale=False)
    assert recon.shape == x0.shape
    print(f"  标准重建: {recon.shape}  [OK]")

    # 多尺度重建
    recon_ms = model.reconstruct(x0, use_multiscale=True)
    assert recon_ms.shape == x0.shape
    print(f"  多尺度重建: {recon_ms.shape}  [OK]")


def test_ae():
    """测试AE"""
    print("\n测试 Autoencoder...")
    W = 100
    model = Autoencoder(input_dim=W, hidden_dim=32, latent_dim=8)
    x = torch.randn(4, 1, W)
    loss = model.compute_loss(x)
    recon = model.reconstruct(x)
    assert recon.shape == x.shape
    print(f"  输入: {x.shape} -> 重建: {recon.shape}, Loss: {loss.item():.4f}  [OK]")


def test_vae():
    """测试VAE"""
    print("\n测试 VAE...")
    W = 100
    model = VAE(input_dim=W, hidden_dim=32, latent_dim=8)
    x = torch.randn(4, 1, W)
    loss = model.compute_loss(x)
    recon = model.reconstruct(x)
    assert recon.shape == x.shape
    print(f"  输入: {x.shape} -> 重建: {recon.shape}, Loss: {loss.item():.4f}  [OK]")


def test_anogan():
    """测试AnoGAN"""
    print("\n测试 AnoGAN...")
    W = 100
    model = AnoGAN(input_dim=W, latent_dim=8, hidden_dim=32)
    x = torch.randn(4, 1, W)
    loss = model.compute_loss(x)
    recon = model.reconstruct(x, n_steps=5)
    assert recon.shape == x.shape
    print(f"  输入: {x.shape} -> 重建: {recon.shape}, Loss: {loss:.4f}  [OK]")


def test_iforest():
    """测试Isolation Forest"""
    print("\n测试 Isolation Forest...")
    from src.models.isolation_forest import IsolationForestDetector
    model = IsolationForestDetector(contamination=0.1)
    x = np.random.randn(100, 50)
    model.fit(x)
    scores = model.score(x)
    preds = model.predict(x)
    assert len(scores) == 100 and len(preds) == 100
    print(f"  训练数据: {x.shape} -> 分数: {scores.shape}  [OK]")


def main():
    print("="*50)
    print("  冒烟测试: 验证所有模型结构")
    print("="*50)

    unet = test_unet()
    test_diffusion(unet)
    test_ae()
    test_vae()
    test_anogan()
    test_iforest()

    print("\n" + "="*50)
    print("  所有测试通过!")
    print("="*50)


if __name__ == "__main__":
    main()
