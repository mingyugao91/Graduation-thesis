"""扩散模型训练脚本"""
import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.unet_1d import UNet1D
from src.models.diffusion import DiffusionModel
from src.data.ucr_loader import get_dataloaders
from src.data.augmentation import TimeSeriesAugmenter
from src.utils.config import load_config


def train_diffusion(config, save_dir="experiments/results", save_name="diffusion"):
    """训练扩散模型"""
    device = config["train"]["device"]
    window_size = config["data"]["window_size"]
    use_aug = config["experiment"]["use_augmentation"]
    use_cond = config["experiment"]["use_conditional"]

    # 数据增强
    augment_fn = TimeSeriesAugmenter() if use_aug else None

    # 数据加载
    train_loader, test_loader, train_set, test_set = get_dataloaders(config, augment_fn)
    print(f"训练样本: {len(train_set)}, 测试样本: {len(test_set)}")
    print(f"窗口大小: {window_size}, 少样本比例: {config['data']['few_shot_ratio']}")
    print(f"数据增强: {use_aug}, 条件扩散: {use_cond}")

    # 模型
    cond_dim = config["unet"]["cond_dim"]
    if not use_cond:
        cond_dim = None

    unet = UNet1D(
        in_channels=config["unet"]["in_channels"],
        base_channels=config["unet"]["base_channels"],
        channel_mults=tuple(config["unet"]["channel_mults"]),
        num_res_blocks=config["unet"]["num_res_blocks"],
        time_dim=config["unet"]["time_dim"],
        dropout=config["unet"]["dropout"],
        cond_dim=cond_dim,
    ).to(device)

    diffusion = DiffusionModel(
        unet=unet,
        n_timesteps=config["diffusion"]["n_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        schedule=config["diffusion"]["schedule"],
        device=device,
    )

    optimizer = torch.optim.Adam(
        diffusion.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    # 训练循环
    best_loss = float("inf")
    losses = []

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config["train"]["epochs"]):
        diffusion.train()
        epoch_losses = []

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            cond = None
            if cond_dim is not None:
                # 简单条件：使用窗口的统计特征作为条件
                cond = torch.cat([
                    batch_x.mean(dim=2),
                    batch_x.std(dim=2),
                    batch_x.max(dim=2).values,
                    batch_x.min(dim=2).values,
                ], dim=1)  # (B, 4)

            loss = diffusion.compute_loss(batch_x, cond)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['train']['epochs']}  Loss: {avg_loss:.6f}")

    # 保存模型
    save_path = os.path.join(save_dir, f"{save_name}.pth")
    diffusion.save(save_path)
    print(f"\n模型已保存: {save_path}")

    # 保存训练曲线
    np.save(os.path.join(save_dir, f"{save_name}_losses.npy"), np.array(losses))

    return diffusion, test_loader


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    train_diffusion(config)
