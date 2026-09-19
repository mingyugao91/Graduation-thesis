"""基线模型训练脚本"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.anogan import AnoGAN
from src.models.isolation_forest import IsolationForestDetector
from src.data.ucr_loader import get_dataloaders
from src.utils.config import load_config


def train_ae(config, train_loader, device="cpu"):
    """训练自编码器"""
    window_size = config["data"]["window_size"]
    model = Autoencoder(
        input_dim=window_size,
        hidden_dim=config["baselines"]["ae"]["hidden_dim"],
        latent_dim=config["baselines"]["ae"]["latent_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["baselines"]["ae"]["lr"])

    for epoch in range(config["baselines"]["ae"]["epochs"]):
        model.train()
        losses = []
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            loss = model.compute_loss(batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  AE Epoch {epoch+1}: Loss={np.mean(losses):.6f}")

    return model


def train_vae(config, train_loader, device="cpu"):
    """训练VAE"""
    window_size = config["data"]["window_size"]
    model = VAE(
        input_dim=window_size,
        hidden_dim=config["baselines"]["vae"]["hidden_dim"],
        latent_dim=config["baselines"]["vae"]["latent_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["baselines"]["vae"]["lr"])

    for epoch in range(config["baselines"]["vae"]["epochs"]):
        model.train()
        losses = []
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            loss = model.compute_loss(batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  VAE Epoch {epoch+1}: Loss={np.mean(losses):.6f}")

    return model


def train_anogan(config, train_loader, device="cpu"):
    """训练AnoGAN"""
    window_size = config["data"]["window_size"]
    model = AnoGAN(
        input_dim=window_size,
        latent_dim=config["baselines"]["anogan"]["latent_dim"],
        hidden_dim=config["baselines"]["anogan"]["hidden_dim"],
        device=device,
    ).to(device)

    opt_d = torch.optim.Adam(model.discriminator.parameters(),
                            lr=config["baselines"]["anogan"]["lr"], betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(),
                            lr=config["baselines"]["anogan"]["lr"], betas=(0.5, 0.999))

    for epoch in range(config["baselines"]["anogan"]["epochs"]):
        model.train()
        losses = []
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            loss = model.compute_loss(batch_x, opt_d, opt_g)
            losses.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f"  AnoGAN Epoch {epoch+1}: Loss={np.mean(losses):.6f}")

    return model


def train_iforest(config, train_loader):
    """训练孤立森林"""
    model = IsolationForestDetector(
        contamination=config["baselines"]["iforest"]["contamination"],
        n_estimators=config["baselines"]["iforest"]["n_estimators"],
    )

    # 收集训练数据
    all_x = []
    for batch_x, _ in train_loader:
        x = batch_x.numpy()
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        all_x.append(x)
    all_x = np.concatenate(all_x)
    model.fit(all_x)
    return model


def train_all_baselines(config, train_loader, device="cpu", save_dir="experiments/results"):
    """训练所有基线模型"""
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    print("\n--- 训练 Autoencoder ---")
    ae = train_ae(config, train_loader, device)
    ae.save(os.path.join(save_dir, "ae.pth"))
    results["ae"] = ae

    print("\n--- 训练 VAE ---")
    vae = train_vae(config, train_loader, device)
    vae.save(os.path.join(save_dir, "vae.pth"))
    results["vae"] = vae

    print("\n--- 训练 AnoGAN ---")
    anogan = train_anogan(config, train_loader, device)
    anogan.save(os.path.join(save_dir, "anogan.pth"))
    results["anogan"] = anogan

    print("\n--- 训练 Isolation Forest ---")
    iforest = train_iforest(config, train_loader)
    iforest.save(os.path.join(save_dir, "iforest.pkl"))
    results["iforest"] = iforest

    return results


if __name__ == "__main__":
    from src.data.augmentation import TimeSeriesAugmenter

    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    augment_fn = TimeSeriesAugmenter() if config["experiment"]["use_augmentation"] else None
    train_loader, _, _, _ = get_dataloaders(config, augment_fn)
    train_all_baselines(config, train_loader, config["train"]["device"])
