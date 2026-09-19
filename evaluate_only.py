"""仅评估脚本（Point-level Point Adjustment 版）

直接加载已训练好的权重，在 point（点）级粒度上计算含 Point Adjustment
（点调整）协议的指标，无需重新训练模型。point 级是 UCR / TSB-AD 等
基准采用的评测粒度，能真实体现点调整的增益。

运行:
  python scripts/evaluate_only.py                 # 对比实验 (含 point-level PA)
  python scripts/evaluate_only.py --ablation      # 消融实验 (含 point-level PA)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config, get_ablation_config
from src.data.ucr_loader import get_dataloaders
from src.models.unet_1d import UNet1D
from src.models.diffusion import DiffusionModel
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.anogan import AnoGAN
from src.models.isolation_forest import IsolationForestDetector
from src.evaluate.metrics import (
    get_recon_scores, get_iforest_scores, compute_metrics,
    print_metrics, point_level_pa,
)


def build_unet(config):
    u = config["unet"]
    return UNet1D(
        in_channels=u.get("in_channels", 1),
        base_channels=u.get("base_channels", 32),
        channel_mults=tuple(u.get("channel_mults", [1, 2, 4, 4])),
        num_res_blocks=u.get("num_res_blocks", 2),
        time_dim=u.get("time_dim", 128),
        dropout=u.get("dropout", 0.1),
        cond_dim=u.get("cond_dim", None),
    )


def load_diffusion(config, weight_path):
    unet = build_unet(config)
    model = DiffusionModel(
        unet,
        n_timesteps=config["diffusion"]["n_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        schedule=config["diffusion"]["schedule"],
        device="cpu",
    )
    model.load(weight_path)
    return model


def add_point_pa(m, scores, test_set):
    """用 point-level PA 覆盖 f1_pa/precision_pa/recall_pa，并补充 auroc_pa"""
    pa = point_level_pa(scores, test_set)
    m["f1_pa"] = pa["f1_pa"]
    m["precision_pa"] = pa["precision_pa"]
    m["recall_pa"] = pa["recall_pa"]
    m["auroc_pa"] = pa["auroc"]
    return m


def run_comparison(config, save_dir):
    device = "cpu"
    _, test_loader, _, test_set = get_dataloaders(config, None)
    results = {}

    # ===== Diffusion (Ours) =====
    diff = load_diffusion(config, os.path.join(save_dir, "diffusion_main.pth"))
    use_ms = config["experiment"]["use_multiscale"]
    scores, labels = get_recon_scores(diff, test_loader, device,
                                      use_multiscale=use_ms, model_type="diffusion")
    m = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    m = add_point_pa(m, scores, test_set)
    print_metrics("Diffusion Model (ours)", m)
    results["Diffusion (Ours)"] = m

    # ===== AE =====
    ws = config["data"]["window_size"]
    ae = Autoencoder(input_dim=ws, hidden_dim=config["baselines"]["ae"]["hidden_dim"],
                     latent_dim=config["baselines"]["ae"]["latent_dim"]).to(device)
    ae.load(os.path.join(save_dir, "ae.pth"))
    scores, labels = get_recon_scores(ae, test_loader, device, model_type="ae")
    m = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    m = add_point_pa(m, scores, test_set)
    print_metrics("Autoencoder", m)
    results["AE"] = m

    # ===== VAE =====
    vae = VAE(input_dim=ws, hidden_dim=config["baselines"]["vae"]["hidden_dim"],
              latent_dim=config["baselines"]["vae"]["latent_dim"]).to(device)
    vae.load(os.path.join(save_dir, "vae.pth"))
    scores, labels = get_recon_scores(vae, test_loader, device, model_type="vae")
    m = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    m = add_point_pa(m, scores, test_set)
    print_metrics("VAE", m)
    results["VAE"] = m

    # ===== AnoGAN =====
    anogan = AnoGAN(input_dim=ws, latent_dim=config["baselines"]["anogan"]["latent_dim"],
                    hidden_dim=config["baselines"]["anogan"]["hidden_dim"], device=device).to(device)
    anogan.load(os.path.join(save_dir, "anogan.pth"))
    scores, labels = get_recon_scores(anogan, test_loader, device, model_type="anogan")
    m = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    m = add_point_pa(m, scores, test_set)
    print_metrics("AnoGAN", m)
    results["AnoGAN"] = m

    # ===== Isolation Forest =====
    iforest = IsolationForestDetector(
        contamination=config["baselines"]["iforest"]["contamination"],
        n_estimators=config["baselines"]["iforest"]["n_estimators"],
    )
    iforest.load(os.path.join(save_dir, "iforest.pkl"))
    scores, labels = get_iforest_scores(iforest, test_loader)
    m = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    m = add_point_pa(m, scores, test_set)
    print_metrics("Isolation Forest", m)
    results["IF"] = m

    with open(os.path.join(save_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  对比实验结果汇总 (Point-level Point Adjustment)")
    print("=" * 70)
    print(f"{'Method':<20}{'AUROC':>9}{'F1':>9}{'F1_PA':>9}{'Prec_PA':>9}{'Rec_PA':>9}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<20}{m['auroc']:>9.4f}{m['f1']:>9.4f}{m['f1_pa']:>9.4f}"
              f"{m['precision_pa']:>9.4f}{m['recall_pa']:>9.4f}")
    print("=" * 70)
    return results


def run_ablation(config, save_dir):
    device = "cpu"
    _, test_loader, _, test_set = get_dataloaders(config, None)
    ablation_types = ["full", "no_aug", "no_cond", "no_multiscale"]
    results = {}

    for ab_type in ablation_types:
        ab_config = get_ablation_config(config, ab_type)
        ab_save_dir = os.path.join(save_dir, f"ablation_{ab_type}")
        weight_path = os.path.join(ab_save_dir, f"diffusion_{ab_type}.pth")

        diff = load_diffusion(ab_config, weight_path)
        use_ms = ab_config["experiment"]["use_multiscale"]
        scores, labels = get_recon_scores(diff, test_loader, device,
                                          use_multiscale=use_ms, model_type="diffusion")
        m = compute_metrics(scores, labels, ab_config["experiment"]["threshold_percentile"])
        m = add_point_pa(m, scores, test_set)
        print_metrics(f"Ablation: {ab_type}", m)
        results[ab_type] = m

    with open(os.path.join(save_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  消融实验结果汇总 (Point-level Point Adjustment)")
    print("=" * 70)
    labels_map = {
        "full": "Full (全部优化)",
        "no_aug": "w/o Data Augmentation",
        "no_cond": "w/o Conditional Diffusion",
        "no_multiscale": "w/o Multi-scale Reconstruction",
    }
    print(f"{'Setting':<40}{'AUROC':>9}{'F1':>9}{'F1_PA':>9}")
    print("-" * 64)
    for ab_type, m in results.items():
        label = labels_map.get(ab_type, ab_type)
        print(f"{label:<40}{m['auroc']:>9.4f}{m['f1']:>9.4f}{m['f1_pa']:>9.4f}")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description="仅评估（point-level PA）")
    parser.add_argument("--ablation", action="store_true", help="运行消融实验")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    parser.add_argument("--save_dir", default=os.path.join(PROJECT_ROOT, "experiments", "results"))
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.ablation:
        run_ablation(config, args.save_dir)
    else:
        run_comparison(config, args.save_dir)


if __name__ == "__main__":
    main()
