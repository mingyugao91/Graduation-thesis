"""主实验脚本：扩散模型 vs 基线 + 消融实验

运行方式:
  python scripts/run_experiment.py              # 对比实验
  python scripts/run_experiment.py --ablation   # 消融实验
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
from src.data.augmentation import TimeSeriesAugmenter
from src.models.unet_1d import UNet1D
from src.models.diffusion import DiffusionModel
from src.train.train_diffusion import train_diffusion
from src.train.train_baselines import train_all_baselines, train_ae, train_vae, train_anogan, train_iforest
from src.evaluate.metrics import get_recon_scores, get_iforest_scores, compute_metrics, print_metrics


def run_comparison(config, save_dir):
    """对比实验：扩散模型 vs 4个基线"""
    device = config["train"]["device"]
    results = {}

    # ===== 扩散模型 =====
    print("\n" + "="*60)
    print("  训练扩散模型 (Diffusion Model)")
    print("="*60)
    diffusion, test_loader = train_diffusion(config, save_dir, "diffusion_main")

    use_ms = config["experiment"]["use_multiscale"]
    scores, labels = get_recon_scores(diffusion, test_loader, device,
                                       use_multiscale=use_ms, model_type="diffusion")
    metrics = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    print_metrics("Diffusion Model (ours)", metrics)
    results["Diffusion (Ours)"] = metrics

    # ===== 基线模型 =====
    print("\n" + "="*60)
    print("  训练基线模型")
    print("="*60)

    augment_fn = TimeSeriesAugmenter() if config["experiment"]["use_augmentation"] else None
    train_loader, test_loader, _, _ = get_dataloaders(config, augment_fn)

    # AE
    print("\n--- Autoencoder ---")
    ae = train_ae(config, train_loader, device)
    ae.save(os.path.join(save_dir, "ae.pth"))
    scores, labels = get_recon_scores(ae, test_loader, device, model_type="ae")
    metrics = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    print_metrics("Autoencoder", metrics)
    results["AE"] = metrics

    # VAE
    print("\n--- VAE ---")
    vae = train_vae(config, train_loader, device)
    vae.save(os.path.join(save_dir, "vae.pth"))
    scores, labels = get_recon_scores(vae, test_loader, device, model_type="vae")
    metrics = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    print_metrics("VAE", metrics)
    results["VAE"] = metrics

    # AnoGAN
    print("\n--- AnoGAN ---")
    anogan = train_anogan(config, train_loader, device)
    anogan.save(os.path.join(save_dir, "anogan.pth"))
    scores, labels = get_recon_scores(anogan, test_loader, device, model_type="anogan")
    metrics = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    print_metrics("AnoGAN", metrics)
    results["AnoGAN"] = metrics

    # Isolation Forest
    print("\n--- Isolation Forest ---")
    iforest = train_iforest(config, train_loader)
    iforest.save(os.path.join(save_dir, "iforest.pkl"))
    scores, labels = get_iforest_scores(iforest, test_loader)
    metrics = compute_metrics(scores, labels, config["experiment"]["threshold_percentile"])
    print_metrics("Isolation Forest", metrics)
    results["IF"] = metrics

    # 保存结果
    with open(os.path.join(save_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 打印汇总表
    print("\n" + "="*60)
    print("  对比实验结果汇总")
    print("="*60)
    print(f"{'Method':<20} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("-"*60)
    for name, m in results.items():
        print(f"{name:<20} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}")
    print("="*60)

    return results


def run_ablation(config, save_dir):
    """消融实验：逐个去掉优化策略"""
    device = config["train"]["device"]
    ablation_types = ["full", "no_aug", "no_cond", "no_multiscale"]
    results = {}

    for ab_type in ablation_types:
        print(f"\n{'#'*60}")
        print(f"  消融实验: {ab_type}")
        print(f"{'#'*60}")

        ab_config = get_ablation_config(config, ab_type)
        ab_save_dir = os.path.join(save_dir, f"ablation_{ab_type}")
        os.makedirs(ab_save_dir, exist_ok=True)

        # 训练扩散模型
        diffusion, test_loader = train_diffusion(ab_config, ab_save_dir, f"diffusion_{ab_type}")

        use_ms = ab_config["experiment"]["use_multiscale"]
        scores, labels = get_recon_scores(diffusion, test_loader, device,
                                           use_multiscale=use_ms, model_type="diffusion")
        metrics = compute_metrics(scores, labels, ab_config["experiment"]["threshold_percentile"])
        print_metrics(f"Ablation: {ab_type}", metrics)
        results[ab_type] = metrics

    # 保存结果
    with open(os.path.join(save_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 汇总表
    print("\n" + "="*60)
    print("  消融实验结果汇总")
    print("="*60)
    labels_map = {
        "full": "Full (全部优化)",
        "no_aug": "w/o Data Augmentation",
        "no_cond": "w/o Conditional Diffusion",
        "no_multiscale": "w/o Multi-scale Reconstruction",
    }
    print(f"{'Setting':<40} {'AUROC':>8} {'AUPRC':>8} {'F1':>8}")
    print("-"*64)
    for ab_type, m in results.items():
        label = labels_map.get(ab_type, ab_type)
        print(f"{label:<40} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f}")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(description="扩散模型少样本时序异常检测实验")
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
