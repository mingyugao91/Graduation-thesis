"""生成实验图表用于论文

运行方式: python scripts/generate_figures.py
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(PROJECT_ROOT, "experiments", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_comparison_bar():
    """对比实验柱状图"""
    path = os.path.join(PROJECT_ROOT, "experiments", "results", "comparison_results.json")
    if not os.path.exists(path):
        print("对比实验结果不存在")
        return

    with open(path) as f:
        results = json.load(f)

    methods = list(results.keys())
    metrics = ["auroc", "auprc", "f1"]
    metric_names = ["AUROC", "AUPRC", "F1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#534AB7", "#1D9E75", "#378ADD", "#BA7517", "#D85A30"]

    for ax, metric, mname in zip(axes, metrics, metric_names):
        values = [results[m][metric] for m in methods]
        bars = ax.bar(range(len(methods)), values, color=colors[:len(methods)])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(mname, fontsize=12)
        ax.set_title(mname, fontsize=14)
        ax.set_ylim(0, 1.0)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "comparison_bar.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {save_path}")


def plot_ablation_bar():
    """消融实验柱状图"""
    path = os.path.join(PROJECT_ROOT, "experiments", "results", "ablation_results.json")
    if not os.path.exists(path):
        print("消融实验结果不存在")
        return

    with open(path) as f:
        results = json.load(f)

    labels_map = {
        "full": "Full",
        "no_aug": "w/o Aug",
        "no_cond": "w/o Cond",
        "no_multiscale": "w/o Multi-scale",
    }
    settings = list(results.keys())
    labels = [labels_map.get(s, s) for s in settings]

    metrics = ["auroc", "auprc", "f1"]
    metric_names = ["AUROC", "AUPRC", "F1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#534AB7", "#AFA9EC", "#7F77DD", "#3C3489"]

    for ax, metric, mname in zip(axes, metrics, metric_names):
        values = [results[s][metric] for s in settings]
        bars = ax.bar(range(len(settings)), values, color=colors[:len(settings)])
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel(mname, fontsize=12)
        ax.set_title(mname, fontsize=14)
        ax.set_ylim(0, 1.0)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "ablation_bar.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {save_path}")


def plot_training_curve():
    """训练损失曲线"""
    path = os.path.join(PROJECT_ROOT, "experiments", "results", "diffusion_main_losses.npy")
    if not os.path.exists(path):
        print("训练损失数据不存在")
        return

    losses = np.load(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, color="#534AB7", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("扩散模型训练损失曲线", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "training_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {save_path}")


def plot_detection_example():
    """异常检测可视化示例"""
    import torch
    from src.utils.config import load_config
    from src.data.ucr_loader import UCRAnomalyDataset
    from src.models.unet_1d import UNet1D
    from src.models.diffusion import DiffusionModel

    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    model_path = os.path.join(PROJECT_ROOT, "experiments", "results", "diffusion_main.pth")

    if not os.path.exists(model_path):
        print("扩散模型未训练")
        return

    # 加载数据
    test_set = UCRAnomalyDataset(
        config["data"]["data_dir"], config["data"]["dataset_idx"],
        config["data"]["window_size"], 1, "test", few_shot_ratio=1.0
    )
    train_set = UCRAnomalyDataset(
        config["data"]["data_dir"], config["data"]["dataset_idx"],
        config["data"]["window_size"], 1, "train", few_shot_ratio=1.0
    )
    test_set.set_scaler(train_set.scaler)

    # 加载模型
    unet = UNet1D(
        in_channels=config["unet"]["in_channels"],
        base_channels=config["unet"]["base_channels"],
        channel_mults=tuple(config["unet"]["channel_mults"]),
        num_res_blocks=config["unet"]["num_res_blocks"],
        time_dim=config["unet"]["time_dim"],
        dropout=config["unet"]["dropout"],
        cond_dim=config["unet"].get("cond_dim"),
    )
    model = DiffusionModel(unet, config["diffusion"]["n_timesteps"],
                           config["diffusion"]["beta_start"],
                           config["diffusion"]["beta_end"],
                           config["diffusion"]["schedule"])
    model.load(model_path)
    model.eval()

    # 取前30个窗口
    n_show = 30
    windows = torch.from_numpy(test_set.windows[:n_show]).float().unsqueeze(1)
    labels = test_set.labels[:n_show]

    cond_dim = config["unet"].get("cond_dim")
    use_cond = config["experiment"].get("use_conditional", False) and cond_dim is not None

    with torch.no_grad():
        cond = None
        if use_cond:
            cond = torch.cat([
                windows.mean(dim=2),
                windows.std(dim=2),
                windows.max(dim=2).values,
                windows.min(dim=2).values,
            ], dim=1)
        recon = model.reconstruct(windows, cond=cond, use_multiscale=True)

    diff = (recon - windows) ** 2
    scores = diff.mean(dim=(1, 2)).numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 原始时序
    ax = axes[0]
    ts_values = windows[:, 0, -1].numpy()
    anomaly_mask = labels == 1
    ax.plot(ts_values, color="#378ADD", linewidth=1, label="原始时序")
    if anomaly_mask.any():
        ax.scatter(np.where(anomaly_mask), ts_values[anomaly_mask],
                  color="#E24B4A", s=30, label="真实异常", zorder=5)
    ax.set_ylabel("值", fontsize=11)
    ax.set_title("时间序列原始数据", fontsize=13)
    ax.legend(fontsize=10)

    # 重建时序
    ax = axes[1]
    recon_values = recon[:, 0, -1].numpy()
    ax.plot(recon_values, color="#1D9E75", linewidth=1, label="重建时序")
    ax.set_ylabel("值", fontsize=11)
    ax.set_title("扩散模型重建结果", fontsize=13)
    ax.legend(fontsize=10)

    # 重建误差
    ax = axes[2]
    colors = ["#E24B4A" if l == 1 else "#534AB7" for l in labels]
    ax.bar(range(n_show), scores, color=colors, width=0.8)
    ax.set_ylabel("重建误差", fontsize=11)
    ax.set_xlabel("窗口索引", fontsize=11)
    ax.set_title("异常分数（重建误差）", fontsize=13)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "detection_example.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {save_path}")


def main():
    print("="*50)
    print("  生成实验图表")
    print("="*50)
    plot_comparison_bar()
    plot_ablation_bar()
    plot_training_curve()
    plot_detection_example()
    print("\n所有图表生成完成！")
    print(f"输出目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
