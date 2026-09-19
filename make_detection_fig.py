# -*- coding: utf-8 -*-
"""生成论文「检测可视化」配图（5.7 节）。

加载扩散模型（000 权重），对某一 UCR 子集做异常检测，绘制：
  - 上：原始时间序列，异常区间红色阴影标注
  - 下：窗口级异常分数曲线 + 阈值线 + 检出点

依赖 matplotlib（系统 Python 自带）+ 项目模型代码。
运行: python scripts/make_detection_fig.py [--idx 0]
"""
import os
import sys
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

_FONT = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.data.ucr_loader import get_dataloaders, UCRAnomalyDataset
from src.models.unet_1d import UNet1D
from src.models.diffusion import DiffusionModel
from src.evaluate.metrics import get_recon_scores

FIG_PATH = os.path.join(PROJECT_ROOT, "paper", "figures", "detection_demo.png")


def load_raw(name, data_dir):
    """读取原始序列与异常区间（来自文件名 _START_END_LEN）。"""
    raw = np.loadtxt(os.path.join(data_dir, "raw", name + ".txt"))
    ts = raw[:-100].astype(np.float32)
    parts = name.split("_")
    try:
        s, e = int(parts[-3]), int(parts[-2])
    except (ValueError, IndexError):
        s, e = len(ts) - 100, len(ts)
    return ts, s, e


def build_unet(config):
    u = config["unet"]
    return UNet1D(in_channels=u.get("in_channels", 1), base_channels=u.get("base_channels", 32),
                  channel_mults=tuple(u.get("channel_mults", [1, 2, 4, 4])),
                  num_res_blocks=u.get("num_res_blocks", 2), time_dim=u.get("time_dim", 128),
                  dropout=u.get("dropout", 0.1), cond_dim=u.get("cond_dim", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="子集索引（默认0）")
    args = ap.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    config["data"]["dataset_idx"] = args.idx
    data_dir = config["data"]["data_dir"]
    ws = config["data"]["window_size"]
    stride = config["data"]["stride"]
    pct = config["experiment"]["threshold_percentile"]
    use_ms = config["experiment"]["use_multiscale"]
    res_dir = os.path.join(PROJECT_ROOT, "experiments", "results")

    names = UCRAnomalyDataset.get_available_datasets(data_dir)
    name = names[args.idx]
    ts, a_start, a_end = load_raw(name, data_dir)

    # 模型
    unet = build_unet(config)
    diff = DiffusionModel(unet, n_timesteps=config["diffusion"]["n_timesteps"],
                          beta_start=config["diffusion"]["beta_start"],
                          beta_end=config["diffusion"]["beta_end"],
                          schedule=config["diffusion"]["schedule"], device="cpu")
    diff.load(os.path.join(res_dir, "diffusion_main.pth"))

    _, test_loader, _, _ = get_dataloaders(config, None)
    scores, labels = get_recon_scores(diff, test_loader, "cpu",
                                       use_multiscale=use_ms, model_type="diffusion")
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # 阈值（与论文评估一致）
    thr = np.percentile(scores, pct)
    detected = scores > thr

    # 窗口中心时间坐标
    centers = np.arange(len(scores)) * stride + ws // 2
    # 限制绘图长度（过长则截断到前 1500 点更易读）
    plot_len = min(len(ts), 1600)
    t = np.arange(plot_len)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)

    # 上：原始序列 + 异常区间阴影
    ax1.plot(t, ts[:plot_len], color="#1f4e79", linewidth=0.8)
    sa, ea = max(a_start, 0), min(a_end, plot_len)
    ax1.add_patch(Rectangle((sa, ax1.get_ylim()[0]), ea - sa,
                            ax1.get_ylim()[1] - ax1.get_ylim()[0],
                            facecolor="red", alpha=0.18, zorder=0))
    ax1.set_ylabel("归一化幅值", fontsize=10)
    ax1.set_title("原始时间序列（红色阴影为真实异常区间）", fontsize=11)
    ax1.grid(linestyle=":", alpha=0.4)

    # 下：分数曲线 + 阈值 + 检出点
    ax2.plot(centers, scores, color="#c0392b", linewidth=0.9, label="异常分数")
    ax2.axhline(thr, color="#2c3e50", linestyle="--", linewidth=1,
                label="阈值 (P%d)" % pct)
    det_idx = np.where(detected)[0]
    if len(det_idx) > 0:
        ax2.scatter(centers[det_idx], scores[det_idx], color="#e67e22",
                    s=14, zorder=3, label="检出异常窗口")
    ax2.set_xlim(0, plot_len)
    ax2.set_ylabel("重建误差分数", fontsize=10)
    ax2.set_xlabel("时间步", fontsize=10)
    ax2.set_title("扩散模型窗口级异常分数", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print("已生成检测可视化图:", FIG_PATH)
    print("  子集=%s  异常区间=[%d,%d]  阈值=%.4f  检出窗口=%d/%d"
          % (name, a_start, a_end, thr, int(detected.sum()), len(scores)))


if __name__ == "__main__":
    main()
