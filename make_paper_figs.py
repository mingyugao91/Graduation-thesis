# -*- coding: utf-8 -*-
"""论文配图生成：读取 multi_dataset_results.json，绘制多数据集 AUROC 对比图。

产出：
  paper/figures/multidataset_auroc.png   8 数据集 × 4 方法分组柱状图
  paper/figures/multidataset_avg.png     4 方法跨数据集平均 AUROC 对比

仅依赖 matplotlib + numpy（系统 Python 自带）。容错：跳过 error 项。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示器后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 注册中文字体（避免标题中文变方框）
_FONT = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJECT_ROOT, "experiments", "results", "multi_dataset_results.json")
FIGDIR = os.path.join(PROJECT_ROOT, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["Diffusion (Ours)", "AE", "VAE", "IF"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
SHORT = {"Diffusion (Ours)": "Diffusion", "AE": "AE", "VAE": "VAE", "IF": "IF"}


def load():
    with open(RES, encoding="utf-8") as f:
        return json.load(f)


def main():
    data = load()
    names = list(data.keys())
    n = len(names)
    if n == 0:
        print("JSON 为空，跳过绘图")
        return

    # 构建矩阵 auroc[i, j]
    matrix = np.full((n, len(METHODS)), np.nan)
    for i, name in enumerate(names):
        for j, m in enumerate(METHODS):
            cell = data[name].get(m, {})
            if isinstance(cell, dict) and "auroc" in cell:
                matrix[i, j] = cell["auroc"]

    # 图1：分组柱状图
    x = np.arange(n)
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for j, m in enumerate(METHODS):
        vals = matrix[:, j]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=SHORT[m], color=COLORS[j])
        # 在柱顶标注数值（仅非 nan）
        for xi, v in zip(x + (j - 1.5) * w, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=0)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("图 N  跨数据集异常检测 AUROC 对比（固定 000 权重跨场景推断）",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([nm[:22] for nm in names], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    p1 = os.path.join(FIGDIR, "multidataset_auroc.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", p1)

    # 图2：平均性能
    mean_vals = np.nanmean(matrix, axis=0)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    bars = ax2.bar([SHORT[m] for m in METHODS], mean_vals, color=COLORS, width=0.6)
    for b, v in zip(bars, mean_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("平均 AUROC", fontsize=11)
    ax2.set_title("图 N+1  各方法跨 %d 数据集平均 AUROC" % n, fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    fig2.tight_layout()
    p2 = os.path.join(FIGDIR, "multidataset_avg.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("已生成:", p2)

    # 控制台摘要
    print("\n=== 各方法平均 AUROC（%d 个数据集）===" % n)
    for j, m in enumerate(METHODS):
        print("  %-16s %.4f" % (m, mean_vals[j]))


if __name__ == "__main__":
    main()
