# -*- coding: utf-8 -*-
"""论文少样本对比图（5.4 节）：读 few_shot_results.json，按不同少样本比例
画各方法 AUROC 的分组柱状图，凸显 Diffusion 在极少样本下的稳定性。

输出: paper/figures/fewshot_auroc.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_FONT = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJECT_ROOT, "experiments", "results", "few_shot_results.json")
FIG = os.path.join(PROJECT_ROOT, "paper", "figures", "fewshot_auroc.png")

METHODS = ["Diffusion (Ours)", "AE", "VAE", "IF"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
SHORT = {"Diffusion (Ours)": "Diffusion(本文)", "AE": "AE", "VAE": "VAE", "IF": "IF"}


def main():
    with open(RES, encoding="utf-8") as f:
        data = json.load(f)
    # 比例按从小到大排序：0.1, 0.3, 0.5, 1.0
    ratios = sorted(data.keys(), key=lambda x: float(x))
    n = len(ratios)
    matrix = np.zeros((n, len(METHODS)))
    for i, r in enumerate(ratios):
        for j, m in enumerate(METHODS):
            v = data[r].get(m, {}).get("auroc", float("nan"))
            matrix[i, j] = v

    x = np.arange(n)
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for j, m in enumerate(METHODS):
        ax.bar(x + (j - 1.5) * w, matrix[:, j], w,
               label=SHORT[m], color=COLORS[j])
        for xi, v in zip(x + (j - 1.5) * w, matrix[:, j]):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5)

    ax.set_xticks(x)
    labels = ["{}%".format(int(float(r) * 100)) for r in ratios]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("训练正常样本比例", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=10)
    ax.set_ylim(0.4, 1.02)
    ax.set_title("图：少样本设定下各方法 AUROC 对比（UCR 子集 000）", fontsize=11)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已生成少样本对比图:", FIG)
    print("  比例:", ratios)
    for j, m in enumerate(METHODS):
        print("  %-16s auroc=%.4f ~ %.4f (Δ=%.4f)"
              % (m, matrix[:, j].min(), matrix[:, j].max(),
                 matrix[:, j].max() - matrix[:, j].min()))


if __name__ == "__main__":
    main()
