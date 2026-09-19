"""生成少样本实验折线图（AUROC vs few-shot ratio）

运行方式: python scripts/gen_few_shot_fig.py
"""
import os
import sys
import json
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

RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "results", "few_shot_results.json")


def plot_few_shot_curve():
    if not os.path.exists(RESULTS_PATH):
        print("少样本结果不存在:", RESULTS_PATH)
        return

    with open(RESULTS_PATH) as f:
        d = json.load(f)

    ratios = sorted(d.keys(), key=float)
    xlabels = [f"{int(round(float(r) * 100))}%" for r in ratios]

    methods = ["Diffusion (Ours)", "AE", "VAE", "IF"]
    colors = {
        "Diffusion (Ours)": "#534AB7",
        "AE": "#1D9E75",
        "VAE": "#378ADD",
        "IF": "#D85A30",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        ys = [d[r].get(m, {}).get("auroc") for r in ratios]
        if any(v is None for v in ys):
            continue
        lw = 2.8 if m == "Diffusion (Ours)" else 1.6
        ms = 9 if m == "Diffusion (Ours)" else 7
        ax.plot(xlabels, ys, marker="o", linewidth=lw, markersize=ms,
                color=colors[m], label=m)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(xlabels[0], 0.505, "随机猜测 (0.5)", color="gray", fontsize=9, va="bottom")
    ax.set_xlabel("训练数据比例 (Few-shot ratio)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("少样本条件下各方法 AUROC 对比", fontsize=14)
    ax.set_ylim(0.45, 0.80)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "few_shot_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {save_path}")


if __name__ == "__main__":
    plot_few_shot_curve()
