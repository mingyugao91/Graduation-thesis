"""生成Streamlit平台界面截图（用matplotlib模拟）

如果streamlit已安装，可以运行真实平台；
否则用matplotlib生成模拟界面截图用于论文。

运行方式: python scripts/generate_streamlit_screenshot.py
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
FIG_DIR = os.path.join(PROJECT_ROOT, "experiments", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 颜色主题（模拟Streamlit浅色主题）
COLORS = {
    "bg": "#FFFFFF",
    "sidebar": "#F0F2F6",
    "primary": "#FF4B4B",
    "text": "#31333F",
    "accent": "#0068C9",
    "green": "#1D9E75",
    "purple": "#534AB7",
    "orange": "#BA7517",
    "red": "#E24B4A",
    "blue": "#378ADD",
    "light_blue": "#E8F0FE",
}


def generate_streamlit_mockup():
    """用matplotlib生成模拟的Streamlit界面截图"""
    # 加载实验结果
    comp_path = os.path.join(RESULTS_DIR, "comparison_results.json")
    comp_results = {}
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comp_results = json.load(f)

    abl_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    abl_results = {}
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            abl_results = json.load(f)

    # 创建画布
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS["bg"])
    gs = GridSpec(10, 12, figure=fig, hspace=0.4, wspace=0.3,
                  left=0.03, right=0.97, top=0.95, bottom=0.05)

    # ===== 标题栏 =====
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(COLORS["bg"])
    ax_title.axis("off")
    ax_title.text(0.01, 0.5, "基于扩散模型的少样本时间序列异常检测系统",
                 fontsize=18, fontweight="bold", color=COLORS["text"],
                 transform=ax_title.transAxes, va="center")

    # ===== 侧边栏 =====
    ax_sidebar = fig.add_subplot(gs[1:, 0:2])
    ax_sidebar.set_facecolor(COLORS["sidebar"])
    ax_sidebar.axis("off")
    ax_sidebar.set_xlim(0, 1)
    ax_sidebar.set_ylim(0, 1)

    # 侧边栏内容
    y = 0.95
    ax_sidebar.text(0.1, y, "参数设置", fontsize=13, fontweight="bold",
                   color=COLORS["text"], transform=ax_sidebar.transAxes)
    y -= 0.08
    ax_sidebar.text(0.1, y, "选择数据集", fontsize=10, color=COLORS["text"],
                   transform=ax_sidebar.transAxes)
    # 模拟下拉框
    rect = mpatches.FancyBboxPatch((0.1, y-0.04), 0.8, 0.04,
                                     boxstyle="round,pad=0.01",
                                     facecolor="white", edgecolor="#CCCCCC")
    ax_sidebar.add_patch(rect)
    ax_sidebar.text(0.15, y-0.02, "SYNTHETIC3860...", fontsize=8,
                   color=COLORS["text"], transform=ax_sidebar.transAxes, va="center")

    y -= 0.10
    ax_sidebar.text(0.1, y, "窗口大小: 100", fontsize=10, color=COLORS["text"],
                   transform=ax_sidebar.transAxes)
    # 模拟滑块
    rect = mpatches.FancyBboxPatch((0.1, y-0.04), 0.8, 0.03,
                                     boxstyle="round,pad=0.01",
                                     facecolor="#E0E0E0", edgecolor="none")
    ax_sidebar.add_patch(rect)
    rect = mpatches.FancyBboxPatch((0.35, y-0.04), 0.2, 0.03,
                                     boxstyle="round,pad=0.01",
                                     facecolor=COLORS["primary"], edgecolor="none")
    ax_sidebar.add_patch(rect)

    y -= 0.10
    ax_sidebar.text(0.1, y, "少样本比例: 1.0", fontsize=10, color=COLORS["text"],
                   transform=ax_sidebar.transAxes)

    y -= 0.10
    ax_sidebar.text(0.1, y, "选择检测算法", fontsize=10, color=COLORS["text"],
                   transform=ax_sidebar.transAxes)
    y -= 0.04
    rect = mpatches.FancyBboxPatch((0.1, y-0.04), 0.8, 0.04,
                                     boxstyle="round,pad=0.01",
                                     facecolor="white", edgecolor="#CCCCCC")
    ax_sidebar.add_patch(rect)
    ax_sidebar.text(0.15, y-0.02, "Diffusion", fontsize=8,
                   color=COLORS["text"], transform=ax_sidebar.transAxes, va="center")

    y -= 0.08
    ax_sidebar.text(0.1, y, "☑ 多尺度重建", fontsize=10, color=COLORS["text"],
                   transform=ax_sidebar.transAxes)

    y -= 0.08
    # 模拟按钮
    rect = mpatches.FancyBboxPatch((0.1, y-0.05), 0.8, 0.06,
                                     boxstyle="round,pad=0.02",
                                     facecolor=COLORS["primary"], edgecolor="none")
    ax_sidebar.add_patch(rect)
    ax_sidebar.text(0.5, y-0.02, "运行检测", fontsize=10, color="white",
                   transform=ax_sidebar.transAxes, ha="center", va="center",
                   fontweight="bold")

    # ===== 主区域：时序图 =====
    ax_ts = fig.add_subplot(gs[1:4, 2:7])
    ax_ts.set_facecolor(COLORS["bg"])
    # 生成模拟时序数据
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    normal = np.sin(t * 0.1) + np.random.normal(0, 0.1, n)
    # 添加异常
    normal[120:140] += 2.5 * np.sin(t[120:140] * 0.5)
    ax_ts.plot(t, normal, color=COLORS["blue"], linewidth=0.8, label="时序值")
    # 标注异常
    ax_ts.scatter(t[120:140], normal[120:140], color=COLORS["red"], s=10, label="真实异常", zorder=5)
    ax_ts.set_title("原始时序窗口", fontsize=12, color=COLORS["text"])
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.tick_params(labelsize=8)

    # ===== 异常分数图 =====
    ax_score = fig.add_subplot(gs[4:6, 2:7])
    ax_score.set_facecolor(COLORS["bg"])
    scores = np.random.exponential(0.01, n)
    scores[120:140] += np.random.uniform(0.05, 0.15, 20)
    colors = [COLORS["red"] if 120 <= i < 140 else COLORS["purple"] for i in range(n)]
    ax_score.bar(range(n), scores, color=colors, width=1.0)
    ax_score.set_title("异常分数", fontsize=12, color=COLORS["text"])
    ax_score.set_ylabel("重建误差", fontsize=10)
    ax_score.tick_params(labelsize=8)

    # ===== 检测统计 =====
    ax_stats = fig.add_subplot(gs[1:3, 7:12])
    ax_stats.set_facecolor(COLORS["bg"])
    ax_stats.axis("off")
    ax_stats.text(0.0, 0.9, "检测结果统计", fontsize=14, fontweight="bold",
                 color=COLORS["text"], transform=ax_stats.transAxes)

    # 模拟指标卡片
    metrics_data = [
        ("TP", "13", COLORS["green"]),
        ("FP", "6", COLORS["orange"]),
        ("FN", "14", COLORS["red"]),
        ("TN", "167", COLORS["blue"]),
    ]
    for i, (label, value, color) in enumerate(metrics_data):
        x = 0.05 + i * 0.24
        rect = mpatches.FancyBboxPatch((x, 0.4), 0.2, 0.3,
                                         boxstyle="round,pad=0.02",
                                         facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5)
        ax_stats.add_patch(rect)
        ax_stats.text(x + 0.1, 0.58, value, fontsize=22, fontweight="bold",
                     color=color, transform=ax_stats.transAxes, ha="center")
        ax_stats.text(x + 0.1, 0.45, label, fontsize=10,
                     color=COLORS["text"], transform=ax_stats.transAxes, ha="center")

    # Precision, Recall, F1
    p_r_f1 = [("Precision", "0.6842"), ("Recall", "0.4815"), ("F1", "0.5652")]
    for i, (label, value) in enumerate(p_r_f1):
        x = 0.05 + i * 0.32
        ax_stats.text(x, 0.25, f"{label}", fontsize=9, color=COLORS["text"],
                     transform=ax_stats.transAxes)
        ax_stats.text(x, 0.10, value, fontsize=18, fontweight="bold",
                     color=COLORS["accent"], transform=ax_stats.transAxes)

    # ===== 分数分布直方图 =====
    ax_hist = fig.add_subplot(gs[3:6, 7:12])
    ax_hist.set_facecolor(COLORS["bg"])
    normal_scores = scores[:120].tolist() + scores[140:].tolist()
    anomaly_scores = scores[120:140].tolist()
    ax_hist.hist(normal_scores, bins=20, alpha=0.7, color=COLORS["purple"], label="正常")
    if anomaly_scores:
        ax_hist.hist(anomaly_scores, bins=10, alpha=0.7, color=COLORS["red"], label="异常")
    ax_hist.axvline(x=0.04, color="#888888", linestyle="--", linewidth=1)
    ax_hist.text(0.041, ax_hist.get_ylim()[1]*0.9, "阈值", fontsize=8, color="#888888")
    ax_hist.set_title("异常分数分布", fontsize=12, color=COLORS["text"])
    ax_hist.legend(fontsize=8)
    ax_hist.tick_params(labelsize=8)

    # ===== 对比实验结果表格 =====
    ax_comp = fig.add_subplot(gs[6:8, 2:12])
    ax_comp.set_facecolor(COLORS["bg"])
    ax_comp.axis("off")
    ax_comp.text(0.0, 0.95, "对比实验", fontsize=14, fontweight="bold",
                 color=COLORS["text"], transform=ax_comp.transAxes)

    if comp_results:
        methods = list(comp_results.keys())
        headers = ["方法", "AUROC", "AUPRC", "F1", "Precision", "Recall"]
        cell_text = []
        for m in methods:
            r = comp_results[m]
            cell_text.append([m, f"{r['auroc']:.4f}", f"{r['auprc']:.4f}",
                            f"{r['f1']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}"])
        table = ax_comp.table(cellText=cell_text, colLabels=headers,
                             loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        # 设置表格样式
        for i in range(len(headers)):
            table[0, i].set_facecolor(COLORS["accent"])
            table[0, i].set_text_props(color="white", fontweight="bold")
        # 高亮Diffusion行
        for j in range(len(headers)):
            table[1, j].set_facecolor(COLORS["light_blue"])

    # ===== 消融实验结果表格 =====
    ax_abl = fig.add_subplot(gs[8:10, 2:12])
    ax_abl.set_facecolor(COLORS["bg"])
    ax_abl.axis("off")
    ax_abl.text(0.0, 0.95, "消融实验", fontsize=14, fontweight="bold",
               color=COLORS["text"], transform=ax_abl.transAxes)

    if abl_results:
        labels_map = {
            "full": "Full (全部优化)",
            "no_aug": "w/o Data Augmentation",
            "no_cond": "w/o Conditional Diffusion",
            "no_multiscale": "w/o Multi-scale",
        }
        headers = ["设置", "AUROC", "AUPRC", "F1"]
        cell_text = []
        for key in ["full", "no_aug", "no_cond", "no_multiscale"]:
            if key in abl_results:
                r = abl_results[key]
                cell_text.append([labels_map.get(key, key),
                                f"{r['auroc']:.4f}", f"{r['auprc']:.4f}", f"{r['f1']:.4f}"])
        table = ax_abl.table(cellText=cell_text, colLabels=headers,
                            loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for i in range(len(headers)):
            table[0, i].set_facecolor(COLORS["purple"])
            table[0, i].set_text_props(color="white", fontweight="bold")
        # 高亮Full行
        if len(cell_text) > 0:
            for j in range(len(headers)):
                table[1, j].set_facecolor(COLORS["light_blue"])

    # 保存
    save_path = os.path.join(FIG_DIR, "streamlit_screenshot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"保存: {save_path}")
    return save_path


if __name__ == "__main__":
    generate_streamlit_mockup()
