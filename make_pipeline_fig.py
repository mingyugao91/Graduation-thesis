# -*- coding: utf-8 -*-
"""论文方法流程图（4.1 系统总体架构）：扩散模型异常检测技术路线。

纯 matplotlib 绘制，不加载模型，不参与评测计算。
输出: paper/figures/pipeline.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_FONT = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PROJECT_ROOT, "paper", "figures", "pipeline.png")


def box(ax, x, y, w, h, text, fc, tc="white", fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.2, edgecolor="#2c3e50", facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold", wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, color="#34495e"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.6, color=color)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center",
                va="bottom", fontsize=7.5, color=color)


def main():
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    C_IN = "#1f4e79"      # 输入类
    C_DIFF = "#8e44ad"    # 扩散/加噪
    C_NET = "#16a085"     # 网络
    C_ERR = "#c0392b"     # 误差/异常
    C_OUT = "#e67e22"     # 输出

    # ---- 训练分支（顶部）----
    box(ax, 0.4, 4.7, 1.8, 0.9, "正常时序\n窗口", C_IN, fs=8.5)
    box(ax, 5.6, 4.7, 2.6, 0.9, "一维 U-Net 去噪模型\n(部分扩散 · 仅正常样本训练)", C_NET, fs=8.5)
    arrow(ax, 2.2, 5.15, 5.6, 5.15, label="训练")

    # 训练→推理 共享权重
    arrow(ax, 6.9, 4.7, 6.9, 3.75, label="共享权重", color="#16a085")
    ax.text(7.35, 4.22, "共享权重", fontsize=7.5, color="#16a085")

    # ---- 推理分支（底部，左→右）----
    y = 2.2
    h = 1.05
    box(ax, 0.4, y, 1.7, h, "测试\n时间序列", C_IN, fs=8.5)
    box(ax, 2.3, y, 1.7, h, "滑动窗口\n切片", C_IN, fs=8.5)
    box(ax, 4.2, y, 1.8, h, "部分扩散\n加噪(ρ)", C_DIFF, fs=8.5)
    box(ax, 6.3, y, 2.3, h, "U-Net 去噪\n(条件向量)", C_NET, fs=8.5)
    box(ax, 8.9, y, 1.6, h, "重建\n序列", C_NET, fs=8.5)
    box(ax, 10.7, y, 1.7, h, "重建误差\n(MSE)", C_ERR, fs=8.5)
    box(ax, 12.6, y, 1.5, h, "阈值\n判定", C_OUT, fs=8.5)

    arrow(ax, 2.1, y + h / 2, 2.3, y + h / 2)
    arrow(ax, 4.0, y + h / 2, 4.2, y + h / 2)
    arrow(ax, 6.0, y + h / 2, 6.3, y + h / 2)
    arrow(ax, 8.6, y + h / 2, 8.9, y + h / 2)
    arrow(ax, 10.5, y + h / 2, 10.7, y + h / 2)
    arrow(ax, 12.4, y + h / 2, 12.6, y + h / 2)

    # 末端输出
    box(ax, 12.6, 0.5, 1.5, 0.95, "异常 / 正常", C_OUT, fs=8.5)
    arrow(ax, 13.35, y, 13.35, 1.45, color=C_OUT)

    # 条件向量注解
    box(ax, 6.3, 0.5, 2.3, 0.8, "条件: 均值/方差/最值(多尺度)", "#7f8c8d", fs=7.5)
    arrow(ax, 7.45, 2.2, 7.45, 1.3, label="多尺度重建", color="#7f8c8d")

    ax.text(7.0, 5.95, "图：基于扩散模型的时间序列异常检测总体架构",
            fontsize=12, weight="bold", ha="center", color="#1a1a1a")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已生成方法流程图:", FIG)


if __name__ == "__main__":
    main()
