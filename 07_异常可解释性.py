"""异常可解释性 —— 对选定窗口做重建，可视化逐点误差并高亮驱动异常分数的区段。"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
import streamlit_app as sa  # noqa: E402

RECON_MODELS = ["Diffusion", "AE", "VAE", "AnoGAN"]  # 支持重建误差的模型


def _plot(inp, rec, err, start, end):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=inp, mode="lines", name="原始输入",
                             line=dict(color="#6366f1", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(y=rec, mode="lines", name="模型重建",
                             line=dict(color="#10b981", width=1.4, dash="dot")), row=1, col=1)
    if end > start:
        fig.add_vrect(x0=start, x1=end, fillcolor="#ef4444", opacity=0.12,
                      line_width=0, row=1, col=1, annotation_text="标注异常",
                      annotation_position="top left")
    fig.add_trace(go.Scatter(y=err, mode="lines", name="逐点误差",
                             line=dict(color="#f59e0b", width=1.2),
                             fill="tozeroy", fillcolor="rgba(245,158,11,0.18)"), row=2, col=1)
    if end > start:
        fig.add_vrect(x0=start, x1=end, fillcolor="#ef4444", opacity=0.12,
                      line_width=0, row=2, col=1)
    fig.update_layout(template="plotly_white", height=460,
                      margin=dict(l=40, r=20, t=20, b=30),
                      font=dict(family="-apple-system, Segoe UI, Microsoft YaHei, sans-serif",
                                size=12, color="#334155"),
                      paper_bgcolor="white", plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text="时间点", row=2, col=1)
    return fig


def _top_region(err, thresh=2.0):
    """返回误差最显著区段 (start, end)。"""
    if len(err) == 0:
        return 0, 0
    mean, std = err.mean(), err.std() + 1e-9
    mask = err > mean + thresh * std
    if not mask.any():
        k = int(np.argmax(err))
        half = max(1, len(err) // 10)
        return max(0, k - half), min(len(err), k + half)
    idx = np.where(mask)[0]
    return int(idx.min()), int(idx.max()) + 1


def main():
    st.set_page_config(page_title="异常可解释性", layout="wide", page_icon="🔦",
                       initial_sidebar_state="expanded")
    sa.inject_css()
    st.markdown('<div class="hero"><div class="hero-title">🔦 异常可解释性</div>'
                '<div class="hero-sub">从「分数高」到「为什么高」：逐点重建误差揭示异常来源</div></div>',
                unsafe_allow_html=True)

    config = sa.load_config_cached()
    ddir = config["data"]["data_dir"]
    if not os.path.isabs(ddir):
        ddir = os.path.join(PROJECT_ROOT, ddir)
    from src.data.ucr_loader import UCRAnomalyDataset
    datasets = UCRAnomalyDataset.get_available_datasets(ddir)

    # 接收主页下钻预设
    preset = st.session_state.get("xpreset", {})
    def_idx = preset.get("dataset_idx", 0) if isinstance(preset, dict) else 0
    if not isinstance(def_idx, int) or def_idx >= len(datasets):
        def_idx = 0

    with st.sidebar:
        st.markdown("## 🎛 分析设置")
        ds_idx = st.selectbox("数据集", range(len(datasets)),
                              index=def_idx,
                              format_func=lambda i: f"{i:02d} · {datasets[i][:42]}")
        model_name = st.selectbox("检测算法", RECON_MODELS, index=0)
        ms = st.checkbox("多尺度重建", value=False)
        nr = st.slider("噪声比例", 0.05, 0.5, 0.15, step=0.01)
        st.caption(f"ℹ️ 窗口大小固定为模型训练尺寸 **{config['data']['window_size']}**"
                   f"（基线模型尺寸锁死，扩散模型支持变长，为可靠对比统一用训练尺寸）。")
        run = st.button("🔍 分析该窗口", type="primary")

    if not run:
        st.info("👈 在左侧选择数据集与窗口索引，点击「分析该窗口」查看重建误差与异常来源。")
        if preset:
            st.caption("（已接收来自「异常检测」主页的下钻预设）")
        return

    with st.spinner("加载数据集与模型（首次约 10~30 秒）..."):
        sys.path.insert(0, APP_DIR)
        import _shared as sh
        windows, labels, ds_name, scaler = sh.load_test_dataset(ds_idx, config["data"]["window_size"], config)
        n = len(windows)
        win_idx = st.session_state.get("xpreset", {}).get("window_idx", None)
        if not isinstance(win_idx, int) or win_idx >= n:
            win_idx = int(np.argmax(labels)) if labels.sum() > 0 else 0
        win_idx = st.slider("选择窗口索引", 0, n - 1, win_idx)
        w = windows[win_idx]
        inp, rec, err = sh.reconstruct_window(model_name, w, scaler, config,
                                              "cpu", ms, nr)

    st.markdown(f"**数据集** `{ds_name}` · 窗口 #{win_idx} · 算法 **{model_name}**")
    is_anom = bool(labels[win_idx]) if labels is not None else False
    st.markdown(f"窗口标签：{'🔴 异常窗口' if is_anom else '🟢 正常窗口'}")

    s, e = (0, len(w))  # 窗口内坐标（w 为该窗口实际长度，等于训练窗口大小）
    # 标注异常在窗口内的位置（标签按窗口内是否有异常点）
    fig = _plot(inp, rec, err, s, e if is_anom else 0)
    st.plotly_chart(fig, width='stretch')

    rs, re = _top_region(err, thresh=2.0)
    peak = float(err.max())
    st.markdown('<div class="section-title"><span class="icon">💡</span> 解读</div>',
                unsafe_allow_html=True)
    st.success(
        f"该窗口内第 **{rs}~{re}** 点重建误差最高（峰值误差 {peak:.4f}），"
        f"是异常分数的主要来源。模型在「见过的正常模式」上重建误差小，"
        f"在偏离正常模式的区段上重建误差陡增——这正是扩散模型以「重建误差」判定异常的白盒解释。"
    )

    with st.expander("📐 数值明细"):
        detail = pd.DataFrame({"点序": np.arange(len(err)),
                               "原始值": inp, "重建值": rec, "逐点误差": err})
        st.dataframe(detail, width='stretch', height=240)

    st.session_state.pop("xpreset", None)


if __name__ == "__main__":
    main()
