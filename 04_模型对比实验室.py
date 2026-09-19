"""模型对比实验室 —— 默认展示预计算的 5 模型对比结果；支持选定数据集实时复现。"""
import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
import streamlit_app as sa  # noqa: E402

RES = os.path.join(PROJECT_ROOT, "experiments", "results")
DESC = {
    "Diffusion (Ours)": "扩散模型（本文方法）：部分扩散重建 + 多尺度 + 条件向量",
    "AE": "自编码器：学习正常模式，重建误差作异常分数",
    "VAE": "变分自编码器：在 AE 基础上引入潜变量分布约束",
    "AnoGAN": "对抗生成网络：通过潜变量搜索逼近异常样本",
    "IF": "孤立森林：基于样本孤立难易度无监督打分",
}


def _plot_bars(results, metric, color):
    models = list(results.keys())
    ys = [results[m].get(metric) for m in models]
    fig = go.Figure(go.Bar(x=models, y=ys,
                           marker_color=[color if "Ours" in m else "#cbd5e1" for m in models]))
    best = max(ys)
    fig.update_layout(template="plotly_white", height=340,
                      margin=dict(l=50, r=20, t=30, b=60),
                      font=dict(family="-apple-system, Segoe UI, Microsoft YaHei, sans-serif",
                                size=12, color="#334155"),
                      paper_bgcolor="white", plot_bgcolor="white",
                      yaxis_title=metric, xaxis_tickangle=-15)
    return fig


def main():
    st.set_page_config(page_title="模型对比实验室", layout="wide", page_icon="⚖",
                       initial_sidebar_state="expanded")
    sa.inject_css()
    st.markdown('<div class="hero"><div class="hero-title">⚖ 模型对比实验室</div>'
                '<div class="hero-sub">扩散模型 vs 4 个基线：AUROC / AUPRC / F1 全面对比</div></div>',
                unsafe_allow_html=True)

    fp = os.path.join(RES, "comparison_results.json")
    if not os.path.exists(fp):
        st.error("未找到 `comparison_results.json`，请先运行 `python scripts/run_experiment.py`。")
        return
    with open(fp) as f:
        results = json.load(f)

    # ── 预计算总表 + 柱状图 ──
    st.markdown('<div class="section-title"><span class="icon">🏆</span> 离线对比结果（全量训练集）</div>'
                '<div class="section-desc">以下为已保存的完整实验结果</div>', unsafe_allow_html=True)
    df = pd.DataFrame(results).T[["auroc", "auprc", "f1", "precision", "recall"]]
    df.columns = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
    st.dataframe(df.style.format("{:.4f}"), width='stretch')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(_plot_bars(results, "auroc", "#6366f1"), width='stretch')
    with c2:
        st.plotly_chart(_plot_bars(results, "auprc", "#0ea5e9"), width='stretch')
    with c3:
        st.plotly_chart(_plot_bars(results, "f1", "#ec4899"), width='stretch')

    st.markdown('<div class="section-title"><span class="icon">🧠</span> 各模型说明</div>',
                unsafe_allow_html=True)
    for m, d in DESC.items():
        st.markdown(f"- **{m}**：{d}")

    # ── 实时复现 ──
    st.markdown('<div class="section-title"><span class="icon">🧪</span> 实时复现实验</div>'
                '<div class="section-desc">选定数据集，现场加载 5 个模型并计算 AUROC（会加载模型权重，首次较慢）</div>',
                unsafe_allow_html=True)
    config = sa.load_config_cached()
    ddir = config["data"]["data_dir"]
    if not os.path.isabs(ddir):
        ddir = os.path.join(PROJECT_ROOT, ddir)
    from src.data.ucr_loader import UCRAnomalyDataset
    datasets = UCRAnomalyDataset.get_available_datasets(ddir)
    ds_idx = st.selectbox("数据集", range(len(datasets)),
                          format_func=lambda i: f"{i:02d} · {datasets[i][:46]}")
    ms = st.checkbox("多尺度重建", value=False)
    nr = st.slider("噪声比例 (去噪步数)", 0.05, 0.5, 0.15, step=0.01,
                   help="越小去噪步数越少、越快；越大越精确（仅影响扩散模型）")
    st.caption(f"ℹ️ 模型按训练窗口大小 **{config['data']['window_size']}** 计算"
               f"（基线模型尺寸固定，扩散模型支持变长，为统一对比此处固定为该尺寸）。")
    if st.button("🚀 实时计算 AUROC", type="primary"):
        with st.spinner("加载模型并计算（首次约 10~30 秒）..."):
            sys.path.insert(0, APP_DIR)
            import _shared as sh
            live, ds_name = sh.evaluate_models(ds_idx, config["data"]["window_size"],
                                               ms, nr, config, max_windows=300)
        st.success(f"✅ 数据集 `{ds_name}` 实时评估完成")
        live_df = pd.DataFrame([(k, (f"{v:.4f}" if v is not None else "权重缺失"))
                                for k, v in live.items()],
                               columns=["模型", "AUROC(实时)"])
        st.dataframe(live_df, width='stretch')
        valid = {k: v for k, v in live.items() if v is not None}
        if valid:
            fig = go.Figure(go.Bar(
                x=list(valid.keys()), y=list(valid.values()),
                marker_color=["#6366f1" if "Ours" in k else "#cbd5e1" for k in valid]))
            fig.update_layout(template="plotly_white", height=340,
                              margin=dict(l=0, r=0, t=30, b=60),
                              font=dict(family="-apple-system, Segoe UI, Microsoft YaHei, sans-serif",
                                        size=12, color="#334155"),
                              paper_bgcolor="white", plot_bgcolor="white",
                              yaxis_title="AUROC", xaxis_tickangle=-15)
            st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()
