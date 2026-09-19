"""消融实验 —— 逐项去掉优化策略，量化数据增强 / 条件扩散 / 多尺度重建的贡献。"""
import os
import json
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
import sys
sys.path.insert(0, APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
import streamlit_app as sa  # noqa: E402

RES = os.path.join(PROJECT_ROOT, "experiments", "results")
FIG = os.path.join(PROJECT_ROOT, "experiments", "figures")

LABELS = {
    "full": "完整模型",
    "no_aug": "去掉·数据增强",
    "no_cond": "去掉·条件扩散",
    "no_multiscale": "去掉·多尺度",
}
METRICS = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1")]


def main():
    st.set_page_config(page_title="消融实验", layout="wide", page_icon="🔬",
                       initial_sidebar_state="expanded")
    sa.inject_css()
    st.markdown('<div class="hero"><div class="hero-title">🔬 消融实验</div>'
                '<div class="hero-sub">固定其余设置，逐项剔除优化策略，定位每个模块对性能的贡献</div></div>',
                unsafe_allow_html=True)

    fp = os.path.join(RES, "ablation_results.json")
    if not os.path.exists(fp):
        st.error("未找到 `ablation_results.json`，请先运行 `python scripts/run_experiment.py --ablation`。")
        return
    with open(fp) as f:
        data = json.load(f)

    st.markdown('<div class="section-title"><span class="icon">📊</span> 分组柱状图</div>'
                '<div class="section-desc">每个设置下 AUROC / AUPRC / F1 的对比，完整模型应为最高</div>',
                unsafe_allow_html=True)

    settings = [k for k in ["full", "no_aug", "no_cond", "no_multiscale"] if k in data]
    fig = go.Figure()
    for key, name in METRICS:
        ys = [data[s].get(key) for s in settings]
        fig.add_trace(go.Bar(name=name, x=[LABELS.get(s, s) for s in settings], y=ys))
    fig.update_layout(barmode="group", template="plotly_white", height=400,
                      margin=dict(l=50, r=20, t=30, b=40),
                      font=dict(family="-apple-system, Segoe UI, Microsoft YaHei, sans-serif",
                                size=12, color="#334155"),
                      paper_bgcolor="white", plot_bgcolor="white",
                      yaxis_title="指标值", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width='stretch')

    # 贡献表：完整 - 变体
    st.markdown('<div class="section-title"><span class="icon">➖</span> 各优化贡献（完整模型 − 变体）</div>',
                unsafe_allow_html=True)
    full = data.get("full", {})
    rows = []
    for s in settings:
        if s == "full":
            continue
        d = data[s]
        rows.append(dict(
            剔除模块=LABELS.get(s, s),
            AUROC=f"{d.get('auroc',0):.4f}",
            AUPRC=f"{d.get('auprc',0):.4f}",
            F1=f"{d.get('f1',0):.4f}",
            ΔAUROC=f"{full.get('auroc',0)-d.get('auroc',0):+.4f}",
        ))
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), width='stretch', height=160)

    # 数值总表
    st.markdown('<div class="section-title"><span class="icon">📋</span> 原始数值</div>',
                unsafe_allow_html=True)
    df = pd.DataFrame({LABELS.get(s, s): data[s] for s in settings}).T[METRICS[0][0], METRICS[1][0], METRICS[2][0]]
    df.columns = [n for _, n in METRICS]
    st.dataframe(df.style.format("{:.4f}"), width='stretch')

    shot = os.path.join(FIG, "ablation_bar.png")
    if os.path.exists(shot):
        st.markdown('<div class="section-title"><span class="icon">🖼</span> 论文章节配图</div>',
                    unsafe_allow_html=True)
        st.image(shot, caption="消融柱状图（论文 5.x 节可直接引用）", width='stretch')


if __name__ == "__main__":
    main()
