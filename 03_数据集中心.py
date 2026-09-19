"""数据集中心 —— 列出 UCR 内置子集、解析异常区间、预览波形，并一键送入检测。"""
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
import streamlit_app as sa  # noqa: E402  (仅用其 inject_css / apply_light_theme，不触发模型)


RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def _list_datasets():
    if os.path.isdir(RAW_DIR):
        files = sorted([f[:-4] for f in os.listdir(RAW_DIR) if f.endswith(".txt")])
        if files:
            return files
    return sa.UCRAnomalyDataset.FALLBACK_NAMES


def _load_meta(name):
    fp = os.path.join(RAW_DIR, name + ".txt")
    raw = np.loadtxt(fp)
    ts = raw[:-100].astype(np.float32)
    n = len(ts)
    parts = name.split("_")
    try:
        start = int(parts[-3])
        end = int(parts[-2])
    except (ValueError, IndexError):
        start, end = n - 100, n
    if start < 0:
        start = 0
    if end > n:
        end = n
    anom_len = max(0, end - start)
    ratio = anom_len / n if n else 0.0
    return ts, start, end, anom_len, ratio


def _plot_waveform(name, ts, start, end):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ts, mode="lines", name="时序",
                             line=dict(color="#6366f1", width=1.4)))
    if start < end:
        fig.add_vrect(x0=start, x1=end, fillcolor="#ef4444", opacity=0.14,
                      line_width=0, annotation_text="异常区间",
                      annotation_position="top left")
    fig.update_layout(template="plotly_white", height=360,
                      margin=dict(l=40, r=20, t=30, b=30),
                      font=dict(family="-apple-system, Segoe UI, Microsoft YaHei, sans-serif",
                                size=12, color="#334155"),
                      paper_bgcolor="white", plot_bgcolor="white",
                      xaxis_title="时间点", yaxis_title="数值")
    return fig


def main():
    st.set_page_config(page_title="数据集中心", layout="wide", page_icon="📚",
                       initial_sidebar_state="expanded")
    sa.inject_css()
    st.markdown('<div class="hero"><div class="hero-title">📚 数据集中心</div>'
                '<div class="hero-sub">UCR Anomaly Archive · 内置子集一览、异常区间解析与波形预览</div></div>',
                unsafe_allow_html=True)

    datasets = _list_datasets()
    if not datasets:
        st.warning("未找到数据集，请先运行 `python scripts/download_ucr.py` 下载 UCR 数据。")
        return

    rows = []
    for i, name in enumerate(datasets):
        try:
            ts, s, e, al, ratio = _load_meta(name)
            rows.append(dict(编号=i, 数据集=name, 序列长度=len(ts),
                             异常起点=s, 异常终点=e, 异常点数=al,
                             异常占比=f"{ratio*100:.1f}%"))
        except Exception as ex:
            rows.append(dict(编号=i, 数据集=name, 序列长度="-",
                             异常起点="-", 异常终点="-", 异常点数="-",
                             异常占比=f"解析失败:{ex}"))
    df = pd.DataFrame(rows)

    st.markdown('<div class="section-title"><span class="icon">📋</span> 内置数据集总览</div>'
                '<div class="section-desc">文件名末尾编码了异常起止（_START_END_LENGTH），下表自动解析</div>',
                unsafe_allow_html=True)
    st.dataframe(df, width='stretch', height=320)

    st.markdown('<div class="section-title"><span class="icon">🌊</span> 波形预览</div>'
                '<div class="section-desc">选择数据集查看完整序列，红色区域为标注异常区间</div>',
                unsafe_allow_html=True)
    sel = st.selectbox("选择数据集", range(len(datasets)),
                       format_func=lambda i: f"{i:02d} · {datasets[i][:46]}")
    name = datasets[sel]
    try:
        ts, s, e, al, ratio = _load_meta(name)
        st.plotly_chart(_plot_waveform(name, ts, s, e), width='stretch')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("序列长度", len(ts))
        c2.metric("异常起点", s)
        c3.metric("异常终点", e)
        c4.metric("异常占比", f"{ratio*100:.1f}%")
        if st.button("🚀 送入异常检测", type="primary"):
            st.session_state["xpreset"] = {"dataset_idx": sel}
            st.switch_page(os.path.join(APP_DIR, "streamlit_app.py"))
    except Exception as ex:
        st.error(f"加载失败：{ex}")


if __name__ == "__main__":
    main()
