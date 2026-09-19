# -*- coding: utf-8 -*-
"""关于与版本 —— 独立的说明页面（Streamlit multipage 结构）。

与「异常检测」主页完全独立：左侧导航切换即可打开，
显示应用版本、运行环境依赖版本、模型权重自检状态与论文信息。
"""
import os
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
st.set_page_config(page_title="关于版本", page_icon="ℹ️", layout="wide")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT + "/app")
import streamlit_app  # noqa: E402  复用主应用的样式

streamlit_app.inject_css()


def render_about():
    """关于与版本页：版本号、依赖、权重自检、论文信息。"""
    import sys as _sys
    import torch as _torch
    import streamlit as _st
    import numpy as _np
    import pandas as _pd
    import plotly as _plotly
    import sklearn as _sklearn
    import matplotlib as _mpl

    APP_VERSION = "1.0.0"
    st.markdown("""
    <div class="section-title"><span class="icon">ℹ️</span> 关于与版本</div>
    <div class="section-desc">系统版本、运行环境与模型权重状态自检。</div>
    """, unsafe_allow_html=True)

    # 版本信息卡
    ver_rows = [
        ("应用版本", APP_VERSION),
        ("Python", _sys.version.split()[0]),
        ("PyTorch", _torch.__version__),
        ("Streamlit", _st.__version__),
        ("NumPy", _np.__version__),
        ("Pandas", _pd.__version__),
        ("Plotly", _plotly.__version__),
        ("scikit-learn", _sklearn.__version__),
        ("Matplotlib", _mpl.__version__),
    ]
    st.markdown("### 🧩 运行环境")
    for k, v in ver_rows:
        st.markdown(f"- **{k}**：`{v}`")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # 模型权重自检
    st.markdown("### 🔧 模型权重自检")
    st.caption("检测前请确认以下权重文件已生成（缺失会阻断对应算法的检测）。")
    _weights = [
        ("Diffusion", "diffusion_main.pth"),
        ("AE", "ae.pth"),
        ("VAE", "vae.pth"),
        ("AnoGAN", "anogan.pth"),
        ("Isolation Forest", "iforest.pkl"),
    ]
    _res_root = os.path.join(PROJECT_ROOT, "experiments", "results")
    for name, fname in _weights:
        fpath = os.path.join(_res_root, fname)
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath) / (1024 * 1024)
            st.success(f"✅ **{name}**：`{fname}`（{sz:.2f} MB）")
        else:
            st.error(f"❌ **{name}**：`{fname}` 缺失 —— 运行 `python scripts/run_experiment.py` 生成")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # 论文信息
    st.markdown("### 📄 论文信息")
    st.markdown("""
    - **题目**：基于扩散模型的少样本时间序列异常检测方法研究
    - **方法**：DDPM 扩散模型 + 一维 U-Net 去噪主干 + 多尺度重建误差检测
    - **数据集**：UCR Anomaly Archive（SYNTHETIC 子集）
    - **基线**：孤立森林 / AE / VAE / AnoGAN
    - **单位**：南京审计大学 计算机科学与技术
    """)
    st.code("""@article{diffusion_ts_ad_2026,
  title={基于扩散模型的少样本时间序列异常检测方法研究},
  author={XXX},
  journal={南京审计大学 学年论文},
  year={2026}
}""", language="bibtex")
    st.caption("引用格式示意，作者/年份请按实际填写。")


render_about()
