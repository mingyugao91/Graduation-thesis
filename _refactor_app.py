# -*- coding: utf-8 -*-
"""精确重构 streamlit_app.py：把 main 改为 st.tabs 三视图，并注入
render_tutorial / render_about 两个说明页函数。基于行标记处理，避免手工缩进出错。"""
import io

PATH = r"C:\Users\21020\OneDrive\桌面\thesis\app\streamlit_app.py"

with open(PATH, encoding="utf-8") as f:
    lines = f.read().split("\n")

new_lines = []
i = 0
n = len(lines)
inserted_session = False
tabs_head_emitted = False

while i < n:
    line = lines[i]
    stripped = line.strip()

    # 1) session_state 初始化（在 config = load_config_cached() 之后）
    if stripped == "config = load_config_cached()" and not inserted_session:
        new_lines.append(line)
        new_lines.append('    # 交互状态：多视图导航 + 检测结果缓存（切 tab 不丢结果）')
        new_lines.append('    if "det" not in st.session_state:')
        new_lines.append('        st.session_state.det = None')
        new_lines.append('    if "run_requested" not in st.session_state:')
        new_lines.append('        st.session_state.run_requested = False')
        inserted_session = True
        i += 1
        continue

    # 2) run_btn 不再就地执行，改为设置标志并重跑，由 tab0 接管检测
    if stripped.startswith('run_btn = st.button("🚀 启动检测"'):
        new_lines.append(line)
        new_lines.append('        if run_btn:')
        new_lines.append('            st.session_state.run_requested = True')
        new_lines.append('            st.rerun()')
        i += 1
        continue

    # 3) 检测执行块开头 → 重构为 tabs 结构
    if stripped == "# ===== 检测执行 =====" and not tabs_head_emitted:
        new_lines.append('    # ===== 顶层多视图导航 =====')
        new_lines.append('    _tabs = st.tabs(["🔍 异常检测", "📚 使用教程", "ℹ️ 关于版本"])')
        new_lines.append('')
        new_lines.append('    with _tabs[0]:  # 异常检测')
        new_lines.append('        if st.session_state.run_requested:')
        new_lines.append('            st.session_state.run_requested = False')
        i += 2  # 跳过 "# ===== 检测执行 =====" 与紧随的 "    if run_btn:"
        # 检测+结果展示块：加 4 缩进（进入 with tabs[0] + if run_requested）
        while i < n and lines[i].strip() != "# ===== 实验结果对比 =====":
            new_lines.append("    " + lines[i])
            i += 1
        # 现在 lines[i] 是 "    # ===== 实验结果对比 ====="，它在 tab0 内、if 外，加 4 缩进
        new_lines.append("    " + lines[i])
        i += 1
        # 实验结果对比块：加 4 缩进，直到页脚前（保持在 tab0 内、if 外）
        while i < n and not lines[i].strip().startswith("# 页脚"):
            new_lines.append("    " + lines[i])
            i += 1
        # 关闭 tab0，追加 tab1 / tab2
        new_lines.append('')
        new_lines.append('    with _tabs[1]:  # 使用教程')
        new_lines.append('        render_tutorial()')
        new_lines.append('')
        new_lines.append('    with _tabs[2]:  # 关于版本')
        new_lines.append('        render_about()')
        new_lines.append('')
        tabs_head_emitted = True
        continue

    new_lines.append(line)
    i += 1

# 在 if __name__ 之前插入两个说明页函数
src = "\n".join(new_lines)
funcs = '''

# ═══════════════════════════════════════════════════════════
#  说明页：使用教程 / 关于与版本
# ═══════════════════════════════════════════════════════════

def render_tutorial():
    """使用教程页：快速上手 + 参数速查 + 结果解读 + FAQ。"""
    st.markdown("""
    <div class="section-title"><span class="icon">📚</span> 使用教程</div>
    <div class="section-desc">四步即可完成一次异常检测；下方可展开查看参数说明、结果解读与常见问题。</div>
    """, unsafe_allow_html=True)

    # 快速上手
    with st.container():
        st.markdown("### 🚀 快速上手（4 步）")
        steps = [
            ("① 选择数据", "左侧「数据源」选 **UCR 内置数据集**（默认 000）或切到「上传数据」拖入你的 CSV/TXT/XLSX。"),
            ("② 调参数", "按需设置窗口大小、检测算法、多尺度开关与检测档位（推荐「快速」）。"),
            ("③ 启动", "点击左侧「🚀 启动检测」，进度条跑完即出结果。"),
            ("④ 看结果 / 导出", "查看指标卡、时序与分数图、异常分布；点「下载检测结果为 CSV」留存，或用「多模型对比」叠加其他算法。"),
        ]
        for title, desc in steps:
            st.markdown(f"- **{title}**：{desc}")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # 数据源说明
    with st.expander("📁 数据源与上传格式", expanded=False):
        st.markdown("""
        - **UCR 内置数据集**：直接复用论文预训练权重（000 训练），覆盖 8 个 SYNTHETIC 子集，含真实异常标签。
        - **上传数据**：
          - CSV / XLSX：每行一个时间点；可含 0/1 标签列（上传后在左侧勾选「包含异常标签列」并指定列）。
          - TXT：每行一个数值（单列）。
          - 上传数据使用 UCR 预训练模型直接检测，结果作演示参考；若含标签则自动计算 P/R/F1 与 AUROC。
        """)

    # 参数速查
    with st.expander("🎛 参数速查", expanded=False):
        st.markdown("""
        | 参数 | 说明 | 建议 |
        |---|---|---|
        | 窗口大小 | 每个检测窗口的时间步数 | 50–300，默认取自配置 |
        | 少样本比例 | 仅 UCR 模式生效，控制训练用样本占比 | 1.0=全量；论文重点验证低比例 |
        | 检测算法 | Diffusion / AE / VAE / AnoGAN / Isolation Forest | 默认 Diffusion |
        | 多尺度重建 | 多尺度特征融合，提升小异常检出 | 默认关；必要时开 |
        | 检测档位 | 标准(质量优先)/快速(推荐)/极速 | 极速去噪步数最少、秒级出结果、精度略降 |
        | 检测窗口数 | 单次检测覆盖的窗口数量 | 20–500，越大覆盖越全越慢 |
        """)

    # 结果解读
    with st.expander("🔍 如何解读结果", expanded=False):
        st.markdown("""
        - **指标卡**：有标签时给出 TP/FP/FN/TN、Precision/Recall/F1；无标签时给出检出数、阈值、总窗口。
        - **时序波形与异常分数图**：上行原始时序、红色阴影=真实异常区间（若有标签），下行异常分数、橙色虚线=阈值。
        - **异常分数分布图**：正常/异常分数叠加直方图，直观看阈值区分度。
        - **导出 CSV**：含窗口索引、异常分数、预测、真实标签，可用于论文附录或二次分析。
        - **多模型对比**：同图叠加其他算法分数曲线，便于横向比较异常峰值位置。
        """)

    # FAQ
    with st.expander("❓ 常见问题 FAQ", expanded=False):
        st.markdown("""
        - **提示「未找到模型权重」？** 先运行 `python scripts/run_experiment.py` 生成权重，再启动检测（见「关于版本」页的权重自检）。
        - **上传后无数值列？** 确保文件以数值为主，CSV/XLSX 的时序列应为数字。
        - **极速档精度略低？** 去噪步数减少所致，属预期；追求精度请选「标准」或「快速」。
        - **无标签时怎么判定异常？** 采用 95 百分位阈值自动判定，指标仅作参考。
        - **想看完整离线实验？** 在「异常检测」页底部已内置对比实验 / 消融实验结果表。
        """)


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
    st.code(
        "@article{diffusion_ts_ad_2026,\n"
        "  title={基于扩散模型的少样本时间序列异常检测方法研究},\n"
        "  author={XXX},\n"
        "  journal={南京审计大学 学年论文},\n"
        "  year={2026}\n"
        "}",
        language="bibtex",
    )
    st.caption("引用格式示意，作者/年份请按实际填写。")


'''

marker = 'if __name__ == "__main__":'
idx = src.find(marker)
if idx != -1:
    out = src[:idx] + funcs + "\n" + src[idx:]
else:
    out = src + funcs

with open(PATH, "w", encoding="utf-8") as f:
    f.write(out)

print("重构完成：main 已改为 st.tabs 三视图，并注入 render_tutorial / render_about。")
