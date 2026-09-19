"""论文图表画廊 —— 集中展示已生成的论文配图，支持预览、下载与一键重新生成。"""
import os
import sys
import subprocess
import streamlit as st
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
import streamlit_app as sa  # noqa: E402

FIG = os.path.join(PROJECT_ROOT, "experiments", "figures")

CAPTIONS = {
    "pipeline": ("流程架构图", "扩散模型训练 + 部分扩散重建 + 异常判定的整体流程（论文 3.x 节）"),
    "detection_example": ("检测示例图", "单条序列检测：原始序列 / 异常分数 / 阈值判定的可视化（论文 5.x 节）"),
    "comparison_bar": ("对比柱状图", "5 个模型 AUROC 对比（论文 5.5 节）"),
    "ablation_bar": ("消融柱状图", "完整模型 vs 各变体的指标对比（论文 5.6 节）"),
    "few_shot_curve": ("少样本曲线", "AUROC 随少样本比例变化（论文 5.7 节）"),
    "training_curve": ("训练曲线", "扩散模型训练 loss 收敛过程（论文 5.x 节）"),
    "streamlit_screenshot": ("平台截图", "Streamlit 平台界面截图（论文系统演示）"),
    "multi_dataset": ("多数据集柱状图", "8 个子集上的 AUROC 汇总（论文 5.x 节）"),
}


def _collect():
    out = []
    for root in [FIG, os.path.join(PROJECT_ROOT, "figures")]:
        if not os.path.isdir(root):
            continue
        for f in sorted(os.listdir(root)):
            if f.lower().endswith(".png"):
                stem = f[:-4]
                title, desc = CAPTIONS.get(stem, (stem, ""))
                out.append((os.path.join(root, f), title, desc))
    return out


def main():
    st.set_page_config(page_title="论文图表画廊", layout="wide", page_icon="🎨",
                       initial_sidebar_state="expanded")
    sa.inject_css()
    st.markdown('<div class="hero"><div class="hero-title">🎨 论文图表画廊</div>'
                '<div class="hero-sub">集中查看、下载论文所需配图，可一键重新生成</div></div>',
                unsafe_allow_html=True)

    figs = _collect()
    if not figs:
        st.warning("未找到任何配图，可点击下方按钮生成。")
    else:
        st.success(f"共找到 **{len(figs)}** 张配图。")

    if st.button("🔄 重新生成全部图表", type="primary"):
        with st.spinner("正在调用生成脚本（含训练曲线/检测示例，可能耗时数十秒）..."):
            try:
                env = dict(os.environ)
                env["PYTHONHOME"] = "D:\\"
                scripts = [
                    os.path.join(PROJECT_ROOT, "scripts", "generate_figures.py"),
                    os.path.join(PROJECT_ROOT, "scripts", "make_fewshot_fig.py"),
                    os.path.join(PROJECT_ROOT, "scripts", "make_pipeline_fig.py"),
                    os.path.join(PROJECT_ROOT, "scripts", "make_detection_fig.py"),
                ]
                for sc in scripts:
                    if os.path.exists(sc):
                        subprocess.run([os.path.join("D:/Scripts/python.exe"), sc],
                                       cwd=PROJECT_ROOT, env=env, check=False,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.success("✅ 图表已重新生成，向下滚动查看。")
                figs = _collect()
            except Exception as e:
                st.error(f"生成失败：{e}")

    cols = st.columns(2)
    for i, (path, title, desc) in enumerate(figs):
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            st.image(path, width='stretch')
            if desc:
                st.caption(desc)
            with open(path, "rb") as f:
                st.download_button("⬇ 下载", f.read(),
                                   file_name=os.path.basename(path),
                                   key=f"dl_{i}", width='stretch')
            st.markdown("---")


if __name__ == "__main__":
    main()
