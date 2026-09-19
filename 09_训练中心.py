"""训练中心 —— UI 配置训练参数，启动 run_experiment.py，实时显示 loss 曲线与日志，可停止。"""
import os
import sys
import json
import time
import threading
import subprocess
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
import streamlit_app as sa  # noqa: E402

PY = os.path.join("D:/Scripts/python.exe")
EXP_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_experiment.py")
SAVE_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
TMP_CFG = os.path.join(SAVE_DIR, "_train_config_tmp.yaml")


def _build_config(opts):
    import yaml
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    cfg.setdefault("experiment", {})
    cfg["data"]["dataset_idx"] = opts["dataset_idx"]
    cfg["data"]["window_size"] = opts["window_size"]
    cfg["data"]["few_shot_ratio"] = opts["few_shot_ratio"]
    cfg["train"]["epochs"] = opts["epochs"]
    cfg["train"]["batch_size"] = opts["batch_size"]
    cfg["experiment"]["use_augmentation"] = opts["use_aug"]
    cfg["experiment"]["use_conditional"] = opts["use_cond"]
    cfg["experiment"]["use_multiscale"] = opts["use_ms"]
    if "seed" in cfg.get("train", {}):
        cfg["train"]["seed"] = opts["seed"]
    with open(TMP_CFG, "w") as f:
        yaml.safe_dump(cfg, f)
    return TMP_CFG


def _reader(proc, key_lines, key_loss):
    """后台线程：读取 stdout，留存日志并解析 loss。"""
    import re
    pat = re.compile(r"loss[\" :=]*([0-9]+\.[0-9]+)", re.IGNORECASE)
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            line = line.rstrip("\n")
            st.session_state[key_lines].append(line)
            m = pat.search(line)
            if m:
                try:
                    st.session_state[key_loss].append(float(m.group(1)))
                except ValueError:
                    pass
    except Exception:
        pass


def main():
    st.set_page_config(page_title="训练中心", layout="wide", page_icon="🛠",
                       initial_sidebar_state="expanded")
    sa.inject_css()
    st.markdown('<div class="hero"><div class="hero-title">🛠 训练中心</div>'
                '<div class="hero-sub">可视化配置训练参数，启动训练并实时观察 loss 收敛（也可仅用命令行）</div></div>',
                unsafe_allow_html=True)

    config = sa.load_config_cached()
    ddir = config["data"]["data_dir"]
    if not os.path.isabs(ddir):
        ddir = os.path.join(PROJECT_ROOT, ddir)
    from src.data.ucr_loader import UCRAnomalyDataset
    datasets = UCRAnomalyDataset.get_available_datasets(ddir)

    # 初始化 session 状态
    for k in ["train_proc", "train_thread", "train_log", "train_loss", "training", "train_done"]:
        if k not in st.session_state:
            st.session_state[k] = [] if k in ("train_log", "train_loss") else None
    if st.session_state["train_proc"] is not None and st.session_state["training"] is not False:
        # 兼容历史
        pass

    with st.sidebar:
        st.markdown("## ⚙ 训练参数")
        exp_type = st.radio("实验类型", ["对比实验（扩散+基线）", "消融实验"],
                            help="消融实验会逐项去掉优化策略，耗时约 4 倍")
        ds_idx = st.selectbox("数据集", range(len(datasets)),
                              format_func=lambda i: f"{i:02d} · {datasets[i][:42]}")
        epochs = st.number_input("训练轮数 (epochs)", 1, 500, int(config["train"].get("epochs", 50)))
        batch_size = st.number_input("批大小 (batch_size)", 8, 256, int(config["train"].get("batch_size", 32)))
        ws = st.slider("窗口大小", 50, 300, config["data"]["window_size"], step=10)
        fs = st.slider("少样本比例", 0.05, 1.0, 1.0, step=0.05)
        use_aug = st.checkbox("数据增强", value=bool(config["experiment"].get("use_augmentation", True)))
        use_cond = st.checkbox("条件扩散", value=True)
        use_ms = st.checkbox("多尺度重建", value=bool(config["experiment"].get("use_multiscale", True)))
        seed = st.number_input("随机种子", 0, 9999, int(config["train"].get("seed", 42)))
        auto = st.checkbox("自动刷新日志", value=True,
                           help="训练期间每 2 秒刷新一次页面以更新曲线")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 启动训练", type="primary", disabled=st.session_state["training"] is True):
            opts = dict(dataset_idx=ds_idx, window_size=ws, few_shot_ratio=fs,
                        epochs=epochs, batch_size=batch_size, use_aug=use_aug,
                        use_cond=use_cond, use_ms=use_ms, seed=seed)
            cfg_file = _build_config(opts)
            cmd = [PY, EXP_SCRIPT, "--config", cfg_file, "--save_dir", SAVE_DIR]
            if exp_type.startswith("消融"):
                cmd.insert(3, "--ablation")
            env = dict(os.environ)
            env["PYTHONHOME"] = "D:\\"
            st.session_state["train_log"] = []
            st.session_state["train_loss"] = []
            st.session_state["training"] = True
            st.session_state["train_done"] = False
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, encoding="utf-8", errors="replace")
            st.session_state["train_proc"] = proc
            t = threading.Thread(target=_reader, args=(proc, "train_log", "train_loss"),
                                 daemon=True)
            t.start()
            st.session_state["train_thread"] = t
            st.rerun()
    with col2:
        if st.button("⏹ 停止训练", disabled=not st.session_state["training"]):
            proc = st.session_state["train_proc"]
            if proc is not None and proc.poll() is None:
                proc.terminate()
            st.session_state["training"] = False
            st.session_state["train_done"] = True
            st.success("已停止训练。")

    # 实时展示
    if st.session_state["training"]:
        st.info("🔄 训练进行中…（日志与 loss 曲线实时更新）")
        losses = st.session_state["train_loss"]
        if losses:
            fig = go.Figure(go.Scatter(y=losses, mode="lines", name="loss",
                                       line=dict(color="#6366f1", width=1.6)))
            fig.update_layout(template="plotly_white", height=300,
                              margin=dict(l=40, r=20, t=20, b=30),
                              font=dict(family="-apple-system, Segoe UI, Microsoft YaHei, sans-serif",
                                        size=12, color="#334155"),
                              paper_bgcolor="white", plot_bgcolor="white",
                              xaxis_title="step", yaxis_title="loss")
            st.plotly_chart(fig, use_container_width=True)
        log = "\n".join(st.session_state["train_log"][-200:])
        st.code(log if log else "（等待输出…）", language="bash", height=300)

        proc = st.session_state["train_proc"]
        if proc is not None and proc.poll() is not None:
            st.session_state["training"] = False
            st.session_state["train_done"] = True
            st.success("✅ 训练进程已结束，结果已写入 `experiments/results/`。")
        elif auto:
            time.sleep(2)
            st.rerun()
    else:
        if st.session_state["train_done"]:
            st.success("上次训练已结束/停止。可重新配置并再次启动。")
        else:
            st.info("👈 配置左侧参数后点击「启动训练」。训练会写到 `experiments/results/`，"
                    "完成后「关于版本」页的权重自检将全部变绿。")


if __name__ == "__main__":
    main()
