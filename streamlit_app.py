"""Streamlit 时序异常检测可视化平台 — 浅色简洁版

运行方式:
  streamlit run app/streamlit_app.py
"""
import os
import sys
import io
import json
import time
import threading
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.data.ucr_loader import UCRAnomalyDataset
from src.data.augmentation import TimeSeriesAugmenter
from src.models.unet_1d import UNet1D
from src.models.diffusion import DiffusionModel
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.anogan import AnoGAN
from src.models.isolation_forest import IsolationForestDetector


# ═══════════════════════════════════════════════════════════
#  样式与展示工具
# ═══════════════════════════════════════════════════════════

def inject_css():
    """注入浅色主题样式（section-title / metric-card / status-bar / hero 等）。"""
    st.markdown("""
    <style>
    .stApp { background: #f8fafc; }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #1e293b;
        margin: 6px 0 2px; display: flex; align-items: center; gap: 8px;
    }
    .section-title .icon { font-size: 1.25rem; }
    .section-desc { font-size: 0.85rem; color: #64748b; margin-bottom: 10px; }
    .metric-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 6px 0; }
    .metric-card {
        flex: 1 1 120px; min-width: 110px; background: #fff; border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1; border-radius: 10px; padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card.tp { border-left-color: #10b981; }
    .metric-card.fp { border-left-color: #f59e0b; }
    .metric-card.fn { border-left-color: #ef4444; }
    .metric-card.tn { border-left-color: #3b82f6; }
    .metric-card.prec { border-left-color: #8b5cf6; }
    .metric-card.rec { border-left-color: #0ea5e9; }
    .metric-card.f1 { border-left-color: #ec4899; }
    .metric-label { font-size: 0.72rem; color: #94a3b8; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #1e293b; }
    .metric-desc { font-size: 0.7rem; color: #94a3b8; }
    .status-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 4px 0 8px; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #6366f1; display: inline-block; }
    .status-text { font-size: 0.85rem; color: #475569; }
    .tag { background: #eef2ff; color: #6366f1; border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; font-weight: 600; }
    .tag.cyan { background: #ecfeff; color: #0891b2; }
    .hero { padding: 8px 0 16px; }
    .hero-title { font-size: 1.5rem; font-weight: 800; color: #0f172a; }
    .hero-sub { font-size: 0.9rem; color: #64748b; margin-top: 4px; }
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🔍 基于扩散模型的时序异常检测平台</div>
      <div class="hero-sub">DDPM + 一维 U-Net 去噪主干 · 少样本时间序列异常检测 · UCR Anomaly Archive</div>
    </div>
    """, unsafe_allow_html=True)


def render_status_bar(parts):
    """parts: list of (text, style)。style ∈ { '', 'dot', 'tag', 'tag cyan' }。"""
    html = '<div class="status-bar">'
    for text, style in parts:
        if style == "dot":
            html += '<span class="status-dot"></span>'
        elif style == "tag":
            html += f'<span class="tag">{text}</span>'
        elif style == "tag cyan":
            html += f'<span class="tag cyan">{text}</span>'
        else:
            html += f'<span class="status-text">{text}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_metric_row(items):
    """items: list of (label, value, kind, desc)。kind 决定卡片强调色。"""
    cards = "".join(
        f'<div class="metric-card {kind}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-desc">{desc}</div>'
        f'</div>' for (label, value, kind, desc) in items
    )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


def apply_light_theme(fig, height=None):
    """给 plotly 图应用浅色主题。"""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(family="-apple-system, Segoe UI, Microsoft YaHei, sans-serif",
                  size=12, color="#334155"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ═══════════════════════════════════════════════════════════
#  配置与模型加载（带缓存）
# ═══════════════════════════════════════════════════════════

_CFG = None


def load_config_cached():
    global _CFG
    if _CFG is None:
        _CFG = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    return _CFG


MODEL_CACHE = {}


def _build_model(model_name, config, device):
    if model_name == "Diffusion":
        unet_cfg = config["unet"]
        unet = UNet1D(
            in_channels=unet_cfg.get("in_channels", 1),
            base_channels=unet_cfg.get("base_channels", 32),
            channel_mults=tuple(unet_cfg.get("channel_mults", [1, 2, 4, 4])),
            num_res_blocks=unet_cfg.get("num_res_blocks", 2),
            time_dim=unet_cfg.get("time_dim", 128),
            dropout=unet_cfg.get("dropout", 0.1),
            cond_dim=unet_cfg.get("cond_dim", 4),
        )
        diff_cfg = config["diffusion"]
        model = DiffusionModel(
            unet,
            n_timesteps=diff_cfg.get("n_timesteps", 200),
            beta_start=diff_cfg.get("beta_start", 1e-4),
            beta_end=diff_cfg.get("beta_end", 0.02),
            schedule=diff_cfg.get("schedule", "cosine"),
            device=device,
        )
        wpath = os.path.join(PROJECT_ROOT, "experiments", "results", "diffusion_main.pth")
        model.load(wpath)
        model.to(device)
        model.eval()
        return model
    elif model_name == "AE":
        ae_cfg = config["baselines"]["ae"]
        input_dim = config["data"]["window_size"]
        model = Autoencoder(input_dim, ae_cfg.get("hidden_dim", 64), ae_cfg.get("latent_dim", 16))
        wpath = os.path.join(PROJECT_ROOT, "experiments", "results", "ae.pth")
        model.load(wpath, device)
        model.eval()
        return model
    elif model_name == "VAE":
        vae_cfg = config["baselines"]["vae"]
        input_dim = config["data"]["window_size"]
        model = VAE(input_dim, vae_cfg.get("hidden_dim", 64), vae_cfg.get("latent_dim", 16))
        wpath = os.path.join(PROJECT_ROOT, "experiments", "results", "vae.pth")
        model.load(wpath, device)
        model.eval()
        return model
    elif model_name == "AnoGAN":
        anogan_cfg = config["baselines"]["anogan"]
        input_dim = config["data"]["window_size"]
        model = AnoGAN(input_dim, anogan_cfg.get("latent_dim", 16),
                       anogan_cfg.get("hidden_dim", 64), device)
        wpath = os.path.join(PROJECT_ROOT, "experiments", "results", "anogan.pth")
        model.load(wpath)
        model.to(device)
        model.eval()
        return model
    elif model_name == "Isolation Forest":
        model = IsolationForestDetector()
        wpath = os.path.join(PROJECT_ROOT, "experiments", "results", "iforest.pkl")
        model.load(wpath)
        return model
    else:
        raise ValueError(f"未知模型: {model_name}")


def load_model(model_name, config, device):
    """加载模型（带缓存，避免重复构建神经网络）。"""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    m = _build_model(model_name, config, device)
    MODEL_CACHE[model_name] = m
    return m


def compute_window_scores(model_name, tw_tensor, config, device,
                          use_multiscale, fast_mode, use_cond, noise_ratio=None):
    """计算逐窗口异常分数（重建误差均值）。供检测主循环与多模型对比复用。"""
    model = load_model(model_name, config, device)
    if model_name == "Isolation Forest":
        x_np = tw_tensor.detach().cpu().numpy().reshape(tw_tensor.shape[0], -1)
        return model.score(x_np)
    if noise_ratio is None:
        noise_ratio = 0.5
    total = len(tw_tensor)
    batch_size = 64
    cond_dim = config["unet"].get("cond_dim")
    use_cond_eff = bool(use_cond) and cond_dim is not None

    scores = []
    import contextlib
    _ctx = contextlib.nullcontext() if model_name == "AnoGAN" else torch.inference_mode()
    with _ctx:
        for i in range(0, total, batch_size):
            batch = tw_tensor[i:i + batch_size]
            if model_name == "Diffusion":
                cond = None
                if use_cond_eff:
                    cond = torch.cat([
                        batch.mean(dim=2), batch.std(dim=2),
                        batch.max(dim=2).values, batch.min(dim=2).values,
                    ], dim=1)
                recon = model.reconstruct(batch, cond=cond, use_multiscale=use_multiscale,
                                           noise_ratio=noise_ratio)
            else:
                if model_name == "AnoGAN":
                    recon = model.reconstruct(batch, n_steps=15)
                else:
                    recon = model.reconstruct(batch)
            if recon.shape != batch.shape:
                recon = recon.view_as(batch)
            diff = (recon - batch) ** 2
            scores.append(diff.mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores)


# ═══════════════════════════════════════════════════════════
#  上传文件解析
# ═══════════════════════════════════════════════════════════

def parse_uploaded_file(uploaded_file):
    """把上传文件解析为 DataFrame。支持 CSV / XLSX / TXT（单列数值）。"""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".txt"):
        raw = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
        vals = [float(x.strip()) for x in raw if x.strip()]
        return pd.DataFrame({"value": vals})
    else:
        raise ValueError("不支持的文件格式，请上传 CSV / TXT / XLSX")


def build_upload_windows(df, series, label_col, window_size):
    """从上传 DataFrame 构建滑动窗口与（可选）标签。"""
    data = df[series].dropna().values.astype(float)
    labels_full = None
    if label_col is not None and label_col in df.columns:
        labels_full = df[label_col].fillna(0).values.astype(float)
    n = len(data)
    if n < window_size:
        raise ValueError(f"数据点数量({n}) 少于窗口大小({window_size})")
    windows = []
    labels = []
    stride = max(1, window_size // 2)
    for start in range(0, n - window_size + 1, stride):
        windows.append(data[start:start + window_size])
        if labels_full is not None:
            labels.append(1.0 if labels_full[start:start + window_size].sum() > 0 else 0.0)
    windows = np.array(windows)
    labels = np.array(labels) if labels_full is not None else None
    return windows, labels


# ═══════════════════════════════════════════════════════════
#  主页面：异常检测（核心功能，含控制面板 + 检测主体）
# ═══════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="时序异常检测平台", layout="wide",
                       page_icon="📊", initial_sidebar_state="expanded")
    inject_css()
    render_hero()

    config = load_config_cached()
    # 交互状态：检测结果缓存（切到其它页面再返回不丢结果）
    if "det" not in st.session_state:
        st.session_state.det = None
    if "run_requested" not in st.session_state:
        st.session_state.run_requested = False
    device = "cpu"

    # ===== 侧边栏 =====
    with st.sidebar:
        st.markdown("## ⚙ 控制面板")

        st.markdown("### 📊 数据源")
        source_mode = st.radio(
            "数据来源",
            ["UCR 内置数据集", "📤 上传数据"],
            horizontal=True,
            index=0,
        )
        dataset_idx = None  # UCR 分支内赋值；上传数据分支保持 None

        window_size = st.slider("窗口大小", 50, 300, config["data"]["window_size"], step=10)

        upload_data = None
        if source_mode == "📤 上传数据":
            st.markdown("---")
            st.markdown("#### 📁 上传你的时序数据")
            uploaded_file = st.file_uploader(
                "支持 CSV / TXT / XLSX",
                type=["csv", "txt", "xlsx", "xls"],
                help="CSV/XLSX：每行一个时间点；TXT：每行一个数值（单列）。可包含 0/1 标签列。",
            )
            if uploaded_file is not None:
                try:
                    _df = parse_uploaded_file(uploaded_file)
                    _numeric_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
                    if not _numeric_cols:
                        st.error("❌ 文件中未找到数值列，无法解析为时间序列。")
                    else:
                        up_series = st.selectbox("选择时序列", _numeric_cols, key="up_series")
                        up_has_label = st.checkbox(
                            "数据包含异常标签列 (0/1)？", value=False, key="up_label_flag"
                        )
                        up_label_col = None
                        if up_has_label:
                            up_label_col = st.selectbox(
                                "选择标签列", _df.columns.tolist(), key="up_label_col"
                            )
                        windows_u, labels_u = build_upload_windows(
                            _df, up_series, up_label_col, window_size
                        )
                        upload_data = (windows_u, labels_u, uploaded_file.name)
                        if labels_u is None:
                            st.success(
                                f"✅ 已加载 `{uploaded_file.name}`：{len(windows_u)} 个窗口"
                                f"（无标签，将自动判定异常）"
                            )
                        else:
                            n_anom = int(labels_u.sum())
                            st.success(
                                f"✅ 已加载 `{uploaded_file.name}`：{len(windows_u)} 个窗口，"
                                f"其中异常窗口 {n_anom} 个"
                            )
                except Exception as e:
                    st.error(f"❌ 解析失败：{e}")
            st.caption("📌 上传数据使用 UCR 预训练模型直接检测，结果作为演示参考。")
        else:
            available_datasets = UCRAnomalyDataset.get_available_datasets(config["data"]["data_dir"])
            _preset_idx = st.session_state.get("xpreset", {}).get("dataset_idx", 0)
            if not isinstance(_preset_idx, int) or _preset_idx >= len(available_datasets):
                _preset_idx = 0
            dataset_idx = st.selectbox(
                "选择 UCR 数据集",
                range(len(available_datasets)),
                index=_preset_idx,
                format_func=lambda i: available_datasets[i][:40],
            )
            if "xpreset" in st.session_state:
                st.session_state.pop("xpreset", None)
            few_shot_ratio = st.slider("少样本比例", 0.05, 1.0, 1.0, step=0.05)

        st.markdown("### 🤖 检测引擎")
        model_name = st.selectbox(
            "检测算法",
            ["Diffusion", "AE", "VAE", "AnoGAN", "Isolation Forest"],
        )
        use_multiscale = st.checkbox("多尺度重建", value=False)
        speed_mode = st.radio("检测档位",
                              ["标准(质量优先)", "快速(推荐)", "极速"],
                              index=1,
                              help="标准：去噪步数最多、精度最高、最慢；"
                                   "快速：默认档，平衡速度与质量；"
                                   "极速：去噪步数最少，检测压到秒级，精度略降")
        # 噪声比例映射：极速档进一步减少去噪步数（noise_ratio 越小步数越少）
        _NR_MAP = {"标准(质量优先)": 0.5, "快速(推荐)": 0.15, "极速": 0.08}
        noise_ratio = _NR_MAP[speed_mode]
        fast_mode = speed_mode != "标准(质量优先)"   # 兼容底层 fast_mode 语义
        max_windows = st.slider("检测窗口数", 20, 500, 50, step=10)

        st.markdown("---")
        st.caption("💡 调节参数后点击下方按钮启动检测")

        run_btn = st.button("🚀 启动检测", type="primary")
        if run_btn:
            st.session_state.run_requested = True
            st.rerun()

    config["data"]["window_size"] = window_size
    config["experiment"]["use_multiscale"] = use_multiscale
    if source_mode == "UCR 内置数据集":
        config["data"]["dataset_idx"] = dataset_idx
        config["data"]["few_shot_ratio"] = few_shot_ratio

    # ===== 异常检测主视图 =====
    if st.session_state.run_requested:
        st.session_state.run_requested = False
        # 准备数据（UCR 或 上传，统一输出 windows / labels / ds_name）
        if source_mode == "📤 上传数据":
            if upload_data is None:
                st.warning("⚠️ 请先在左侧「数据源」中上传并解析数据文件，再启动检测。")
                st.stop()
            windows, labels, ds_name = upload_data
            has_labels = labels is not None
        else:
            with st.spinner("正在加载数据集..."):
                augment_fn = TimeSeriesAugmenter() if config["experiment"]["use_augmentation"] else None
                train_set = UCRAnomalyDataset(
                    config["data"]["data_dir"], dataset_idx, window_size, 1, "train",
                    few_shot_ratio=few_shot_ratio, augment_fn=augment_fn
                )
                test_set = UCRAnomalyDataset(
                    config["data"]["data_dir"], dataset_idx, window_size, 1, "test",
                    few_shot_ratio=1.0, augment_fn=None
                )
                test_set.set_scaler(train_set.scaler)
            windows = test_set.windows
            labels = test_set.labels
            ds_name = available_datasets[dataset_idx][:30]
            has_labels = True

        # 友好报错：模型权重缺失时明确提示，而不是加载空模型输出无意义结果
        _model_files = {
            "Diffusion": "diffusion_main.pth",
            "AE": "ae.pth",
            "VAE": "vae.pth",
            "AnoGAN": "anogan.pth",
            "Isolation Forest": "iforest.pkl",
        }
        _mf = os.path.join(PROJECT_ROOT, "experiments", "results", _model_files[model_name])
        if not os.path.exists(_mf):
            st.error(
                f"❌ 未找到 **{model_name}** 的模型权重文件：\n\n"
                f"`{_mf}`\n\n"
                f"请先运行以下脚本生成权重，再启动检测：\n"
                f"```bash\n"
                f"python scripts/run_experiment.py        # 生成扩散模型与基线权重\n"
                f"python scripts/evaluate_only.py         # 或仅加载已有权重重新评估\n"
                f"```"
            )
            st.stop()
        with st.spinner(f"正在加载 {model_name} 模型..."):
            model = load_model(model_name, config, device)

        # 检测计算 —— 均匀采样以保证覆盖整个测试集（含异常段）
        n_total = len(windows)
        if max_windows >= n_total:
            sel_idx = np.arange(n_total)
            n_eval = n_total
        else:
            # 在 [0, n_total-1] 范围内均匀采 max_windows 个点
            sel_idx = np.linspace(0, n_total - 1, max_windows).astype(int)
            # 去重（保留首次出现的顺序）
            sel_idx = np.unique(sel_idx)
            n_eval = len(sel_idx)
        test_windows = torch.from_numpy(windows[sel_idx]).float().unsqueeze(1).to(device)
        test_labels = labels[sel_idx] if has_labels else None
        # 保留采样后的 numpy 数组供后续可视化使用
        sampled_windows = windows[sel_idx]

        st.markdown("""
        <div class="section-title"><span class="icon">⚡</span> 检测进行中</div>
        """, unsafe_allow_html=True)

        if model_name == "Isolation Forest":
            x_np = windows[sel_idx]
            scores = model.score(x_np)
        else:
            cond_dim = config["unet"].get("cond_dim")
            use_cond = config["experiment"].get("use_conditional", False) and cond_dim is not None
            # noise_ratio 由上层 UI 按「标准/快速/极速」档位统一计算后传入
            total = len(test_windows)
            batch_size = 64
            torch.set_num_threads(max(1, os.cpu_count() or 4))

            # 后台线程跑检测，on_step 每步回报真实进度；主线程缓动显示 → 进度条连续滑动
            _state = {"frac": 0.0, "done": False}
            _scores = []
            _batch_i = {"i": 0}

            def _on_step(local_frac):
                _state["frac"] = min(1.0, (_batch_i["i"] + local_frac * batch_size) / total)

            def _detect():
                import contextlib
                # AnoGAN 的 reconstruct 内部要 loss.backward() 搜索潜变量，需开启 autograd；
                # 其余模型禁用梯度以省显存。用上下文变量统一处理。
                _ctx = contextlib.nullcontext() if model_name == "AnoGAN" else torch.inference_mode()
                with _ctx:
                    for i in range(0, total, batch_size):
                        _batch_i["i"] = i
                        batch = test_windows[i:i + batch_size]
                        if model_name == "Diffusion":
                            cond = None
                            if use_cond:
                                cond = torch.cat([
                                    batch.mean(dim=2), batch.std(dim=2),
                                    batch.max(dim=2).values, batch.min(dim=2).values,
                                ], dim=1)
                            recon = model.reconstruct(batch, cond=cond, use_multiscale=use_multiscale,
                                                       noise_ratio=noise_ratio, on_step=_on_step)
                        else:
                            if model_name == "AnoGAN":
                                # 平台检测用 15 步潜变量搜索（论文评估脚本仍用默认 30 步，结果不变）
                                recon = model.reconstruct(batch, n_steps=15)
                            else:
                                recon = model.reconstruct(batch)
                        if recon.shape != batch.shape:
                            recon = recon.view_as(batch)
                        diff = (recon - batch) ** 2
                        _scores.append(diff.mean(dim=(1, 2)).cpu().numpy())
                        _state["frac"] = min(1.0, (i + batch_size) / total)
                _state["frac"] = 1.0
                _state["done"] = True

            _th = threading.Thread(target=_detect, daemon=True)
            _th.start()

            _bar = st.progress(0, text="检测准备中...")
            _disp = 0.0
            while True:
                _target = _state["frac"]
                _disp += (_target - _disp) * 0.18   # 缓动插值：进度条连续滑动而非跳变
                _pct = int(_disp * 100)
                _done_n = int(_disp * total)
                _bar.progress(_pct, text=f"检测进度 {_pct}%  ({_done_n}/{total} 窗口)")
                if _state["done"] and _disp >= 0.999:
                    break
                time.sleep(0.07)
            _th.join()
            scores = np.concatenate(_scores)
            _bar.progress(100, text="检测完成 ✓")
            _bar.empty()

        # 缓存所需的占位量（_render_detection 内部会按标签重算阈值/预测/异常区间）
        threshold = float(np.percentile(scores, 95))
        pred = (scores > threshold).astype(int)
        spans = []

        st.session_state.det = dict(
            scores=scores, sel_idx=sel_idx, pred=pred, test_labels=test_labels,
            ds_name=ds_name, model_name=model_name, has_labels=has_labels,
            sampled_windows=sampled_windows, threshold=threshold, n_eval=n_eval,
            spans=spans, window_size=window_size, config=config, device=device,
            use_multiscale=use_multiscale, fast_mode=fast_mode, use_cond=use_cond,
            noise_ratio=noise_ratio, test_windows=test_windows, source_mode=source_mode,
            dataset_idx=(dataset_idx if source_mode == "UCR 内置数据集" else None))
        _render_detection(st.session_state.det)
    elif st.session_state.det is not None:
        _render_detection(st.session_state.det)
    else:
        st.info("👈 请在左侧配置参数，点击「🚀 启动检测」开始分析")

    # ===== 完整实验结果 =====
    st.markdown("""
    <div class="section-title"><span class="icon">🏆</span> 完整实验结果</div>
    <div class="section-desc">以下为离线运行的完整实验数据（非实时检测）</div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        results_path = os.path.join(PROJECT_ROOT, "experiments", "results", "comparison_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                comp_results = json.load(f)
            st.markdown("""
            <div class="section-title"><span class="icon">⚖</span> 对比实验</div>
            """, unsafe_allow_html=True)
            df = pd.DataFrame(comp_results).T[["auroc", "auprc", "f1", "precision", "recall"]]
            df.columns = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]

            styled = df.style.format("{:.4f}")
            best_auroc = df["AUROC"].idxmax()
            styled = styled.apply(
                lambda row: ["background: #eef2ff; color: #6366f1; font-weight:700"
                            if row.name == best_auroc else "" for _ in row],
                axis=1
            )
            st.dataframe(styled, width='stretch', height=200)
        else:
            st.info("对比实验结果未生成，请先运行 `python scripts/run_experiment.py`")

    with col_b:
        abl_path = os.path.join(PROJECT_ROOT, "experiments", "results", "ablation_results.json")
        if os.path.exists(abl_path):
            with open(abl_path) as f:
                abl_results = json.load(f)
            st.markdown("""
            <div class="section-title"><span class="icon">🔬</span> 消融实验</div>
            """, unsafe_allow_html=True)
            labels_map = {
                "full": "Full (完整模型)",
                "no_aug": "w/o 数据增强",
                "no_cond": "w/o 条件扩散",
                "no_multiscale": "w/o 多尺度",
            }
            abl_df = pd.DataFrame({
                labels_map.get(k, k): v for k, v in abl_results.items()
            }).T[["auroc", "auprc", "f1"]]
            abl_df.columns = ["AUROC", "AUPRC", "F1"]

            styled_abl = abl_df.style.format("{:.4f}")
            best_abl = abl_df["AUROC"].idxmax()
            styled_abl = styled_abl.apply(
                lambda row: ["background: #ecfeff; color: #0891b2; font-weight:700"
                            if row.name == best_abl else "" for _ in row],
                axis=1
            )
            st.dataframe(styled_abl, width='stretch', height=200)
        else:
            st.info("消融实验结果未生成，请先运行 `python scripts/run_experiment.py --ablation`")

    # 页脚
    st.markdown("""
    <div style="text-align:center; padding: 30px 0 10px; color: #94a3b8; font-size: 0.75rem;">
        基于扩散模型的少样本时间序列异常检测 · DDPM + 1D U-Net · UCR Anomaly Archive
    </div>
    """)


# ═══════════════════════════════════════════════════════════
#  检测结果渲染（供异常检测主页复用，支持切页面后缓存）
# ═══════════════════════════════════════════════════════════

def _render_detection(ctx):
    """根据缓存的检测结果 dict 渲染检测结果视图（支持切 tab 复用）。"""
    scores = ctx['scores']
    sel_idx = ctx['sel_idx']
    pred = ctx['pred']
    test_labels = ctx['test_labels']
    ds_name = ctx['ds_name']
    model_name = ctx['model_name']
    has_labels = ctx['has_labels']
    sampled_windows = ctx['sampled_windows']
    threshold = ctx['threshold']
    n_eval = ctx['n_eval']
    spans = ctx['spans']
    window_size = ctx['window_size']
    config = ctx['config']
    device = ctx['device']
    use_multiscale = ctx['use_multiscale']
    fast_mode = ctx['fast_mode']
    use_cond = ctx['use_cond']
    noise_ratio = ctx['noise_ratio']
    test_windows = ctx['test_windows']
    source_mode = ctx['source_mode']

    # ===== 结果展示 =====
    desc_text = ("以下指标基于最优 F1 阈值自动搜索得到"
                 if has_labels else
                 "未提供标签：基于 95 百分位阈值自动判定异常（演示参考）")
    st.markdown(f"""
    <div class="section-title"><span class="icon">📊</span> 检测结果</div>
    <div class="section-desc">{desc_text}</div>
    """, unsafe_allow_html=True)

    # 状态条
    status_parts = [
        ("", "dot"),
        (f"数据源: {ds_name}", ""),
        (model_name, "tag"),
        (f"窗口: {window_size}", "tag cyan"),
        (f"多尺度: {'ON' if use_multiscale else 'OFF'}", "tag"),
        (f"检测: {n_eval} 个窗口", ""),
    ]
    if source_mode == "📤 上传数据":
        status_parts.append(("UCR 预训练模型", "tag cyan"))
    render_status_bar(status_parts)

    # 阈值确定（有标签→搜索最优 F1 阈值；无标签→95 百分位）
    if has_labels:
        from sklearn.metrics import f1_score as _f1
        best_f1, best_thr = 0.0, np.percentile(scores, 95)
        for p in range(50, 100):
            thr = np.percentile(scores, p)
            pred_t = (scores > thr).astype(int)
            if pred_t.sum() == 0 or pred_t.sum() == len(pred_t):
                continue
            f1_t = _f1(test_labels, pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thr = thr
        threshold = best_thr
    else:
        threshold = np.percentile(scores, 95)
    pred = (scores > threshold).astype(int)

    if has_labels:
        tp = int((pred * test_labels).sum())
        fp = int((pred * (1 - test_labels)).sum())
        fn = int(((1 - pred) * test_labels).sum())
        tn = int(((1 - pred) * (1 - test_labels)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 混淆矩阵 4 卡
        render_metric_row([
            ("True Positive", tp, "tp", "正确检出异常"),
            ("False Positive", fp, "fp", "误报"),
            ("False Negative", fn, "fn", "漏检"),
            ("True Negative", tn, "tn", "正确判为正常"),
        ])

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # 评估指标 3 卡
        render_metric_row([
            ("Precision", f"{precision:.4f}", "prec", "精确率"),
            ("Recall", f"{recall:.4f}", "rec", "召回率"),
            ("F1 Score", f"{f1:.4f}", "f1", f"最优阈值={threshold:.4f}"),
        ])
    else:
        detected = int(pred.sum())
        st.info(
            f"📎 未提供标签，跳过 P/R/F1 计算。基于 95 百分位阈值（={threshold:.4f}）"
            f"自动判定，共检出 **{detected}** 个异常窗口。"
        )
        render_metric_row([
            ("检出异常", detected, "fp", "无标签自动判定"),
            ("异常阈值", f"{threshold:.4f}", "f1", "95 百分位"),
            ("总窗口", f"{n_eval}", "tn", "检测样本数"),
        ])

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ===== 可视化图表 =====
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-title"><span class="icon">📈</span> 时序波形与异常分数</div>
        """, unsafe_allow_html=True)
        n_show = min(50, len(scores))
        labels_show = test_labels[:n_show] if has_labels else None

        # 提取连续异常窗口区间，用于红色阴影高亮
        def _anomaly_spans(lbl):
            spans = []
            in_a, s = False, 0
            for i, v in enumerate(lbl):
                if v == 1 and not in_a:
                    in_a, s = True, i
                elif v == 0 and in_a:
                    in_a = False
                    spans.append((s, i))
            if in_a:
                spans.append((s, len(lbl)))
            return spans

        spans = _anomaly_spans(labels_show) if labels_show is not None else []

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4],
                            vertical_spacing=0.08,
                            subplot_titles=("原始时序窗口（红色阴影=真实异常区间）",
                                            "异常分数（橙色虚线=阈值）"))

        # 原始时序
        fig.add_trace(
            go.Scatter(
                y=sampled_windows[:n_show, -1],
                name="时序值",
                line=dict(color="#6366f1", width=1.5),
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)",
            ),
            row=1, col=1
        )
        # 真实异常点
        if labels_show is not None:
            anomaly_idx = np.where(labels_show == 1)[0]
            if len(anomaly_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_idx,
                        y=sampled_windows[anomaly_idx, -1],
                        name="真实异常点",
                        mode="markers",
                        marker=dict(color="#ef4444", size=7, symbol="x",
                                    line=dict(width=1, color="#f87171")),
                    ),
                    row=1, col=1
                )
        # 异常分数折线 + 阈值线
        fig.add_trace(
            go.Scatter(
                y=scores[:n_show],
                name="异常分数",
                mode="lines",
                line=dict(color="#6366f1", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.08)",
            ),
            row=2, col=1
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
                      line_width=2, row=2, col=1,
                      annotation_text=f"阈值={threshold:.3f}",
                      annotation_position="top right",
                      annotation_font=dict(color="#f59e0b", size=10))
        # 异常区间阴影（上下两行均高亮）
        for (s, e) in spans:
            for r in (1, 2):
                fig.add_vrect(x0=s - 0.5, x1=e - 0.5,
                              fillcolor="rgba(239,68,68,0.12)", line_width=0,
                              row=r, col=1)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.14))
        apply_light_theme(fig, height=480)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("""
        <div class="section-title"><span class="icon">🎯</span> 异常分数分布</div>
        """, unsafe_allow_html=True)
        hist_fig = go.Figure()
        if has_labels:
            normal_scores = scores[test_labels == 0]
            anomaly_scores = scores[test_labels == 1]
            hist_fig.add_trace(go.Histogram(
                x=normal_scores, name="正常",
                opacity=0.75,
                marker=dict(color="#818cf8", line=dict(width=0)),
                nbinsx=30,
            ))
            if len(anomaly_scores) > 0:
                hist_fig.add_trace(go.Histogram(
                    x=anomaly_scores, name="异常",
                    opacity=0.75,
                    marker=dict(color="#ef4444", line=dict(width=0)),
                    nbinsx=30,
                ))
        else:
            hist_fig.add_trace(go.Histogram(
                x=scores, name="全部分数",
                opacity=0.75,
                marker=dict(color="#6366f1", line=dict(width=0)),
                nbinsx=30,
            ))
        hist_fig.add_vline(
            x=threshold, line_dash="dash",
            line_color="#f59e0b", line_width=2,
            annotation_text=f"阈值={threshold:.4f}",
            annotation_font=dict(color="#f59e0b", size=10),
        )
        hist_fig.update_layout(barmode="overlay")
        apply_light_theme(hist_fig, height=300)
        st.plotly_chart(hist_fig, width='stretch')

        # AUROC 概览
        st.markdown("""
        <div class="section-title"><span class="icon">🌐</span> 本次检测概览</div>
        """, unsafe_allow_html=True)
        if has_labels:
            from sklearn.metrics import roc_auc_score
            try:
                auroc = roc_auc_score(test_labels, scores)
            except Exception:
                auroc = 0.5
            auroc_disp = f"{auroc:.4f}"
            auroc_desc = "ROC 曲线下面积"
        else:
            auroc_disp = "N/A"
            auroc_desc = "无标签不可计算"
        render_metric_row([
            ("AUROC", auroc_disp, "prec", auroc_desc),
            ("阈值", f"{threshold:.4f}", "f1", "95 百分位阈值" if not has_labels else "最优 F1 阈值"),
            ("总窗口", f"{n_eval}", "tn", "检测样本数"),
        ])

        # 异常分布统计（不依赖是否有标签，始终展示）
        _peak = float(np.max(scores)) if len(scores) else 0.0
        _n_pred = int(pred.sum())
        _ratio = (_n_pred / len(scores)) if len(scores) else 0.0
        render_metric_row([
            ("峰值分数", f"{_peak:.4f}", "tp", "异常分数最大值"),
            ("异常窗口数", f"{_n_pred}", "fp", "超过阈值的窗口数"),
            ("异常占比", f"{_ratio * 100:.1f}%", "fn", "异常窗口 / 总窗口"),
        ])

    st.markdown("---")

    # ===== 导出检测结果 =====
    st.markdown("""
    <div class="section-title"><span class="icon">💾</span> 导出检测结果</div>
    """, unsafe_allow_html=True)
    _export = pd.DataFrame({
        "window_index": sel_idx,
        "anomaly_score": np.round(scores, 6),
        "prediction": pred,
        "true_label": (test_labels.astype(int) if has_labels else ["未标注"] * len(scores)),
    })
    _csv_bytes = _export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 下载检测结果为 CSV",
        data=_csv_bytes,
        file_name=f"detection_{ds_name}_{model_name}.csv",
        mime="text/csv",
    )
    st.caption("CSV 包含：窗口索引、异常分数、预测判定、真实标签（若有）。")

    # ===== 多模型异常分数对比 =====
    with st.expander("📊 多模型异常分数对比（同图叠加）", expanded=False):
        _others = [m for m in ["Diffusion", "AE", "VAE", "AnoGAN", "Isolation Forest"]
                   if m != model_name]
        cmp_models = st.multiselect("选择要叠加对比的模型", _others,
                                    default=_others, key="cmp_models")
        if st.button("🔬 计算并对比", key="cmp_btn"):
            if not cmp_models:
                st.warning("请至少选择一个对比模型。")
            else:
                cmp_scores = {}
                for mname in cmp_models:
                    with st.spinner(f"正在计算 {mname} ..."):
                        try:
                            cmp_scores[mname] = compute_window_scores(
                                mname, test_windows, config, device,
                                use_multiscale, fast_mode, use_cond,
                                noise_ratio=noise_ratio)
                        except Exception as e:
                            st.error(f"{mname} 计算失败：{e}")
                if cmp_scores:
                    cfig = make_subplots(rows=1, cols=1)
                    cfig.add_trace(go.Scatter(y=scores[:n_show], name=model_name,
                                             mode="lines",
                                             line=dict(color="#6366f1", width=2.2)))
                    for mname, cs in cmp_scores.items():
                        cfig.add_trace(go.Scatter(y=cs[:n_show], name=mname,
                                                 mode="lines",
                                                 line=dict(width=1.4, dash="dot")))
                    cfig.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
                                   annotation_text=f"当前模型阈值={threshold:.3f}",
                                   annotation_font=dict(color="#f59e0b", size=10))
                    for (s, e) in spans:
                        cfig.add_vrect(x0=s - 0.5, x1=e - 0.5,
                                       fillcolor="rgba(239,68,68,0.10)", line_width=0)
                    cfig.update_layout(showlegend=True,
                                       legend=dict(orientation="h", y=1.12),
                                       xaxis_title="窗口索引")
                    apply_light_theme(cfig, height=420)
                    st.plotly_chart(cfig, width='stretch')
                    st.caption("各模型分数量纲不同，曲线仅用于直观对比异常峰值位置；"
                               "橙色虚线为当前模型阈值，供参考。")

                    # 导出多模型对比 CSV
                    _cmp_df = pd.DataFrame({
                        "window_index": sel_idx,
                        model_name: np.round(scores, 6),
                    })
                    for mname, cs in cmp_scores.items():
                        _cmp_df[mname] = np.round(cs, 6)
                    if has_labels:
                        _cmp_df["true_label"] = test_labels.astype(int)
                    _cmp_csv = _cmp_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ 下载多模型对比 CSV",
                        data=_cmp_csv,
                        file_name=f"compare_{ds_name}.csv",
                        mime="text/csv",
                    )
                    st.caption("CSV 含各模型逐窗口异常分数及真实标签（若有），"
                               "可直接用于论文附录或后续分析。")

    # ===== 下钻到异常可解释性 =====
    if ctx.get("dataset_idx") is not None:
        st.markdown("---")
        st.markdown("""
        <div class="section-title"><span class="icon">🔦</span> 异常窗口可解释性（下钻）</div>
        <div class="section-desc">把检测到的异常窗口送入「异常可解释性」页，逐点查看重建误差与异常来源</div>
        """, unsafe_allow_html=True)
        if has_labels and test_labels is not None and int(test_labels.sum()) > 0:
            _widx = int(np.argmax(test_labels))
        else:
            _widx = int(np.argmax(scores))
        if st.button("🔦 下钻分析异常窗口", key="drill_btn", type="primary"):
            st.session_state["xpreset"] = {
                "dataset_idx": ctx["dataset_idx"],
                "window_idx": _widx,
            }
            st.switch_page(os.path.join(PROJECT_ROOT, "app", "pages", "07_异常可解释性.py"))


if __name__ == "__main__":
    main()
