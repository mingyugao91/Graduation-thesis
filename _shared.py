"""共享评估工具（供重页面复用，已验证的 streamlit_app 接口）。

注意：本模块导入 streamlit_app（含 torch），因此只应由需要模型推理的页面
（模型对比实验室的实时模式、异常可解释性）导入；纯展示页（少样本/消融/画廊/
数据集中心）不要导入本模块，以保持首屏轻量。
"""
import os
import sys
import contextlib
import numpy as np
import torch

from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import streamlit_app as sa  # noqa: E402

MODEL_KEYS = ["Diffusion", "AE", "VAE", "AnoGAN", "Isolation Forest"]
_WEIGHT_FILES = {
    "Diffusion": "diffusion_main.pth",
    "AE": "ae.pth",
    "VAE": "vae.pth",
    "AnoGAN": "anogan.pth",
    "Isolation Forest": "iforest.pkl",
}


def get_config():
    return sa.load_config_cached()


def resolve_data_dir(config):
    d = config["data"]["data_dir"]
    if not os.path.isabs(d):
        d = os.path.join(PROJECT_ROOT, d)
    return d


def list_datasets(config):
    return sa.UCRAnomalyDataset.get_available_datasets(resolve_data_dir(config))


def weight_exists(model_name):
    p = os.path.join(PROJECT_ROOT, "experiments", "results", _WEIGHT_FILES[model_name])
    return os.path.exists(p)


def load_test_dataset(dataset_idx, window_size, config):
    """返回 (windows[N,W], labels[N], ds_name, scaler)。

    注意：基线模型（AE/VAE/AnoGAN/IF）按训练时的固定窗口大小构建，线性层维度锁死，
    必须用 config["data"]["window_size"]（训练尺寸），不能传任意窗口大小，否则形状不匹配。
    因此这里强制使用训练窗口大小，忽略传入的 window_size 参数，保证各模型都能正确推理。
    """
    from src.data.ucr_loader import UCRAnomalyDataset
    ddir = resolve_data_dir(config)
    wsize = int(config["data"]["window_size"])
    train_set = UCRAnomalyDataset(ddir, dataset_idx, wsize, 1, "train",
                                  few_shot_ratio=1.0, augment_fn=None)
    test_set = UCRAnomalyDataset(ddir, dataset_idx, wsize, 1, "test",
                                 few_shot_ratio=1.0, augment_fn=None)
    test_set.set_scaler(train_set.scaler)
    names = list_datasets(config)
    ds_name = names[dataset_idx][:30] if dataset_idx < len(names) else f"ds{dataset_idx}"
    return test_set.windows, test_set.labels, ds_name, train_set.scaler


def evaluate_models(dataset_idx, window_size, use_multiscale, noise_ratio,
                    config, device="cpu", max_windows=300):
    """对所选数据集的测试集，实时计算 5 个模型的 AUROC。"""
    from sklearn.metrics import roc_auc_score

    windows, labels, ds_name, _ = load_test_dataset(dataset_idx, window_size, config)
    n = len(windows)
    if max_windows and n > max_windows:
        idx = np.unique(np.linspace(0, n - 1, max_windows).astype(int))
        windows = windows[idx]
        labels = labels[idx]

    tw = torch.from_numpy(windows).float().unsqueeze(1).to(device)
    use_cond = bool(config["experiment"].get("use_conditional", False)) and \
        config["unet"].get("cond_dim") is not None

    results = {}
    for mk in MODEL_KEYS:
        if not weight_exists(mk):
            results[mk] = None
            continue
        try:
            scores = sa.compute_window_scores(mk, tw, config, device,
                                              use_multiscale, False, use_cond,
                                              noise_ratio=noise_ratio)
            results[mk] = float(roc_auc_score(labels, scores))
        except Exception as e:
            results[mk] = None
    return results, ds_name


def reconstruct_window(model_name, window_1d, scaler, config, device,
                       use_multiscale, noise_ratio):
    """对单窗口做重建，返回 (input_1d, recon_1d, error_1d)，均为原始尺度。"""
    model = sa.load_model(model_name, config, device)
    w = np.asarray(window_1d, dtype=np.float32)
    if scaler is not None:
        w = scaler.transform(w.reshape(-1, 1)).reshape(-1)
    x = torch.from_numpy(w).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,W]

    ctx = contextlib.nullcontext() if model_name == "AnoGAN" else torch.inference_mode()
    with ctx:
        if model_name == "Diffusion":
            use_cond = bool(config["experiment"].get("use_conditional", False)) and \
                config["unet"].get("cond_dim") is not None
            cond = None
            if use_cond:
                b = x
                cond = torch.cat([b.mean(dim=2), b.std(dim=2),
                                  b.max(dim=2).values, b.min(dim=2).values], dim=1)
            recon = model.reconstruct(x, cond=cond, use_multiscale=use_multiscale,
                                      noise_ratio=noise_ratio)
        elif model_name == "AnoGAN":
            recon = model.reconstruct(x, n_steps=15)
        else:
            recon = model.reconstruct(x)
    if recon.shape != x.shape:
        recon = recon.view_as(x)

    inp = x.detach().cpu().numpy().reshape(-1)
    rec = recon.detach().cpu().numpy().reshape(-1)
    err = (rec - inp) ** 2
    if scaler is not None:
        inp = scaler.inverse_transform(inp.reshape(-1, 1)).reshape(-1)
        rec = scaler.inverse_transform(rec.reshape(-1, 1)).reshape(-1)
    return inp, rec, err
