"""多数据集评测：用 000 子集训练的权重，对全部 8 个 UCR 子集做跨场景推断评测。

补论文「单数据集」硬伤，呈现 Diffusion 在多个场景下的泛化能力。
权重只在 000 上训练、仅加载一次；遍历 data/raw 下全部子集，对每个子集
独立标准化(scaler)后做推断，计算 AUROC / F1 / point-level PA。

健壮性：
  - 每子集、每方法独立 try/except，单点失败只记 error 不影响其他
  - 启动预载已有 multi_dataset_results.json，已完成子集自动跳过（断点续跑）
  - 每完成一个子集即增量落盘，防止中途崩溃丢数据

运行:
  python scripts/multi_dataset_evaluate.py            # 全量（跳过已完成）
  python scripts/multi_dataset_evaluate.py --start 2  # 从 idx=2 续跑
  python scripts/multi_dataset_evaluate.py --limit 1  # 冒烟测试
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.data.ucr_loader import get_dataloaders, UCRAnomalyDataset
from src.models.unet_1d import UNet1D
from src.models.diffusion import DiffusionModel
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.anogan import AnoGAN
from src.models.isolation_forest import IsolationForestDetector
from src.evaluate.metrics import (
    get_recon_scores, get_iforest_scores, compute_metrics, point_level_pa,
)

# 跨域推断 AnoGAN 极慢且不稳定，多数据集表聚焦核心 4 方法（单数据集结果见主表）
SKIP_ANOGAN = True


def build_unet(config):
    u = config["unet"]
    return UNet1D(
        in_channels=u.get("in_channels", 1),
        base_channels=u.get("base_channels", 32),
        channel_mults=tuple(u.get("channel_mults", [1, 2, 4, 4])),
        num_res_blocks=u.get("num_res_blocks", 2),
        time_dim=u.get("time_dim", 128),
        dropout=u.get("dropout", 0.1),
        cond_dim=u.get("cond_dim", None),
    )


def load_diffusion(config, path):
    unet = build_unet(config)
    model = DiffusionModel(
        unet,
        n_timesteps=config["diffusion"]["n_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        schedule=config["diffusion"]["schedule"],
        device="cpu",
    )
    model.load(path)
    return model


def add_pa(m, scores, test_set):
    pa = point_level_pa(scores, test_set)
    m["f1_pa"] = pa["f1_pa"]
    m["precision_pa"] = pa["precision_pa"]
    m["recall_pa"] = pa["recall_pa"]
    m["auroc_pa"] = pa["auroc"]
    return m


def _done(results, name):
    d = results.get(name, {}).get("Diffusion (Ours)")
    return isinstance(d, dict) and "auroc" in d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="只评测 N 个子集（0=全部），配合 --start 使用")
    parser.add_argument("--start", type=int, default=0,
                        help="从第 N 个子集开始（默认0，断点续跑用）")
    args = parser.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    data_dir = config["data"]["data_dir"]
    names = UCRAnomalyDataset.get_available_datasets(data_dir)
    n = len(names)
    save_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    out_path = os.path.join(save_dir, "multi_dataset_results.json")
    device = "cpu"
    pct = config["experiment"]["threshold_percentile"]
    ws = config["data"]["window_size"]

    # 预载已有结果（断点续跑）
    results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                results = json.load(f)
        except Exception:
            results = {}

    print(">>> 加载固定权重（000 子集训练）...")
    diff = load_diffusion(config, os.path.join(save_dir, "diffusion_main.pth"))
    use_ms = config["experiment"]["use_multiscale"]
    ae = Autoencoder(ws, config["baselines"]["ae"]["hidden_dim"],
                    config["baselines"]["ae"]["latent_dim"]).to(device)
    ae.load(os.path.join(save_dir, "ae.pth"))
    vae = VAE(ws, config["baselines"]["vae"]["hidden_dim"],
              config["baselines"]["vae"]["latent_dim"]).to(device)
    vae.load(os.path.join(save_dir, "vae.pth"))
    iforest = IsolationForestDetector(
        contamination=config["baselines"]["iforest"]["contamination"],
        n_estimators=config["baselines"]["iforest"]["n_estimators"])
    iforest.load(os.path.join(save_dir, "iforest.pkl"))
    anogan = None
    if not SKIP_ANOGAN:
        anogan = AnoGAN(ws, config["baselines"]["anogan"]["latent_dim"],
                        config["baselines"]["anogan"]["hidden_dim"], device=device).to(device)
        anogan.load(os.path.join(save_dir, "anogan.pth"))
    print(">>> 权重加载完成，遍历 %d 个子集 (start=%d)..." % (n, args.start))

    def dump():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    end = n if args.limit <= 0 else min(args.start + args.limit, n)
    for idx in range(args.start, end):
        name = names[idx]
        if _done(results, name):
            print("[%d/%d] %s 已存在，跳过" % (idx + 1, n, name[:30]))
            continue
        config["data"]["dataset_idx"] = idx
        try:
            _, test_loader, _, test_set = get_dataloaders(config, None)
        except Exception as e:
            results[name] = {"error": "dataloader: %s" % e}
            dump()
            print("[%d/%d] %s DATALOADER ERROR: %s" % (idx + 1, n, name[:30], e))
            continue

        methods = {}
        # Diffusion (Ours)
        try:
            sc, lb = get_recon_scores(diff, test_loader, device,
                                      use_multiscale=use_ms, model_type="diffusion")
            methods["Diffusion (Ours)"] = add_pa(compute_metrics(sc, lb, pct), sc, test_set)
        except Exception as e:
            methods["Diffusion (Ours)"] = {"error": str(e)}
        # AE
        try:
            sc, lb = get_recon_scores(ae, test_loader, device, model_type="ae")
            methods["AE"] = add_pa(compute_metrics(sc, lb, pct), sc, test_set)
        except Exception as e:
            methods["AE"] = {"error": str(e)}
        # VAE
        try:
            sc, lb = get_recon_scores(vae, test_loader, device, model_type="vae")
            methods["VAE"] = add_pa(compute_metrics(sc, lb, pct), sc, test_set)
        except Exception as e:
            methods["VAE"] = {"error": str(e)}
        # IF
        try:
            sc, lb = get_iforest_scores(iforest, test_loader)
            methods["IF"] = add_pa(compute_metrics(sc, lb, pct), sc, test_set)
        except Exception as e:
            methods["IF"] = {"error": str(e)}
        # AnoGAN（可选）
        if anogan is not None:
            try:
                sc, lb = get_recon_scores(anogan, test_loader, device, model_type="anogan")
                methods["AnoGAN"] = add_pa(compute_metrics(sc, lb, pct), sc, test_set)
            except Exception as e:
                methods["AnoGAN"] = {"error": str(e)}

        results[name] = methods
        dump()
        d_ = methods["Diffusion (Ours)"]; a_ = methods["AE"]
        v_ = methods["VAE"]; i_ = methods["IF"]
        print("[%d/%d] %s  Diff=%.4f AE=%.4f VAE=%.4f IF=%.4f"
              % (idx + 1, n, name[:30],
                 d_.get("auroc", float("nan")), a_.get("auroc", float("nan")),
                 v_.get("auroc", float("nan")), i_.get("auroc", float("nan"))))

    # 汇总打印
    order = ["Diffusion (Ours)", "AE", "VAE", "IF"] + ([] if SKIP_ANOGAN else ["AnoGAN"])
    print("\n=== 多数据集 AUROC 汇总 ===")
    print("Dataset".ljust(38) + "".join(m[:10].rjust(12) for m in order))
    for name, ms in results.items():
        row = name[:36].ljust(38)
        for m in order:
            v = ms.get(m, {})
            row += (f"{v.get('auroc', 0):.4f}" if "auroc" in v else "ERR").rjust(12)
        print(row)
    print("\n结果已保存:", out_path)


if __name__ == "__main__":
    main()
