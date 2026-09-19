"""少样本对比实验：不同训练数据比例下各方法的性能

论文核心实验——验证"少样本"场景下扩散模型(+数据增强+条件扩散)的鲁棒性。

运行方式:
  python scripts/few_shot_experiment.py

产出:
  experiments/results/few_shot_results.json
  结构: {ratio: {method: metrics}}
"""
import os
import sys
import json
import copy
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.data.ucr_loader import get_dataloaders
from src.data.augmentation import TimeSeriesAugmenter
from src.train.train_diffusion import train_diffusion
from src.train.train_baselines import train_ae, train_vae, train_iforest
from src.evaluate.metrics import get_recon_scores, get_iforest_scores, compute_metrics, print_metrics

RATIOS = [0.1, 0.3, 0.5]  # 1.0 直接复用 comparison_results.json 已有结果
RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "results", "few_shot_results.json")


def slim(m):
    """只保留论文需要的核心指标"""
    keys = ["auroc", "auprc", "f1", "precision", "recall"]
    return {k: m[k] for k in keys if k in m}


def run_one_ratio(base_config, ratio, save_root):
    config = copy.deepcopy(base_config)
    config["data"]["few_shot_ratio"] = ratio
    device = config["train"]["device"]
    pct = config["experiment"]["threshold_percentile"]

    save_dir = os.path.join(save_root, f"few_shot_{int(ratio*100)}")
    os.makedirs(save_dir, exist_ok=True)

    results = {}
    t0 = time.time()

    # ===== 扩散模型 =====
    print(f"\n{'='*60}\n  [ratio={ratio}] 训练扩散模型\n{'='*60}", flush=True)
    diffusion, test_loader = train_diffusion(config, save_dir, f"diffusion_fs{int(ratio*100)}")
    use_ms = config["experiment"]["use_multiscale"]
    scores, labels = get_recon_scores(diffusion, test_loader, device,
                                      use_multiscale=use_ms, model_type="diffusion")
    m = compute_metrics(scores, labels, pct)
    print_metrics(f"Diffusion ratio={ratio}", m)
    results["Diffusion (Ours)"] = slim(m)

    # ===== 基线 =====
    augment_fn = TimeSeriesAugmenter() if config["experiment"]["use_augmentation"] else None
    train_loader, test_loader, _, _ = get_dataloaders(config, augment_fn)

    print(f"\n--- [ratio={ratio}] AE ---", flush=True)
    ae = train_ae(config, train_loader, device)
    scores, labels = get_recon_scores(ae, test_loader, device, model_type="ae")
    results["AE"] = slim(compute_metrics(scores, labels, pct))

    print(f"\n--- [ratio={ratio}] VAE ---", flush=True)
    vae = train_vae(config, train_loader, device)
    scores, labels = get_recon_scores(vae, test_loader, device, model_type="vae")
    results["VAE"] = slim(compute_metrics(scores, labels, pct))

    print(f"\n--- [ratio={ratio}] Isolation Forest ---", flush=True)
    iforest = train_iforest(config, train_loader)
    scores, labels = get_iforest_scores(iforest, test_loader)
    results["IF"] = slim(compute_metrics(scores, labels, pct))

    print(f"\n[ratio={ratio}] 完成，耗时 {time.time()-t0:.0f}s", flush=True)
    return results


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    save_root = os.path.join(PROJECT_ROOT, "experiments", "results")

    all_results = {}

    # 复用 ratio=1.0 已有结果
    comp_path = os.path.join(save_root, "comparison_results.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            comp = json.load(f)
        all_results["1.0"] = {k: slim(v) for k, v in comp.items() if k != "AnoGAN"}
        print("已复用 ratio=1.0 的对比实验结果", flush=True)

    for ratio in RATIOS:
        all_results[str(ratio)] = run_one_ratio(config, ratio, save_root)
        # 每档跑完立即落盘，防止中途断掉全丢
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[ratio={ratio}] 结果已保存 -> {RESULTS_PATH}", flush=True)

    # ===== 汇总表 =====
    print(f"\n{'='*70}\n  少样本实验结果汇总 (AUROC)\n{'='*70}", flush=True)
    methods = ["Diffusion (Ours)", "AE", "VAE", "IF"]
    ratios_sorted = sorted(all_results.keys(), key=float)
    header = f"{'Method':<20}" + "".join(f"{'r='+r:>12}" for r in ratios_sorted)
    print(header)
    print("-" * len(header))
    for name in methods:
        row = f"{name:<20}"
        for r in ratios_sorted:
            m = all_results[r].get(name)
            row += f"{m['auroc']:>12.4f}" if m else f"{'--':>12}"
        print(row)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
