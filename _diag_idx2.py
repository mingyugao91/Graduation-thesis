# -*- coding: utf-8 -*-
"""诊断 idx=2 子集为何崩溃：加载权重后逐方法评测，完整 traceback 写日志。"""
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.data.ucr_loader import get_dataloaders
from src.models.unet_1d import UNet1D
from src.models.diffusion import DiffusionModel
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.isolation_forest import IsolationForestDetector
from src.evaluate.metrics import (
    get_recon_scores, get_iforest_scores, compute_metrics,
)

log = []


def L(m):
    log.append(str(m))
    print(m, flush=True)


config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
config["data"]["dataset_idx"] = 2
device = "cpu"
pct = config["experiment"]["threshold_percentile"]
ws = config["data"]["window_size"]
rd = os.path.join(PROJECT_ROOT, "experiments", "results")

L(">>> 加载权重...")
u = config["unet"]
unet = UNet1D(in_channels=1, base_channels=u.get("base_channels", 32),
              channel_mults=tuple(u.get("channel_mults", [1, 2, 4, 4])),
              num_res_blocks=u.get("num_res_blocks", 2), time_dim=u.get("time_dim", 128),
              dropout=u.get("dropout", 0.1), cond_dim=u.get("cond_dim", None))
diff = DiffusionModel(unet, n_timesteps=config["diffusion"]["n_timesteps"],
                      beta_start=config["diffusion"]["beta_start"], beta_end=config["diffusion"]["beta_end"],
                      schedule=config["diffusion"]["schedule"], device="cpu")
diff.load(os.path.join(rd, "diffusion_main.pth"))
ae = Autoencoder(ws, config["baselines"]["ae"]["hidden_dim"], config["baselines"]["ae"]["latent_dim"]).to(device)
ae.load(os.path.join(rd, "ae.pth"))
vae = VAE(ws, config["baselines"]["vae"]["hidden_dim"], config["baselines"]["vae"]["latent_dim"]).to(device)
vae.load(os.path.join(rd, "vae.pth"))
iforest = IsolationForestDetector(contamination=config["baselines"]["iforest"]["contamination"],
                                 n_estimators=config["baselines"]["iforest"]["n_estimators"])
iforest.load(os.path.join(rd, "iforest.pkl"))
L(">>> 权重加载完成")

try:
    _, test_loader, _, test_set = get_dataloaders(config, None)
    L("dataloader OK, test_set len=%d" % len(test_set))
except Exception:
    L("DATALOADER ERROR:\n" + traceback.format_exc())
    open(os.path.join(rd, "_diag2.log"), "w", encoding="utf-8").write("\n".join(log))
    raise SystemExit

for tag, fn in [
    ("Diffusion", lambda: get_recon_scores(diff, test_loader, device, use_multiscale=config["experiment"]["use_multiscale"], model_type="diffusion")),
    ("AE", lambda: get_recon_scores(ae, test_loader, device, model_type="ae")),
    ("VAE", lambda: get_recon_scores(vae, test_loader, device, model_type="vae")),
    ("IF", lambda: get_iforest_scores(iforest, test_loader)),
]:
    try:
        sc, lb = fn()
        m = compute_metrics(sc, lb, pct)
        L("%s auroc=%.4f" % (tag, m["auroc"]))
    except Exception:
        L("%s ERROR:\n%s" % (tag, traceback.format_exc()))

L("=== idx=2 诊断结束 ===")
open(os.path.join(rd, "_diag2.log"), "w", encoding="utf-8").write("\n".join(log))
