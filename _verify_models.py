# -*- coding: utf-8 -*-
"""验证重建的 load_model / compute_window_scores 能正确加载权重并出分。"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from app.streamlit_app import load_model, load_config_cached, compute_window_scores

config = load_config_cached()

print("=== 加载 5 个模型权重 ===")
for name in ["Diffusion", "AE", "VAE", "AnoGAN", "Isolation Forest"]:
    try:
        m = load_model(name, config, "cpu")
        print(f"[OK] {name}: {type(m).__name__}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")

print("\n=== 计算逐窗口异常分数（随机张量，验证重建通路）===")
tw = torch.randn(8, 1, 100)
for name in ["Diffusion", "AE", "VAE", "AnoGAN"]:
    try:
        s = compute_window_scores(name, tw, config, "cpu", False, True, True, noise_ratio=0.15)
        print(f"[OK] scores {name}: shape={s.shape}, mean={float(s.mean()):.4f}")
    except Exception as e:
        print(f"[FAIL] scores {name}: {e}")

# Isolation Forest 走 .score 分支
try:
    s = compute_window_scores("Isolation Forest", tw, config, "cpu", False, True, True, noise_ratio=0.15)
    print(f"[OK] scores Isolation Forest: shape={s.shape}")
except Exception as e:
    print(f"[FAIL] scores Isolation Forest: {e}")
