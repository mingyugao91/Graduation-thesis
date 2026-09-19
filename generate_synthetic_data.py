"""合成时间序列异常数据生成器

当无法下载UCR Anomaly Archive时，生成模拟数据用于验证流程。
包含多种异常模式：尖峰、电平偏移、频率变化、噪声注入。
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


def generate_synthetic_series(length=5000, anomaly_ratio=0.1, seed=42):
    """生成单条合成时序 + 标签

    Args:
        length: 时序长度
        anomaly_ratio: 异常占比
        seed: 随机种子
    Returns:
        ts: (length,) 时间序列
        labels: (length,) 0/1标签
    """
    rng = np.random.RandomState(seed)
    ts = np.zeros(length, dtype=np.float32)
    labels = np.zeros(length, dtype=np.int64)

    # 基础信号：正弦波 + 缓慢趋势
    t = np.arange(length)
    base = np.sin(2 * np.pi * t / 200) * 0.5 + np.sin(2 * np.pi * t / 50) * 0.2
    # 加趋势
    trend = np.linspace(0, 0.3, length)
    # 加噪声
    noise = rng.normal(0, 0.05, length)
    ts = base + trend + noise

    # 随机插入异常段
    n_anomalies = max(1, int(length * anomaly_ratio / 50))
    anomaly_types = ["spike", "level_shift", "noise_burst", "freq_change"]

    for _ in range(n_anomalies):
        atype = rng.choice(anomaly_types)
        seg_len = rng.randint(20, 80)
        start = rng.randint(length // 2, length - seg_len - length // 10)
        end = start + seg_len
        labels[start:end] = 1

        if atype == "spike":
            ts[start:end] += rng.uniform(2, 4) * np.sign(rng.randn())
        elif atype == "level_shift":
            ts[start:end] += rng.uniform(1, 2) * np.sign(rng.randn())
        elif atype == "noise_burst":
            ts[start:end] += rng.normal(0, 0.5, seg_len)
        elif atype == "freq_change":
            freq = rng.uniform(3, 8)
            ts[start:end] = np.sin(2 * np.pi * np.arange(seg_len) / freq) * 0.8

    return ts.astype(np.float32), labels


def save_synthetic_datasets(data_dir, n_datasets=8, seed=42):
    """保存多条合成数据集，格式与UCR Anomaly Archive兼容"""
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    for i in range(n_datasets):
        length = rng.randint(3000, 6000)
        anomaly_ratio = rng.uniform(0.05, 0.15)
        ts, labels = generate_synthetic_series(length, anomaly_ratio, seed + i)

        # 找到异常区间
        anomaly_starts = np.where(np.diff(labels) == 1)[0] + 1
        anomaly_ends = np.where(np.diff(labels) == -1)[0] + 1

        if len(anomaly_starts) > 0:
            a_start = anomaly_starts[0]
            a_end = anomaly_ends[0] if len(anomaly_ends) > 0 else length
        else:
            a_start = length // 2
            a_end = a_start + 50

        # 构造文件名（与UCR格式兼容）
        name = f"{i:03d}_UCR_Anomaly_SYNTHETIC{length}s{length}D{length}_{a_start}_{a_end}_{length}"

        # UCR格式：时序值 + 100个标签值
        label_tail = np.zeros(100, dtype=np.float64)
        label_tail[:50] = 0  # 前50个正常
        label_tail[50:] = 1  # 后50个异常
        combined = np.concatenate([ts, label_tail])

        filepath = os.path.join(raw_dir, name + ".txt")
        np.savetxt(filepath, combined, fmt="%.6f")
        print(f"  生成: {name} (长度={length}, 异常={a_start}:{a_end})")

    print(f"\n共生成 {n_datasets} 个合成数据集到 {raw_dir}")


if __name__ == "__main__":
    data_dir = os.path.join(PROJECT_ROOT, "data")
    save_synthetic_datasets(data_dir)
