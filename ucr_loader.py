"""UCR Anomaly Archive 时间序列异常检测数据集加载器"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class UCRAnomalyDataset(Dataset):
    """UCR Anomaly Archive 单条时间序列数据集

    将连续时序切分为滑动窗口，仅使用正常段训练，
    测试时在完整序列上逐窗口计算重建误差。
    """

    # UCR Anomaly Archive 子集名 (部分代表性数据集)
    # 如果数据目录中有.txt文件，优先使用目录中的文件
    FALLBACK_NAMES = [
        "028_UCR_Anomaly_DISTORTED1s540D4000_3500_5400_5400",
        "029_UCR_Anomaly_DISTORTED2s540D4000_3600_5400_5400",
        "047_UCR_Anomaly_Twave1s540D4000_3500_5400_5400",
        "063_UCR_Anomaly_AMIGA1s440D3000_3500_4400_4400",
        "079_UCR_Anomaly_POWER1s440D3000_1000_2000_4400",
        "088_UCR_Anomaly_RESPIRATION1s400D3000_1400_1600_4000",
        "096_UCR_Anomaly_S10943s415D3000_700_800_4150",
        "121_UCR_Anomaly_YAHOOweb1s640D5000_2300_2400_6400",
    ]

    @classmethod
    def get_available_datasets(cls, data_dir):
        """扫描数据目录，返回可用的数据集名列表"""
        raw_dir = os.path.join(data_dir, "raw")
        if os.path.isdir(raw_dir):
            files = sorted([f[:-4] for f in os.listdir(raw_dir) if f.endswith(".txt")])
            if files:
                return files
        return cls.FALLBACK_NAMES

    @property
    def DATASET_NAMES(self):
        """动态获取数据集名列表"""
        return UCRAnomalyDataset.get_available_datasets(self._data_dir)

    def __init__(self, data_dir, dataset_idx=0, window_size=100, stride=1,
                 mode="train", train_ratio=0.5, few_shot_ratio=1.0, augment_fn=None):
        """
        Args:
            data_dir: 数据根目录
            dataset_idx: 数据集索引
            window_size: 滑动窗口大小
            stride: 窗口步长
            mode: "train" 或 "test"
            train_ratio: 训练集占正常数据的比例
            few_shot_ratio: 少样本比例 (0.0~1.0)，1.0表示使用全部正常数据
            augment_fn: 数据增强函数，仅训练时使用
        """
        self._data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.augment_fn = augment_fn

        name = self.DATASET_NAMES[dataset_idx]
        ts, labels = self._load_single(data_dir, name)

        # 分割训练/测试：前 train_ratio 用于训练（仅正常），后部分测试（含异常）
        split_idx = int(len(ts) * train_ratio)

        if mode == "train":
            train_ts = ts[:split_idx]
            train_labels = labels[:split_idx]
            self._build_windows(train_ts, train_labels, few_shot_ratio)
            # point 级标签与窗口起始索引，供 point-level 评测（Point Adjustment）使用
            self.point_labels = train_labels
            self.win_starts = np.arange(0, len(train_ts) - self.window_size + 1, self.stride)
        else:
            # 测试：使用 split_idx 之后的全部数据
            test_ts = ts[split_idx:]
            test_labels = labels[split_idx:]
            self._build_windows(test_ts, test_labels, 1.0)
            # point 级标签与窗口起始索引，供 point-level 评测（Point Adjustment）使用
            self.point_labels = test_labels
            self.win_starts = np.arange(0, len(test_ts) - self.window_size + 1, self.stride)

        # 标准化（仅用训练数据拟合）
        if mode == "train":
            self.scaler = StandardScaler()
            all_vals = np.concatenate([w.flatten() for w in self.windows])
            self.scaler.fit(all_vals.reshape(-1, 1))
        else:
            self.scaler = None  # 由外部传入

    def set_scaler(self, scaler):
        """设置外部传入的标准化器（测试集用训练集的scaler）"""
        self.scaler = scaler

    def _build_windows(self, ts, labels, sample_ratio):
        """滑动窗口切片"""
        windows = []
        window_labels = []

        for i in range(0, len(ts) - self.window_size + 1, self.stride):
            win = ts[i:i + self.window_size]
            win_labels = labels[i:i + self.window_size]
            # 窗口标签：只要窗口内有一个异常点，标记为异常
            win_label = 1 if win_labels.sum() > 0 else 0
            windows.append(win)
            window_labels.append(win_label)

        windows = np.array(windows, dtype=np.float32)
        window_labels = np.array(window_labels, dtype=np.int64)

        # 训练集：只保留正常窗口，按少样本比例采样
        if self.mode == "train":
            normal_idx = np.where(window_labels == 0)[0]
            if len(normal_idx) == 0:
                raise ValueError("训练数据中没有正常窗口！")
            n_keep = max(1, int(len(normal_idx) * sample_ratio))
            keep_idx = np.sort(np.random.choice(normal_idx, n_keep, replace=False))
            windows = windows[keep_idx]
            window_labels = window_labels[keep_idx]

        self.windows = windows  # (N, W)
        self.labels = window_labels  # (N,)

    def _load_single(self, data_dir, name):
        """加载单条UCR Anomaly Archive时间序列"""
        filepath = os.path.join(data_dir, "raw", name + ".txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"数据文件不存在: {filepath}\n"
                f"请先运行 scripts/download_ucr.py 下载数据集"
            )
        # UCR Anomaly Archive格式：每行一个数值，最后100个值为标签(0/1)
        raw = np.loadtxt(filepath)
        ts = raw[:-100].astype(np.float32)
        label = raw[-100:].astype(np.int64)

        # 将标签扩展到与时间序列等长（UCR格式中标签对应最后部分）
        labels = np.zeros(len(ts), dtype=np.int64)
        # 异常区间信息编码在文件名中: ..._START_END_LENGTH
        parts = name.split("_")
        try:
            anomaly_start = int(parts[-3])
            anomaly_end = int(parts[-2])
        except (ValueError, IndexError):
            # 文件名格式不符合预期，使用文件末尾标签推断
            anomaly_start = len(ts) - 100
            anomaly_end = len(ts)
        # 调整索引到时间序列范围
        n_ts = len(ts)
        if anomaly_start < n_ts and anomaly_end <= n_ts:
            labels[anomaly_start:anomaly_end] = 1
        elif anomaly_start < n_ts:
            labels[anomaly_start:] = 1

        return ts, labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        win = self.windows[idx]  # (W,)
        if self.scaler is not None:
            win = self.scaler.transform(win.reshape(-1, 1)).flatten().astype(np.float32)

        if self.mode == "train" and self.augment_fn is not None:
            win = self.augment_fn(win)

        # (1, W) - 单通道一维时序
        return win[np.newaxis, :], self.labels[idx]


def get_dataloaders(config, augment_fn=None):
    """创建训练/测试 DataLoader"""
    data_dir = config["data"]["data_dir"]
    dataset_idx = config["data"]["dataset_idx"]
    window_size = config["data"]["window_size"]
    stride = config["data"]["stride"]
    few_shot_ratio = config["data"]["few_shot_ratio"]
    batch_size = config["train"]["batch_size"]

    train_set = UCRAnomalyDataset(
        data_dir, dataset_idx, window_size, stride, "train",
        few_shot_ratio=few_shot_ratio, augment_fn=augment_fn
    )

    test_set = UCRAnomalyDataset(
        data_dir, dataset_idx, window_size, stride, "test",
        few_shot_ratio=1.0, augment_fn=None
    )
    test_set.set_scaler(train_set.scaler)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, train_set, test_set
