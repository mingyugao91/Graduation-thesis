"""少样本时序数据增强模块

提供多种增强策略，增加训练数据多样性，提升少样本场景下的泛化能力。
"""
import numpy as np


def jitter(x, sigma=0.03):
    """添加高斯噪声"""
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def scaling(x, sigma=0.1):
    """随机缩放"""
    factor = np.random.normal(1.0, sigma)
    return (x * factor).astype(np.float32)


def time_warp(x, sigma=0.2, knot=4):
    """时间扭曲：通过随机控制点生成非线性时间映射"""
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(len(x))
    random_warps = np.random.normal(1.0, sigma, knot + 2)
    warp_steps = np.linspace(0, len(x) - 1, knot + 2)
    warper = CubicSpline(warp_steps, random_warps)(orig_steps)
    warped_steps = np.clip(orig_steps * warper, 0, len(x) - 1).astype(int)
    return x[warped_steps]


def window_slice(x, reduce_ratio=0.9):
    """窗口裁剪：随机截取子序列再重采样到原长度"""
    seq_len = len(x)
    sliced_len = int(seq_len * reduce_ratio)
    start = np.random.randint(0, seq_len - sliced_len + 1)
    sliced = x[start:start + sliced_len]
    # 线性插值回原长度
    indices = np.linspace(0, sliced_len - 1, seq_len)
    return np.interp(indices, np.arange(sliced_len), sliced).astype(np.float32)


def magnitude_warp(x, sigma=0.1, knot=4):
    """幅值扭曲：用平滑随机曲线调制幅值"""
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(len(x))
    random_warps = np.random.normal(1.0, sigma, knot + 2)
    warp_steps = np.linspace(0, len(x) - 1, knot + 2)
    warper = CubicSpline(warp_steps, random_warps)(orig_steps)
    return (x * warper).astype(np.float32)


class TimeSeriesAugmenter:
    """组合多种增强策略的数据增强器

    每次调用随机选择一种增强方式（含不增强），增强后返回。
    """

    def __init__(self, augment_list=None, prob=0.5):
        if augment_list is None:
            augment_list = ["jitter", "scaling", "window_slice", "magnitude_warp"]
        self.augment_list = augment_list
        self.prob = prob

    def __call__(self, x):
        """x: (W,) 一维时序"""
        if np.random.rand() > self.prob:
            return x

        method = np.random.choice(self.augment_list)

        if method == "jitter":
            return jitter(x)
        elif method == "scaling":
            return scaling(x)
        elif method == "time_warp":
            return time_warp(x)
        elif method == "window_slice":
            return window_slice(x)
        elif method == "magnitude_warp":
            return magnitude_warp(x)
        else:
            return x
