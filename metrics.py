"""异常检测逻辑与评估指标

统一接口：各模型训练完成后，通过重建误差计算异常分数，再计算评估指标。
"""
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
)


def point_adjustment(labels, pred):
    """点调整（Point Adjustment）

    对连续的真实异常段，只要段内至少有一个点被预测为异常，
    就将整段预测调整为异常。这是时间序列异常检测领域的标准评测协议，
    能够容忍"仅检测到异常段中一个点"的情况，评测更贴近实际应用。

    Args:
        labels: (N,) 真实标签 (0正常, 1异常)
        pred: (N,) 预测标签 (0/1)
    Returns:
        pred_adj: (N,) 点调整后的预测标签
    """
    labels = np.asarray(labels)
    pred = np.asarray(pred)
    pred_adj = pred.copy()
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            j = i
            while j < n and labels[j] == 1:
                j += 1
            # 异常段 [i, j) 内只要有一点命中，整段判为异常
            if pred[i:j].sum() > 0:
                pred_adj[i:j] = 1
            i = j
        else:
            i += 1
    return pred_adj


def point_adjustment_best_f1(scores, labels):
    """在点调整协议下搜索最优 F1 对应的阈值

    遍历候选阈值（分数唯一值 + 关键分位数），对每个阈值：
    1. 得到原始预测 pred = (scores > thr)
    2. 对 pred 做点调整得到 pred_adj
    3. 计算 F1(pred_adj, labels)
    取使 F1 最大的阈值与其指标。

    Returns:
        dict: f1_pa / precision_pa / recall_pa / threshold
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    sorted_scores = np.sort(scores)
    candidates = np.unique(sorted_scores)
    for p in [50, 75, 80, 85, 90, 92.5, 95, 97.5, 99]:
        candidates = np.append(candidates, np.percentile(scores, p))

    best_f1 = 0.0
    best = None
    for thr in candidates:
        pred = (scores > thr).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        pred_adj = point_adjustment(labels, pred)
        f1 = f1_score(labels, pred_adj, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best = {
                "f1_pa": float(f1),
                "precision_pa": float(precision_score(labels, pred_adj, zero_division=0)),
                "recall_pa": float(recall_score(labels, pred_adj, zero_division=0)),
                "threshold_pa": float(thr),
            }
    if best is None:
        best = {"f1_pa": 0.0, "precision_pa": 0.0, "recall_pa": 0.0, "threshold_pa": 0.0}
    return best


def point_level_pa(window_scores, test_set, aggregate="mean"):
    """Point-level Point Adjustment 评测（时间序列异常检测标准协议）

    将 window 级重建误差分数映射回 point 级序列（每个点取覆盖它的
    所有窗口分数的聚合），再用 point 级真实标签做 Point Adjustment 评测。
    这是 UCR / TSB-AD 等基准采用的评测粒度，能显著体现点调整的增益。

    Args:
        window_scores: (N,) window 级异常分数
        test_set: UCRAnomalyDataset (test 模式)，需含 point_labels 与 win_starts 属性
        aggregate: 多窗口覆盖同一点的聚合方式 ("mean" | "max")
    Returns:
        dict: auroc / f1_pa / precision_pa / recall_pa / threshold_pa
    """
    L = len(test_set.point_labels)
    point_scores = np.zeros(L)
    count = np.zeros(L)
    W = test_set.window_size
    starts = test_set.win_starts
    for i, s in enumerate(starts):
        end = min(s + W, L)
        if aggregate == "max":
            seg = point_scores[s:end]
            np.maximum(seg, window_scores[i], out=seg)
        else:
            point_scores[s:end] += window_scores[i]
            count[s:end] += 1
    if aggregate != "max":
        count[count == 0] = 1
        point_scores /= count

    point_labels = np.asarray(test_set.point_labels)

    # AUROC（point 级，阈值无关）
    try:
        auroc = float(roc_auc_score(point_labels, point_scores))
    except ValueError:
        auroc = 0.5

    # Point Adjustment 最优 F1（在 point 级）
    pa = point_adjustment_best_f1(point_scores, point_labels)
    pa["auroc"] = auroc
    return pa


@torch.no_grad()
def get_recon_scores(model, dataloader, device="cpu", use_multiscale=False,
                     model_type="diffusion"):
    """计算重建误差分数

    Args:
        model: 训练好的模型
        dataloader: 测试数据加载器
        device: 计算设备
        use_multiscale: 是否使用多尺度重建（仅扩散模型）
        model_type: "diffusion" | "ae" | "vae" | "anogan" | "iforest"
    Returns:
        scores: (N,) 异常分数（越大越异常）
        labels: (N,) 真实标签 (0正常, 1异常)
    """
    # AnoGAN 的 reconstruct 需要梯度，单独处理
    if model_type == "anogan":
        return _get_anogan_scores(model, dataloader, device)

    all_scores = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)

        if model_type == "iforest":
            # Isolation Forest: 直接用decision_function
            scores = model.score(batch_x.cpu().numpy())
        else:
            # 基于重建误差的模型
            if model_type == "diffusion":
                # 检查模型是否使用条件扩散
                cond = None
                if hasattr(model, 'unet') and hasattr(model.unet, 'cond_dim') and model.unet.cond_dim is not None:
                    cond = torch.cat([
                        batch_x.mean(dim=2),
                        batch_x.std(dim=2),
                        batch_x.max(dim=2).values,
                        batch_x.min(dim=2).values,
                    ], dim=1)
                recon = model.reconstruct(batch_x, cond=cond, use_multiscale=use_multiscale)
            else:
                recon = model.reconstruct(batch_x)

            if recon.shape != batch_x.shape:
                recon = recon.view_as(batch_x)

            # 逐样本计算MSE
            diff = (recon - batch_x) ** 2
            scores = diff.mean(dim=(1, 2)).cpu().numpy()

        all_scores.append(scores)
        all_labels.append(batch_y.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def _get_anogan_scores(model, dataloader, device="cpu"):
    """AnoGAN 专用评分（reconstruct 需要梯度）"""
    all_scores = []
    all_labels = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        with torch.enable_grad():
            recon = model.reconstruct(batch_x)
        with torch.no_grad():
            if recon.shape != batch_x.shape:
                recon = recon.view_as(batch_x)
            diff = (recon - batch_x) ** 2
            scores = diff.mean(dim=(1, 2)).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(batch_y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


@torch.no_grad()
def get_iforest_scores(detector, dataloader):
    """Isolation Forest 专用评分"""
    all_scores = []
    all_labels = []
    for batch_x, batch_y in dataloader:
        x_np = batch_x.numpy()
        if x_np.ndim == 3:
            x_np = x_np.reshape(x_np.shape[0], -1)
        scores = detector.score(x_np)
        all_scores.append(scores)
        all_labels.append(batch_y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def compute_metrics(scores, labels, threshold_percentile=95):
    """计算评估指标

    Args:
        scores: (N,) 异常分数
        labels: (N,) 真实标签
        threshold_percentile: 兼容参数（未使用），保留向后兼容
    Returns:
        dict: 包含各指标的字典
    """
    metrics = {}

    # AUROC（阈值无关指标）
    try:
        metrics["auroc"] = roc_auc_score(labels, scores)
    except ValueError:
        metrics["auroc"] = 0.5

    # AUPRC（阈值无关指标）
    try:
        metrics["auprc"] = average_precision_score(labels, scores)
    except ValueError:
        metrics["auprc"] = 0.0

    # 最优 F1 阈值搜索：遍历所有可能的阈值，取使 F1 最大的
    sorted_scores = np.sort(scores)
    best_f1, best_pred = 0.0, np.zeros_like(labels)
    candidates = np.unique(sorted_scores)
    # 额外加入分位数候选以减少搜索空间
    for p in [50, 75, 80, 85, 90, 92.5, 95, 97.5, 99]:
        candidates = np.append(candidates, np.percentile(scores, p))
    for thr in candidates:
        pred = (scores > thr).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_pred = pred

    metrics["f1"] = f1_score(labels, best_pred, zero_division=0)
    metrics["precision"] = precision_score(labels, best_pred, zero_division=0)
    metrics["recall"] = recall_score(labels, best_pred, zero_division=0)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, best_pred, labels=[0, 1]).ravel()
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)

    # Point Adjustment（点调整）协议下的指标 —— 时间序列异常检测领域标准评测
    pa = point_adjustment_best_f1(scores, labels)
    metrics["f1_pa"] = pa["f1_pa"]
    metrics["precision_pa"] = pa["precision_pa"]
    metrics["recall_pa"] = pa["recall_pa"]
    metrics["threshold_pa"] = pa["threshold_pa"]

    return metrics


def print_metrics(name, metrics):
    """格式化打印指标"""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    print(f"{'='*50}")
