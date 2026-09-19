"""孤立森林（Isolation Forest）基线模型

直接调用 sklearn 的 IsolationForest，统一接口便于对比。
"""
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """孤立森林异常检测器"""

    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, x):
        """x: (N, W) 或 (N, C, W)"""
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        self.model.fit(x)

    def score(self, x):
        """返回异常分数（越大越异常）"""
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        # decision_function: 越小越异常，取反使其越大越异常
        scores = -self.model.decision_function(x)
        return scores

    def predict(self, x):
        """返回0(正常)/1(异常)标签"""
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return (self.model.predict(x) == -1).astype(int)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)
