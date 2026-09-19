"""生成用于演示「文档上传」功能的样例数据。

覆盖：CSV(带标签) / CSV(无标签) / TXT(单列) / XLSX(带标签) / 多列CSV(选列)
每个文件都注入了清晰的异常段，方便直接上传演示异常检测效果。

用法：
    python scripts/make_demo_data.py
输出目录：data/sample_uploads/
"""
import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_uploads")
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(42)


def make_series(n, base, noise, anomaly_range, anomaly_delta, smooth=True):
    """生成一条带平稳段 + 异常段的时间序列。"""
    t = np.arange(n)
    if smooth:
        # 用随机游走 + 周期项模拟真实传感器
        x = np.cumsum(rng.normal(0, noise, n))
        x = x - x.min() + base
        x += 2.0 * np.sin(2 * np.pi * t / 50)
    else:
        x = base + rng.normal(0, noise, n)
    a0, a1 = anomaly_range
    x[a0:a1] += anomaly_delta          # 异常段整体抬升
    x[a0:a1] += rng.normal(0, noise * 2, a1 - a0)  # 异常段更剧烈抖动
    return x


# 1) CSV —— 机器传感器（多列 + 标签）
def build_sensor_csv():
    n = 1000
    temp = make_series(n, base=60.0, noise=1.2, anomaly_range=(650, 720), anomaly_delta=14.0)
    vib = make_series(n, base=0.5, noise=0.08, anomaly_range=(650, 720), anomaly_delta=0.6)
    label = np.zeros(n, dtype=int)
    label[650:720] = 1
    df = pd.DataFrame({
        "timestamp": np.arange(n),
        "temperature": rng.normal(0, 0.0, n) + temp,
        "vibration": vib,
        "label": label,
    })
    # 让 temperature 列带有轻微的相关性噪声以更真实
    df["temperature"] = temp
    path = os.path.join(OUT_DIR, "demo_sensor_with_label.csv")
    df.to_csv(path, index=False)
    print(f"[CSV·带标签]   {os.path.basename(path)}  行={n} 异常段=650~720 选'temperature'+'label'")


# 2) CSV —— 电商日销量（单列序列 + 日期，无标签）
def build_sales_csv():
    n = 365
    base = 1000 + 300 * np.sin(2 * np.pi * np.arange(n) / 30)   # 月度周期
    base += rng.normal(0, 40, n)
    base[200:215] += 1500   # 大促尖峰（异常）
    base[200:215] += rng.normal(0, 120, 15)
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d"),
        "daily_sales": base.astype(int),
    })
    path = os.path.join(OUT_DIR, "demo_sales_no_label.csv")
    df.to_csv(path, index=False)
    print(f"[CSV·无标签]   {os.path.basename(path)}  行={n} 异常段=第200~214天 选'daily_sales'(无标签)")


# 3) TXT —— 心电式单变量（单列数值）
def build_ecg_txt():
    n = 1200
    t = np.arange(n)
    beat = np.sin(2 * np.pi * t / 25)                 # 窦性心律
    beat += 0.3 * np.sin(2 * np.pi * t / 7)
    beat += rng.normal(0, 0.05, n)
    beat[800:860] += 1.4 * np.sin(2 * np.pi * (t[800:860]) / 6)  # 心律不齐段
    beat[800:860] += rng.normal(0, 0.2, 60)
    path = os.path.join(OUT_DIR, "demo_ecg.txt")
    np.savetxt(path, beat, fmt="%.4f")
    print(f"[TXT·单列]    {os.path.basename(path)}  行={n} 异常段=800~860 直接单列当序列")


# 4) XLSX —— 交通流量（多列 + 标签）
def build_traffic_xlsx():
    n = 900
    flow = make_series(n, base=500.0, noise=25.0, anomaly_range=(400, 470), anomaly_delta=-300.0)
    occ = make_series(n, base=0.4, noise=0.03, anomaly_range=(400, 470), anomaly_delta=-0.25)
    label = np.zeros(n, dtype=int)
    label[400:470] = 1
    df = pd.DataFrame({
        "hour": np.arange(n),
        "flow": flow,
        "occupancy": occ,
        "label": label,
    })
    path = os.path.join(OUT_DIR, "demo_traffic_with_label.xlsx")
    df.to_excel(path, index=False)
    print(f"[XLSX·带标签] {os.path.basename(path)}  行={n} 异常段=400~470 选'flow'+'label'")


# 5) CSV —— 服务器多指标（多列 + 标签，演示选列）
def build_server_csv():
    n = 1100
    cpu = make_series(n, base=35.0, noise=3.0, anomaly_range=(700, 760), anomaly_delta=45.0)
    mem = make_series(n, base=55.0, noise=2.0, anomaly_range=(700, 760), anomaly_delta=10.0)
    net_in = make_series(n, base=120.0, noise=15.0, anomaly_range=(700, 760), anomaly_delta=400.0)
    label = np.zeros(n, dtype=int)
    label[700:760] = 1
    df = pd.DataFrame({
        "ts": np.arange(n),
        "cpu_percent": cpu,
        "mem_percent": mem,
        "net_in_kbps": net_in,
        "label": label,
    })
    path = os.path.join(OUT_DIR, "demo_server_multimetric.csv")
    df.to_csv(path, index=False)
    print(f"[CSV·多指标]  {os.path.basename(path)}  行={n} 异常段=700~760 选'net_in_kbps'+'label'(演示选列)")


if __name__ == "__main__":
    build_sensor_csv()
    build_sales_csv()
    build_ecg_txt()
    build_traffic_xlsx()
    build_server_csv()
    print(f"\n全部样例已生成到: {OUT_DIR}")
