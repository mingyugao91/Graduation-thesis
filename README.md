# 基于扩散模型的少样本时间序列异常检测方法研究

基于 DDPM 扩散模型的时间序列异常检测系统，针对少样本场景优化。

## 项目结构

```
thesis/
├── src/
│   ├── data/              # 数据加载与增强
│   │   ├── ucr_loader.py  # UCR Anomaly Archive 加载器
│   │   └── augmentation.py # 少样本数据增强（jitter/scaling/warp等）
│   ├── models/            # 模型实现
│   │   ├── unet_1d.py     # 一维U-Net去噪网络
│   │   ├── diffusion.py   # DDPM扩散模型（前向/反向/多尺度）
│   │   ├── autoencoder.py # AE基线
│   │   ├── vae.py         # VAE基线
│   │   ├── anogan.py     # AnoGAN基线
│   │   └── isolation_forest.py # 孤立森林基线
│   ├── train/             # 训练脚本
│   ├── evaluate/          # 评估指标
│   └── utils/             # 配置工具
├── configs/               # 配置文件
├── scripts/               # 实验脚本
│   ├── download_ucr.py    # 下载UCR数据集
│   ├── generate_synthetic_data.py # 生成合成数据（备用）
│   ├── run_experiment.py  # 主实验（对比+消融）
│   └── smoke_test.py      # 冒烟测试
├── app/
│   └── streamlit_app.py   # 可视化检测平台
├── experiments/           # 实验结果输出
└── data/                  # 数据目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 方式一：下载UCR Anomaly Archive
python scripts/download_ucr.py

# 方式二：生成合成数据（UCR下载失败时备用）
python scripts/generate_synthetic_data.py
```

### 3. 冒烟测试

```bash
python scripts/smoke_test.py
```

### 4. 运行实验

```bash
# 对比实验（扩散模型 vs IF/AE/VAE/AnoGAN）
python scripts/run_experiment.py

# 消融实验（去数据增强/去条件扩散/去多尺度）
python scripts/run_experiment.py --ablation
```

### 5. 启动可视化平台

```bash
streamlit run app/streamlit_app.py
```

## 核心方法

### 扩散模型异常检测原理

1. **训练阶段**：对正常时序逐步添加高斯噪声，训练一维U-Net学习逆向去噪
2. **检测阶段**：从纯噪声出发重建，计算原始时序与重建时序的误差
3. **异常判定**：误差大的窗口判定为异常（模型未见过异常模式，无法准确重建）

### 少样本优化策略

| 策略 | 说明 |
|------|------|
| 数据增强 | Jitter/Scaling/TimeWarp/WindowSlice/MagnitudeWarp |
| 条件扩散 | 将时序统计特征作为条件信息注入U-Net |
| 多尺度重建 | 原始/半/四分之一分辨率分别重建后加权融合 |
