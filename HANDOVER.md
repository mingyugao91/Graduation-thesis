# 项目交接文档 —— 基于扩散模型的少样本时间序列异常检测可视化平台

> 本文档供接手本项目（后续由其他 AI / 开发者继续开发）使用，包含背景、现状、架构、接口、已知坑、待办与运行方式。读完即可直接上手。
> 最后更新：2026-09-19

---

## 1. 项目背景与当前状态

| 项 | 内容 |
|---|---|
| 选题 | 《基于扩散模型的少样本时间序列异常检测方法研究》 |
| 性质 | 学年论文（**与后续毕业论文同选题**，可复用） |
| 学校 / 专业 | 南京审计大学 · 计算机科学与技术（2023 级） |
| 硬性截止 | **2026 年秋学期成绩录入，9 月 30 日前必须完成**（时间紧，接手优先保交付） |
| 论文要求 | 正文 ≥ 3000 字；查重 ≤ 20%；结构：中英文摘要/关键词 → 正文（背景/现状/技术原理/架构设计/实验/总结）→ 参考文献；**必须附系统截图与实验图表** |
| 当前进度 | 算法原型 ✅、对比/消融/少样本实验 ✅、Streamlit 平台 10 视图 ✅、论文初稿（docx）已交付老师审核待反馈 |
| 技术栈 | Python + PyTorch + Streamlit + Plotly + scikit-learn |

**核心方法一句话**：用扩散模型（DDPM）训练只"认识正常"的一维 U-Net 去噪网络；检测时给一段时序加噪→去噪→重建，重建误差越大越可能是异常（异常段模型"画不准"）。辅以数据增强、条件扩散（均值/标准差/最大/最小 4 维条件）、多尺度重建三大优化，使其在**少样本（训练正常样本极少）**场景下仍优于 AE/VAE/AnoGAN/孤立森林等基线。

---

## 2. 运行环境与启动（⚠️ 有坑，务必先看）

### 2.1 Python 解释器
- 系统可用解释器：`D:\Scripts\python.exe`（项目全程用此路径跑 torch）
- 注意：本机另有 managed 解释器 `C:\Users\21020\.workbuddy\binaries\python\versions\3.13.12\python.exe`，但本项目依赖与 torch DLL 是按 `D:\Scripts\python.exe` 环境配的，**优先用 `D:\Scripts\python.exe`**。

### 2.2 🔴 最关键的环境坑：torch 必须配 `PYTHONHOME='D:\'`
当前 `D:\Scripts\python.exe` 的 `sys.prefix` 形如 `D:`，torch 拼接 DLL 路径时会失败。
**直接 `python run_app.py` 或 `streamlit run ...` 会报 torch 加载错。**
- 正确做法：`run_app.py` 已内置 `_fix_pythonhome()` 自动修复，因此**一律用 `python run_app.py` 启动**，不要直接 `streamlit run`。
- 若需在脚本/命令行里加载 torch（如训练、验证、AppTest）：前置 `PYTHONHOME='D:\'`，例如：
  ```bash
  cd /c/Users/21020/OneDrive/桌面/thesis
  PYTHONHOME='D:\\' D:/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.port 8501
  ```

### 2.3 依赖
见 `requirements.txt`（torch>=2.2, torchvision, numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, streamlit>=1.32, plotly, pyyaml, tqdm, python-docx, openpyxl）。已安装，通常无需重装。

### 2.4 启动方式
```bash
cd /c/Users/21020/OneDrive/桌面/thesis
D:/Scripts/python.exe run_app.py        # 自动修复 PYTHONHOME、无控制台窗口启动、就绪后自动开浏览器
```
启动后访问 **http://localhost:8501**。首次加载 torch 较慢（约 10~30 秒首屏）。
桌面启动器：`异常检测平台.lnk`（已配置 wscript 隐藏窗口 + 自定义 icon.ico + 自动开浏览器，调用 `run_app.py`）。

### 2.5 关键配置 `configs/default.yaml`（数值务必记住）
- `data.window_size: 100`（**训练窗口大小，见 §7 坑②**）
- `data.stride: 5`、`data.few_shot_ratio: 1.0`（0.1=只用 10% 训练样本）
- `diffusion.n_timesteps: 200`、`schedule: cosine`
- `unet.cond_dim: 4`（条件向量维度 = 均值/标准差/最大/最小）
- `train.epochs: 50`、`train.device: "cpu"`
- `experiment.use_augmentation/use_conditional/use_multiscale: true`（三大优化开关，消融实验用）
- `experiment.threshold_percentile: 95`（异常分数阈值百分位）

---

## 3. 目录结构

```
thesis/
├── HANDOVER.md                # 本文件
├── README.md
├── requirements.txt
├── run_app.py                 # 一键启动（含 PYTHONHOME 修复 + 无窗口 + 自动开浏览器）
├── configs/
│   └── default.yaml           # 全局配置（见 §2.5）
├── app/
│   ├── streamlit_app.py       # ★ 主页（异常检测），含全部可复用工具函数
│   ├── _shared.py             # ★ 共享评估模块（需 torch 的页面才导入）
│   └── pages/                 # Streamlit 多页面（左侧导航自动生成）
│       ├── 01_使用教程.py
│       ├── 02_关于版本.py
│       ├── 03_数据集中心.py
│       ├── 04_模型对比实验室.py
│       ├── 05_少样本分析.py
│       ├── 06_消融实验.py
│       ├── 07_异常可解释性.py
│       ├── 08_论文图表画廊.py
│       └── 09_训练中心.py
├── src/                       # 算法核心（勿随意改接口）
│   ├── models/                # diffusion, unet_1d, autoencoder, vae, anogan, isolation_forest
│   ├── data/                  # ucr_loader.py（UCR 数据集）、augmentation.py
│   └── utils/config.py        # load_config
├── scripts/                   # run_experiment.py（训练入口）、各类 figure 生成脚本
├── data/raw/                  # 8 个 UCR SYNTHETIC 子集（000~007，文件名含异常区间）
├── experiments/
│   ├── results/               # ★ 模型权重 + 预计算实验 JSON
│   └── figures/               # 论文用图表 PNG（画廊页读取）
└── paper/                     # 论文相关
    ├── 学年论文.docx          # 论文初稿（已交付老师审核）
    ├── draft.md               # 初稿 Markdown 源
    ├── 毕业论文增强方案.md
    ├── results_dashboard.html
    └── figures/               # pipeline / fewshot_auroc / multidataset_* 配图
```

---

## 4. 系统架构：10 视图（Streamlit 多页面）

Streamlit 多页面机制：`app/streamlit_app.py` 是主页，`app/pages/NN_xxx.py` 自动成为左侧导航独立页面。

| # | 文件 | 视图 | 职责 | 是否加载 torch |
|---|---|---|---|---|
| 主页 | `streamlit_app.py` | 异常检测 | 控制面板（数据源/窗口/算法/多尺度/档位）+ 检测 + 指标卡 + 时序/分数图 + 对比/消融表 + CSV 导出 +「下钻到可解释性」按钮 | 是 |
| 01 | `01_使用教程.py` | 使用教程 | **先系统说明**（是什么/能做什么/怎么实现），**再全展开教程**（数据源/参数逐项详解/结果解读/FAQ） | 否 |
| 02 | `02_关于版本.py` | 关于版本 | 版本号 v1.0.0 + 依赖 + 5 权重自检（✅/❌）+ 论文信息 + BibTeX | 否 |
| 03 | `03_数据集中心.py` | 数据集中心 | UCR 8 子集元信息（长/异常起止/占比）+ Plotly 波形预览 +「送入检测」跳主页 | 否 |
| 04 | `04_模型对比实验室.py` | 模型对比实验室 | 读预计算对比 JSON 出表/柱状图；可选「实时复现」现场算 5 模型 AUROC | 实时模式才加载 |
| 05 | `05_少样本分析.py` | 少样本分析 | AUROC vs 少样本比例折线（论文 headline）+ 数值表 + 解读 | 否（读 JSON） |
| 06 | `06_消融实验.py` | 消融实验 | 分组柱状图量化「数据增强/条件扩散/多尺度」各自贡献 | 否（读 JSON） |
| 07 | `07_异常可解释性.py` | 异常可解释性 | 选窗口→重建→逐点误差曲线 + 高亮异常驱动段（黑盒变白盒） | 是 |
| 08 | `08_论文图表画廊.py` | 论文图表画廊 | 集中预览/下载论文配图 + 一键重生成 | 否 |
| 09 | `09_训练中心.py` | 训练中心 | UI 配参 → 写临时 config → 子进程跑 `run_experiment.py` → 实时 loss 曲线 + 日志 + 可停止 | 训练子进程加载 |

---

## 5. 各文件职责与关键接口

### 5.1 主页 `app/streamlit_app.py`（核心，最常被复用）
- **样式工具**：`inject_css()`（浅色主题：`.section-title`/`.metric-card`/`.status-bar`/`.hero` 等 class，子页面通过 `import streamlit_app; streamlit_app.inject_css()` 复用）。
- **配置缓存**：`load_config_cached()`（带 `_CFG` 全局缓存，避免重复读 YAML）。
- **★ 模型加载**：`load_model(model_name, config, device)`（带 `MODEL_CACHE` 缓存，不重复构建网络）。`model_name ∈ {Diffusion, AE, VAE, AnoGAN, Isolation Forest}`。
- **★ 异常分数计算**：`compute_window_scores(model_name, tw_tensor, config, device, use_multiscale, fast_mode, use_cond, noise_ratio=None)` → 返回逐窗口异常分数（重建误差均值）。`tw_tensor` 形状 `[N,1,W]`。
- **上传解析**：`parse_uploaded_file` / `build_upload_windows`（支持 CSV/XLSX/TXT）。
- **检测主循环**：在 `main()` 内联（控制面板 + 后台线程缓动进度条 + 阈值判定 + 指标计算 + 下钻按钮）。
- **下钻**：检测结果区「🔦 下钻分析异常窗口」按钮 → `st.switch_page("app/pages/07_异常可解释性.py")` 并把 `dataset_idx` 写入 `session_state["drill_dataset_idx"]`。
- **读取数据集预设**：主页启动时读 `session_state["preset_dataset_idx"]`（数据集中心「送入检测」写入），自动选中对应数据集后清空。

### 5.2 共享模块 `app/_shared.py`
> 仅**需要模型推理**的页面（04 实时模式、07）才 `import _shared`（因它 import streamlit_app→torch）。纯展示页（03/05/06/08）**不要**导入，保首屏轻量。
- `get_config()` → `sa.load_config_cached()`
- `list_datasets(config)` / `resolve_data_dir(config)`
- `weight_exists(model_name)`（检查 `experiments/results/<权重文件>`）
- `load_test_dataset(dataset_idx, window_size, config)` → `(windows[N,W], labels[N], ds_name, scaler)`
- `evaluate_models(dataset_idx, window_size, use_multiscale, noise_ratio, config, device="cpu", max_windows=300)` → `(results_dict, ds_name)`，`results_dict` 含 5 模型 AUROC（失败为 None）
- `reconstruct_window(model_name, window_1d, scaler, config, device, use_multiscale, noise_ratio)` → `(input_1d, recon_1d, error_1d)`，原始尺度
- **权重映射**：`Diffusion→diffusion_main.pth, AE→ae.pth, VAE→vae.pth, AnoGAN→anogan.pth, Isolation Forest→iforest.pkl`

### 5.3 各子页面要点
- **01 使用教程**：结构 = 系统说明（通俗介绍/作用/原理）→ 全展开教程（数据源、参数逐项详解、结果解读、FAQ）。所有 `expander` 默认 `expanded=True`。内容贴合真实实现，可直接截图嵌论文。
- **02 关于版本**：权重自检用 `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent`（**子页面需三层 parent**，见坑①）。自检 5 文件并显示大小。
- **03 数据集中心**：文件名编码异常区间（`..._START_END_LENGTH`），解析展示；Plotly 画波形；「送入检测」写 `session_state["preset_dataset_idx"]` + `st.switch_page("app/streamlit_app.py")`。
- **04 模型对比实验室**：默认读 `experiments/results/comparison_results.json` 出表/柱状图（快）；可选「实时复现」→ `sh.evaluate_models(...)`（慢，加载 torch）。**窗口大小强制用训练尺寸，无滑块**（见坑②）。
- **05 少样本分析**：读 `experiments/results/few_shot_results.json`，keys=`"0.1"/"0.3"/"0.5"/"1.0"`，每含 `Diffusion (Ours)/AE/VAE/IF` 指标。画 AUROC vs 比例折线。
- **06 消融实验**：读 `experiments/results/ablation_results.json`，画分组柱状图（full vs no_aug vs no_cond vs no_multiscale）。
- **07 异常可解释性**：读 `session_state["drill_dataset_idx"]`（无则用下拉）；`sh.load_test_dataset` + `sh.reconstruct_window` → 画逐点误差 + 高亮 Top 异常段。**窗口大小强制训练尺寸，无滑块**。
- **08 论文图表画廊**：展示 `experiments/figures/*.png`（comparison_bar/ablation_bar/training_curve/detection_example/few_shot_curve/streamlit_screenshot），可预览/下载/一键重生成（`scripts/generate_figures.py` 等）。
- **09 训练中心**：UI 改参 → 写临时 config → `subprocess` 调 `scripts/run_experiment.py --config <tmp>` → 后台线程读 stdout → 实时 loss 曲线 + 可停止。**注意 `run_experiment.py` 只吃 `--config/--ablation/--save_dir`，不暴露 epochs 等 CLI**（见坑③）。

### 5.4 训练脚本 `scripts/run_experiment.py`
- CLI：`--config <yaml>`、`--ablation <name>`、`--save_dir <dir>`
- `--ablation` 对应 `no_aug`/`no_cond`/`no_multiscale`，关闭 `experiment` 中对应开关后训练，输出到 `comparison_results.json`/`ablation_results.json`
- 训练中心页以「UI 改默认 config → 写临时 yaml → 传 `--config`」方式间接暴露参数

---

## 6. 数据流与实验产物位置

### 6.1 模型权重（`experiments/results/`）
| 文件 | 模型 | 大小 |
|---|---|---|
| `diffusion_main.pth` | Diffusion (Ours) | ~9.06 MB |
| `ae.pth` | AE | ~0.07 MB |
| `vae.pth` | VAE | ~0.11 MB |
| `anogan.pth` | AnoGAN | ~0.17 MB |
| `iforest.pkl` | Isolation Forest | ~1.44 MB |

### 6.2 预计算实验 JSON（`experiments/results/`，分析页直接读取，无需重训）
- `comparison_results.json` —— 5 模型在 UCR 上的对比指标（04 页用）
- `few_shot_results.json` —— 4 比例 × 4 模型 AUROC（05 页用，论文 headline）
- `ablation_results.json` —— 3 组消融 AUROC（06 页用）
- `multi_dataset_results.json` —— 8 子集多数据集结果（平均 AUROC≈0.807）

### 6.3 图表
- 画廊页用：`experiments/figures/*.png`（6 张）
- 论文版：`paper/figures/{pipeline.png, fewshot_auroc.png, multidataset_auroc.png, multidataset_avg.png}`

### 6.4 数据集
- `data/raw/000~007_UCR_Anomaly_SYNTHETIC*.txt`（8 个子集，文件名末段 `_START_END_LENGTH` 为异常区间）
- 加载器：`src/data/ucr_loader.py` 的 `UCRAnomalyDataset`，`get_available_datasets(data_dir)` 列文件名

---

## 7. 🔴 已知 Bug 与坑（按严重程度排序，接手必读）

**坑①：`app/pages/*.py` 取项目根需三层 `.parent.parent.parent`**
- 主页 `streamlit_app.py` 在 `app/` 下，取根用 `os.path.dirname(os.path.dirname(__file__))`（两层）。
- 子页面在 `app/pages/` 下，取根必须 `Path(__file__).resolve().parent.parent.parent`（三层），否则会去查 `app/experiments/...` 找不到文件。
- **已修**：02/01 页面已改对。新增子页面务必照此。

**坑②：基线模型推理必须用训练窗口大小（window_size=100），不能传任意窗口**
- 根因：AE/VAE/AnoGAN/IF 按**固定训练窗口 100** 构建，线性层输入维度锁死。传非 100 的窗口 → 形状不匹配 → 异常（被 try 吞掉返回 None，表现成"该模型无结果"）。
- Diffusion 是卷积 U-Net，支持变长，所以不受限。
- **已修**：`_shared.load_test_dataset` 强制 `wsize = config["data"]["window_size"]`；04/07 页面已去掉误导性窗口滑块改用训练尺寸。
- **⚠️ 仍存在的隐患**：主页检测页若用户手动把侧边栏「窗口大小」滑块改成非 100，再用 AE/VAE 检测会崩（见待办①）。

**坑③：训练脚本 CLI 不暴露 epochs/lr 等参数**
- `run_experiment.py` 只吃 `--config/--ablation/--save_dir`。想改训练超参必须经「临时 config 文件」。训练中心页已按此实现。

**坑④：torch 加载需 `PYTHONHOME='D:\'`（见 §2.2）**
- 任何加载 torch 的脚本/AppTest/训练，都需前置该环境变量，最省事是统一用 `python run_app.py` 启动。

**坑⑤：`use_container_width` 已统一换 `width='stretch'`**
- Streamlit 1.60 弃用前者，所有页面已批量替换为 `width='stretch'`，新增图表/表格请用新写法。

**坑⑥：AnoGAN 重建必须 `nullcontext()` 而非 `inference_mode()`**
- AnoGAN 的 `reconstruct` 内部有 `loss.backward()` 搜潜变量，推理模式会报错。代码已对 AnoGAN 特判 `nullcontext()`（见 `compute_window_scores` 与 `_shared.reconstruct_window`）。

---

## 8. 已验证状态（截至 2026-09-19，本次交接前最后一轮）

| 验证项 | 结果 |
|---|---|
| `py_compile` 全部 11 个 py 文件 | ✅ 全 OK |
| AppTest 10 视图渲染 | ✅ 零 exception（含 torch 页用 60s 超时） |
| headless `evaluate_models`（5 模型 AUROC 实时算出） | ✅ Diffusion 0.74 / AE 0.65 / VAE 0.42 / AnoGAN 0.30 / IF 0.48 |
| headless `reconstruct_window`（Diffusion/AE/VAE/AnoGAN 单窗口重建） | ✅ 全部 recon OK |
| 真实启动 `localhost:8501`，10 视图 HTTP 200 | ✅ 全通 |
| 权重自检（02 页） | ✅ 5 文件全 ✅（修复坑①后） |

---

## 9. 待办清单（接手优先处理，按推荐顺序）

1. **【中】主页窗口大小锁死防护**：让用户改侧边栏「窗口大小」滑块后用 AE/VAE 也不崩。建议：主页检测时对基线模型也强制用训练尺寸，或滑块选项限制为固定值。时间充足可做。
2. **【高，论文必需】把新页面截图嵌进论文**：老师要求"附系统截图"。建议用 headless 方式启动 App 对各视图（尤其主页异常检测、使用教程、模型对比、少样本、消融、可解释性）截图，插入论文 5.x 节（系统演示 / 实验分析）。已有 `paper/figures/` 放实验图，系统截图另存。
3. **【低】整理旧桌面入口**：`.bat`/旧版 `.lnk` 若仍指向旧结构，统一到 `异常检测平台.lnk`（调用 `run_app.py`）。当前 `.lnk` 已正确，主要是清理多余旧文件。
4. **【视老师反馈】论文修订**：初稿已交付审核，**等老师反馈**后据意见改 `paper/draft.md` 与 `paper/学年论文.docx`。核心实验（对比/少样本/消融）数据已全部就绪，可直接引用。
5. **【可选增强】批量检测 / 检测历史 / 报告生成（Word）**：此前规划过但未实现（本轮选了"论文核心 + 训练中心"范围）。如需更完整系统可补这三个页面。

---

## 10. 论文进度与文件

- `paper/学年论文.docx` —— 论文初稿（已交付老师审核，等待反馈）
- `paper/draft.md` —— 初稿 Markdown 源（改论文优先改这里再导出 docx）
- `paper/毕业论文增强方案.md` —— 向毕业论文扩展的备选方案
- `paper/figures/` —— pipeline / fewshot_auroc / multidataset_auroc / multidataset_avg 四张已生成配图
- `paper/results_dashboard.html` —— 实验指标看板（离线 HTML）
- **模板结构**：中英文摘要/关键词 → 正文（背景/现状/技术原理/架构设计/实验/总结）→ 参考文献。正文 ≥3000 字、查重 ≤20%。

---

## 11. 给接手者的"第一件事"建议

1. 先 `cd` 到项目根，用 `D:/Scripts/python.exe run_app.py` 启动，浏览器打开 http://localhost:8501，**亲手点一遍 10 个视图**，建立整体印象（耗时约 5 分钟，首屏含 torch 加载）。
2. 改任何代码前，先 `PYTHONHOME='D:\' D:/Scripts/python.exe -m py_compile <文件>` 确认能编译（torch 页用 AppTest 60s 超时验证）。
3. 真正要动算法/接口前，读 `src/models/*.py` 与 `src/data/ucr_loader.py` 的接口（主页与 `_shared` 已严格按这些接口复用，改模型类签名会牵连多个页面）。
4. 论文截图（待办②）是当前最紧迫的"交付物"之一，建议优先做。

---

## 12. 用户偏好（协作方式）

- 用户称呼本助手为"宝宝"；偏好「陈述需求 → 直接拿结果」，排斥冗长排查步骤与分步提问。
- 交付时给结论/方案概要，再展开细节；重视即时产出与高交互体验。
- 当前重点是**保 9/30 截止 + 论文通过查重 + 系统截图齐备**。

---
*本文件由上一轮开发收尾时生成，覆盖项目全貌。如与代码实际不一致，以代码 + `.workbuddy/memory/MEMORY.md` 为准。*
