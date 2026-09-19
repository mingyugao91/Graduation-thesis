# -*- coding: utf-8 -*-
"""生成自包含的「结果汇总」HTML 看板（答辩用）。

聚合少样本/对比/多数据集结果，嵌入 pipeline/fewshot/multidataset 配图，
纯单文件 HTML（图片 base64 内嵌），无需联网、无需起服务。

输出: paper/results_dashboard.html
"""
import os
import json
import base64
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(PROJECT_ROOT, "experiments", "results")
FIGDIR = os.path.join(PROJECT_ROOT, "paper", "figures")
OUT = os.path.join(PROJECT_ROOT, "paper", "results_dashboard.html")


def img_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def fmt(v, d=4):
    if isinstance(v, (int, float)) and v == v:
        return ("%." + str(d) + "f") % v
    return "—"


def load_json(name):
    p = os.path.join(RESDIR, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def section_fewshot():
    data = load_json("few_shot_results.json")
    if not data:
        return "<p>暂无数据</p>"
    ratios = sorted(data.keys(), key=lambda x: float(x))
    methods = ["Diffusion (Ours)", "AE", "VAE", "IF"]
    head = "<tr><th>方法</th>" + "".join("<th>%s%%</th>" % int(float(r) * 100) for r in ratios) + "</tr>"
    rows = ""
    for m in methods:
        cells = "".join("<td>%.4f</td>" % data[r][m]["auroc"] for r in ratios if m in data[r])
        rows += "<tr><td><b>%s</b></td>%s</tr>" % (m, cells)
    img = "<img src='%s'/>" % ("data:image/png;base64," + img_b64(os.path.join(FIGDIR, "fewshot_auroc.png")).replace("data:image/png;base64,", ""))
    return "<table class='tbl'>%s%s</table><br>%s" % (head, rows, img)


def section_comparison():
    data = load_json("comparison_results.json")
    if not data:
        return "<p>暂无数据</p>"
    metrics = ["auroc", "auprc", "f1", "precision", "recall",
               "f1_pa", "precision_pa", "recall_pa", "auroc_pa"]
    methods = list(data.keys())
    head = "<tr><th>方法</th>" + "".join("<th>%s</th>" % m for m in metrics) + "</tr>"
    rows = ""
    for m in methods:
        cells = "".join("<td>%.4f</td>" % data[m].get(k, float("nan")) for k in metrics)
        rows += "<tr><td><b>%s</b></td>%s</tr>" % (m, cells)
    return "<table class='tbl'>%s%s</table>" % (head, rows)


def section_multidataset():
    data = load_json("multi_dataset_results.json")
    if not data:
        return "<p>暂无数据（评测进行中）</p>"
    methods = ["Diffusion (Ours)", "AE", "VAE", "IF"]
    head = "<tr><th>数据集</th>" + "".join("<th>%s</th>" % m for m in methods) + "<th>最优</th></tr>"
    rows = ""
    for name, ms in data.items():
        aucs = {m: ms.get(m, {}).get("auroc", float("nan")) for m in methods}
        valid = {k: v for k, v in aucs.items() if v == v}
        best = max(valid, key=lambda k: valid[k]) if valid else "—"
        cells = "".join("<td>%s</td>" % ("%.4f" % v if v == v else "—") for v in aucs.values())
        rows += "<tr><td>%s</td>%s<td><b>%s</b></td></tr>" % (name[:34], cells, best)
    img_tag = ""
    fig_path = os.path.join(FIGDIR, "multidataset_auroc.png")
    if os.path.exists(fig_path):
        img_tag = "<br><img src='data:image/png;base64," + img_b64(fig_path).replace("data:image/png;base64,", "") + "'/>"
    return "<table class='tbl'>%s%s</table>%s" % (head, rows, img_tag)


HTML = """<!DOCTYPE html>
<html lang='zh-CN'><head><meta charset='utf-8'><title>结果汇总 - 少样本扩散异常检测</title>
<style>
body { font-family: 'Microsoft YaHei','SimHei',sans-serif; margin: 24px; background: #fafbfc; color: #2c3e50; }
h1 { color: #1f4e79; border-bottom: 3px solid #1f4e79; padding-bottom: 8px; }
h2 { color: #16a085; margin-top: 32px; padding-left: 10px; border-left: 4px solid #16a085; }
table.tbl { border-collapse: collapse; margin: 12px 0; font-size: 13px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
table.tbl th, table.tbl td { border: 1px solid #e0e0e0; padding: 6px 12px; text-align: center; }
table.tbl th { background: #1f4e79; color: #fff; }
table.tbl tr:nth-child(even) { background: #f7f9fb; }
img { max-width: 100%%; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 8px 0; }
.meta { color: #7f8c8d; font-size: 12px; }
</style></head><body>
<h1>基于扩散模型的少样本时间序列异常检测 · 结果汇总</h1>
<p class='meta'>南京审计大学 · 计算机科学与技术 · 学年论文</p>

<h2>① 方法总体架构</h2>
<img src='data:image/png;base64,PIPELINE_B64'/>

<h2>② 少样本对比实验（子集 000）</h2>
FEWSHOT_HTML

<h2>③ 单数据集对比（子集 000，去噪重建 vs 基线）</h2>
COMPARISON_HTML

<h2>④ 跨数据集泛化能力（固定 000 权重跨场景推断）</h2>
MULTIDATASET_HTML

</body></html>"""


def main():
    pipeline_b64 = img_b64(os.path.join(FIGDIR, "pipeline.png")).replace("data:image/png;base64,", "")
    html = (HTML.replace("PIPELINE_B64", pipeline_b64)
                 .replace("FEWSHOT_HTML", section_fewshot())
                 .replace("COMPARISON_HTML", section_comparison())
                 .replace("MULTIDATASET_HTML", section_multidataset()))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("已生成结果看板:", OUT)
    print("  文件大小:", os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
