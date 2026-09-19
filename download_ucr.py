# UCR Anomaly Archive 数据下载脚本
# 数据源: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
#
# 用法: python scripts/download_ucr.py
import os
import sys
import urllib.request
import zipfile
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets.zip"


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    zip_path = os.path.join(DATA_DIR, "ucr_anomaly.zip")

    # 检查是否已有数据
    existing = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if existing:
        print(f"数据目录已有 {len(existing)} 个数据文件，跳过下载。")
        return

    print(f"正在下载 UCR Anomaly Archive...")
    print(f"URL: {URL}")

    try:
        urllib.request.urlretrieve(URL, zip_path)
        print("下载完成，正在解压...")
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n请手动下载:")
        print(f"  1. 访问 {URL}")
        print(f"  2. 解压 zip 文件")
        print(f"  3. 将 .txt 文件放到 {DATA_DIR}")
        sys.exit(1)

    # 解压
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    # 清理嵌套目录
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".txt"):
                src = os.path.join(root, f)
                dst = os.path.join(DATA_DIR, f)
                if src != dst:
                    shutil.move(src, dst)

    # 清理子目录和zip
    for d in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, d)
        if os.path.isdir(full):
            shutil.rmtree(full)
    os.remove(zip_path)

    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"完成！共加载 {len(txt_files)} 个数据文件。")
    print(f"数据目录: {DATA_DIR}")


if __name__ == "__main__":
    download_and_extract()
