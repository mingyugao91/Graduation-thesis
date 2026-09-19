"""
一键启动「基于扩散模型的少样本时间序列异常检测」可视化平台。

特性：
- 自动修复 PYTHONHOME（D:\\Scripts 的 torch DLL 坑）
- 以无控制台窗口方式启动 Streamlit 子进程（Windows 下 CREATE_NO_WINDOW）
- 服务就绪后自动用默认浏览器打开 http://localhost:8501
- 运行日志写入项目根目录 launch.log，便于排查

用法：
    python run_app.py      # 由桌面 .vbs 启动器隐藏调用
"""
import os
import sys
import time
import threading
import subprocess
import urllib.request
import webbrowser


def _fix_pythonhome():
    """修复 sys.prefix 形如 'D:' 时 torch 拼 DLL 路径失败的问题。"""
    prefix = sys.prefix
    if not prefix:
        return
    if prefix.endswith(":") or (len(prefix) <= 3 and not os.path.isdir(prefix)):
        fixed = prefix.rstrip("\\") + "\\"
        # 直接覆盖，确保子进程一定拿到正确值
        os.environ["PYTHONHOME"] = fixed
        if fixed not in sys.path:
            sys.path.insert(0, fixed)


def _log(msg, log_path):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except Exception:
        pass


def _open_browser_when_ready(url, log_path, timeout=40.0):
    """轮询直到 Streamlit 就绪，然后用默认浏览器打开。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        _log("浏览器就绪检测超时，请手动打开 " + url, log_path)
        return
    try:
        ok = webbrowser.open(url, new=1, autoraise=True)
        _log("已自动打开浏览器: " + url + (" (成功)" if ok else " (调用失败，请手动打开)"),
             log_path)
    except Exception as e:
        _log("自动打开浏览器失败: " + str(e) + " 请手动打开 " + url, log_path)


def main():
    _fix_pythonhome()
    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    log_path = os.path.join(here, "launch.log")
    cmd = [py, "-m", "streamlit", "run", "app/streamlit_app.py",
           "--server.headless", "true", "--server.port", "8501"]
    _log(">>> 启动命令: " + " ".join(cmd), log_path)

    # Windows 下让 Streamlit 子进程也无控制台窗口
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            cmd, cwd=here, creationflags=flags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        _log("[错误] 未找到 streamlit，请先安装依赖：pip install -r requirements.txt",
             log_path)
        return 1

    # 服务就绪后自动打开浏览器（放后台线程，不阻塞 Streamlit）
    t = threading.Thread(target=_open_browser_when_ready,
                         args=("http://localhost:8501", log_path), daemon=True)
    t.start()

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    _log(">>> Streamlit 已退出", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
