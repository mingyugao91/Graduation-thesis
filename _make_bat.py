# 生成桌面双击启动器（UTF-8 BOM 避免中文路径乱码）
content = (
    "@echo off\r\n"
    "chcp 65001 >nul\r\n"
    "set PYTHONHOME=D:\\\r\n"
    'start "" "D:\\Scripts\\python.exe" "C:\\Users\\21020\\OneDrive\\桌面\\thesis\\run_app.py"\r\n'
    "exit\r\n"
)
desktop = r"C:\Users\21020\OneDrive\桌面"
out = desktop + r"\启动异常检测系统.bat"
with open(out, "w", encoding="utf-8-sig") as f:
    f.write(content)
print("written:", out)
with open(out, "rb") as f:
    print("BOM ok:", f.read(3) == b"\xef\xbb\xbf")
with open(out, "r", encoding="utf-8-sig") as f:
    print("---- content ----")
    print(f.read())
