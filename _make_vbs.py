# -*- coding: utf-8 -*-
"""在桌面生成「启动异常检测系统.vbs」——双击无黑窗口、自动开浏览器。"""
import os
content = (
    'Set ws = CreateObject("WScript.Shell")\n'
    'ws.Run """D:\\Scripts\\python.exe"" '
    '""C:\\Users\\21020\\OneDrive\\桌面\\thesis\\run_app.py""", 0, False\n'
)
desktop = r"C:\Users\21020\OneDrive\桌面"
path = os.path.join(desktop, "启动异常检测系统.vbs")
with open(path, "w", encoding="utf-16") as f:
    f.write(content)
print("written:", path)
with open(path, "rb") as f:
    head = f.read(2)
print("UTF-16 BOM ok:", head == b"\xff\xfe")
