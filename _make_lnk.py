# -*- coding: utf-8 -*-
"""在桌面生成「异常检测平台.lnk」：自定义图标 + 隐藏启动 run_app.vbs。"""
import os
import subprocess

VBS = r'''Set ws = CreateObject("WScript.Shell")
Set sc = ws.CreateShortcut("C:\Users\21020\OneDrive\桌面\异常检测平台.lnk")
sc.TargetPath = "C:\Windows\System32\wscript.exe"
sc.Arguments = """C:\Users\21020\OneDrive\桌面\thesis\run_app.vbs"""
sc.WorkingDirectory = "C:\Users\21020\OneDrive\桌面\thesis"
sc.IconLocation = "C:\Users\21020\OneDrive\桌面\thesis\icon.ico,0"
sc.Description = "基于扩散模型的少样本时间序列异常检测平台"
sc.WindowStyle = 0
sc.Save
Set sc2 = ws.CreateShortcut("C:\Users\21020\OneDrive\桌面\异常检测平台.lnk")
WScript.Echo "OK Target=" & sc2.TargetPath & " | Icon=" & sc2.IconLocation
'''

vbs_path = r"C:\Users\21020\OneDrive\桌面\thesis\scripts\_do_lnk.vbs"
with open(vbs_path, "w", encoding="utf-16") as f:
    f.write(VBS)

r = subprocess.run(["cscript", "//nologo", vbs_path],
                  capture_output=True, text=True, encoding="utf-8", errors="replace")
print("STDOUT:", r.stdout.strip())
print("STDERR:", r.stderr.strip())
print("returncode:", r.returncode)
