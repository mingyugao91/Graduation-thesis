# 用 ctypes 直接调用 Windows Shell API 在桌面生成 .lnk 快捷方式（不依赖 pywin32 / WScript.Shell COM）。
import os
import sys
import ctypes
from ctypes import wintypes, POINTER, byref, c_void_p, c_wchar_p, c_int

class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8)]
    def __init__(self, d1, d2, d3, d4):
        super().__init__(d1, d2, d3, d4)

CLSID_ShellLink = GUID(0x00021401, 0, 0, (ctypes.c_ubyte * 8)(0xC0, 0, 0, 0, 0, 0, 0, 0x46))
IID_IShellLinkW = GUID(0x000214F9, 0, 0, (ctypes.c_ubyte * 8)(0xC0, 0, 0, 0, 0, 0, 0, 0x46))
IID_IPersistFile = GUID(0x0000010B, 0, 0, (ctypes.c_ubyte * 8)(0xC0, 0, 0, 0, 0, 0, 0, 0x46))

ole32 = ctypes.windll.ole32
shell32 = ctypes.windll.shell32

ole32.CoInitialize(None)
ole32.CoCreateInstance.argtypes = [POINTER(GUID), c_void_p, ctypes.c_ulong, POINTER(GUID), POINTER(c_void_p)]
ole32.CoCreateInstance.restype = ctypes.HRESULT
shell32.SHGetFolderPathW.argtypes = [c_void_p, c_int, c_void_p, c_int, ctypes.c_wchar_p]
shell32.SHGetFolderPathW.restype = c_int

p_sl = c_void_p()
hr = ole32.CoCreateInstance(byref(CLSID_ShellLink), None, 1, byref(IID_IShellLinkW), byref(p_sl))
if hr < 0:
    print("CoCreateInstance failed hr=", hr); sys.exit(1)

vt = ctypes.cast(p_sl, POINTER(c_void_p)).contents
meth = ctypes.cast(vt, POINTER(c_void_p * 40)).contents

QIProto = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, POINTER(GUID), POINTER(c_void_p))
StrProto = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, c_wchar_p)
IntProto = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, c_int)
RelProto = ctypes.WINFUNCTYPE(ctypes.c_int, c_void_p)

QI = QIProto(meth[0])
SetPath = StrProto(meth[5])
SetDesc = StrProto(meth[7])
SetWD = StrProto(meth[9])
SetArgs = StrProto(meth[11])
SetShow = IntProto(meth[15])
Release = RelProto(meth[2])

target = r"C:\Users\21020\OneDrive\桌面\thesis\run_app.py"
workdir = r"C:\Users\21020\OneDrive\桌面\thesis"
SetPath(p_sl, target)
SetDesc(p_sl, "基于扩散模型的少样本时间序列异常检测可视化平台")
SetWD(p_sl, workdir)
SetArgs(p_sl, "")
SetShow(p_sl, 1)

p_pf = c_void_p()
hr = QI(p_sl, byref(IID_IPersistFile), byref(p_pf))
if hr < 0:
    print("QI IPersistFile failed hr=", hr); sys.exit(1)

pf_vt = ctypes.cast(p_pf, POINTER(c_void_p)).contents
pf_meth = ctypes.cast(pf_vt, POINTER(c_void_p * 10)).contents
SaveProto = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, c_wchar_p, c_int)
Save = SaveProto(pf_meth[6])
ReleasePF = RelProto(pf_meth[2])

buf = ctypes.create_unicode_buffer(260)
shell32.SHGetFolderPathW(0, 0, None, 0, buf)
desktop = buf.value
lnk_path = os.path.join(desktop, "异常检测系统.lnk")
Save(p_pf, lnk_path, 1)

ReleasePF(p_pf)
Release(p_sl)
ole32.CoUninitialize()
print("CREATED:", lnk_path)
