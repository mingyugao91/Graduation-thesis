# -*- coding: utf-8 -*-
"""把 streamlit_app.py 从「单页三 tab」还原为「多页面主页」：

- 主页 = 异常检测：移除顶部 st.tabs，控制面板 + 检测主体 + 底部对比/消融实验
  全部保留在主页面内（即用户要的「核心功能异常检测」单一视图）。
- 移除 main() 中对 render_tutorial / render_about 的调用（它们已迁到 app/pages/ 下的独立页面）。
- 删除主文件里的 render_tutorial / render_about 函数定义（已迁移，避免重复）。
"""
import re

PATH = "app/streamlit_app.py"
with open(PATH, encoding="utf-8") as f:
    lines = f.readlines()

out = []
i = 0
n = len(lines)
while i < n:
    s = lines[i].rstrip("\n")
    # 删除「顶层多视图导航」注释
    if s == "    # ===== 顶层多视图导航 =====":
        i += 1
        continue
    # 删除 tabs 定义
    if s.startswith("    _tabs = st.tabs("):
        i += 1
        continue
    # 进入异常检测 tab：丢弃 with 行，内部整体降一级缩进
    if s.startswith("    with _tabs[0]:"):
        i += 1
        while i < n and not lines[i].rstrip("\n").startswith("    with _tabs[1]:"):
            cur = lines[i].rstrip("\n")
            if cur.startswith("    "):
                cur = cur[4:]
            out.append(cur)
            i += 1
        # 此时 lines[i] == '    with _tabs[1]:'，交给下方跳过逻辑
        continue
    # 跳过 tabs[1]（使用教程）/ tabs[2]（关于版本）调用块，直到页脚
    if s.startswith("    with _tabs[1]:"):
        while i < n and lines[i].rstrip("\n") != "    # 页脚":
            i += 1
        # 停在 '    # 页脚'，下一轮会被正常 append
        continue
    out.append(lines[i].rstrip("\n"))
    i += 1

src = "\n".join(out)

# 注释语义更新（已无 tab，但缓存仍用于多页面切回保留结果）
src = src.replace(
    "# 交互状态：多视图导航 + 检测结果缓存（切 tab 不丢结果）",
    "# 交互状态：检测结果缓存（切到其它页面再返回不丢结果）",
)

# 删除「说明页」注释块 + render_tutorial + render_about 两个函数定义
pat = re.compile(r"\n# .*?说明页.*?def _render_detection\(ctx\):", re.DOTALL)
src = pat.sub("\n\ndef _render_detection(ctx):", src, count=1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("refactor done; render_tutorial/render_about removed from main; tabs removed")
