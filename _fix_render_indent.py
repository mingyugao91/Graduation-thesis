# -*- coding: utf-8 -*-
"""修复 _render_detection 函数体缩进：body 部分（从「结果展示」注释起）相对缩进对齐到最小 4 空格。"""
PATH = r"C:\Users\21020\OneDrive\桌面\thesis\app\streamlit_app.py"
with open(PATH, encoding="utf-8") as f:
    lines = f.read().split("\n")

def_start = next(i for i, l in enumerate(lines) if l.strip().startswith("def _render_detection("))
main_idx = next(i for i, l in enumerate(lines) if l.strip() == 'if __name__ == "__main__":')
func_lines = lines[def_start + 1:main_idx]

rel = next(i for i, l in enumerate(func_lines) if "# ===== 结果展示 =====" in l)
header = func_lines[:rel]
body = func_lines[rel:]
min_ind = min(len(l) - len(l.lstrip()) for l in body if l.strip())
new_body = []
for l in body:
    if l.strip() == "":
        new_body.append("")
    else:
        s = l.lstrip()
        ci = len(l) - len(s)
        new_body.append(" " * (ci - min_ind + 4) + s)

lines[def_start + 1:main_idx] = header + new_body
with open(PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"已修复 _render_detection 缩进：body 最小缩进由 {min_ind} 对齐到 4。")
