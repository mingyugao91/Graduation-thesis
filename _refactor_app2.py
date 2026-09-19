# -*- coding: utf-8 -*-
"""第二步重构：把「结果展示」内联块提取为 _render_detection(ctx)，
并在原位置改为：检测后缓存 st.session_state.det + 调用渲染；切 tab 后用 elif 复用缓存。"""
PATH = r"C:\Users\21020\OneDrive\桌面\thesis\app\streamlit_app.py"

with open(PATH, encoding="utf-8") as f:
    lines = f.read().split("\n")

rs = next(i for i, l in enumerate(lines) if l.strip() == "# ===== 结果展示 =====")
es = next(i for i, l in enumerate(lines) if l.strip() == "# ===== 实验结果对比 =====")
render_block = lines[rs:es]

# 去 12 空格公共缩进（渲染块在 run_requested 内，缩进 12）
def unindent(line, n=12):
    return line[n:] if line.startswith(" " * n) else line

body = [unindent(l, 12) for l in render_block]

header = [
    "",
    "def _render_detection(ctx):",
    '    """根据缓存的检测结果 dict 渲染检测结果视图（支持切 tab 复用）。"""',
    "    scores = ctx['scores']",
    "    sel_idx = ctx['sel_idx']",
    "    pred = ctx['pred']",
    "    test_labels = ctx['test_labels']",
    "    ds_name = ctx['ds_name']",
    "    model_name = ctx['model_name']",
    "    has_labels = ctx['has_labels']",
    "    sampled_windows = ctx['sampled_windows']",
    "    threshold = ctx['threshold']",
    "    n_eval = ctx['n_eval']",
    "    spans = ctx['spans']",
    "    window_size = ctx['window_size']",
    "    config = ctx['config']",
    "    device = ctx['device']",
    "    use_multiscale = ctx['use_multiscale']",
    "    fast_mode = ctx['fast_mode']",
    "    use_cond = ctx['use_cond']",
    "    noise_ratio = ctx['noise_ratio']",
    "    test_windows = ctx['test_windows']",
    "",
]
func_text = "\n".join(header + body)

replace_block = [
    '            st.session_state.det = dict(',
    '                scores=scores, sel_idx=sel_idx, pred=pred, test_labels=test_labels,',
    '                ds_name=ds_name, model_name=model_name, has_labels=has_labels,',
    '                sampled_windows=sampled_windows, threshold=threshold, n_eval=n_eval,',
    '                spans=spans, window_size=window_size, config=config, device=device,',
    '                use_multiscale=use_multiscale, fast_mode=fast_mode, use_cond=use_cond,',
    '                noise_ratio=noise_ratio, test_windows=test_windows)',
    '            _render_detection(st.session_state.det)',
    '        elif st.session_state.det is not None:',
    '            _render_detection(st.session_state.det)',
    '        else:',
    '            st.info("👈 请在左侧配置参数，点击「🚀 启动检测」开始分析")',
]

new_lines = lines[:rs] + replace_block + lines[es:]

src = "\n".join(new_lines)
marker = 'if __name__ == "__main__":'
idx = src.find(marker)
if idx != -1:
    out = src[:idx] + "\n" + func_text + "\n" + src[idx:]
else:
    out = src + "\n" + func_text + "\n"

with open(PATH, "w", encoding="utf-8") as f:
    f.write(out)

print("第二步重构完成：渲染段已提取为 _render_detection，并加 session_state.det 缓存。")
