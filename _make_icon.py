# -*- coding: utf-8 -*-
"""生成贴合系统的图标 icon.ico：深蓝圆角底 + 浅蓝时间序列折线 + 红色异常点。"""
from PIL import Image, ImageDraw

SIZE = 256
OUT = r"C:\Users\21020\OneDrive\桌面\thesis\icon.ico"

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 圆角深蓝背景
d.rounded_rectangle([14, 14, SIZE - 14, SIZE - 14], radius=52, fill=(21, 49, 79, 255))

# 时间序列折线（模拟波动）
pts = [(44, 182), (82, 150), (120, 168), (160, 120), (198, 142), (214, 92)]
d.line(pts, fill=(127, 209, 255, 255), width=11, joint="curve")

# 异常点（红色高亮 + 光晕）
ax, ay = 160, 120
d.ellipse([ax - 28, ay - 28, ax + 28, ay + 28], fill=(255, 77, 79, 55))
d.ellipse([ax - 15, ay - 15, ax + 15, ay + 15], fill=(255, 77, 79, 255))

# 其余正常点：浅色小圆点
for (x, y) in pts:
    if (x, y) != (ax, ay):
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(222, 240, 255, 235))

img.save(OUT, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
print("icon saved ->", OUT)
