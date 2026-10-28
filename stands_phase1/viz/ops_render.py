#!/usr/bin/env python3
"""17.10.2026. Рендер ops_anim.html у відео: playwright знімає кадри renderFrame(t), ffmpeg збирає mp4.
Запуск: /home/trim/notqa-venv/bin/python3 ops_render.py [fps=30] [out=viz/ops_anim.mp4]"""
import sys, os, subprocess, shutil
from playwright.sync_api import sync_playwright
here = os.path.dirname(os.path.abspath(__file__)); fps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "ops_anim.mp4"); fr = os.path.join(here, "_frames"); shutil.rmtree(fr, ignore_errors=True); os.makedirs(fr)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
    pg.add_init_script("window.__capture=true"); pg.goto("file://" + os.path.join(here, "ops_anim.html")); pg.wait_for_function("typeof window.renderFrame === 'function'")
    total = pg.evaluate("window.TOTAL"); n = int(total * fps)
    for i in range(n):
        pg.evaluate("t=>renderFrame(t)", i / fps); pg.screenshot(path=os.path.join(fr, "f%05d.png" % i))
        if i % (fps * 5) == 0: print("кадр %d/%d" % (i, n), flush=True)
    b.close()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(fr, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
shutil.rmtree(fr); print("готово", out, "%.1f с" % total)
