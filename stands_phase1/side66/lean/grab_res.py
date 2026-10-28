import sys, time, os, json
from playwright.sync_api import sync_playwright
OUT = "/home/trim/wb2lean/pw/out"; sid = sys.argv[1]
URL = "https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/%s" % sid
got = []
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context("/home/trim/pw_aicrowd", headless=False, viewport={"width": 1500, "height": 1000})
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    def on_resp(r):
        try:
            ct = r.headers.get("content-type", "")
            if "json" in ct or r.url.endswith(".json") or "text" in ct:
                b = r.body()
                if b and b"residual_wall_time_s" in b:
                    fn = "%s/res_%s_%d.bin" % (OUT, sid, len(got)); open(fn, "wb").write(b); got.append((r.url, len(b), fn))
        except Exception as e:
            pass
    page.on("response", on_resp)
    page.goto(URL, wait_until="domcontentloaded", timeout=120000)
    t0 = time.time()
    while time.time() - t0 < 150 and not got: time.sleep(3)
    time.sleep(5); print("знайдено:", got, flush=True); ctx.close()
