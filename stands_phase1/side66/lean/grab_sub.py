"""Headed Playwright: open AIcrowd submission pages, wait for user (login/captcha),
dump innerText + HTML + screenshot for each id. Usage: grab_sub.py ID [ID...]"""
import sys, time, os, re
from playwright.sync_api import sync_playwright
OUT = "/home/trim/wb2lean/pw/out"; os.makedirs(OUT, exist_ok=True)
BASE = "https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/"
ids = sys.argv[1:]
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context("/home/trim/pw_aicrowd", headless=False,
            viewport={"width":1500,"height":1000}, args=["--start-maximized"])
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    for sid in ids:
        page.goto(BASE + sid, wait_until="domcontentloaded", timeout=120000)
        print(f"[{sid}] opened; waiting for ledger text (login/captcha in window if needed)", flush=True)
        t0 = time.time(); ok = False
        while time.time() - t0 < 1500:
            try:
                txt = page.evaluate("document.body.innerText")
            except Exception:
                txt = ""
            if re.search(r"WALL|TIME BREAKDOWN|FLOPS", txt):
                ok = True; break
            time.sleep(3)
        print(f"[{sid}] ledger visible={ok} after {time.time()-t0:.0f}s", flush=True)
        # try to expand everything: click buttons with 'more'/'all'/'expand'
        for pat in ["show all", "show more", "load more", "expand", "view all"]:
            try:
                for b in page.get_by_text(re.compile(pat, re.I)).all()[:5]:
                    b.click(timeout=2000); time.sleep(1)
            except Exception: pass
        for _ in range(30):
            page.mouse.wheel(0, 1500); time.sleep(0.3)
        time.sleep(2)
        # hover every table row to trigger tooltips, collect tooltip text
        tips = []
        try:
            if os.environ.get('SKIP_HOVER'): raise RuntimeError('skip')
            rows = page.locator("tr, [role=row]").all()
            for r in rows[:120]:
                try:
                    r.hover(timeout=1500); time.sleep(0.25)
                    for sel in ["[role=tooltip]", ".tooltip", "[class*=tooltip]", "[class*=Tooltip]", "[class*=popover]"]:
                        for tnode in page.locator(sel).all():
                            t = tnode.inner_text().strip()
                            if t and t not in tips: tips.append(t)
                except Exception: pass
        except Exception as e:
            print("hover err", e)
        txt = page.evaluate("document.body.innerText")
        html = page.content()
        open(f"{OUT}/{sid}.txt","w").write(txt)
        open(f"{OUT}/{sid}.html","w").write(html)
        open(f"{OUT}/{sid}_tips.txt","w").write("\n=====\n".join(tips))
        try: page.screenshot(path=f"{OUT}/{sid}.png", full_page=True)
        except Exception as e: print("shot err", e)
        print(f"[{sid}] saved: text {len(txt)} chars, html {len(html)}, tips {len(tips)}", flush=True)
    print("DONE_ALL", flush=True)
    time.sleep(5)
    ctx.close()
