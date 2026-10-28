import sys, time, os, re
from playwright.sync_api import sync_playwright
OUT="/home/trim/wb2lean/pw/out"; os.makedirs(OUT, exist_ok=True)
URL="https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards"
with sync_playwright() as p:
    ctx=p.chromium.launch_persistent_context("/home/trim/pw_aicrowd", headless=False, viewport={"width":1500,"height":1000})
    page=ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120000)
    t0=time.time()
    while time.time()-t0<900:
        txt=page.evaluate("document.body.innerText")
        if re.search(r"joe_wanza|J2W|marius", txt): break
        time.sleep(3)
    for _ in range(20): page.mouse.wheel(0,2000); time.sleep(0.3)
    txt=page.evaluate("document.body.innerText"); html=page.content()
    open(f"{OUT}/lb.txt","w").write(txt); open(f"{OUT}/lb.html","w").write(html)
    links=sorted(set(re.findall(r'/submissions/(\d+)', html)))
    print("submission links:", len(links), links[:60]); print("DONE_LB", flush=True)
    ctx.close()
