"""embed_pt (19.10.2026, white-box-89): вшити ваги LT2 з torch state_dict (ep*.pt від train.py white-box-16) у
_LT2_BLOB естиматора. Міняються лише ключі W1 b1 Wm0 bm0 Wm1 bm1 Wo bo; форми звіряються зі старим блобом,
решта (usc es mz iz m2 i2 flags) і весь інший текст файлу лишаються байт у байт (за зразком lab/lt2_embed.py Susupro, MIT).
usage: embed_pt.py <ep.pt> <out_estimator.py> [--src estimator_sb2.py]
Пломба: ep0.pt (= відправлені ваги) мусить дати файл, ідентичний src."""
import argparse, base64, hashlib, re, sys
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch

KEYS = ("W1", "b1", "Wm0", "bm0", "Wm1", "bm1", "Wo", "bo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt"); ap.add_argument("out")
    ap.add_argument("--src", default="/home/trim/wb2lean/v39fija/estimator_sb2.py")
    a = ap.parse_args()
    src = open(a.src).read()
    sd = torch.load(a.pt, map_location="cpu")
    head, rest = src.split("\n_LT2_BLOB = {\n")
    body, tail = rest.split("\n}\n", 1)
    lines = body.split("\n")
    done = set()
    for i, ln in enumerate(lines):
        m = re.match(r'    "(\w+)": \((\([0-9, ]*\)), "(\w)", "', ln)
        assert m, ln[:60]
        k, shp, tc = m.group(1), eval(m.group(2)), m.group(3)
        if k not in KEYS:
            continue
        arr = sd[k].detach().cpu().numpy().astype("<f4")
        assert tuple(arr.shape) == tuple(shp) and tc == "f", (k, arr.shape, shp)
        lines[i] = f'    "{k}": ({tuple(int(x) for x in arr.shape)!r}, "f", "{base64.b64encode(arr.tobytes()).decode()}"),'
        done.add(k)
    assert done == set(KEYS), done
    text = head + "\n_LT2_BLOB = {\n" + "\n".join(lines) + "\n}\n" + tail
    text = text.replace('"""sb2 (white-box-89', f'"""LT2 retrained ({a.pt}) on top of sb2 (white-box-89', 1) if text != src else text
    open(a.out, "w").write(text)
    print("out", a.out, "identical_to_src", text == src, "sha256", hashlib.sha256(text.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
