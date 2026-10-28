"""Чутливість фінальної MSE до сталих воріт модуля κ₄-дерев (26.09 ніч): чи має сім'я важіль узагалі."""
import os, sys, time, numpy as np, torch
os.environ["EPOCHS"] = "0"
sys.argv = ["x"]; exec(open("/home/trim/wb2lean/train_k4tree.py").read().split("for ep in range(state")[0])
class Const(torch.nn.Module):
    def __init__(self, v): super().__init__(); self.v = torch.tensor(v, device=dev, dtype=torch.float32)
    def forward(self, x): return self.v[None, :].expand(x.shape[0], 3)
for k in [0, 3]:
    nm, w, gt = prep(k); base = state["base"][nm]; print("мережа %s база %.4e" % (nm, base), flush=True)
    for typ in range(3):
        for c in [-3.0, -1.0, -0.3, 0.3, 1.0, 3.0]:
            v = [0.0, 0.0, 0.0]; v[typ] = c; gate = Const(v)
            with torch.no_grad(): r = float(run(w, gt, False, True, state["coef"][nm])) / base
            print("   тип %d c=%+.1f  ratio %.5f" % (typ + 1, c, r), flush=True)
print("DONE_SENS")
