"""Генерує estimator_v30k31.py = v29 + memoryless κ₃₁ (3 моди: D21·(D3/var), C_off·g4row, C_off; таблиця 15×3 = середнє LS по 8 публічних mini)."""
import json, numpy as np, re
tabs = [json.load(open("memcoef_%d.json" % i)) for i in range(8)]
T = {l: np.mean([t[str(l)] for t in tabs], axis=0) for l in range(1, 16)}
sd = {l: np.std([t[str(l)] for t in tabs], axis=0)/np.abs(T[l]) for l in range(1, 16)}
print("розкид між мережами (sd/|mean|):", " ".join("l%d:%s" % (l, "/".join("%.2f" % v for v in sd[l])) for l in (1, 4, 8, 12, 15)))
rows = ",\n".join("    [%.6e, %.6e, %.6e]" % tuple(T[l]) for l in range(1, 16))
src = open("/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py").read()
old = "                    wk431 = None if trim else fnp.multiply(C_off, (0.5 * METRIC_C * lam_prev), out=NN(\"wk431\"))"
assert src.count(old) == 1
new = """                    if trim or K31_MODE == 0:
                        wk431 = None if trim else fnp.multiply(C_off, (0.5 * METRIC_C * lam_prev), out=NN("wk431"))
                    else:
                        # V30 (18.09.2026, trim_qewas): memoryless (3,1)-slice of kappa4 on live modes
                        # kappa31 ~ a1 * D21 * (D3/var)_i + a2 * C_off * g4row_i + a3 * C_off   (LS on 8 public mini MLPs,
                        # lean chain: R2 0.41 (L1) -> 0.83 (L15) vs 0.40 -> 0.60 for the lambda*C_off form; true kappa31 = x1.22 raw)
                        k31c = K31_TAB[min(li, len(K31_TAB)) - 1]
                        rw = fnp.multiply(D3 / var, k31c[0])[:, None]
                        wk431 = fnp.multiply(D21, rw, out=NN("wk431"))
                        rw2 = fnp.add(fnp.multiply(g4row, k31c[1]), k31c[2])[:, None]
                        wk431 = fnp.add(wk431, fnp.multiply(C_off, rw2), out=NN("wk431"))
                        wk431 = _zero_diag(wk431)"""
src = src.replace(old, new)
anchor = "NO_WK431 = _os.environ.get(\"V17_NO_WK431\", \"0\") == \"1\""
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + """
K31_MODE = int(_os.environ.get("V30_K31", "1"))
K31_TAB = [   # rows = layer 1..15: [a1 (D21*D3/var), a2 (C_off*g4row), a3 (C_off)]
%s
]""" % rows)
src = src.replace('"""K3-simple factored', '"""V30 = V29 + memoryless kappa31 (trim_qewas, 18.09.2026). Below: V29 docstring.\n\nK3-simple factored', 1)
open("estimator_v30k31.py", "w").write(src); print("written estimator_v30k31.py; table rows:", len(T))
