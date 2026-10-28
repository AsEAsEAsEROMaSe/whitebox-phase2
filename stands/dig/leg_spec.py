"""17.10.2026: чи губиться інформація при стисненні ніг молодих джерел, чи лише «відновлення» погане.
Спектр Грама ніг кожного молодого джерела (X*, Y*, M, P) на шарах 4, 8, 12: частка енергії в топ-128/256/512.
Якщо топ-256 тримає мало, жоден лінійний декодер з 256 чисел не відновить (QQᵀ уже оптимальний за нормою)."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
i = 0; d = np.load(lo.DUMP.format(i)); w = d["weights"]; st = lk.build_static_coefs(w.shape[1])
out = []
def probe(l, K3, D3, D21):
    if l not in (4, 8, 12): return
    for s in K3.sources:
        if s["P"] is None or s.get("Q") is not None: continue
        mats = [s[k] for k in K3.KEYS + ("P",) if s.get(k) is not None]
        G = sum(M.double() @ M.double().T for M in mats); ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(0); c = torch.cumsum(ev, 0) / ev.sum()
        # окремо: стовпці ноги P після зняття відомого масштабу рядків не роблять (масштаб уже в нозі)
        r = ev[1:]; cr = torch.cumsum(r, 0) / r.sum()
        out.append("шар %2d вік %d: топ-1 %.4f енергії · ЗАЛИШОК без топ-1: топ-127 %.3f · топ-255 %.3f · топ-511 %.3f · ефект. ранг залишку %.0f" % (l, s.get("age", 0), ev[0]/ev.sum(), cr[126], cr[254], cr[510], float(r.sum()**2 / (r**2).sum())))
lk.K3_PROBE = probe; lk.REGEN_STATE.clear()
lk.lean_k3_predict(w, dtype=torch.float32, statics=st, k4_aug=True); lk.K3_PROBE = None
print("\n".join(out))
