"""equil104 — РОЗКЛАД НАКОПИЧЕНОЇ ПОХИБКИ ПО ШАРАХ ПОХОДЖЕННЯ.

ЗВІДКИ: питання ЮЗЕРА про турбіни реактивного двигуна. Математика справді та
сама — moment closure problem (ієрархія Фрідмана-Келлера). Але корисним
виявився не бік турбулентності, а бік САМОЇ ТУРБІНИ: багатоступенева машина
має `repeating-stage condition` — асимптотичний режим, у якому профіль на
вході ступеня = профіль на виході. Наша мережа рівно такий каскад, і ми це
вже зміряли, не знаючи назви:
    похибка m НАСИЧУЄТЬСЯ 0.0046%(ш0) · 0.526%(ш4) · 0.793%(ш12) · 0.907%(ш30)
    frozen 0.943 · eff_rank → 2.8 · cos(h,h′) → 0.974

ЩО МІРЯЄМО: чи є накопичена похибка ЛІНІЙНО ПЕРЕНЕСЕНОЮ СУМОЮ крокових.
    крок:     b_l = крок(m_l^іст, C_l^іст) − m_{l+1}^іст     (ВЕКТОР, не норма)
    якобіан:  ∂m'/∂mp = Φ(u) + u·φ(u)·(1−gam)                (вивід аналітичний)
    перенос:  δm_{l+1} = D_l ⊙ (δm_l @ W_l) + b_l
    ⟹ δm_31^pred порівнюємо з РЕАЛЬНИМ δm_31 = m^наше − m^іст

🔒 ПЛОМБА, ЯКА ВИРІШУЄ ВСЕ: якщо лінійна теорія вірна, `δm_pred` мусить збігтися
   з `δm_real` І ЗА НОРМОЮ, І ЗА НАПРЯМКОМ (cos). Якщо ні — похибка НЕ є сумою
   крокових, і весь клас «виправити крок» мертвий уже не за величиною, а
   структурно (це вже підозрюється: крок 0.139–0.162% проти накопиченої 0.668%).

ЩО ЦЕ ДАЄ, ЯКЩО ПЛОМБА ПРОЙДЕ: `b_l` — це γ₄-член Еджворта, і його можна
   порахувати БЕЗ істини. Тоді `δm = Σ перенесене b_l` віднімається як явна
   поправка. Це не ручка і не поправка до кроку — це обчислення накопичення.

ЩО ЦЕ ДАЄ, ЯКЩО ПЛОМБА ВПАДЕ: закриває клас остаточно і показує, що працює
   канал C (похибка в C впливає на m нелінійно), а не канал m.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - cos(δm_pred, δm_real) > 0.8  (лінійна теорія тримає):        40%
   - ‖δm_pred‖/‖δm_real‖ у смузі 0.5…2.0:                         45%
   - внесок останніх 4 шарів > 50% усієї похибки:                 55%
   - розклад виявиться РІВНОМІРНИМ по шарах 9..31:                25%
"""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NPAIRS = 131072
CHUNK = 4096
KSIG, KAP, GAM = 8, 0.9985, 1.048
OUT = "/home/trim/wblogs/equil104.json"

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не рівно одну карту — пінування зламане")
    sys.exit(2)
state = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    with open(OUT + ".tmp", "w") as fh:
        json.dump(state, fh)
    os.replace(OUT + ".tmp", OUT)


def stream(nmax):
    got = 0
    for fp in sorted(glob.glob(SRC + "/data-*.arrow")):
        if got >= nmax:
            break
        for b in pa.ipc.open_stream(pa.memory_map(fp, "rb")):
            if got >= nmax:
                break
            xw = b.column("weights")
            while not pa.types.is_primitive(xw.type):
                xw = xw.flatten()
            xt = b.column("final_means")
            while not pa.types.is_primitive(xt.type):
                xt = xt.flatten()
            nb = b.num_rows
            wv = np.asarray(xw).reshape(nb, DEPTH, W_, W_)
            tv = np.asarray(xt).reshape(nb, W_)
            for j in range(min(nb, nmax - got)):
                yield np.array(wv[j], dtype=np.float32), \
                    np.array(tv[j], dtype=np.float64)
            got += min(nb, nmax - got)
            del xw, xt, wv, tv


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def step(m, C, w, i):
    """Один крок НАШОГО замикання. Повертає (m', D), де D = ∂m'/∂mp."""
    cpm = w.T @ C @ w
    if i > KSIG:
        cpm = cpm * KAP
    s2 = cp.maximum(cp.diagonal(cpm), 1e-30)
    s = cp.sqrt(s2)
    mp = m @ w
    u = GAM * (mp / s)
    Ph, ph = ndtr(u), phi(u)
    mn = mp * Ph + s * ph
    D = Ph + u * ph * (1.0 - GAM)          # 🔒 аналітичний вивід, gam=1 ⟹ D=Φ
    if i == DEPTH - 1:                     # 🚨 ЦІЛЬ — це E[ReLU(p_31)], а не
        return mn, cpm, D                  #    E[p_31]: пломба приладу дала 72%
    Co = (cpm * cp.outer(Ph, Ph)
          + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
    d = (mp * mp + s2) * Ph + (mp * s) * ph - mn * mn
    Co = Co + cp.diag(d - cp.diagonal(Co))
    return mn, Co, D


def one_net(net, Wn, T):
    Wg = [cp.asarray(Wn[L]) for L in range(DEPTH)]
    W64 = [w.astype(cp.float64) for w in Wg]
    N = 2 * NPAIRS
    rng = cp.random.default_rng(1104 + net)
    S1 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    S2 = cp.zeros((DEPTH, W_, W_), dtype=cp.float64)
    OUT1 = cp.zeros(W_, dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            a64 = a.astype(cp.float64)
            S1[i] += a64.sum(0)             # вхід шару i (h_i); i=0 ⟹ z
            S2[i] += a64.T @ a64
            p = a @ Wg[i]
            if i < DEPTH - 1:
                a = cp.maximum(p, 0.0)
            else:
                OUT1 += cp.maximum(p, 0.0).astype(cp.float64).sum(0)
        done += nb
    del z, a, a64, p
    cp.get_default_memory_pool().free_all_blocks()
    mt = S1 / N                                   # істинні m_i (вхід шару i)
    Ct = S2 / N - mt[:, :, None] * mt[:, None, :]
    Ct = 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1)))
    out_t = OUT1 / N                              # істинний вихід E[p_31]
    del S1, S2
    cp.get_default_memory_pool().free_all_blocks()

    # ---- b_l: похибка ОДНОГО кроку при ТОЧНИХ входах, як ВЕКТОР
    B, Dl = [], []
    for i in range(DEPTH):
        mn, _, D = step(mt[i], Ct[i], W64[i], i)
        tgt = out_t if i == DEPTH - 1 else mt[i + 1]   # ціль з ReLU на обох
        B.append(mn - tgt)
        Dl.append(D)

    # ---- наша повна рекурсія (для РЕАЛЬНОЇ похибки)
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(DEPTH):
        m, C, _ = step(m, C, W64[i], i)
    d_real = m - out_t

    # ---- лінійний перенос: δm_{l+1} = D ⊙ (δm_l @ W) + b_l
    def transport(src):
        d = cp.zeros(W_, dtype=cp.float64)
        for i in range(DEPTH):
            if i > 0:
                d = Dl[i] * (d @ W64[i])
            if src is None or src == i:
                d = d + B[i]
        return d

    d_pred = transport(None)
    contrib = [float(cp.linalg.norm(transport(i))) for i in range(DEPTH)]
    nrm = float(cp.linalg.norm(out_t))
    r = {"real": float(cp.linalg.norm(d_real)) / nrm,
         "pred": float(cp.linalg.norm(d_pred)) / nrm,
         "cos": float((d_real @ d_pred)
                      / cp.maximum(cp.linalg.norm(d_real)
                                   * cp.linalg.norm(d_pred), 1e-300)),
         "bnrm": [float(cp.linalg.norm(B[i])) / nrm for i in range(DEPTH)],
         "contrib": [c / nrm for c in contrib],
         "cosT": float((m @ T) / cp.maximum(
             cp.linalg.norm(m) * cp.linalg.norm(T), 1e-300)),
         "dsT": float(cp.linalg.norm(out_t - T) / nrm)}
    del Wg, W64, B, Dl, Ct, mt
    cp.get_default_memory_pool().free_all_blocks()
    return r


print(f"equil104 · {NNET} мереж × {2*NPAIRS} семплів", flush=True)
t0 = time.time()
for k, (wn, tn) in enumerate(stream(NNET)):
    if str(k) in state:
        continue
    state[str(k)] = one_net(k, wn, cp.asarray(tn))
    save()
    s = state[str(k)]
    print(f"  ✅ мережа {k} ({time.time()-t0:.1f} с) · реальна "
          f"{100*s['real']:.4f}% · передбачена {100*s['pred']:.4f}% · "
          f"cos {s['cos']:+.4f}", flush=True)

# ------------------------------------------------------------------ аналіз
ns = sorted(state, key=int)
n = len(ns)


def av(f):
    return sum(f(state[k]) for k in ns) / n


print(f"\n🔒 ПЛОМБА ПРИЛАДУ: ‖істина_МК − final_means‖/‖·‖ = "
      f"{100*av(lambda s: s['dsT']):.4f}%  (шум МК, мусить бути ≪ похибки)")
print(f"\n{'величина':<34}{'значення'}")
print(f"{'РЕАЛЬНА похибка ‖δm‖/‖m‖':<34}{100*av(lambda s: s['real']):.4f}%")
print(f"{'ПЕРЕДБАЧЕНА лінійним переносом':<34}"
      f"{100*av(lambda s: s['pred']):.4f}%")
print(f"{'відношення pred/real':<34}"
      f"{av(lambda s: s['pred'])/av(lambda s: s['real']):.4f}")
print(f"{'🔒 cos(pred, real)':<34}{av(lambda s: s['cos']):+.4f}")

print(f"\n{'шар':>4} {'‖b_l‖ (крок)':>13} {'внесок у вихід':>15} {'частка':>8}")
tot = sum(av(lambda s: s["contrib"][i]) for i in range(DEPTH))
for i in list(range(0, 9)) + list(range(12, DEPTH, 4)) + [DEPTH - 1]:
    c = av(lambda s: s["contrib"][i])
    print(f"{i:>4} {100*av(lambda s: s['bnrm'][i]):>12.4f}% "
          f"{100*c:>14.4f}% {100*c/max(tot,1e-300):>7.2f}%")
last4 = sum(av(lambda s: s["contrib"][i]) for i in range(DEPTH - 4, DEPTH))
print(f"\nвнесок ОСТАННІХ 4 шарів: {100*last4/max(tot,1e-300):.1f}% "
      f"(передбачення до прогону: >50% з шансом 55%)")
cc = av(lambda s: s["cos"])
rr = av(lambda s: s["pred"]) / av(lambda s: s["real"])
if cc > 0.8 and 0.5 < rr < 2.0:
    print("\n🟢 ПЛОМБА ПРОЙДЕНА: похибка Є лінійно перенесеною сумою крокових")
    print("   ⟹ `b_l` рахується без істини (γ₄-член) ⟹ поправку МОЖНА відняти")
else:
    print("\n⛔ ПЛОМБА ВПАЛА: похибка НЕ є сумою крокових, перенесених лінійно")
    print("   ⟹ клас «виправити крок» мертвий СТРУКТУРНО, а не за величиною;")
    print("      працює канал C (він входить у m нелінійно), а не канал m")
