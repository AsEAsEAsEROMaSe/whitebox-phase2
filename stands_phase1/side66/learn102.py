"""learn102 — ЧИ ПЕРЕДБАЧУВАНІ `a_k` З ДЕТЕРМІНОВАНИХ ОЗНАК? (ідея ЮЗЕРА)

ЛАНЦЮГ: поправка до `m` у базисі напрямків змішування має оракульну стелю
×1.4149 (K=8) / ×1.70 (K=16), 🔒чужі напрямки ×1.0167, знаковий 24/24.
АЛЕ коефіцієнти з ВИБІРКИ вбивають бленд: −4.4% (шум корельований з МК-гілкою)
або −41% (пілот, шум незалежний але ×6.9). Стиснення не рятує — градієнт у
нулі від'ємний.
⟹ ЄДИНИЙ ВИХІД: зробити `a_k` ДЕТЕРМІНОВАНИМИ. Тоді шуму немає взагалі.

🟢 ЩО ЦЕ УМОЖЛИВЛЮЄ: спліт `full` датасету має **1000 мереж** і колонку
`final_means` — ТОЧНУ істину (n_samples = 1e9). Тобто навчальна вибірка вже
лежить на диску, МК для неї не потрібна, а замикання рахується за мілісекунди.
🚨 С100 закрив мікромережу «2209 параметрів на 50 спостережень». Тут інша
арифметика: 1000 спостережень і 2–4 параметри на кожен `a_k`.

ЩО РОБИМО:
  1) замикання (κ=0.9985 при i>8, старт m=0,C=I — як у сабміті, без обрізання)
  2) eigh фінальної C → V (топ-16), λ
  3) ОРАКУЛЬНИЙ фіт a = lstsq([m | (v_k·m)v_k], final_means)
  4) ДЕТЕРМІНОВАНІ ознаки: спектр, cos(v_k,m), frozen, eff_rank по шарах,
     квантилі t, норми — усе, що замикання знає безкоштовно
  5) зберігаємо; вирок — LOO-R² передбачення a_k з ознак (окремим кроком)

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 оракульна стеля на 1000 мереж відтворить ×1.4 (K=8)              80%
 (б) знайдеться ознака з |corr| > 0.3 хоча б до одного a_k               35%
 (в) LOO-R² для a_1 перевищить 0.2                                      30%
 (г) 🔒 перемішані ознаки дадуть LOO-R² ≤ 0                             90%

ВИРОК: LOO-R² < 0.1 для всіх k ⟹ `a_k` НЕ передбачувані детерміновано,
і вся гілка «поправка в базисі змішування» закрита остаточно — інформація
є, але доставити її в бал нічим.
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KAPPA, KSIG = 0.9985, 8
KMAX = 16
PROBE = [4, 8, 16, 24, 31]
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
OUT = "/home/trim/learn102.npz"
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

if cp.cuda.runtime.getDeviceCount() != 1:
    print("ПІНУВАННЯ НЕ СПРАЦЮВАЛО", flush=True)
    sys.exit(2)

files = sorted(glob.glob(SRC + "/data-*.arrow"))
A, F, LM, CS, EFF, FRZ, TQ, NRM, IDS = [], [], [], [], [], [], [], [], []
GG, HH, YY = [], [], []
done = 0
for fp in files:
    if done >= NMAX:
        break
    tb = pa.ipc.open_stream(pa.memory_map(fp, "rb")).read_all()
    xw = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xw.type):
        xw = xw.flatten()
    Wall = np.asarray(xw).reshape(tb.num_rows, DEPTH, W_, W_)
    xt = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(xt.type):
        xt = xt.flatten()
    Tall = np.asarray(xt).reshape(tb.num_rows, W_)
    names = tb.column("mlp_name").to_pylist()
    for r in range(tb.num_rows):
        if done >= NMAX:
            break
        m = cp.zeros(W_, dtype=cp.float64)
        C = cp.eye(W_, dtype=cp.float64)
        eff, tqs = [], []
        for i in range(DEPTH):
            w = cp.asarray(np.ascontiguousarray(Wall[r][i], dtype=np.float64))
            cp_ = w.T @ C @ w
            if i > KSIG:
                cp_ = cp_ * KAPPA
            s2 = cp.maximum(cp.diagonal(cp_), 1e-30)
            s = cp.sqrt(s2)
            mp = m @ w
            t = mp / s
            Ph = ndtr(t)
            ph = cp.exp(-0.5 * t * t) / SQ2PI
            m = mp * Ph + s * ph
            C = (cp_ * cp.outer(Ph, Ph)
                 + (0.5 * (cp_ * cp_)) * cp.outer(ph / s, ph / s))
            d = (mp * mp + s2) * Ph + (mp * s) * ph - m * m
            C = C + cp.diag(d - cp.diagonal(C))
            if (i + 1) in PROBE:
                tr = float(cp.trace(C))
                eff.append(tr * tr / max(float((C * C).sum()), 1e-300))
                tqs.append([float(cp.percentile(t, q)) for q in (25, 50, 75)])
        wv, vv = cp.linalg.eigh(C)
        V = vv[:, ::-1][:, :KMAX]
        lam = cp.maximum(wv[::-1][:KMAX], 0.0)
        pj = m @ V
        B = cp.concatenate([cp.reshape(m, (W_, 1)), V * pj], axis=1)
        y = cp.asarray(Tall[r].astype(np.float64))
        G = B.T @ B
        G = G + (1e-12 * float(cp.trace(G)) / (KMAX + 1)) * cp.eye(KMAX + 1)
        h = B.T @ y
        a = cp.linalg.inv(G) @ h
        GG.append(cp.asnumpy(G))
        HH.append(cp.asnumpy(h))
        YY.append(float((y * y).sum()))
        trC = float(cp.trace(C))
        nm2 = float((m * m).sum())
        A.append(cp.asnumpy(a))
        F.append(cp.asnumpy(y))
        LM.append(cp.asnumpy(lam / max(trC, 1e-300)))
        CS.append(cp.asnumpy(cp.abs(pj) / math.sqrt(max(nm2, 1e-300))))
        EFF.append(eff)
        FRZ.append(nm2 / (nm2 + trC))
        TQ.append(np.array(tqs).ravel())
        NRM.append([math.sqrt(nm2), trC,
                    float(cp.asnumpy(cp.sqrt((cp.asarray(Tall[r]) ** 2).sum())))])
        IDS.append(names[r])
        # стеля: похибка після a₀ і після повного a
        done += 1
        if done % 100 == 0:
            print(f"  {done} мереж", flush=True)
    del Wall, Tall
    cp.get_default_memory_pool().free_all_blocks()

np.savez(OUT, A=np.array(A), F=np.array(F), LM=np.array(LM), CS=np.array(CS),
         EFF=np.array(EFF), FRZ=np.array(FRZ), TQ=np.array(TQ),
         NRM=np.array(NRM), IDS=np.array(IDS),
         GG=np.array(GG), HH=np.array(HH), YY=np.array(YY))
print(f"\nзбережено {done} мереж → {OUT}", flush=True)

A = np.array(A)
R = A[:, 1:] / A[:, :1]
print("\n" + "=" * 70)
print("ОРАКУЛЬНІ a_k/a₀ на 1000 мережах")
print(f"{'k':>3} {'медіана':>10} {'сер':>10} {'sd':>10} {'|сер|/sd':>9} {'знак+':>9}")
for k in range(KMAX):
    c = R[:, k]
    pos = int((c > 0).sum())
    print(f"{k+1:>3} {np.median(c):>10.5f} {c.mean():>10.5f} {c.std():>10.5f} "
          f"{abs(c.mean())/max(c.std(),1e-12):>9.3f} {max(pos,len(c)-pos):>5}/{len(c)}")
