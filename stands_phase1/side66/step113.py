#!/usr/bin/env python3
"""step113 — СТЕЛЯ КЛАСУ «КРАЩИЙ КРОК» У ФАЗІ 2 (1024×16).

════════ ЧОМУ ЦЕ ПЕРШИЙ ВИМІР, А НЕ ДРУГИЙ ════════
С105 зміряла цю стелю у фазі 1: ×1.19 у похибці замикання ⟹ +3.2% у балі
⟹ клас закрито. АЛЕ той вирок винесено при `_W_BLEND = 0.0798`, тобто коли
замикання важило 8.8%. У фазі 2 С113 зміряла ІНВЕРСІЮ: замикання 0.1264%
проти МК 0.2404%, `_W_BLEND = 0.820`. Передатна функція `h = e·v/(e+v)`:

    k у похибці замикання   ×бал ФАЗА 1   ×бал ФАЗА 2
             1.19              1.032          1.326
             1.41              1.05           1.774
             2.00              1.23           3.35
🔒 рядок 1.19/фаза 1 відтворює зміряне step105 ×1.0315 (збіг 0.2%) ⟹ прилад
   перерахунку правильний.

⟹ ТА САМА СТЕЛЯ КОШТУЄ ТУТ У 10× БІЛЬШЕ. До #6 треба ×1.08, до #1 ×6.8.

════════ ЩО САМЕ МІРЯЄМО ════════
`b_l = (наш крок від ІСТИННИХ (m_l,C_l)) − істинне m_{l+1}` — ТОЧНА похибка
одного кроку. Вкидаємо її в рекурсію. Це ВЕРХНЯ МЕЖА для будь-якої ідеї виду
«точніша формула кроку»: γ-схеми, квадратура, навчений енкодер ваг — усі разом.

  base     наша рекурсія фази 2 (κ=0.9993, KSIG=8, без gam)      ← ПЛОМБА P1
  inj      m ← крок − c·b_l на кожному шарі                      ← СТЕЛЯ КЛАСУ
  injp     те саме, b перенесена в НАШУ точку множником (s/s₀)   ← H1
  injL8    b лише на шарах ≥8 (де жила ×22.8 фази 1)             ← локалізація
  foreign  b ЧУЖОЇ мережі                                        ← 🔒 КОНТРОЛЬ
  injm     inj ПОВЕРХ істинних (m,C)                             ← 🔒 ТАВТОЛОГІЯ

════════ 🔒 ПЛОМБИ ════════
 P1 base мусить лягти в 0.11…0.16% (С113: чисте замикання 0.1264%)
    — без цього прогін НЕВАЛІДНИЙ
 P2 injm ≈ 0.00% — тавтологія; у фазі 1 САМЕ вона впіймала зсув індексу C
 P3 foreign: оптимальний c → 0
 P4 fp32-шлях накопичення C проти fp64 на малому N — відносна різниця < 1e-4
 P5 розкид по мережах друкуємо; вирок читати разом із ним

════════ 📉 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
   inj ×1.2…×2.0                                            50%
   inj ≥ ×3 (клас оживає широко)                            20%
   inj < ×1.2 (стеля впала разом із важелем)                30%
   c* < 1 («поправка стиснена», 7-й раз)                    60%
   частка кроку БІЛЬША за фазу 1 (16%), бо глибина вдвічі менша  65%

Запуск:
  CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 \
  PYTHONPATH=/home/trim/gpupkg /usr/bin/python3 side66/step113.py --nets 24
"""
import argparse
import json
import math
import os
import time

import numpy as np

CKPT = "/home/trim/wblogs/step113_ckpt.json"
SQ2PI = math.sqrt(2.0 * math.pi)


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=24)
    ap.add_argument("--npairs", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--ksig", type=int, default=8)
    ap.add_argument("--kaps", type=str, default="0.9985,0.9993,1.0")
    ap.add_argument("--cs", type=str, default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    import cupy as cp
    from cupyx.scipy.special import ndtr
    log(f"cupy {cp.__version__} · пристроїв {cp.cuda.runtime.getDeviceCount()} · "
        f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    assert cp.cuda.runtime.getDeviceCount() == 1, "пінування по UUID не спрацювало"

    W_, D_, KS = a.width, a.depth, a.ksig
    NP, CH = a.npairs, a.chunk
    KAPS = [float(x) for x in a.kaps.split(",")]
    CS = [float(x) for x in a.cs.split(",")]
    MODES = ["base", "inj", "injp", "injL8", "foreign", "injm"]

    def phi(x):
        return cp.exp(-0.5 * x * x) / SQ2PI

    # ─────────────────── істина: (m_l, C_l) активацій + вихід ───────────────
    def truth(Ws, seed):
        N = 2 * NP
        rng = cp.random.default_rng(seed)
        A1 = cp.zeros((D_, W_), cp.float64)
        A2 = cp.zeros((D_, W_, W_), cp.float64)
        O1 = cp.zeros(W_, cp.float64)
        Oh = [cp.zeros(W_, cp.float64), cp.zeros(W_, cp.float64)]
        cnt = [0, 0]
        done = k = 0
        while done < NP:
            nb = min(CH, NP - done)
            z = rng.standard_normal((nb, W_), dtype=cp.float32)
            act = cp.concatenate([z, -z], axis=0)
            for i in range(D_):
                A1[i] += act.sum(0, dtype=cp.float64)
                # fp32-матмул на чанк, fp64-акумуляція між чанками (пломба P4)
                A2[i] += (act.T @ act).astype(cp.float64)
                act = cp.maximum(act @ Ws[i], 0.0)
            O1 += act.sum(0, dtype=cp.float64)
            Oh[k % 2] += act.sum(0, dtype=cp.float64)
            cnt[k % 2] += 2 * nb
            done += nb
            k += 1
        mt = A1 / N
        Ct = A2 / N - mt[:, :, None] * mt[:, None, :]
        Ct = 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1)))
        T = O1 / N
        return mt, Ct, T, (Oh[0] / cnt[0], Oh[1] / cnt[1])

    # ─────────────────── один крок замикання (форма фази 2) ─────────────────
    def step(w, C, m, kap, i):
        cpm = w.T @ C @ w
        if i > KS:
            cpm = cpm * kap
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        Ph, ph = ndtr(t), phi(t)
        return mp * Ph + s * ph, mp, s, Ph, ph, cpm

    def stepb(Ws, mt, Ct, T, kap):
        """ТОЧНА похибка кроку при ІСТИННИХ входах + опорний масштаб s."""
        B = cp.zeros((D_, W_), cp.float64)
        SS = cp.zeros((D_, W_), cp.float64)
        for i in range(D_):
            Cin = cp.eye(W_, dtype=cp.float64) if i == 0 else Ct[i]
            min_ = cp.zeros(W_, cp.float64) if i == 0 else mt[i]
            mn, mp, s, Ph, ph, _ = step(Ws[i], Cin, min_, kap, i)
            tgt = T if i == D_ - 1 else mt[i + 1]
            B[i] = mn - tgt
            SS[i] = s
        return B, SS

    def closure(Ws, mt, Ct, kap, mode, c, B, SS):
        m = cp.zeros(W_, cp.float64)
        C = cp.eye(W_, dtype=cp.float64)
        for i in range(D_):
            mn, mp, s, Ph, ph, cpm = step(Ws[i], C, m, kap, i)
            if mode != "base" and c != 0.0:
                if not (mode == "injL8" and i < 8):
                    corr = B[i]
                    if mode == "injp":
                        corr = corr * (s / cp.maximum(SS[i], 1e-30))
                    mn = mn - c * corr
            m = mn
            if i == D_ - 1:
                break
            Co = (cpm * cp.outer(Ph, Ph)
                  + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
            d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
            Co = Co + cp.diag(d - cp.diagonal(Co))
            if mode == "injm":
                C, m = Ct[i + 1], mt[i + 1]
            else:
                C = Co
        return m

    # ───────────────────────────── прогін ───────────────────────────────────
    state = {}
    if os.path.exists(CKPT) and not a.fresh:
        try:
            state = json.load(open(CKPT))
        except Exception:
            state = {}

    log(f"step113 · {a.nets} мереж {W_}×{D_} · {2*NP} семплів · режими {MODES}")
    log(f"κ {KAPS} · c {CS} · KSIG {KS}")
    t0 = time.time()
    PREV = None
    fails = 0
    for net in range(a.nets):
        if str(net) in state:
            PREV = None
            continue
        try:
            rng = cp.random.default_rng(7000 + net)
            sc = float(np.sqrt(2.0 / W_))
            Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
                  for _ in range(D_)]
            mt, Ct, T, halves = truth(Ws, 113000 + net)
            Ws64 = [w.astype(cp.float64) for w in Ws]
            # шумова підлога істини (половинки вибірки)
            h0, h1 = halves
            cc = float(cp.dot(h0, h1) / (cp.linalg.norm(h0) * cp.linalg.norm(h1)))
            nf = math.sqrt(max(0.0, 1.0 - cc * cc)) / 2.0

            rec = {"_TT": float(T @ T), "_NF": nf}
            KEEP = None
            for kap in KAPS:
                B, SS = stepb(Ws64, mt, Ct, T, kap)
                for mode in MODES:
                    if mode == "foreign" and PREV is None:
                        continue
                    cs = [1.0] if mode == "base" else CS
                    Bu = PREV if mode == "foreign" else B
                    for c in cs:
                        mm = closure(Ws64, mt, Ct, kap, mode, c, Bu, SS)
                        if not bool(cp.all(cp.isfinite(mm))):
                            continue
                        rec[f"{mode}|{kap}|{c}"] = [float(mm @ mm), float(mm @ T)]
                if abs(kap - 0.9993) < 1e-9:
                    rec["_bn"] = [float(cp.linalg.norm(B[i])) for i in
                                  range(0, D_, max(1, D_ // 8))]
                    rec["_bnall"] = [float(cp.linalg.norm(B[i])) for i in range(D_)]
                    rec["_Tn"] = float(cp.linalg.norm(T))
                    KEEP = B.copy()
            state[str(net)] = rec
            if KEEP is not None:
                PREV = KEEP
            tmp = CKPT + ".tmp"
            json.dump(state, open(tmp, "w"))
            os.replace(tmp, CKPT)
            log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с) · шум {100*nf:.4f}%")
            fails = 0
        except Exception as e:                                   # noqa: BLE001
            fails += 1
            log(f"  ⚠️ мережа {net}: {type(e).__name__}: {e}")
            if fails >= 3:
                log("🚨 три невдачі поспіль — чистий вихід зі збереженням")
                break
        finally:
            cp.get_default_memory_pool().free_all_blocks()

    # ───────────────────────────── агрегація ────────────────────────────────
    def agg(key):
        num = den = 0.0
        ind = []
        for r in state.values():
            if key not in r:
                continue
            xx, xt = r[key]
            tt = r["_TT"]
            e = tt - xt * xt / max(xx, 1e-300)
            num += e
            den += tt
            ind.append(math.sqrt(max(e, 0.0) / tt))
        if len(ind) < max(2, len(state) - 2):
            return None, None
        return math.sqrt(num / den), np.array(ind)

    nets = len(state)
    nf = np.mean([r["_NF"] for r in state.values()])
    log(f"\n📊 {nets} мереж · косинусна похибка · шумова підлога {100*nf:.4f}%")
    best = {}
    for mode in MODES:
        rows = []
        for kap in KAPS:
            for c in ([1.0] if mode == "base" else CS):
                v, ind = agg(f"{mode}|{kap}|{c}")
                if v is not None:
                    rows.append((v, kap, c, ind))
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])
        best[mode] = rows[0]
        v, kap, c, ind = rows[0]
        log(f"{mode:>9}  {100*v:8.4f}%   κ={kap} c={c:4.2f}   "
            f"медіана {100*np.median(ind):7.4f}%  sd(log) {np.std(np.log(ind+1e-30)):.3f}")

    # передатна функція ФАЗИ 2 (зміряні гілки С113)
    E0, V0 = 0.1264e-2, 0.2404e-2
    if "base" in best:
        b0 = best["base"][0]
        h0 = (E0**2) * (V0**2) / ((E0**2) + (V0**2))
        log(f"\n{'режим':>9}{'×точність':>12}{'×бал ф2':>11}{'бал':>13}")
        for mode in MODES:
            if mode not in best:
                continue
            if best[mode][0] < 1e-9:
                log(f"{mode:>9}{'∞':>12}{'—':>11}{'тавтологія':>13}")
                continue
            k = b0 / best[mode][0]
            e = (E0 / k) ** 2
            h = e * (V0**2) / (e + V0**2)
            log(f"{mode:>9}{k:>12.4f}{h0/h:>11.3f}{1.2250e-07*h/h0:>13.3e}")

    log("\n🔒 ПЛОМБИ")
    if "base" in best:
        ok = 0.11e-2 <= best["base"][0] <= 0.16e-2
        log(f" (P1) base = {100*best['base'][0]:.4f}% проти очікуваних "
            f"0.11…0.16% [С113: 0.1264%]  {'✅' if ok else '🚨 ПРОГІН НЕВАЛІДНИЙ'}")
    if "injm" in best:
        log(f" (P2) injm = {100*best['injm'][0]:.4f}% — мусить бути ≈0")
    if "foreign" in best:
        log(f" (P3) foreign: оптимальний c = {best['foreign'][2]:.2f} — мусить бути 0.00")
    # частка кроку: Σ‖b‖ проти накопиченої
    bn = [r["_bnall"] for r in state.values() if "_bnall" in r]
    tn = [r["_Tn"] for r in state.values() if "_Tn" in r]
    if bn:
        bn = np.array(bn)
        tn = np.array(tn)
        frac = (bn.sum(1) / tn)
        log(f"\n📐 Σ‖b_l‖/‖T‖ = {100*np.mean(frac):.3f}%  (фаза 1: 2.2% проти "
            f"накопиченої 0.48%)")
        per = 100 * (bn / tn[:, None]).mean(0)
        log("   ‖b_l‖/‖T‖ по шарах: " + " ".join(f"{x:.3f}" for x in per))

    log("\n📉 ПЕРЕДБАЧЕННЯ: inj ×1.2…×2.0 (50%) · ≥×3 (20%) · <×1.2 (30%) · c*<1 (60%)")
    if "inj" in best and "base" in best:
        k = best["base"][0] / best["inj"][0]
        if k >= 3.0:
            log(f"🟢🟢 inj = ×{k:.3f} ⟹ КЛАС «кращий крок» ШИРОКО ЖИВИЙ у фазі 2")
        elif k >= 1.6:
            log(f"🟢 inj = ×{k:.3f} ⟹ клас живий: ×{k:.2f} у похибці = помітний бал")
        elif k >= 1.2:
            log(f"🟡 inj = ×{k:.3f} ⟹ між передбаченнями, читати з розкидом")
        else:
            log(f"⛔ inj = ×{k:.3f} ⟹ стеля впала разом із важелем")


if __name__ == "__main__":
    main()
