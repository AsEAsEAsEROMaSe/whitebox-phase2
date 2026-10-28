"""orac101 — СКІЛЬКИ КОШТУЄ ІДЕАЛЬНА `C` У РЕКУРСІЇ? ВИМІР З НУЛЯ, БЕЗ ЦИТАТ.

ЮЗЕР (С101): «пам'ять може бути невірна, тому краще міряти з 0».
⟹ записане «позадіагональ = найдешевший канал ×1.16» НЕ використовується ні як
   орієнтир, ні як передбачення. Стенд рахує стелю каналу заново.

ПИТАННЯ. С100 закрила розклад `E²`: формула кроку практично точна (Мелер K=2
проти точної квадратури = 0.1%) ⟹ винне саме ГАУСОВЕ ПРИПУЩЕННЯ. Маски ReLU
зміряли порушення цього припущення прямо: реальна `P(D∧D')` розходиться з
orthant-передбаченням із тих самих `(m,C)` на 4.7–4.9% на глибині.
Якщо маски й позадіагональ `C` — той самий об'єкт, то стеля «ідеальної `C`»
і є стелею всієї маскової гілки.

ЯК МІРЯЄМО. Женемо реальні семпли, збираємо ІСТИННУ `Cov(a_l)` і `E[a_l]` на
кожному шарі. Тоді ганяємо нашу рекурсію, підставляючи істину в один канал:

    base  наша рекурсія цілком                          ← база таблиці
    off   істинна ПОЗАДІАГОНАЛЬ C, діагональ наша       ← об'єкт масок
    diag  істинна ДІАГОНАЛЬ C, позадіагональ наша       ← контрольний канал
    full  істинна C цілком
    allm  істинні C і m                                 ← 🔒 ПЛОМБА

🔒 ПЛОМБА `allm`: підставивши істинні `(m,C)` на кожному кроці, лишається ЛИШЕ
   похибка ОДНОГО кроку формули. Записане значення одношагової похибки
   `E[relu]` = 0.139% ⟹ `allm` мусить дати 0.10…0.20%. Якщо дасть 0.0% —
   стенд тавтологічний і всі числа з нього недійсні. Якщо дасть >0.5% —
   зламаний збір істини.
🔒 ПЛОМБА ШАРУ 0: на вході `z ~ N(0,I)` ТОЧНО гаусів ⟹ наша `C_ours[0]` проти
   `C_true[0]` міряє ЧИСТУ похибку формули кроку без накопичення. Це пряма
   перевірка тези «маскова розбіжність 4.9% ≈ позадіагональна похибка кроку».
🔒 ШУМ ОРАКУЛА ЛИШЕ ЗАНИЖУЄ: ціль (`final_means`, 1e9 семплів з датасету)
   НЕЗАЛЕЖНА від наших семплів ⟹ спільного шуму, який завищував `varmap`, нема.
🔒 МАСШТАБУВАННЯ: все рахується на ДВОХ `N` (чверть і повний) ⟹ якщо стеля
   росте з `N`, число ще не збіглось і вирок не виноситься.

ЧЕСНА СХЕМА ПОРІВНЯННЯ (закони проєкту):
  · кожен варіант отримує СВІЙ оптимальний `α` (фітований коефіцієнт, не 1.0)
  · кожен варіант отримує СВІЙ оптимальний `κ` (константи НЕ переносяться між
    конфігураціями — С100 коштувала на цьому цілого прогону)
  · порівнюються ОПТИМУМИ в ОДНІЙ таблиці, абсолюти назовні не виносяться

ЩО ЗБЕРІГАЄМО. Для фіту глобального `α` достатньо трьох сум на (мережа, режим,
κ): Σm², Σ(m·T), ΣT². Тоді MSE = (α²Σm² − 2αΣmT + ΣT²)/256, а вибір `α` і `κ`
робиться на етапі аналізу і лишається ВИДИМИМ.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ, щоб не тлумачити заднім числом):
 (а) 🔒 `allm` дасть 0.10…0.20%                                        70%
 (б) 🔒 позадіагональна похибка на шарі 0 буде 3…6% (≈ маскові 4.9%)   55%
 (в) `off` покращить похибку замикання у ×1.1…×1.4                     50%
 (г) `full` дасть ×1.5…×2.5                                            60%
 (д) `off` у БАЛІ дасть менше за ×1.05                                 65%
 (е) `diag` виявиться сильнішим каналом за `off`                       55%

ЗАПУСК:
  CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-<uuid> /usr/bin/python3 orac101.py <tag> <nnet> <npairs>
"""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
TAG = sys.argv[1] if len(sys.argv) > 1 else "a"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 262144
SEED0 = int(sys.argv[4]) if len(sys.argv) > 4 else 777   # 🔒 незалежна репліка
CHUNK = 16384
KSIG = 8
KGRID = sorted({round(0.980 + 0.002 * i, 4) for i in range(21)}
               | {0.9975, 0.998, 0.9985, 0.999, 0.9995, 1.0})
MODES = ["base", "off", "diag", "full", "allm", "mm"]
# 🚨 чекпоінт — на ext4: /mnt/win_d — NTFS через FUSE, 25–53 КБ/с
OUT = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
       "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/orac101_%s.json" % TAG)

state = {}
if os.path.exists(OUT):
    try:
        state = json.load(open(OUT))
        print(f"↻ підхоплено {len(state)} мереж із {OUT}", flush=True)
    except Exception as e:                                    # noqa: BLE001
        print(f"⚠️ чекпоінт не читається ({e}), з нуля", flush=True)


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, OUT)


import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не одну карту — пінування не спрацювало, вихід", flush=True)
    sys.exit(2)
print(f"orac101[{TAG}] · карта "
      f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} · "
      f"{NNET} мереж × {2*NPAIRS} семплів · {len(KGRID)} точок κ", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure(W, kap, Ctrue, mtrue, mode):
    """наша рекурсія; `mode` каже, який канал брати з істини."""
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    dev0 = None
    for i in range(DEPTH):
        w = W[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * kap
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        m = mp * Ph + s * ph
        if i == DEPTH - 1:
            break
        Co = (cpm * cp.outer(Ph, Ph)
              + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        Co = Co + cp.diag(d - cp.diagonal(Co))
        if i == 0:
            dev0 = _dev(Co, Ctrue[0])          # 🔒 пломба шару 0
        if mode == "base":
            C = Co
        elif mode == "off":
            C = Ctrue[i] + cp.diag(cp.diagonal(Co) - cp.diagonal(Ctrue[i]))
        elif mode == "diag":
            C = Co + cp.diag(cp.diagonal(Ctrue[i]) - cp.diagonal(Co))
        elif mode == "full":
            C = Ctrue[i]
        elif mode == "allm":
            C = Ctrue[i]
            m = mtrue[i]
        elif mode == "mm":
            C = Co
            m = mtrue[i]          # 🆕 істинна m, НАША C — канал `m` соло
    return m, dev0


def _dev(A, B):
    """відносна розбіжність окремо по діагоналі й позадіагоналі"""
    da, db = cp.diagonal(A), cp.diagonal(B)
    dd = float(cp.abs(da - db).mean() / cp.maximum(cp.abs(db).mean(), 1e-30))
    M = 1.0 - cp.eye(W_, dtype=cp.float64)
    do = float((cp.abs(A - B) * M).sum()
               / cp.maximum((cp.abs(B) * M).sum(), 1e-30))
    return (dd, do)


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(tb.num_rows, W_)
NNET = min(NNET, tb.num_rows)
N = 2 * NPAIRS
QN = min(NPAIRS, max(CHUNK, (NPAIRS // 4 // CHUNK) * CHUNK))   # чверть, у чанках


def one_net(net):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
          for L in range(DEPTH)]
    W64 = [w.astype(cp.float64) for w in Wg]
    T = cp.asarray(FM[net].astype(np.float64))
    rng = cp.random.default_rng(SEED0 + net)
    S1 = cp.zeros((DEPTH - 1, W_), dtype=cp.float64)
    S2 = cp.zeros((DEPTH - 1, W_, W_), dtype=cp.float64)
    H1 = cp.zeros((DEPTH - 1, W_), dtype=cp.float64)
    H2 = cp.zeros((DEPTH - 1, W_, W_), dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            p = a @ Wg[i]
            if i < DEPTH - 1:
                a = cp.maximum(p, 0.0)
                s1 = a.sum(0, dtype=cp.float64)
                s2 = (a.T @ a).astype(cp.float64)   # чанк fp32, акумулятор fp64
                S1[i] += s1
                S2[i] += s2
                if done < QN:
                    H1[i] += s1
                    H2[i] += s2
        done += nb
    del z, a, p
    cp.get_default_memory_pool().free_all_blocks()

    TT = float(T @ T)
    out = {"_TT": TT}
    for label, (s1, s2, nn) in (("full", (S1, S2, N)),
                                ("quarter", (H1, H2, 2 * QN))):
        mt = s1 / nn
        Ct = s2 / nn - mt[:, :, None] * mt[:, None, :]
        Ct = 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1)))
        res = {}
        for mode in MODES:
            tab = {}
            for k in KGRID:
                mm, d0 = closure(W64, k, Ct, mt, mode)
                if label == "full" and mode == "base":
                    out["_dev0"] = d0
                if not bool(cp.all(cp.isfinite(mm))):
                    continue
                tab[str(k)] = [float(mm @ mm), float(mm @ T)]
            res[mode] = tab
        out[label] = res
        del mt, Ct
        cp.get_default_memory_pool().free_all_blocks()
    del S1, S2, H1, H2
    cp.get_default_memory_pool().free_all_blocks()
    return out


fails = 0
for net in range(NNET):
    if str(net) in state:
        continue
    try:
        t0 = time.time()
        state[str(net)] = one_net(net)
        save()
        fails = 0
        d0 = state[str(net)]["_dev0"]
        print(f"  ✅ мережа {net}  ({time.time()-t0:.1f} с)  "
              f"🔒шар0: діаг {100*d0[0]:.3f}% · позадіаг {100*d0[1]:.3f}%",
              flush=True)
    except Exception as e:                                    # noqa: BLE001
        fails += 1
        print(f"  ⚠️ мережа {net} ВПАЛА ({type(e).__name__}: {e}) — "
              f"{fails}/3", flush=True)
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(10)
        if fails >= 3:
            print(f"🚨 три невдачі поспіль — збережено {len(state)}, вихід",
                  flush=True)
            save()
            sys.exit(3)

save()
print(f"\n✅ зібрано {len(state)} мереж → {OUT}", flush=True)
print("   аналіз: orac101_an.py (вибір α і κ робиться там і видимий)",
      flush=True)
