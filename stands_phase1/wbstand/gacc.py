"""С115-к: ДЕ НАРОДЖУЄТЬСЯ НАКОПИЧЕННЯ. Крива відгуку на ОДНЕ пере-якорення.

Карта [gclos]: формула ×1.00 · накопичення ×3.5 · поза два моменти ×2.2.
Накопичення — найбільший шматок. Питання: скільки дає ОДИН якір і де його ставити.

СХЕМА: точне двомоментне замикання (без формули); на шарі k підмінюємо `m`
істинним із ПОЛОВИНИ A банку, далі йдемо як завжди; міряємо проти половини B.
🔒 незалежність за побудовою · 🔒 контроль k=-1 мусить відтворити базу БІТ У БІТ.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 глибокий якір (k=12..14) дає найбільше                      55%
  P2 один якір бере <30% від ×3.5                                60%
  P3 крива плоска (всі шари однакові)                            20%
  P4 мілкий якір (k<=4) дає майже нуль (похибка ще не накопичена) 70%
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq

PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LOGN = int(sys.argv[2]) if len(sys.argv) > 2 else 21
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3141
CHUNK = 1 << 16
KS = [-1, 2, 4, 6, 8, 10, 12, 14]
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
N = 1 << LOGN

def step(Wg, m, C, i, sd):
    Wi64 = Wg[i].astype(cp.float64)
    mp32 = (m @ Wi64).astype(cp.float32)
    Cp = Wi64.T @ C @ Wi64; Cp = 0.5*(Cp + Cp.T)
    ev = cp.linalg.eigvalsh(Cp)
    jit = max(0.0, -float(ev[0])) + 1e-9*float(cp.diagonal(Cp).mean())
    L = cp.linalg.cholesky(Cp + cp.eye(1024, dtype=cp.float64)*jit).astype(cp.float32)
    am = cp.zeros(1024, dtype=cp.float64); aS = cp.zeros((1024,1024), dtype=cp.float64)
    done = 0
    while done < N:
        b = min(CHUNK, N-done)
        g = cp.random.RandomState(seed=sd+7919*i+(done//CHUNK)).standard_normal((b,1024), dtype=cp.float32)
        h = cp.maximum(mp32 + g @ L.T, 0.0)
        am += h.sum(0, dtype=cp.float64); aS += (h.T @ h).astype(cp.float64)
        done += b; del g, h
    m2 = am/N
    return m2, aS/N - cp.outer(m2, m2)

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(), 1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

acc = {k: [] for k in KS}
files = sorted([x for x in os.listdir("/home/trim/wblogs") if x.startswith("truthhi_")])[:NNET]
for f in files:
    d = np.load("/home/trim/wblogs/"+f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float32).reshape(16,1024,1024)
    Wg = [cp.asarray(W[i]) for i in range(16)]
    A, B = d["m_half"][:,0,:], d["m_half"][:,1,:]
    # базовий прохід, зберігаємо стан перед кожним шаром
    st = []
    m = cp.zeros(1024, dtype=cp.float64); C = cp.eye(1024, dtype=cp.float64)
    for i in range(16):
        st.append((m.copy(), C.copy()))
        m, C = step(Wg, m, C, i, SEED)
    base = m.get()
    for k in KS:
        if k < 0:
            pred = base
        else:
            m2, C2 = st[k+1] if k+1 < 16 else (m, C)
            m2 = cp.asarray(A[k])                      # 🔑 ЯКІР: істинне m шару k
            C2 = C2.copy()
            for i in range(k+1, 16):
                m2, C2 = step(Wg, m2, C2, i, SEED)
            pred = m2.get()
        acc[k].append(0.5*(cerr(pred, B[15]) + cerr(pred, A[15])))
    log("%-20s база %.4f%%  якір@14 %.4f%%" % (nm, acc[-1][-1]*100, acc[14][-1]*100))
    for g in Wg: del g
    cp.get_default_memory_pool().free_all_blocks()
b = np.median(acc[-1])
print("\n%-6s %10s %8s" % ("якір", "похибка%", "×база"))
for k in KS:
    v = np.median(acc[k]); print("%-6s %10.4f %8.3f" % ("нема" if k < 0 else str(k), v*100, b/v))
log("ГОТОВО")
