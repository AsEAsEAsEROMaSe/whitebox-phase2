"""С115-к: БАНК ІСТИНИ ВИСОКОЇ ТОЧНОСТІ. Прилад під ціль «якість середнього».

ПРИЧИНА: шум істини датасету (N=4e5) ≈ 0.038% — рівно рівень двомоментного
оракула і ВИЩЕ за #1 (0.017%). Нинішнім приладом ціль не видно.

СХЕМА: i.i.d. + АНТИТЕТИКА (точно незміщена; вибілювання НЕ беремо — воно
вносить зсув O(1/n), а тут потрібна саме незміщеність).
🔒 ВЛАСНА ОЦІНКА ШУМУ: половини вибірки рахуються окремо ⟹ |m_A − m_B|/2
дає шум напряму, без припущень.
🔒 ПЛОМБА: розбіжність із `all_layer_means` датасету мусить лягти в шум ДАТАСЕТУ.

стійкість (вимога ЮЗЕРА): чекпоінт після кожної мережі, атомарний запис,
підхоплення зробленого, лови винятків карти.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq

PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
OUT = "/home/trim/wblogs"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LOGN = int(sys.argv[2]) if len(sys.argv) > 2 else 23      # усього 2**LOGN семплів
CHUNK = 1 << 16                                            # пар за раз
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
log("cupy %s · карта %s" % (cp.__version__, cp.cuda.runtime.getDeviceCount()))

tb = pq.read_table(PQ)
NPAIR = (1 << LOGN) // 2
fails = 0
for j in range(min(NNET, tb.num_rows)):
    nm = tb["mlp_name"][j].as_py()
    path = "%s/truthhi_%02d.npz" % (OUT, j)
    if os.path.exists(path):
        log("%2d %s — вже є" % (j, nm)); continue
    try:
        W = np.asarray(tb["weights"][j].as_py(), dtype=np.float32).reshape(16, 1024, 1024)
        Wg = [cp.asarray(W[i]) for i in range(16)]
        acc = [cp.zeros((2, 1024), dtype=cp.float64) for _ in range(16)]
        cnt = np.zeros(2, dtype=np.int64)
        done = 0
        half = 0
        while done < NPAIR:
            b = min(CHUNK, NPAIR - done)
            rs = cp.random.RandomState(seed=770000 + 1000*j + (done // CHUNK))
            z = rs.standard_normal((b, 1024), dtype=cp.float32)
            x = cp.concatenate([z, -z], 0)              # 🔑 антитетика: точна
            for i in range(16):
                x = cp.maximum(x @ Wg[i], 0.0)
                acc[i][half] += x.sum(axis=0, dtype=cp.float64)
            cnt[half] += 2*b
            done += b
            half = 1 - half                              # чергуємо половини
            del x, z
        M = np.stack([(acc[i] / cp.asarray(cnt)[:, None]).get() for i in range(16)])  # (16,2,1024)
        mean = M.mean(1)
        noise = np.abs(M[:, 0] - M[:, 1]) / 2.0
        tmp = path + ".tmp.npz"
        np.savez(tmp, name=nm, n=(1 << LOGN), m=mean, m_half=M, noise=noise)
        os.replace(tmp, path)
        rel = np.sqrt((noise[15]**2).sum()) / np.sqrt((mean[15]**2).sum())
        log("%2d %s  шум фін. шару %.5f%%" % (j, nm, rel*100))
        fails = 0
        for g in Wg: del g
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        fails += 1; log("%2d %s ПОМИЛКА %s" % (j, nm, e))
        if fails >= 3: log("три поспіль — вихід"); break
log("ГОТОВО")
