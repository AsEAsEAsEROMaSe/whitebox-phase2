"""k3gen_alpha97 — ЧИ ТРИМАЄТЬСЯ `K3_gen` ПІСЛЯ ВІЛЬНОГО МАСШТАБУ?
Це вирішує, ЩО вставляти: складну ранг-4 схему з несенням чи просту
БЕЗТЕНЗОРНУ, БЕЗ СТАНУ (лише два зрізи між шарами, 3 матмули).
Якщо K3_gen після α ≳ ×1.15 — складна схема не потрібна взагалі."""
import sys, numpy as np
sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")
import k3rec97 as K
N, NETS = 400000, 3
Wall = np.load(K.W_PATH, mmap_mode="r")
res = {}
for net in range(NETS):
    W = np.array(Wall[net]).astype(np.float64)
    mt, _ = K.truth(W, N, 12345 + net)
    den = np.sqrt((mt**2).mean())
    def err(p):
        e = np.sqrt(((p-mt)**2).mean())/den
        al = float(p @ mt / (p @ p))
        return e, np.sqrt(((al*p-mt)**2).mean())/den
    eg, ega = err(K.run_closure(W, "G"))
    print("\nМЕРЕЖА %d  G = %.4f%%  (після α: %.4f%%)" % (net, 100*eg, 100*ega), flush=True)
    for lam in (0.5, 1.0, 1.5):
        e, ea = err(K.run_closure(W, "K3_gen", lam=lam))
        res.setdefault(lam, []).append((eg/e, ega/ea))
        print("   K3_gen λ=%.1f : %.4f%%  x%.3f  |  після α %.4f%%  x%.3f"
              % (lam, 100*e, eg/e, 100*ea, ega/ea), flush=True)
print("\nПІДСУМОК (середні множники по %d мережах)" % NETS, flush=True)
for lam in (0.5, 1.0, 1.5):
    v = np.array(res[lam])
    print("   λ=%.1f :  x%.3f   після α  x%.3f" % (lam, v[:,0].mean(), v[:,1].mean()), flush=True)
