"""orac_crit (20.10.2026 white-box-89, ідея юзера про критичні переломи): чи сидить виграш оракула (m,C) у КРИТИЧНИХ нейронах
(ті, що справді перемикаються: eps < P(h>0) < 1-eps), чи в заморожених (завжди ввімкнені = лінійні, завжди вимкнені = нуль).
Заморожений нейрон лінійний ⟹ його моменти переносяться точно; якщо виграш у критичному блоці k×k, дорожчий метод (K=4) там коштує k^4.
ПЕРЕДБАЧЕННЯ: критичні рядки (S×все) тримають ≥ 70% log-виграшу mc при |S| ≤ 40% n; шанс 30%.
Базова мітка: наступний щабель оракульної драбини white-box-16 — ПАРНИЙ стан.
На шарі K у ланцюг sb2 (LT_OFF=1, GPU-шим) підставляється справжня позадіагональна коваріація після ReLU (K11),
з власної GPU МК (N семплів, дві незалежні половини A і B), режими: c (лише C), mc (C і середнє), m (лише середнє, = драбина).
Пломби: (1) МК-середнє на шарі 15 проти істини корпусу N=1e8: різниця = шум; (2) c з половини A і з половини B мусять
давати той самий виграш (інакше міряємо шум МК, а не C); (3) m відтворює драбину сусіда.
ПЕРЕДБАЧЕННЯ (до запуску): c на K=12 дає ×1.3…×2 (Var(p) = wᵀCw несе позадіагональ, а середнє на глибині чутливе до σ_p);
mc ≈ добуток m і c. Шанс, що c на K<=12 дасть ×1.5+: 35%.
usage: orac_c.py <a-b> <out.json>   env KS (за замовч. 7,10,12,14), NMC (2^24)"""
import json, os, sys, time, types, importlib.util
os.environ["LT_OFF"] = "1"
os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1)

EST = "/home/trim/wb2lean/lt2re/estimator_sb2_orc.py"
SH = "/home/trim/wb2data/corpus14k"
SCALE = (2.0 / 1024.0) ** 0.5
KS = [int(x) for x in os.environ.get("KS", "12,14").split(",")]
NMC = int(float(os.environ.get("NMC", str(2 ** 24))))
B = 65536


def mc_cov(w, k):
    """two independent halves: post-ReLU mean and covariance at layers KS, plus layer-15 mean (seal)."""
    dev = "cuda"
    W = w.to(dev)
    out = []
    for half in range(2):
        g = torch.Generator(device=dev).manual_seed(990000 + 2 * k + half)
        # P104 float64 is 1/32 rate: per-batch Gram in float32 on h shifted by the first batch's mean, summed in float64
        S1 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in KS + [15]}
        S2 = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in KS}
        S0 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in KS}
        sh = {}
        done = 0
        while done < NMC // 2:
            h = torch.randn(B, 1024, generator=g, device=dev)
            for l in range(16):
                h = torch.relu(h @ W[l])
                if l in S2 and l not in sh:
                    sh[l] = h.mean(0)
                if l in S1:
                    hc = h - sh[l] if l in sh else h
                    S1[l] += hc.sum(0, dtype=torch.float64)
                if l in S2:
                    S2[l] += (hc.T @ hc).double()
                    S0[l] += (h > 0).sum(0, dtype=torch.float64)
            done += B
        r = {}
        for l in S1:
            m = S1[l] / done
            r[("m", l)] = m + sh[l].double() if l in sh else m
            if l in S2:
                r[("C", l)] = S2[l] / done - torch.outer(m, m)
                r[("P", l)] = S0[l] / done
        out.append(r)
    del W
    return out


def main():
    lo, hi = (int(x) for x in sys.argv[1].split("-")); out = sys.argv[2]
    res = json.load(open(out)) if os.path.exists(out) else {}
    saved = gpufnp.install()
    spec = importlib.util.spec_from_file_location("est_orc", EST)
    mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
    gpufnp.restore(saved)
    est = mod.Estimator()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
    for k in range(lo, hi + 1):
        if str(k) in res:
            continue
        t0 = time.time()
        fn = "shard%02d.npz" % (k % 16); row = k // 16
        with np.load(os.path.join(SH, fn), allow_pickle=False) as z:
            assert int(z["idx"][row]) == k
            M = np.asarray(z["moments"][row], dtype=np.float64)   # (4,16,1024)
        m_true = M[0]
        g = torch.Generator(device="cpu").manual_seed(770000 + k)
        w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * SCALE
        cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
        halves = mc_cov(w, k)
        mcm = 0.5 * (halves[0][("m", 15)] + halves[1][("m", 15)]).cpu().numpy()
        seal_m = float(np.mean((mcm - m_true[15]) ** 2))
        assert seal_m < 1e-6, ("convention/seed mismatch", seal_m)
        Wg = [cp.asarray(x.numpy()) for x in w]; del w
        Tg = [[cp.asarray(m_true[l].astype(np.float32))] * 4 for l in range(16)]   # only k1 is used (mode m)
        Cs = {}
        Pon_ = None
        for tag, sel in (("A", [0]), ("B", [1]), ("F", [0, 1])):
            Cs[tag] = {l: cp.asarray((sum(halves[i][("C", l)] for i in sel) / len(sel)).float().cpu().numpy()) for l in KS}
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

        Pon = {l: (0.5 * (halves[0][("P", l)] + halves[1][("P", l)])).cpu().numpy() for l in KS}

        def run(K, mode, tag="F", S=None, Sform="block"):
            mod.ORAC = {"K": K, "mode": mode, "truth": Tg, "rec": [], "C": Cs[tag], "Sform": Sform,
                        "S": None if S is None else cp.asarray(S.astype(np.float32))}
            mlp = types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k))
            p = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41))).astype(np.float64)
            mod.ORAC = None
            return float(np.mean((p[-1] - m_true[15]) ** 2))

        base = run(-1, "")
        r = {"base": base, "seal_m15_mc": seal_m, "lad": {}}
        r["Pon_hist"] = {}
        for K in KS:
            P = Pon[K]
            r["Pon_hist"][str(K)] = np.histogram(P, bins=[0, 1e-4, 0.02, 0.1, 0.5, 0.9, 0.98, 1 - 1e-4, 1.0001])[0].tolist()
            r["lad"]["%d_mc" % K] = run(K, "mc")
            for eps in (0.02, 0.1):
                S = (P > eps) & (P < 1 - eps)
                r["lad"]["%d_size%.2f" % (K, eps)] = float(S.sum())   # stored as a count, not a ratio
                r["lad"]["%d_mcS%.2f_block" % (K, eps)] = run(K, "mc", S=S, Sform="block")
                r["lad"]["%d_mcS%.2f_rows" % (K, eps)] = run(K, "mc", S=S, Sform="rows")
                r["lad"]["%d_mcNotS%.2f_block" % (K, eps)] = run(K, "mc", S=~S, Sform="block")
                r["lad"]["%d_mcNotS%.2f_rows" % (K, eps)] = run(K, "mc", S=~S, Sform="rows")
        res[str(k)] = r
        tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
        line = " ".join(("%s:%d" % (key, v)) if "size" in key else ("%s:%.3f" % (key, base / v)) for key, v in r["lad"].items() if v > 0) + " hist " + str(r["Pon_hist"])
        print("net %d base %.4e mcseal %.1e (%.0fs)  %s" % (k, base, seal_m, time.time() - t0, line), flush=True)


if __name__ == "__main__":
    main()
