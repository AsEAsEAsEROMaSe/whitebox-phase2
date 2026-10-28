"""dead97 — ЧАСТКА МЕРТВИХ НЕЙРОНІВ ПО ШАРАХ І РОЗКИД МІЖ МЕРЕЖАМИ.

Питання ВИМІРУ: сабміт обрізає нейрони, що не відкрилися на 256 пілотних семплах,
і прибирає ~15%. Я зміряла на шарі 31 — 42-76 із 256 (16-30%). Якщо мертвих
удвічі більше, ніж ловить критерій, це +15% семплів МК за ту саму ціну.

МІРЯЮ ТРИ РІЗНІ РЕЧІ (їх легко сплутати):
 (а) СПРАВДІ мертві: жодної активації на N=40000 семплів;
 (б) що ловить критерій 256 пілотних семплів (те, що робить сабміт);
 (в) 🚨 ЦІНА ПОМИЛКИ: яку частку ЕНЕРГІЇ шару (Σ E[relu]^2) несуть нейрони,
     які критерій (б) визнав мертвими — це РИЗИК обрізання, і саме він
     вирішує, чи можна обрізати агресивніше.

ПЕРЕДБАЧЕННЯ: (б) знайде МЕНШЕ за (а), бо 256 семплів — це груба сітка; але
різниця буде в бік ОБЕРЕЖНОСТІ (критерій пропускає мертвих, а не вбиває живих),
бо нейрон, що не відкрився на 256, майже напевно рідкісний.       шанс 75%
"""
import sys, numpy as np
W_PATH = "/mnt/win_d/llm/White-Box/arena60/fs_w.npy"
LAYERS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 31]
N, B, NETS = 40000, 4000, 24
Wall = np.load(W_PATH, mmap_mode="r")
res = {l: [] for l in LAYERS}; res256 = {l: [] for l in LAYERS}; risk = {l: [] for l in LAYERS}
for net in range(NETS):
    W = np.array(Wall[net]).astype(np.float32)
    rng = np.random.default_rng(8000 + net)
    fires = {l: np.zeros(256) for l in LAYERS}
    f256 = {l: np.zeros(256) for l in LAYERS}
    esum = {l: np.zeros(256) for l in LAYERS}
    for i in range(N // B):
        a = rng.standard_normal((B, 256), dtype=np.float32)
        for l in range(max(LAYERS) + 1):
            h = a @ W[l]
            if l in fires:
                pos = (h > 0)
                fires[l] += pos.sum(0)
                if i == 0:
                    f256[l] += pos[:256].sum(0)
                esum[l] += np.maximum(h, 0.0).sum(0)
            a = np.maximum(h, 0.0)
    for l in LAYERS:
        dead = fires[l] == 0
        d256 = f256[l] == 0
        e = (esum[l] / N) ** 2
        res[l].append(dead.mean()); res256[l].append(d256.mean())
        risk[l].append(e[d256].sum() / e.sum() if e.sum() > 0 else 0.0)
print("=" * 92)
print("dead97 — мертві нейрони. %d мереж, N=%d семплів" % (NETS, N))
print("=" * 92)
print(" шар | (а) мертвих на 40000 |  (б) критерій 256  | (в) втрачена енергія від (б)")
print("     |  сер     min–max     |   сер    min–max   |   сер        max")
for l in LAYERS:
    a_, b_, r_ = np.array(res[l]), np.array(res256[l]), np.array(risk[l])
    print(" %3d | %6.1f%%  %5.1f–%5.1f%%  | %6.1f%%  %5.1f–%5.1f%% | %9.2e  %9.2e"
          % (l, 100*a_.mean(), 100*a_.min(), 100*a_.max(),
             100*b_.mean(), 100*b_.min(), 100*b_.max(), r_.mean(), r_.max()))
print("\nрозкид МІЖ мережами на шарі 31: (а) %.1f%%–%.1f%%  (б) %.1f%%–%.1f%%"
      % (100*min(res[31]), 100*max(res[31]), 100*min(res256[31]), 100*max(res256[31])))
print("середнє по ВСІХ шарах: (а) %.1f%%   (б) %.1f%%   ⟹ критерій ловить %.0f%% мертвих"
      % (100*np.mean([np.mean(res[l]) for l in LAYERS]),
         100*np.mean([np.mean(res256[l]) for l in LAYERS]),
         100*np.mean([np.mean(res256[l]) for l in LAYERS])/max(1e-9, np.mean([np.mean(res[l]) for l in LAYERS]))))
