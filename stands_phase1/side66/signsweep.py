"""signsweep — розгортка по частці від'ємних ваг, PR-ранг стану по глибині.

Постала з аудиту write-up: у тексті стояло тлумачення «обидва чисті знаки
тривіальні», взяте з корпусної таблиці, ЧИЙ ШАР НЕ БУВ ЗАПИСАНИЙ. Замість
міркування — прямий вимір.

🔒 Пломби, які прогін мусить дати:
   S1 при 0% мінусів мережа лінійна з шару 2 ⟹ PR-ранг → 1.0 (промінь)
   S2 при 100% мінусів усі затвори шару 2 закриті ⟹ ранг РІВНО 0,
      і 100% семплів мовчать
   S3 при 50% профіль мусить відтворити записаний 165.6 → 2.8
"""
import numpy as np

n, D, N = 256, 32, 4096
CK = (0, 1, 4, 15, 31)


def build(frac_neg, seed=3):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(D):
        w = np.abs(r.standard_normal((n, n)) * np.sqrt(2.0 / n))
        out.append(np.where(r.random(w.shape) < frac_neg, -w, w))
    return out


def run(W, seed=7):
    a = np.random.default_rng(seed).standard_normal((N, n))
    ranks, silent = [], []
    for i, w in enumerate(W):
        a = np.maximum(a @ w, 0.0)
        silent.append(float(np.mean(np.all(a <= 0, axis=1))))
        if i in CK:
            ev = np.clip(np.linalg.eigvalsh(np.cov(a.T)), 0, None)
            ranks.append((i + 1, round(float(ev.sum() ** 2
                                             / max((ev ** 2).sum(), 1e-300)), 1)))
    return ranks, silent


if __name__ == "__main__":
    for f in (0.0, 0.25, 0.50, 1.0):
        r, s = run(build(f))
        print(f"мінусів {int(f*100):>3}%  PR-ранг {r}")
        print(f"           мовчить увесь шар: ш1 {s[0]:.3f} ш2 {s[1]:.3f} "
              f"ш5 {s[4]:.3f} ш32 {s[31]:.3f}")
