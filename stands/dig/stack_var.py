"""Офлайн-збірка дешевих варіантів предиктора з k4pp_{CHAIN}_d*.npz (stack_pred2.py raw).
Варіант: ім'я, LMIN (з якого шару L інжектувати), KB (скільки останніх народжень l ∈ [L−KB, L−1]), K3 (all | young ≤1 | y3 ≤3 |
slice | none), FRAC. Фіт LOO по дампах (γ4 ≈ c + a·гаус + b·κ3 на чутливих FRAC) → /home/trim/wblogs/k4pv_{CHAIN}_{ім'я}_d{i}.npy.
Кошторис (lean, без Штрассена): множення n²×(FRAC·n): напрямки + 3 гаус + член κ3 (лічильник по віку з NMM, ×кількість пар) або 2 (зріз).
1 множення n³ = 2.15e9 флопів = 0.000977 бюджету."""
import os, sys, json, numpy as np
CH = os.environ.get("CHAIN", "reg"); D = list(range(8))
R = {i: np.load('/home/trim/wblogs/k4pp_%s_d%d.npz' % (CH, i)) for i in D}
NL, n = R[0]['kt'].shape; U3 = 2.0 * n**3 / 2.199023255552e12
VARS = json.loads(os.environ["VARS"])
def comps(r, L, KB, K3):
    ls = range(max(0, L-KB), L)
    gs = sum(r['gsP'][L, l] for l in ls) if len(ls) else np.zeros(n)
    if K3 == "all": k3 = sum(r['k3P'][:, L, l].sum(0) for l in ls)
    elif K3 == "young": k3 = sum(r['k3P'][0, L, l] for l in ls)
    elif K3 == "y3": k3 = sum(r['k3P'][0, L, l] + r['k3P'][1, L, l] for l in ls)
    elif K3 == "slice": k3 = sum(r['k3sP'][L, l] for l in ls)
    else: k3 = np.zeros(n)
    return gs, k3
for name, LMIN, KB, K3, FR in VARS:
    out = {i: np.zeros((NL, n)) for i in D}; err = []
    for L in range(max(1, LMIN), NL):
        def feats(i):
            r = R[i]; s, t = r['s'][L], r['t'][L]; v2 = s**4; b = np.abs(t**2-1)*np.exp(-t**2/2)/(24*s**3); sel = np.argsort(-b*v2)[:int(round(FR*n))]
            gs, k3 = comps(r, L, KB, K3); X = np.stack([np.ones(n), gs/v2, k3/v2], 1) if K3 != "none" else np.stack([np.ones(n), gs/v2], 1)
            return X, r['kt'][L]/v2, sel, v2
        F = {i: feats(i) for i in D}
        for i in D:
            Xa = np.concatenate([F[j][0][F[j][2]] for j in D if j != i]); ya = np.concatenate([F[j][1][F[j][2]] for j in D if j != i])
            c = np.linalg.lstsq(Xa, ya, rcond=None)[0]; X, y, sel, v2 = F[i]; out[i][L] = (X @ c)*v2
            if L == NL-1: err.append(np.sqrt((((X @ c) - y)[sel]**2).mean()))
    for i in D: np.save('/home/trim/wblogs/k4pv_%s_%s_d%d.npy' % (CH, name, i), out[i])
    # кошторис на одну мережу (мережа 0: кількість множень у члені κ3 по віку)
    nmm = R[0]['NMM']; cost = 0.0
    for L in range(max(1, LMIN), NL):
        ls = range(max(0, L-KB), L); cost += len(ls) * (1 + 3)
        for l in ls:
            if K3 == "all": cost += nmm[:, l].sum()
            elif K3 == "young": cost += nmm[0, l]
            elif K3 == "y3": cost += nmm[0, l] + nmm[1, l]
            elif K3 == "slice": cost += 2
    print("%-10s LMIN=%2d KB=%2d K3=%-5s FRAC=%.2f | rms γ4 L15 %.4f | кошторис %.0f множень ≈ %.3f бюджету" %
          (name, LMIN, KB, K3, FR, np.mean(err) if err else float('nan'), cost, cost*FR*U3), flush=True)
