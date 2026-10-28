"""18.10.2026 (a2): латка «корекція середнього загальною ridge-таблицею» для будь-якого нащадка v39.
Ознаки за граматикою e30_a (_tab_feats, LordProcciuto, MIT): базові (1, mu, var, sigma, alpha, |alpha|, phi, Phi, pred,
D3, D21n, K4, sigma*phi, pk2..pk4, g4, d21s), h<k> = sigma*phi*He_k(alpha), модулятори s3, s4, s33, dn, v, ds, cr, c2, k4.
Стандартизація колонок згорнута в beta офлайн (delta = X @ B[l]), тож у сабміті лише ~F елементних операцій на шар.
Env у сабміті: ECORR_C (множник, 0 = вимкнено), ECORR_LMIN (з якого шару застосовувати).
Запуск: python3 make_v40corr.py <src_estimator.py> <tables.json> <key> <out.py> [default_c]
"""
import json, sys

src, tabf, key, outp = sys.argv[1:5]
c_def = sys.argv[5] if len(sys.argv) > 5 else "1.0"
T = json.load(open(tabf))[key]
feats, beta, mean, scale = T['feats'], T['beta'], T['mean'], T['scale']
assert feats[0] == '1'
B = []
for l in range(len(beta)):
    b = [beta[l][f] / scale[l][f] for f in range(len(feats))]
    b[0] = beta[l][0] / scale[l][0] - sum(mean[l][f] * beta[l][f] / scale[l][f] for f in range(len(feats)))
    B.append(b)

s = open(src).read()
anchor = 'NO_CORR = _os.environ.get("V17_NO_CORR", "1") == "1"\n'
assert s.count(anchor) == 1
block = anchor + (
    "# V41corr (18.10.2026): online mean correction with a general ridge table, fitted on the chain's own\n"
    "# dumped state (mini 10..59). Feature grammar after LordProcciuto's e30_a _tab_feats (MIT).\n"
    "# Column standardisation folded into the rows offline: delta = X @ CORR_TAB_B[layer].\n"
    "CORR_TAB_KEY = %r\n" % key +
    "CORR_TAB_FEATS = %r\n" % feats +
    "CORR_TAB_B = %r\n" % B +
    "CORR_C = float(_os.environ.get(\"ECORR_C\", \"%s\"))\n" % c_def +
    "CORR_LMIN = int(_os.environ.get(\"ECORR_LMIN\", \"0\"))\n")
s = s.replace(anchor, block)

lam_anchor = 'LAM = [c * float(_os.environ.get("V17_LAM_SCALE", "0.95")) for c in LAM]\n'
assert s.count(lam_anchor) == 1
s = s.replace(lam_anchor, lam_anchor + (
    "# V41corr: optional lambda variants for the scan (both off by default = unchanged):\n"
    "# ECORR_LAM_NEW=1 -> LAM_NEW refitted by LordProcciuto's e30_a (MIT), same V17 scale;\n"
    "# ECORR_LAM_TILT=t -> LAM[i] *= 1 + t*max(0,(i-3)/12)^1.5 (e30_a X_LAM_TILT form).\n"
    "if _os.environ.get(\"ECORR_LAM_NEW\", \"0\") == \"1\":\n"
    "    LAM = [c * float(_os.environ.get(\"V17_LAM_SCALE\", \"0.95\")) for c in [\n"
    "        0.005268912, 0.008429589942857143, 0.01010993142857143, 0.010705728, 0.010863401142857143,\n"
    "        0.01069187657142857, 0.01031066057142857, 0.00964608, 0.009130048457142858, 0.008553545142857142,\n"
    "        0.008061133714285714, 0.007597588114285714, 0.007180902857142857, 0.006738212571428571,\n"
    "        0.0062716032, 0.0061720539428571424]]\n"
    "_LAM_TILT = float(_os.environ.get(\"ECORR_LAM_TILT\", \"0\"))\n"
    "if _LAM_TILT != 0.0:\n"
    "    LAM = [c * (1.0 + _LAM_TILT * (max(0.0, (i - 3) / 12.0) ** 1.5)) for i, c in enumerate(LAM)]\n"))

fn_anchor = 'def _zero_diag(A):\n'
assert s.count(fn_anchor) == 1
fn = '''def _corr_delta(li, mode, mu, var, sigma, alpha, phi, Phi, pk1v, pk2v, pk3v, pk4v, D3, D21, K4_sigma,
                g4row, C_off, ones_n, T1, Brow):
    """V41corr: delta = X @ Brow, X = CORR_TAB_FEATS evaluated per neuron (n x F)."""
    z = ones_n * 0.0
    m1 = mode == 1
    D3f = D3 if (m1 and D3 is not None) else z
    if m1 and D21 is not None:
        d21n = fnp.abs(D21, out=T1) @ ones_n
        d21s = D21 @ ones_n
    else:
        d21n = d21s = z
    k4 = ones_n * (float(K4_sigma) if K4_sigma is not None else 0.0)
    g4f = g4row if (m1 and g4row is not None) else z
    if C_off is not None:
        crs = C_off @ ones_n
        cr2 = fnp.multiply(C_off, C_off, out=T1) @ ones_n
    else:
        crs = cr2 = z
    sp = sigma * phi
    base = {"1": ones_n, "mu": mu, "var": var, "sigma": sigma, "alpha": alpha, "|alpha|": fnp.abs(alpha),
            "phi": phi, "Phi": Phi, "pred": pk1v, "D3": D3f, "D21n": d21n, "K4": k4, "sigma*phi": sp,
            "pk2": pk2v, "pk3": pk3v, "pk4": pk4v, "g4": g4f, "d21s": d21s}
    iv = 1.0 / var
    mods = {}
    he = [ones_n, alpha]
    cols = []
    for nm in CORR_TAB_FEATS:
        if nm in base:
            cols.append(base[nm])
            continue
        hb, _, md = nm.partition("*")
        k = int(hb[6:]) if hb.startswith("sphiHe") else int(hb[1:])
        while len(he) <= k:
            j = len(he) - 1
            he.append(alpha * he[j] - he[j - 1] * float(j))
        v = sp * he[k]
        if md:
            if md not in mods:
                if md == "s3":
                    mods[md] = D3f * (iv / sigma)
                elif md == "s4":
                    mods[md] = g4f * (iv * iv)
                elif md == "s33":
                    t3 = D3f * (iv / sigma)
                    mods[md] = t3 * t3
                elif md == "dn":
                    mods[md] = d21n * iv
                elif md == "v":
                    mods[md] = var
                elif md == "ds":
                    mods[md] = d21s * iv
                elif md == "cr":
                    mods[md] = crs * iv
                elif md == "c2":
                    mods[md] = cr2 * (iv * iv)
                elif md == "k4":
                    mods[md] = k4
                else:
                    raise KeyError(md)
            v = v * mods[md]
        cols.append(v)
    X = fnp.stack(cols, axis=1)
    return X @ Brow


'''
s = s.replace(fn_anchor, fn + fn_anchor)

old = "            if riders and not NO_CORR:\n                if mode == 1:\n                    D3f = D3\n"
assert s.count(old) == 1
new = ("            if riders and CORR_C != 0.0 and li >= CORR_LMIN:\n"
       "                delta = _corr_delta(li, mode, mu, var, sigma, alpha, phi, Phi, pk1v, pk2v, pk3v, pk4v, D3, D21,\n"
       "                                    K4_sigma, g4row, C_off, ones_n, T1, corr_rows[li])\n"
       "                if CORR_C != 1.0:\n"
       "                    delta = delta * CORR_C\n"
       "            elif riders and not NO_CORR:\n                if mode == 1:\n                    D3f = D3\n")
s = s.replace(old, new)

old2 = "        beta_rows = [fnp.asarray(b, dtype=f32) for b in CORR_BETA]\n"
assert s.count(old2) == 1
s = s.replace(old2, old2 + "        corr_rows = [fnp.asarray(b, dtype=f32) for b in CORR_TAB_B]   # V41corr\n")
open(outp, 'w').write(s)
print('ok', outp, key, 'F =', len(feats), 'layers =', len(B))
