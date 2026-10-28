# 21.10.2026 f0: порт коректора LT2 (Susupro MIT, з sb2) у s32refire (ALG058 + зріз народження + fuse1, koushik_rudra MIT)
import re
S = open('sb2.py').read().split('\n'); A = open('alg.py').read().split('\n')
def find(L, pat, start=0, exact=False):
    for k in range(start, len(L)):
        if (L[k] == pat) if exact else (pat in L[k]): return k
    raise SystemExit('not found: ' + pat)
def block(L, a, b): return L[a:b + 1]
ins = []   # (index in A, lines) — insert AFTER index
# 1 module level
a = find(S, 'LT_FE = ["mu_pre"'); b = find(S, 'LT_DUMP_ON = _os.environ.get("LT_DUMP"')
ins.append((find(A, 'import gc as _gc'), block(S, a, b)))
# 2 setup
sa = find(S, '        self._lt = None'); sb = find(S, '                self._lt = lt2', sa)
ins.append((find(A, '        self._setup_rng = fnp.random.default_rng(ctx.seed)'), block(S, sa, sb)))
# 3 predict vars
la = find(S, '        lt = getattr(self, "_lt", None) if riders else None'); lb = find(S, '        fd = None', la)
ins.append((find(A, '        regen = riders and not NO_REGEN'), block(S, la, lb)))
# 4 fd after Phi
fa = find(S, '            if lt_on:', find(S, '            Phi = flops.stats.norm.cdf(alpha).astype(f32)', 9000))
fb = find(S, '                        fd["coff_abs"]', fa)
ins.append((find(A, '            Phi = flops.stats.norm.cdf(alpha).astype(f32)', 3000), block(S, fa, fb)))
# 5 delta, before res_e block (after delta = None else-branch)
k = find(A, '            if riders and li>0:', 3200)
ins.append((k - 1, ['            if lt_ext and delta is not None:', '                fd["delta"] = delta']))
# 6 last layer
k = find(A, '                if riders:_r=fnp.maximum(_r+res_e,0)')
ins.append((k - 1, ['                if lt_on:',
                '                    fd["pred"] = pk1v',
                '                    if lt is not None and lt.get("v2", False):',
                '                        lt_u, lt_prev = self._lt2_layer(lt, li, fd, W, NN("ww"), lt_prev, n, True)',
                '                    else:',
                '                        lt_u, lt_prev = self._lt_layer(lt, li, fd, W, NN("ww"), lt_prev, lt_u, n, None, lt_dump)',
                '                    if lt is not None:',
                '                        _r = _r + lt_u']))
k = find(A, '                res_e=Phi*(W@res_e)+learned')
ins.append((k, ['                if lt_dump:', '                    LT_RES.append((li, fnp.copy(res_e)))']))
ins.append((find(A, 'import gc as _gc') , ['LT_RES = []   # f0 21.10.2026: dump of the ALG058 nn_asset carried residual res_e per layer (li, res_e) after its update']))
# 7 K block before mu_pre_saved
ka = find(S, '            if lt_on:', find(S, '            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]', 9000)); kb = find(S, '                    fd["k11sum"]', ka)
ins.append((find(A, '            mu_pre_saved=mu') - 1, block(S, ka, kb)))
# 8 e_b
ea = find(S, '            if lt_on:', find(S, '            e_b = W_all[:, self._i12] - 2.0 * pk1v * w1', 9000)); eb = find(S, '                fd["s3c"] = S3c', ea)
ins.append((find(A, '            e_b = (W_all[:, self._i12] + m2v * w1)'), block(S, ea, eb)))
# 9 regen
k = find(A, '                lam_prev = float(LAM[min(li, len(LAM) - 1)])', 3600)
ins.append((k, ['                if lt_on:', '                    fd["g_post"] = g_prev', '                    fd["k22row"] = k22row']))
# 10 non-last LT call before rows.append(mu)
k = find(A, '            rows.append(mu)', 3600, exact=True)
ins.append((k - 1, ['            if lt_on:',
                    '                if lt is not None and lt.get("v2", False):',
                    '                    lt_u, lt_prev = self._lt2_layer(lt, li, fd, W, NN("ww") if lt_prev is not None else None,',
                    '                                                    lt_prev, n, False)',
                    '                else:',
                    '                    lt_u, lt_prev = self._lt_layer(lt, li, fd, W, NN("ww") if lt_prev is not None else None,',
                    '                                                   lt_prev, lt_u, n, None, lt_dump)']))
# 11 methods after _predict_core return
ma = find(S, '    def _lt2_layer(self, lt, li, fd, W, WW, st, n, last):')
mb = max(k for k in range(ma, len(S)) if S[k].startswith('        u = g if u is None else fd["Phi"] * (W @ u) + g'))
mb = find(S, '        return', mb)
k = find(A, '        return _out', 3600)
ins.append((k, [''] + block(S, ma, mb)))
for idx, lines in sorted(ins, key=lambda x: -x[0]):
    A[idx + 1:idx + 1] = lines
hdr = ['# s32lt (21.10.2026, team trim_qewas): s32refire + LT2 per-layer corrector of Susupro (MIT, github.com/logxio/fourth-cumulant),',
       '# ported from our sb2; LT_OFF=1 restores s32refire. Weights: lt2_weights.npz next to this file or embedded _LT2_BLOB.']
open('estimator_s32lt.py', 'w').write('\n'.join(hdr + A))
print('ok', len(A))
