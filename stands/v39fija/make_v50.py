"""19.10.2026 white-box-8c: v50 = v47 + готовий LT2-коректор Susupro (github.com/logxio/fourth-cumulant, MIT, сабміт #336537).
Дозвіл юзера 19.10 («перевір щоб не всунути щось що порушує і так»). Перевірено: ваги = 16 масивів float у base64
(лише дані), нові імпорти лише stdlib array/base64, усі обчислення predict через fnp (тарифікуються), розпаковка в setup.
Коректор навчений на 3048 публічних мережах (full 1000 + корпус keenanpepper 0..2047) на ланцюзі B тієї ж лінії V29:
бічний потік u_l = Phi_l*(W_l u_{l-1}) + g_l, g_l = MLP(35 ознак ланцюга + 16 перенесених + стан 32 + 3 від u),
вихід pk1 + delta + u_15. Ланцюг не чіпається. Ознаки беруться в тих самих місцях і з тих самих змінних.
Ознака "delta" лишається нульовою (ланцюг B учився з NO_CORR=1); V50_DFEAT=1 подає наш ECORR delta.
Env: V50_LT=0 вимикає коректор (тоді рядки = v47); ECORR_C=0 вимикає наш ECORR.
Запуск: python3 make_v50.py  ->  estimator_v50.py"""
E = '/home/trim/wb2lean/v39fija/'
src = open(E + 'estimator_v47.py').read()
their = open('/home/trim/ext/fc/repo/estimator.py').read().split('\n')


def lines(a, b):   # 1-based inclusive
    return '\n'.join(their[a - 1:b])


def rep(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


assert their[7776].startswith('LT_FE = [') and their[7811].strip().startswith('for k, (sh, tc, b)')
blob = ('# ---- V50: LT2 corrector of Susupro (github.com/logxio/fourth-cumulant, MIT, submission #336537) ----\n'
        + lines(7773, 7783) + '\n' + lines(7786, 7812) + '\n'
        + 'LT_OFF = _os.environ.get("V50_LT", "1") != "1"\n'
        + 'LT_DFEAT = _os.environ.get("V50_DFEAT", "0") == "1"\n\n\n')
src = rep(src, 'class Estimator(BaseEstimator):', blob + 'class Estimator(BaseEstimator):')

assert their[8683].strip().startswith('path2 = ') and their[8713].strip() == 'self._lt = lt2'
load = lines(8686, 8714).replace('d = _lt2_embedded() if _LT2_BLOB else fnp.load(path2)', 'd = _lt2_embedded()')
assert 'd = _lt2_embedded()' in load
load = '\n'.join(l[4:] if l.startswith('    ') else l for l in load.split('\n'))
setup_add = '        self._lt = None\n        if not LT_OFF:\n' + load + '\n'
head = '    def setup(self, ctx: SetupContext) -> None:\n        self._setup_rng = fnp.random.default_rng(ctx.seed)\n'
src = rep(src, head, head + setup_add)

assert their[9967].strip().startswith('def _lt2_layer') and their[10021].strip() == 'return u, nst'
src = rep(src, '    def setup(self, ctx: SetupContext) -> None:\n',
          lines(9968, 10022) + '\n\n    def setup(self, ctx: SetupContext) -> None:\n')

src = rep(src, '        riders = (n == 1024 and L == len(CORR_BETA))\n',
          '        riders = (n == 1024 and L == len(CORR_BETA))\n'
          '        lt = getattr(self, "_lt", None) if riders else None\n'
          '        lt_on = lt is not None\n'
          '        lt_ext = lt_on and lt["Fo"] > len(LT_FE)\n'
          '        lt_u = None\n        lt_prev = None\n        fd = None\n')

assert their[9469].strip() == 'if lt_on:' and their[9489].strip().startswith('fd["coff_abs"]')
src = rep(src, '            Phi = flops.stats.norm.cdf(alpha).astype(f32)\n',
          '            Phi = flops.stats.norm.cdf(alpha).astype(f32)\n' + lines(9470, 9490) + '\n')

src = rep(src, '            else:\n                delta = None\n\n            if last:\n',
          '            else:\n                delta = None\n'
          '            if lt_ext and LT_DFEAT and delta is not None:\n                fd["delta"] = delta\n\n'
          '            if last:\n')

src = rep(src, '            e_b = W_all[:, self._i12] - 2.0 * pk1v * w1\n',
          '            e_b = W_all[:, self._i12] - 2.0 * pk1v * w1\n'
          '            if lt_on:\n                fd["e_b"] = e_b\n                fd["w1"] = w1\n'
          '                fd["w2"] = w2\n                fd["s3c"] = S3c\n')

assert their[9781].strip() == 'fd["pred"] = K1v' and their[9793].strip() == 'fd["k11sum"] = fnp.sum(K11, axis=1)'
src = rep(src, '            K1v = K[(1,)]\n            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]\n',
          '            K1v = K[(1,)]\n            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]\n'
          '            if lt_on:\n' + lines(9782, 9794) + '\n')

src = rep(src, '                lam_prev = float(LAM[min(li, len(LAM) - 1)])\n',
          '                lam_prev = float(LAM[min(li, len(LAM) - 1)])\n'
          '                if lt_on:\n                    fd["g_post"] = g_prev\n                    fd["k22row"] = k22row\n')

src = rep(src, '            rows.append(mu)\n',
          '            if lt_on:\n'
          '                lt_u, lt_prev = self._lt2_layer(lt, li, fd, W, NN("ww") if lt_prev is not None else None,\n'
          '                                                lt_prev, n, False)\n'
          '            rows.append(mu)\n')

src = rep(src, '                rows.append(pk1v if delta is None else pk1v + delta)\n',
          '                row_ = pk1v if delta is None else pk1v + delta\n'
          '                if lt_on:\n                    fd["pred"] = pk1v\n'
          '                    lt_u, lt_prev = self._lt2_layer(lt, li, fd, W, NN("ww"), lt_prev, n, True)\n'
          '                    row_ = row_ + lt_u\n'
          '                rows.append(row_)\n')

# V50_DIAG=<file>: local diagnostic only (never in a submission): per layer, the standardized corrector inputs'
# |z| percentiles and the carried correction size, appended as JSON lines.
srcd = rep(src, '        x = fnp.concatenate(parts, axis=1)\n',
          '        x = fnp.concatenate(parts, axis=1)\n'
          '        if LT_DIAG:\n'
          '            import json as _json\n'
          '            zz = fnp.abs(x[:, :Fo + 16])\n'
          '            with open(LT_DIAG, "a") as _f:\n'
          '                _f.write(_json.dumps({"li": li, "zmax": fnp.max(zz, axis=0).tolist(),\n'
          '                                      "zmean": fnp.mean(zz, axis=0).tolist()}) + "\\n")\n')
srcd = rep(srcd, '        u = g if st is None else fd["Phi"] * Wu + g\n',
          '        u = g if st is None else fd["Phi"] * Wu + g\n'
          '        if LT_DIAG:\n'
          '            with open(LT_DIAG, "a") as _f:\n'
          '                _f.write(_json.dumps({"li": li, "g_rms": float(fnp.sqrt(fnp.mean(g * g))),\n'
          '                                      "u_rms": float(fnp.sqrt(fnp.mean(u * u))),\n'
          '                                      "pred_rms": float(fnp.sqrt(fnp.mean(fd["pred"] * fd["pred"]))) if fd.get("pred") is not None else -1.0}) + "\\n")\n')
srcd = rep(srcd, 'LT_DFEAT = _os.environ.get("V50_DFEAT", "0") == "1"\n',
          'LT_DFEAT = _os.environ.get("V50_DFEAT", "0") == "1"\nLT_DIAG = _os.environ.get("V50_DIAG", "")\n')

open(E + 'estimator_v50d.py', 'w').write(srcd)

# V50_FIX=1 (v50f): the diagnostic v50d showed our d21_rowsum (|z| mean up to 6.8, max 44), d21_colsum and lam
# (|z| 3.7) outside the corrector's training range: pin them to the training mean (z=0) and clip every z to +-4.
srcf = rep(src, '        Z0 = (fnp.sign(X0) * fnp.log1p(fnp.abs(X0) * 1e3) - lt["mzoL"][li]) * lt["izoL"][li]\n',
           '        Z0 = (fnp.sign(X0) * fnp.log1p(fnp.abs(X0) * 1e3) - lt["mzoL"][li]) * lt["izoL"][li]\n'
           '        if LT_FIX:\n'
           '            Z0 = fnp.clip(Z0, -4.0, 4.0) * lt["fixmask"]\n')
srcf = rep(srcf, '            self._lt = lt2\n',
           '            lt2["fixmask"] = fnp.array([0.0 if k in ("d21_rowsum", "d21_colsum", "lam") else 1.0\n'
           '                                        for k in (LT_FE + LT_FE2)[:fl[0]]], dtype=fnp.float32)\n'
           '            self._lt = lt2\n')
srcf = rep(srcf, 'LT_DFEAT = _os.environ.get("V50_DFEAT", "0") == "1"\n',
           'LT_DFEAT = _os.environ.get("V50_DFEAT", "0") == "1"\nLT_FIX = _os.environ.get("V50_FIX", "1") == "1"\n')
open(E + 'estimator_v50f.py', 'w').write(srcf)
open(E + 'estimator_v50.py', 'w').write(src)
print('estimator_v50.py', len(src))
