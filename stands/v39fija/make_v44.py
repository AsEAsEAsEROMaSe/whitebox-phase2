"""v44 (18.10.2026) = v43 + КОВАРІАЦІЯ ЛИШЕ НА ЖИВИХ ВХОДАХ (частина E-C, як dead_fis у e30_a для C_pre).
Cnext = коваріація після ReLU шару l-1 (вхід W шару l). Рядки і стовпці мертвих нейронів (alpha < -DEAD_TH) майже нульові
(Var(ReLU)/sigma^2 ~ 0.001 при alpha = -2.5), тож
    W Cnext W^T ~ W[:, L] Cnext[L, L] W[:, L]^T,   L = nb_perm[:nR] (живі першими, nR кратне ROWALIGN, як у DEADROW).
Ціна: n*nR^2 + n^2*nR (з симетричним блоком 0.75) замість n^3 + 0.75 n^3. V44_COVDEAD=0 = v43 біт у біт.
Передбачення (до прогону): флопи -1.0…-2.0%, MSE x0.997…1.000, бал x1.01…1.02, шанс 70% що бал > x1.005."""
src = open('/home/trim/wb2lean/v39fija/estimator_v43.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('OLDROWS = _os.environ.get("V40_OLDROWS", "1") == "1"',
    'COVDEAD = _os.environ.get("V44_COVDEAD", "1") == "1"   # V44: W C W^T on live inputs only\n'
    'OLDROWS = _os.environ.get("V40_OLDROWS", "1") == "1"')

# covr exists on every layer
rep('            mu = W @ mu\n            C_pre = None\n',
    '            mu = W @ mu\n            C_pre = None\n            covr = None\n')

# (1) W C on live input rows of C (dead_on branch with a stack)
rep('                    if newborn is not None:\n'
    '                        WCb = pool.get("wc", (n, n))\n'
    '                        smm.mm(W[None, None], Cnext[None, None], WCb[None, None], smm.level(n, n, n, s_lev))\n'
    '                else:\n',
    '                    if newborn is not None and COVDEAD and DEADROW and nb_perm is not None and alive_list and nR < n:\n'
    '                        # V44: C rows/cols of dead inputs ~0: W C W^T = W[:, L] C[L, L] W[:, L]^T\n'
    '                        rsc = nb_perm[:nR]\n'
    '                        Wrc = pool.get("wrc", (n, nR))\n'
    '                        fnp.copyto(Wrc, fnp.take(W, rsc, axis=1))\n'
    '                        Crc = pool.get("crc", (nR, nR))\n'
    '                        fnp.copyto(Crc, fnp.take(fnp.take(Cnext, rsc, axis=0), rsc, axis=1))\n'
    '                        WCr = pool.get("wcr", (n, nR))\n'
    '                        smm.mm(Wrc[None, None], Crc[None, None], WCr[None, None], smm.level(n, nR, nR, s_lev))\n'
    '                        covr = (rsc, Wrc, WCr)\n'
    '                    elif newborn is not None:\n'
    '                        WCb = pool.get("wc", (n, n))\n'
    '                        smm.mm(W[None, None], Cnext[None, None], WCb[None, None], smm.level(n, n, n, s_lev))\n'
    '                else:\n')

# (2) C_pre / var from the restricted product
rep('                WC = WCb if dead_on else legs["AP0"][k, 1]\n'
    '                if trim:\n',
    '                if covr is not None:\n'
    '                    rsc, Wrc, WCr = covr\n'
    '                    # dead inputs keep their own variance on the diagonal: sum_d W_id^2 C_dd\n'
    '                    _dd = nb_perm[Wrc.shape[1]:]\n'
    '                    _Wd = fnp.take(W, _dd, axis=1)\n'
    '                    _vdc = fnp.matmul(fnp.multiply(_Wd, _Wd), fnp.take(fnp.diag(Cnext), _dd))\n'
    '                    if trim:\n'
    '                        var = fnp.maximum(fnp.sum(fnp.multiply(WCr, Wrc, out=pool.get("wcrw", WCr.shape)), axis=1) + _vdc, 1e-10)\n'
    '                    else:\n'
    '                        Yrc = pool.get("yrc", (Wrc.shape[1], n))\n'
    '                        fnp.copyto(Yrc, fnp.take(w32, rsc, axis=0))\n'
    '                        C_pre = self._sym_product(WCr, Yrc, n, NN("cpre"), min(CPRE_LEV, s_lev))\n'
    '                        fnp.add(C_pre, fnp.diag(_vdc), out=C_pre)\n'
    '                WC = None if covr is not None else (WCb if dead_on else legs["AP0"][k, 1])\n'
    '                if covr is not None:\n'
    '                    pass\n'
    '                elif trim:\n')

# (3) _sym_product with a general contraction width m = X.shape[1]
rep('        X3 = pool.get("sym_x", (3, 1, h, n))\n'
    '        Y3 = pool.get("sym_y", (3, 1, n, h))\n'
    '        O3 = pool.get("sym_o", (3, 1, h, h))\n',
    '        m = X.shape[1]   # V44: contraction width (n, or the live-input count)\n'
    '        X3 = pool.get("sym_x", (3, 1, h, m))\n'
    '        Y3 = pool.get("sym_y", (3, 1, m, h))\n'
    '        O3 = pool.get("sym_o", (3, 1, h, h))\n')
rep('        self._smm.mm(X3, Y3, O3, self._smm.level(h, n, h, lev))\n',
    '        self._smm.mm(X3, Y3, O3, self._smm.level(h, m, h, lev))\n')

src = src.replace('"""v40 (старий рівень', '"""v44 (коваріація лише на живих входах, 18.10.2026) поверх v43 = v40 (старий рівень', 1)
open('/home/trim/wb2lean/v39fija/estimator_v44.py', 'w').write(src)
print('ok', len(src))
