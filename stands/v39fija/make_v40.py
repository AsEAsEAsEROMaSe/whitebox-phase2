"""v40 = v39 + СТАРИЙ РІВЕНЬ ЛИШЕ В ПОТРІБНИХ РЯДКАХ (E-C + E-D, 18.10.2026; ідея як dead_fis / X_VB_HOT у e30_a).
Рядки поточного шару з alpha < -V40_OLD_DEAD (мертві, внесок ~0) і alpha > V40_OLD_HOT (розпечені, ReLU ~ тотожність)
не отримують старих ніг: формування Qc@F (tier 1, tier 2 через QU), хаб inner = lap^T F і inner Qc^T рахуються
лише на живих рядках (кількість округлено вгору до ROWALIGN), решта рядків = 0. Молоді джерела НЕ чіпаються.
Формування старих ніг перенесено з лінійного кроку до _dslices (там уже відома alpha цього шару); між цими
місцями старі ноги ніде не читаються. V40_OLDROWS=0 = v39 біт у біт; V40_OLD_HOT<=0 вимикає лише E-D."""
src = open('/home/trim/wb2lean/v39fija/estimator_v39_fija.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('FIJA = _os.environ.get("V39_FIJA", "1") == "1"',
    'OLDROWS = _os.environ.get("V40_OLDROWS", "1") == "1"   # V40: old level only on live, non-hot rows\n'
    'OLD_DEAD = float(_os.environ.get("V40_OLD_DEAD", "2.5"))   # V40: rows with alpha < -OLD_DEAD get no old legs\n'
    'OLD_HOT = float(_os.environ.get("V40_OLD_HOT", "3.0"))     # V40: rows with alpha > OLD_HOT get no old legs (<=0: off)\n'
    'FIJA = _os.environ.get("V39_FIJA", "1") == "1"')

# (0) old_pending must exist on layers where the source block is skipped but the newborn creates A_st
rep('            if A_st is not None and not skip_src:\n',
    '            old_pending = False\n'
    '            if A_st is not None and not skip_src:\n')

# (1) defer the old-leg formation
rep('                if ka > kb:\n'
    '                    # formed dense legs of the tier-1 sources (pre-wick of this layer):\n',
    '                old_pending = OLDROWS and dead_on and ka > 0\n'
    '                if ka > kb and not old_pending:\n'
    '                    # formed dense legs of the tier-1 sources (pre-wick of this layer):\n')
rep('                if kb > 0:\n'
    '                    # V24: tier-2 legs through the sub-basis\n',
    '                if kb > 0 and not old_pending:\n'
    '                    # V24: tier-2 legs through the sub-basis\n')

# (2) form them on the kept rows right before the dslices
rep('                _ds = self._dslices35 if dead_on else self._dslices\n',
    '                bufs["orow"] = None\n'
    '                if A_st is not None and not skip_src and old_pending:\n'
    '                    # V40: kept rows = live and not hot at THIS layer, kept first; padded to ROWALIGN\n'
    '                    _ao = mu / fnp.sqrt(var)\n'
    '                    _kf = (_ao >= -OLD_DEAD).astype(f32)\n'
    '                    if OLD_HOT > 0.0:\n'
    '                        _kf = _kf * (_ao <= OLD_HOT).astype(f32)\n'
    '                    _op = fnp.argsort(1.0 - _kf, kind="stable")\n'
    '                    _oip = fnp.argsort(_op, kind="stable")\n'
    '                    _nk = int(fnp.sum(_kf))\n'
    '                    _oR = min(n, -(-max(_nk, 1) // ROWALIGN) * ROWALIGN)\n'
    '                    _rs = _op[:_oR]\n'
    '                    if ka > kb:\n'
    '                        m1 = ka - kb\n'
    '                        _tc = fnp.zeros((2 * m1, 1, n, n), dtype=f32)\n'
    '                        smm.mm(fnp.take(Qc, _rs, axis=0)[None, None], fap4[2 * fa_off:2 * (fa_off + m1)],\n'
    '                               _tc[:, :, :_oR, :], smm.level(_oR, r_old, n, s_sb))\n'
    '                        fnp.copyto(legs4["AP0"][2 * kb:2 * ka], fnp.take(_tc, _oip, axis=2))\n'
    '                        del _tc\n'
    '                    if kb > 0:\n'
    '                        _tc = fnp.zeros((2 * kb, 1, n, n), dtype=f32)\n'
    '                        _QUr = fnp.matmul(fnp.take(Qc, _rs, axis=0), U)\n'
    '                        smm.mm(_QUr[None, None], fap24[:2 * kb], _tc[:, :, :_oR, :],\n'
    '                               smm.level(_oR, r2, n, s_sb))\n'
    '                        fnp.copyto(legs4["AP0"][:2 * kb], fnp.take(_tc, _oip, axis=2))\n'
    '                        del _tc\n'
    '                    if _oR < n:\n'
    '                        bufs["orow"] = (_op, _oip, _oR)\n'
    '                _ds = self._dslices35 if dead_on else self._dslices\n')

# (3) old-level hub and inner Qc^T on the kept rows only (inside _dslices35)
rep('        smm = self._smm\n'
    '        inner = None\n'
    '        if ka > kb:\n'
    '            r1 = sb1.shape[2]\n'
    '            inner = self._pool.get("inner", (1, n, r1))\n'
    '            smm.hub(bufs["lap4"][2 * kb:2 * ka], sb1, inner, smm.level(n, n, r1, s_sb))\n'
    '            inner = inner[0]\n'
    '        if kb > 0:\n'
    '            r2_ = sb2.shape[2]\n'
    '            inner2 = self._pool.get("inner2", (1, n, r2_))\n'
    '            smm.hub(bufs["lap4"][:2 * kb], sb2, inner2, smm.level(n, n, r2_, s_sb))\n'
    '            lift = fnp.matmul(inner2[0], U2.T, out=self._pool.get("lift", (n, U2.shape[0])))\n'
    '            inner = lift if inner is None else fnp.add(inner, lift, out=inner)\n',
    '        smm = self._smm\n'
    '        inner = None\n'
    '        orow = bufs.get("orow")\n'
    '        nO = n if orow is None else orow[2]\n'
    '        rsO = None if orow is None else orow[0][:nO]\n'
    '        if ka > kb:\n'
    '            r1 = sb1.shape[2]\n'
    '            inner = self._pool.get(("inner", nO), (1, nO, r1))\n'
    '            Lo = bufs["lap4"][2 * kb:2 * ka] if orow is None else fnp.take(bufs["lap4"][2 * kb:2 * ka], rsO, axis=2)\n'
    '            smm.hub(Lo, sb1, inner, smm.level(nO, n, r1, s_sb))\n'
    '            inner = inner[0]\n'
    '        if kb > 0:\n'
    '            r2_ = sb2.shape[2]\n'
    '            inner2 = self._pool.get(("inner2", nO), (1, nO, r2_))\n'
    '            Lo2 = bufs["lap4"][:2 * kb] if orow is None else fnp.take(bufs["lap4"][:2 * kb], rsO, axis=2)\n'
    '            smm.hub(Lo2, sb2, inner2, smm.level(nO, n, r2_, s_sb))\n'
    '            lift = fnp.matmul(inner2[0], U2.T, out=self._pool.get(("lift", nO), (nO, U2.shape[0])))\n'
    '            inner = lift if inner is None else fnp.add(inner, lift, out=inner)\n'
    '        if inner is not None and orow is not None:\n'
    '            # V40: rows of the old-level D21 part on the kept rows only, scattered back (others 0)\n'
    '            # 4-D zeros: a 2-D square zeros is tagged symmetric and [None, None, :nO] keeps the tag\n'
    '            tcO = fnp.zeros((1, 1, n, n), dtype=fnp.float32)\n'
    '            smm.mm(inner[None, None], fnp.swapaxes(Qc[None, None], -1, -2), tcO[:, :, :nO, :],\n'
    '                   smm.level(nO, inner.shape[1], n, min(s_sb, SBR_LEV)))\n'
    '            oldD = fnp.take(tcO[0, 0], orow[1], axis=0)\n'
    '            inner = None\n'
    '        else:\n'
    '            oldD = None\n')

rep('            if inner is not None:\n'
    '                smm.mm(inner[None, None], fnp.swapaxes(Qc[None, None], -1, -2),\n'
    '                       bufs["t1"][None, None], smm.level(n, inner.shape[1], n, min(s_sb, SBR_LEV)))\n'
    '                fnp.add(D21, bufs["t1"], out=D21)\n'
    '        else:\n'
    '            smm.mm(inner[None, None], fnp.swapaxes(Qc[None, None], -1, -2),\n'
    '                   D21[None, None], smm.level(n, inner.shape[1], n, min(s_sb, SBR_LEV)))\n',
    '            if inner is not None:\n'
    '                smm.mm(inner[None, None], fnp.swapaxes(Qc[None, None], -1, -2),\n'
    '                       bufs["t1"][None, None], smm.level(n, inner.shape[1], n, min(s_sb, SBR_LEV)))\n'
    '                fnp.add(D21, bufs["t1"], out=D21)\n'
    '            if oldD is not None:\n'
    '                fnp.add(D21, oldD, out=D21)\n'
    '        elif oldD is not None:\n'
    '            fnp.copyto(D21, oldD)\n'
    '        else:\n'
    '            smm.mm(inner[None, None], fnp.swapaxes(Qc[None, None], -1, -2),\n'
    '                   D21[None, None], smm.level(n, inner.shape[1], n, min(s_sb, SBR_LEV)))\n')

src = src.replace('"""v39 (фіксований базис', '"""v40 (старий рівень лише в живих нерозпечених рядках, 18.10.2026) поверх v39 (фіксований базис', 1)
open('/home/trim/wb2lean/v39fija/estimator_v40_oldrows.py', 'w').write(src)
print('ok', len(src))
