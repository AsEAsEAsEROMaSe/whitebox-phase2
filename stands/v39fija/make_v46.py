"""v46 (18.10.2026) = v45 + ТОЧНІ ЗАОЩАДЖЕННЯ e30_a (E28: ueye, tri, nest2/nestg), ціни з профілю v43 (prof43, s0):
  ueye : під FIJA під-базис U = [I; 0] (вибір перших r2 стовпців) ⟹ take(Qc)[:, :r2] замість take(Qc) @ U (0.67 u)
         і inner[:, :r2] += inner2 замість inner + inner2 @ U^T (0.59 u). Біт у біт (множення на 1 і додавання 0).
  tri  : Mt = R0^-1 Q0^T з трикутною R0 бісекцією на блоки (глибина V46_TRI): matmul + малі solve замість LU 416
         (тариф solve 2r^3/3 + 2 r^2 m ⟹ ~1.1 r^2 m): 1.31 u ⟹ ~0.69 u. Не біт у біт (інше округлення).
  nestg: F2 = R11^-1 [R11 R12] F = F_top + G F_bot, G = R11^-1 R12 (один solve на 192 стовпці, tri):
         1.38 u ⟹ ~0.43 u. Не біт у біт.
  gsum : програма нелінійних членів: IND2 one-hot, групи суцільні ⟹ PK2[g] = sum(ABstack[t0:t1]) замість
         (g, T) @ (T, n^2) за 2gTn^2 і reshape-копії (2.5 + 0.36 u ⟹ ~0.13 u). Не біт у біт (порядок суми).
Разом ~5.4 u = -2.4% флопів (локально). Передбачення (правка 18.10 до прогону, з gsum): флопи x1.020…1.027 (дешевше),
MSE x0.999…1.001, бал x1.019…1.027; шанс 85% що бал > x1.01.
V46_UEYE=0 V46_TRI=0 V46_NESTG=0 V46_GSUM=0 = v45 біт у біт; лише V46_UEYE=1 теж біт у біт."""
src = open('/home/trim/wb2lean/v39fija/estimator_v45.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('SBR_LEV = int(_os.environ.get("V30_SBR_LEV", "3"))  # V30: level cap of those products\n',
    'SBR_LEV = int(_os.environ.get("V30_SBR_LEV", "3"))  # V30: level cap of those products\n'
    'E46_UEYE = _os.environ.get("V46_UEYE", "1") == "1"   # V46: U = [I; 0] under FIJA: products by U are column copies\n'
    'E46_TRI = int(_os.environ.get("V46_TRI", "3"))         # V46: triangular solve by block bisection, this many levels (0: LU solve)\n'
    'E46_NESTG = _os.environ.get("V46_NESTG", "1") == "1"  # V46: F2 = F_top + (R11^-1 R12) F_bot\n'
    '\n'
    '\n'
    'def _tri_solve(R, B, d):\n'
    '    """V46: X = R^-1 B for upper-triangular R: X2 = R22^-1 B2, X1 = R11^-1 (B1 - R12 X2), d levels, LU solve at the leaves."""\n'
    '    r = R.shape[0]\n'
    '    if d <= 0 or r < 32:\n'
    '        return fnp.linalg.solve(R, B)\n'
    '    h = r // 2\n'
    '    X2 = _tri_solve(R[h:, h:], B[h:], d - 1)\n'
    '    X1 = _tri_solve(R[:h, :h], fnp.subtract(B[:h], fnp.matmul(R[:h, h:], X2)), d - 1)\n'
    '    return fnp.concatenate([X1, X2], axis=0)\n')

# tri: the projection operator of the joiner
rep('                        Mt = fnp.linalg.solve(R0, Q0.T)     # (r, n) = R0^-1 Q0^T\n',
    '                        Mt = _tri_solve(R0, Q0.T, E46_TRI) if E46_TRI > 0 else fnp.linalg.solve(R0, Q0.T)     # (r, n) = R0^-1 Q0^T\n')

# nestg: the outgoing tier-1 member's coordinates on the first r2 basis columns
rep('                        F2 = fnp.linalg.solve(R0[:r2, :r2], R0[:r2, :] @ Fcat)          # (r2, 2n)\n',
    '                        if E46_NESTG:\n'
    '                            # V46: R11^-1 [R11 R12] F = F_top + (R11^-1 R12) F_bot\n'
    '                            _G = _tri_solve(R0[:r2, :r2], R0[:r2, r2:], E46_TRI) if E46_TRI > 0 else fnp.linalg.solve(R0[:r2, :r2], R0[:r2, r2:])\n'
    '                            F2 = fnp.add(Fcat[:r2], fnp.matmul(_G, Fcat[r2:]))\n'
    '                        else:\n'
    '                            F2 = fnp.linalg.solve(R0[:r2, :r2], R0[:r2, :] @ Fcat)          # (r2, 2n)\n')

# ueye (1): tier-2 legs on the kept rows
rep('                        _QUr = fnp.matmul(fnp.take(Qc, _rs, axis=0), U)\n',
    '                        if E46_UEYE and FIJA:\n'
    '                            _QUr = fnp.take(Qc[:, :r2], _rs, axis=0)   # V46: U = [I; 0]\n'
    '                        else:\n'
    '                            _QUr = fnp.matmul(fnp.take(Qc, _rs, axis=0), U)\n')

# ueye (2): lift of the tier-2 contraction in _dslices35
rep('            lift = fnp.matmul(inner2[0], U2.T, out=self._pool.get(("lift", nO), (nO, U2.shape[0])))\n'
    '            inner = lift if inner is None else fnp.add(inner, lift, out=inner)\n'
    '        if inner is not None and orow is not None:\n',
    '            if E46_UEYE and FIJA:\n'
    '                # V46: U2 = [I; 0]: the lift is a column copy\n'
    '                r2u = inner2.shape[2]\n'
    '                if inner is None:\n'
    '                    lift = self._pool.get(("lift", nO), (nO, U2.shape[0]))\n'
    '                    fnp.copyto(lift, 0.0)\n'
    '                    fnp.copyto(lift[:, :r2u], inner2[0])\n'
    '                    inner = lift\n'
    '                else:\n'
    '                    fnp.add(inner[:, :r2u], inner2[0], out=inner[:, :r2u])\n'
    '            else:\n'
    '                lift = fnp.matmul(inner2[0], U2.T, out=self._pool.get(("lift", nO), (nO, U2.shape[0])))\n'
    '                inner = lift if inner is None else fnp.add(inner, lift, out=inner)\n'
    '        if inner is not None and orow is not None:\n')

# gsum: PK2[g] = sum of the contiguous block of terms of group g (IND2 is one-hot per term, groups in order)
rep('            FP[mode]["IND2T"] = fnp.asarray([list(r) for r in zip(*p["IND2"])], dtype=f32)\n',
    '            FP[mode]["IND2T"] = fnp.asarray([list(r) for r in zip(*p["IND2"])], dtype=f32)\n'
    '            # V46: group bounds of the one-hot IND2 (terms of a group are contiguous)\n'
    '            _gt = [row.index(1.0) for row in p["IND2"]]\n'
    '            assert all(sum(row) == 1.0 for row in p["IND2"]) and _gt == sorted(_gt)\n'
    '            G2B[mode] = [(_gt.index(g), len(_gt) - _gt[::-1].index(g)) if g in _gt else (0, 0)\n'
    '                         for g in range(len(p["IND2"][0]) if p["IND2"] else 0)]\n')
rep('        FP = {}\n        for mode, p in self._progs.items():\n',
    '        FP = {}\n        G2B = {}\n        for mode, p in self._progs.items():\n')
rep('                    pk2f = pool.get(("pk2f", g_), (g_, n * n))\n'
    '                    fnp.matmul(fpm["IND2T"], fnp.reshape(ABstack, (T_, n * n)), out=pk2f)\n'
    '                    PK2 = fnp.reshape(pk2f, (g_, n, n))\n',
    '                    if E46_GSUM:\n'
    '                        # V46: one-hot group indicator: block sums instead of the (g, T) @ (T, n^2) product\n'
    '                        PK2 = pool.get(("pk2g", g_), (g_, n, n))\n'
    '                        for _g, (_t0, _t1) in enumerate(G2B[pmode]):\n'
    '                            if _t1 > _t0:\n'
    '                                fnp.sum(ABstack[_t0:_t1], axis=0, out=PK2[_g])\n'
    '                            else:\n'
    '                                fnp.copyto(PK2[_g], 0.0)\n'
    '                    else:\n'
    '                        pk2f = pool.get(("pk2f", g_), (g_, n * n))\n'
    '                        fnp.matmul(fpm["IND2T"], fnp.reshape(ABstack, (T_, n * n)), out=pk2f)\n'
    '                        PK2 = fnp.reshape(pk2f, (g_, n, n))\n')
rep('E46_NESTG = _os.environ.get("V46_NESTG", "1") == "1"  # V46: F2 = F_top + (R11^-1 R12) F_bot\n',
    'E46_NESTG = _os.environ.get("V46_NESTG", "1") == "1"  # V46: F2 = F_top + (R11^-1 R12) F_bot\n'
    'E46_GSUM = _os.environ.get("V46_GSUM", "1") == "1"    # V46: nonlin PK2 = block sums (IND2 one-hot)\n')

src = src.replace('"""v45 (ранги','"""v46 (точні заощадження e30_a ueye/tri/nestg, 18.10.2026) поверх v45 (ранги', 1)
open('/home/trim/wb2lean/v39fija/estimator_v46.py', 'w').write(src)
print('ok', len(src))
