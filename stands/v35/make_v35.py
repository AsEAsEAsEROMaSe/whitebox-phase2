"""v35 = v34_m12 + виключення мертвих хаб-стовпців молодих джерел (t_b < -V35_DEAD_TH при народженні).
Хаб кожного джерела переставлено за спаданням t_b при народженні (хаб = німий індекс), мертві стовпці A і P нульові,
молодий стек працює на перших K стовпцях (K = max живих серед молодих, кратне 32). C транспортується окремо."""
src = open('/home/trim/wb2lean/estimator_v34_m12.py').read()
def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:80])
    src = src.replace(a, b)

# ---- константи ----
rep('FASTNL = _os.environ.get("V33_FASTNL", "1") == "1"  # V33: nonlin term program without the 4-operand einsum\n',
    'FASTNL = _os.environ.get("V33_FASTNL", "1") == "1"  # V33: nonlin term program without the 4-operand einsum\n'
    'DEAD_TH = float(_os.environ.get("V35_DEAD_TH", "2.5"))   # V35: hub column j of a source is dead if alpha_b[j] < -DEAD_TH (<=0: off, exact v34 path)\n'
    'KALIGN = int(_os.environ.get("V35_KALIGN", "32"))        # V35: young hub width rounded up to this (Strassen block divisibility)\n')

# ---- стан ----
rep('        w1_prev = None  # wick w(1) of the previous layer, folded into WD\n',
    '        w1_prev = None  # wick w(1) of the previous layer, folded into WD\n'
    '        dead_on = riders and DEAD_TH > 0.0   # V35: suite shape only\n'
    '        alive_list = []   # V35: per source, number of live hub columns (leading, after the birth permutation)\n'
    '        Kcur = n          # V35: hub width of the young slots valid in the current AP0 side\n'
    '        nb_perm = nb_mrow = None   # V35: newborn hub permutation and live-column row mask\n'
    '        Cnext = pool.get("cnext", (n, n))   # V35: C of the previous layer (no longer rides in the newborn P slot)\n')

# ---- joiner: обнулити стовпці за Kcur ----
rep('                    j = ka  # the joiner (exactly one per layer)\n',
    '                    j = ka  # the joiner (exactly one per layer)\n'
    '                    if dead_on and Kcur < n:\n'
    '                        fnp.copyto(legs4["AP0"][2 * j:2 * j + 2, :, :, Kcur:], 0.0)   # V35: stale columns beyond the young width\n')

# ---- транспорт молодих: ширина K, без C ----
rep('''                extra = 2 if newborn is not None else 0
                if 2 * k + extra > 2 * ka:
                    smm.mm(W[None, None], legs4["AP0"][2 * ka:2 * k + extra],
                           legs4["AP1"][2 * ka:2 * k + extra], smm.level(n, n, n, s_lev))
''','''                if dead_on:
                    # V35: young width = max live hub columns among young slots + newborn
                    al = alive_list[ka:k] + ([alive_list[k]] if newborn is not None else [])
                    Knew = n
                    if al:
                        Knew = min(n, -(-max(al) // KALIGN) * KALIGN)
                    if Knew > Kcur and k > ka:
                        fnp.copyto(legs4["AP0"][2 * ka:2 * k, :, :, Kcur:Knew], 0.0)
                    Kcur = Knew
                    extra = 1 if newborn is not None else 0
                    if 2 * k + extra > 2 * ka:
                        smm.mm(W[None, None], legs4["AP0"][2 * ka:2 * k + extra, :, :, :Kcur],
                               legs4["AP1"][2 * ka:2 * k + extra, :, :, :Kcur], smm.level(n, n, Kcur, s_lev))
                    if newborn is not None:
                        WCb = pool.get("wc", (n, n))
                        smm.mm(W[None, None], Cnext[None, None], WCb[None, None], smm.level(n, n, n, s_lev))
                else:
                    extra = 2 if newborn is not None else 0
                    if 2 * k + extra > 2 * ka:
                        smm.mm(W[None, None], legs4["AP0"][2 * ka:2 * k + extra],
                               legs4["AP1"][2 * ka:2 * k + extra], smm.level(n, n, n, s_lev))
''')

# ---- шар 1: новонароджений сам ----
rep('''                    smm.mm(W[None, None], legs4["AP0"][0:2], legs4["AP1"][0:2],
                           smm.level(n, n, n, s_lev))
''','''                    if dead_on:
                        Kcur = min(n, -(-alive_list[0] // KALIGN) * KALIGN)
                        smm.mm(W[None, None], legs4["AP0"][0:1, :, :, :Kcur], legs4["AP1"][0:1, :, :, :Kcur],
                               smm.level(n, n, Kcur, s_lev))
                        WCb = pool.get("wc", (n, n))
                        smm.mm(W[None, None], Cnext[None, None], WCb[None, None], smm.level(n, n, n, s_lev))
                    else:
                        smm.mm(W[None, None], legs4["AP0"][0:2], legs4["AP1"][0:2],
                               smm.level(n, n, n, s_lev))
''')
rep('''                WC = legs["AP0"][k, 1]
''','''                WC = WCb if dead_on else legs["AP0"][k, 1]
''')
rep('''                fnp.copyto(legs["AP0"][k, 1], W)
''','''                if dead_on:
                    # V35: newborn P = W with its hub columns permuted, dead columns zero
                    fnp.multiply(fnp.take(W, nb_perm, axis=1), nb_mrow, out=legs["AP0"][k, 1])
                else:
                    fnp.copyto(legs["AP0"][k, 1], W)
''')

# ---- dslices виклик ----
rep('''                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,''',
'''                _ds = self._dslices35 if dead_on else self._dslices
                if dead_on:
                    bufs["Kh"] = Kcur
                D3, D21 = _ds(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,''')

# ---- wick молодих: лише перші Kcur стовпців ----
rep('''                fnp.multiply(legs4["AP0"][2 * ka:2 * kk], w1col[None, None],
                             out=legs4["AP0"][2 * ka:2 * kk])
''','''                if dead_on:
                    fnp.multiply(legs4["AP0"][2 * ka:2 * kk, :, :, :Kcur], w1col[None, None],
                                 out=legs4["AP0"][2 * ka:2 * kk, :, :, :Kcur])
                else:
                    fnp.multiply(legs4["AP0"][2 * ka:2 * kk], w1col[None, None],
                                 out=legs4["AP0"][2 * ka:2 * kk])
''')

# ---- C у власний буфер ----
rep('''            fnp.copyto(legs["AP0"][k_b, 1], C)
''','''            if dead_on:
                fnp.copyto(Cnext, C)
            else:
                fnp.copyto(legs["AP0"][k_b, 1], C)
''')

# ---- народження: перестановка хабу (перед записом статичних хаб-об'єктів) ----
rep('''            Rr_full = pool.get("rr", (n, r + 2))
''','''            if dead_on:
                # V35: hub order of this source = alpha descending; live = alpha >= -DEAD_TH
                nb_perm = fnp.argsort(-alpha)
                nA = int(fnp.sum(alpha >= -DEAD_TH))
                nA = max(nA, 1)
                nb_mrow = (fnp.arange(n) < nA).astype(f32)[None, :]
                # A leg of the newborn (born into its slot in natural order): permute + zero dead
                fnp.multiply(fnp.take(legs["AP0"][k_b, 0], nb_perm, axis=1), nb_mrow, out=T1)
                fnp.copyto(legs["AP0"][k_b, 0], T1)
                hp = lambda v: fnp.take(v, nb_perm)
            else:
                hp = lambda v: v
            Rr_full = pool.get("rr", (n, r + 2))
''')
rep('''            fnp.copyto(lb[k_b, :, :r], Q)
            fnp.copyto(lb[k_b, :, r], w1sq)
''','''            fnp.copyto(lb[k_b, :, :r], fnp.take(Q, nb_perm, axis=0) if dead_on else Q)
            fnp.copyto(lb[k_b, :, r], hp(w1sq))
''')
rep('''            newborn = (a_b, Rr_full, Lr_full, S3c, e_b, Ff_b)
            if rfb > 0:
                r1b = pool.get("r1t", (L - 1, n, rfb))
                r2b = pool.get("r2t", (L - 1, n, rfb))
                fnp.copyto(r1b[k_b], R1T_b)
                fnp.copyto(r2b[k_b], R2T_b)
''','''            newborn = (a_b, Rr_full, Lr_full, hp(S3c), hp(e_b), Ff_b)
            if rfb > 0:
                r1b = pool.get("r1t", (L - 1, n, rfb))
                r2b = pool.get("r2t", (L - 1, n, rfb))
                fnp.copyto(r1b[k_b], fnp.take(R1T_b, nb_perm, axis=0) if dead_on else R1T_b)
                fnp.copyto(r2b[k_b], fnp.take(R2T_b, nb_perm, axis=0) if dead_on else R2T_b)
''')
rep('''            w2b_list.append(w2)
            # V21: hub-column Gram weights of this source's legs (X1 = 3A, Y1 ~ A d(w2),
            # M ~ P d(s) + 3 A d(e)): A-type 9 + w2^2 + 9 e^2, P-type 1 + s^2
            dA_list.append(9.0 + w2 * w2 + 9.0 * e_b * e_b)
            dP_list.append(1.0 + S3c * S3c)
            c1_list.append(c1_b)
            c2_list.append(dgw)
            y_list.append(y_b)
''','''            w2b_list.append(hp(w2))
            # V21: hub-column Gram weights of this source's legs (X1 = 3A, Y1 ~ A d(w2),
            # M ~ P d(s) + 3 A d(e)): A-type 9 + w2^2 + 9 e^2, P-type 1 + s^2
            dA_list.append(hp(9.0 + w2 * w2 + 9.0 * e_b * e_b))
            dP_list.append(hp(1.0 + S3c * S3c))
            c1_list.append(hp(c1_b))
            c2_list.append(hp(dgw))
            y_list.append(y_b)
            if dead_on:
                alive_list.append(nA)
''')

# ---- новий _dslices35 ----
rep('''    # ------------------------------------------------------------------
    def _dslices(self, A_st,''','''    # ------------------------------------------------------------------
    def _grp35(self, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list, c1_list, c2_list,
               bufs, g0, g1, h, rres, rfb, Zf_st, R1T_st, R2T_st, need_d21):
        """V35: the per-slot part of _dslices for slots g0:g1 on their first h hub columns.
        Returns (D3, D21_fb, R^T Yk, sum_k PPL_k Z_k^T) contributions (row-indexed, n x n)."""
        sl = slice(g0, g1)
        k = g1 - g0
        A = A_st[sl, :, :h]
        P = P_st[sl, :, :h]
        Z = Z_st[sl]
        W2B = fnp.stack([v[:h] for v in w2b_list[g0:g1]], axis=0)[:, None, :]
        Sb = fnp.stack([v[:h] for v in s_list[g0:g1]], axis=0)[:, None, :]
        Eb = fnp.stack([v[:h] for v in e_list[g0:g1]], axis=0)[:, None, :]
        AP = fnp.multiply(A, P, out=bufs["ap"][sl, :, :h])
        PP = fnp.multiply(P, P, out=bufs["pp"][sl, :, :h])
        T = bufs["t"][sl, :, :h]
        MP = fnp.matmul(Z, fnp.swapaxes(L_st[sl, :h], -1, -2), out=bufs["mp"][sl, :, :h])
        fnp.multiply(MP, P, out=MP)
        fnp.multiply(PP, Sb, out=T)
        fnp.add(MP, T, out=MP)
        fnp.multiply(AP, Eb * 3.0, out=T)
        fnp.add(MP, T, out=MP)
        LA = LP = None
        if need_d21:
            LA = fnp.multiply(AP, W2B * 2.0, out=bufs["lap"][sl, 0, :, :h])
            fnp.multiply(PP, Eb, out=T)
            fnp.add(LA, T, out=LA)
            LP = fnp.multiply(A, A, out=bufs["lap"][sl, 1, :, :h])
            fnp.multiply(LP, W2B, out=LP)
            fnp.multiply(PP, Sb * (1.0 / 3.0), out=T)
            fnp.add(LP, T, out=LP)
            fnp.multiply(MP, 2.0 / 3.0, out=T)
            fnp.add(LP, T, out=LP)
        fnp.multiply(P, W2B, out=T)
        D3 = (fnp.einsum("kij,kij,kij->i", A, A, T) * 3.0
              + fnp.einsum("kij,kij->i", MP, P))
        D21_fb = None
        if rfb > 0:
            F1 = Zf_st[sl, :, :rfb]
            F2 = Zf_st[sl, :, rfb:]
            R1T = R1T_st[sl, :h]
            R2T = R2T_st[sl, :h]
            Xt = fnp.einsum("kiq,kjq->kij", F1, R1T, out=bufs["xt"][sl, :, :h])
            Yt = fnp.einsum("kiq,kjq->kij", F2, R2T, out=bufs["yt"][sl, :, :h])
            U = bufs["u"][sl, :, :h]
            fnp.multiply(A, Yt, out=T)
            fnp.multiply(Xt, A, out=U)
            fnp.multiply(U, W2B * (1.0 / 3.0), out=U)
            fnp.add(T, U, out=T)
            fnp.multiply(Xt, Yt, out=U)
            fnp.multiply(U, 1.0 / 3.0, out=U)
            fnp.add(T, U, out=T)
            D3 = D3 + fnp.einsum("kij,kij->i", P, T) * 3.0
            if need_d21:
                fnp.add(LP, T, out=LP)
                fnp.multiply(Xt, P, out=T)
                fnp.multiply(T, W2B * (1.0 / 3.0), out=T)
                fnp.add(LA, T, out=LA)
                fnp.multiply(P, Yt, out=U)
                fnp.add(LA, U, out=LA)
                fnp.multiply(Xt, P, out=T)
                fnp.multiply(T, 1.0 / 3.0, out=T)
                fnp.add(T, AP, out=T)                 # GY
                Xf = bufs["xt"][sl]                   # full (k, n, n) scratch for row x row results
                GYR = fnp.matmul(T, R2T, out=bufs["gyr"][sl])
                fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=Xf)
                D21_fb = fnp.sum(Xf, axis=0, out=bufs["d21fb"])
                fnp.multiply(AP, W2B, out=T)
                fnp.add(T, U, out=T)
                fnp.multiply(T, 1.0 / 3.0, out=T)     # GX
                GXR = fnp.matmul(T, R1T, out=bufs["gxr"][sl])
                fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=Xf)
                fnp.sum(Xf, axis=0, out=bufs["t1"])
                fnp.add(D21_fb, bufs["t1"], out=D21_fb)
                D21_fb = fnp.copy(D21_fb)
        C1 = fnp.stack([v[:h] for v in c1_list[g0:g1]], axis=0)
        C2 = fnp.stack([v[:h] for v in c2_list[g0:g1]], axis=0)
        Yk = Z[:, :, rres + 1]
        R = fnp.einsum("kij,kj->ki", AP, C1) + fnp.einsum("kij,kj->ki", PP, C2)
        D3 = D3 + fnp.einsum("ki,ki->i", R, Yk)
        if not need_d21:
            return D3, None, None, None
        MPs = bufs["mp"][sl, :, :h]
        fnp.multiply(P, C2[:, None, :], out=MPs)
        fnp.multiply(A, C1[:, None, :], out=T)
        fnp.add(T, MPs, out=T)
        fnp.multiply(T, (Yk * (2.0 / 3.0))[:, :, None], out=T)
        fnp.add(LP, T, out=LP)
        RY = fnp.matmul(R.T, Yk)
        PPL = fnp.matmul(PP, L_st[sl, :h], out=bufs["ppl"][sl])
        PZ = fnp.einsum("kiq,kcq->ic", PPL, Z)
        return D3, D21_fb, RY, PZ

    def _dslices35(self, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                   c1_list, c2_list, y_list, n, bufs, rres, rfb, Zf_st, R1T_st, R2T_st,
                   need_d21=True, ka=0, Qc=None, kb=0, U2=None, apb=None, apb4=None,
                   sb1=None, sb2=None, s_sb=0):
        """V35: _dslices with the old (confined, formed) slots 0:ka on the full hub and the
        young slots ka:k on their first Kh hub columns (dead hub columns are zero)."""
        k = len(w2b_list)
        Kh = bufs["Kh"]
        groups = []
        if ka > 0:
            groups.append((0, ka, n))
        if ka < k:
            groups.append((ka, k, Kh))
        D3 = None
        FB = RY = PZ = None
        for (g0, g1, h) in groups:
            d3, fb, ry, pz = self._grp35(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                                         c1_list, c2_list, bufs, g0, g1, h, rres, rfb,
                                         Zf_st, R1T_st, R2T_st, need_d21)
            D3 = d3 if D3 is None else D3 + d3
            if need_d21:
                FB = fb if FB is None else (FB if fb is None else FB + fb)
                RY = ry if RY is None else RY + ry
                PZ = pz if PZ is None else PZ + pz
        if not need_d21:
            return D3, None
        smm = self._smm
        inner = None
        if ka > kb:
            r1 = sb1.shape[2]
            inner = self._pool.get("inner", (1, n, r1))
            smm.hub(bufs["lap4"][2 * kb:2 * ka], sb1, inner, smm.level(n, n, r1, s_sb))
            inner = inner[0]
        if kb > 0:
            r2_ = sb2.shape[2]
            inner2 = self._pool.get("inner2", (1, n, r2_))
            smm.hub(bufs["lap4"][:2 * kb], sb2, inner2, smm.level(n, n, r2_, s_sb))
            lift = fnp.matmul(inner2[0], U2.T, out=self._pool.get("lift", (n, U2.shape[0])))
            inner = lift if inner is None else fnp.add(inner, lift, out=inner)
        D21 = bufs["d21"]
        if ka < k:
            hub = bufs["hub"]
            smm.hub(bufs["lap4"][2 * ka:2 * k, :, :, :Kh], apb4[2 * ka:2 * k, :, :, :Kh], hub[None],
                    smm.level(n, Kh, n, min(STRASSEN_HUB, self._s_hub)))
            fnp.copyto(D21, hub)
            if inner is not None:
                smm.mm(inner[None, None], fnp.swapaxes(Qc[None, None], -1, -2),
                       bufs["t1"][None, None], smm.level(n, inner.shape[1], n, min(s_sb, SBR_LEV)))
                fnp.add(D21, bufs["t1"], out=D21)
        else:
            smm.mm(inner[None, None], fnp.swapaxes(Qc[None, None], -1, -2),
                   D21[None, None], smm.level(n, inner.shape[1], n, min(s_sb, SBR_LEV)))
        fnp.multiply(RY, 1.0 / 3.0, out=RY)
        fnp.add(D21, RY, out=D21)
        fnp.multiply(PZ, 1.0 / 3.0, out=PZ)
        fnp.add(D21, PZ, out=D21)
        if FB is not None:
            fnp.add(D21, FB, out=D21)
        return D3, _zero_diag(D21)

    def _dslices(self, A_st,''')
open('/home/trim/wb2lean/v35/estimator_v35_dead.py', 'w').write(src)
print('ok', len(src))
