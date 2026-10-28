"""v39 = v38_age + ФІКСОВАНИЙ БАЗИС СТАРИХ ДЖЕРЕЛ (X_FIJA з публічного e30_a, MIT; 18.10.2026).
Базис обирає range finder лише на першому уніоні; далі він тільки переноситься Qc <- W diag(w1) Qc
(без реортонормалізації, обрізання й ротацій). Новий старий член проєктується МНК: Qp = Q0 R0,
F = R0^-1 Q0^T [Aj|Pj]. Старі фактори лишаються на місці (слоти не зсуваються, fa_off росте).
Tier 2 = перші r2 стовпців фіксованого базису (U = [I; 0]); базис упорядковується ОДИН раз на першому
уніоні за Gram першого члена (r2 найбільших напрямків першими); перехід у tier 2:
F2 = R0[:r2,:r2]^-1 R0[:r2,:] F (МНК-проєкція на Qp[:, :r2]).
V39_FIJA=0 = v38 біт у біт."""
src = open('/home/trim/wb2sub/v38age/estimator_v38_age.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('SBR = _os.environ.get("V30_SBR", "1") == "1"  # V30: basis-rebuild products through Strassen\n',
    'SBR = _os.environ.get("V30_SBR", "1") == "1"  # V30: basis-rebuild products through Strassen\n'
    'FIJA = _os.environ.get("V39_FIJA", "1") == "1"   # V39: fixed old basis (e30_a X_FIJA): no range finder / rotations after the first join\n'
    'FIJA_PASS = int(_os.environ.get("V39_FIJA_PASS", "8"))   # V39: subspace passes ordering the fixed basis at the first join\n')

# (A) skip the range finder on later joins; least-squares projector onto the transported basis
rep('                    APj = legs4["AP0"][2 * j:2 * j + 2]   # (2, 1, n, n) = (Aj, Pj)\n'
    '                    for _pass in range(QPASS):\n',
    '                    APj = legs4["AP0"][2 * j:2 * j + 2]   # (2, 1, n, n) = (Aj, Pj)\n'
    '                    fija = FIJA and ka > 0\n'
    '                    if fija:\n'
    '                        # V39: the basis is the transported one; joiner = LS projection Qp^+ [Aj|Pj]\n'
    '                        Q0, R0 = fnp.linalg.qr(Qp)\n'
    '                        Mt = fnp.linalg.solve(R0, Q0.T)     # (r, n) = R0^-1 Q0^T\n'
    '                        Qn = Qp\n'
    '                    for _pass in range(0 if fija else QPASS):\n')

# (B) joiner factors: projector Mt (fija) or Qn^T; slot after the untouched old factors
rep('                    fa_side ^= 1\n'
    '                    fap_new, fap4_new = pool.get_pair(("fap", fa_side), (L - 1, 2, r_old, n))\n'
    '                    m1 = ka - kb\n'
    '                    if SBR:\n'
    '                        smm.mm(fnp.swapaxes(Qn[None, None], -1, -2), APj,\n'
    '                               fap4_new[2 * m1:2 * m1 + 2], smm.level(r_old, n, n, min(s_sb, SBR_LEV)))\n'
    '                        FAj = fap_new[m1, 0]\n'
    '                        FPj = fap_new[m1, 1]\n'
    '                    else:\n'
    '                        FAj = fnp.matmul(Qn.T, Aj, out=fap_new[m1, 0])\n'
    '                        FPj = fnp.matmul(Qn.T, Pj, out=fap_new[m1, 1])\n'
    '                    if ka > 0:\n',
    '                    if not fija:\n'
    '                        fa_side ^= 1\n'
    '                    fap_new, fap4_new = pool.get_pair(("fap", fa_side), (L - 1, 2, r_old, n))\n'
    '                    m1 = ka - kb\n'
    '                    sj_ = fa_off + m1 if fija else m1   # V39: joiner slot\n'
    '                    Pm = Mt if fija else fnp.swapaxes(Qn, -1, -2)\n'
    '                    if SBR:\n'
    '                        smm.mm(Pm[None, None], APj,\n'
    '                               fap4_new[2 * sj_:2 * sj_ + 2], smm.level(r_old, n, n, min(s_sb, SBR_LEV)))\n'
    '                        FAj = fap_new[sj_, 0]\n'
    '                        FPj = fap_new[sj_, 1]\n'
    '                    else:\n'
    '                        FAj = fnp.matmul(Pm, Aj, out=fap_new[sj_, 0])\n'
    '                        FPj = fnp.matmul(Pm, Pj, out=fap_new[sj_, 1])\n'
    '                    if ka > 0 and not fija:\n')

# (C) factor slab bookkeeping + Gram core only on the rebuild path; first-join ordering of the fixed basis
rep('                    FAP = fap_new[:m1 + 1]\n'
    '                    fap4 = fap4_new\n'
    '                    fa_off = 0\n'
    '                    FAo = FAP[:, 0]\n'
    '                    FPo = FAP[:, 1]\n'
    '                    Sj = (FAj @ (dAj * FAj.T)) + (FPj @ (dPj * FPj.T))\n'
    '                    Sg = Sj if Sg is None else Sg + Sj\n',
    '                    if fija:\n'
    '                        FAP = fap_new[fa_off:fa_off + m1 + 1]\n'
    '                    else:\n'
    '                        FAP = fap_new[:m1 + 1]\n'
    '                        fa_off = 0\n'
    '                    fap4 = fap4_new\n'
    '                    FAo = FAP[:, 0]\n'
    '                    FPo = FAP[:, 1]\n'
    '                    if not fija:\n'
    '                        Sj = (FAj @ (dAj * FAj.T)) + (FPj @ (dPj * FPj.T))\n'
    '                        Sg = Sj if Sg is None else Sg + Sj\n'
    '                    if FIJA and nest and ka == 0:\n'
    '                        # V39: order the fixed basis once: the r2 leading directions of the first\n'
    '                        # member\'s Gram first (= the tier-2 sub-basis U = [I; 0])\n'
    '                        Ur = fnp.copy(Sg[:, :r2])\n'
    '                        for _pass in range(FIJA_PASS):\n'
    '                            Ur, _ = fnp.linalg.qr(Sg @ Ur)\n'
    '                        Ie = fnp.zeros((r_old, r_old - r2), dtype=f32)\n'
    '                        fnp.fill_diagonal(Ie, 1.0)\n'
    '                        Vr, _ = fnp.linalg.qr(fnp.concatenate([Ur, Ie], axis=1))\n'
    '                        fnp.copyto(FAP[0, 0], Vr.T @ FAP[0, 0])\n'
    '                        fnp.copyto(FAP[0, 1], Vr.T @ FAP[0, 1])\n'
    '                        Qn = Qn @ Vr\n'
    '                        U = fnp.zeros((r_old, r2), dtype=f32)\n'
    '                        fnp.fill_diagonal(U, 1.0)\n')

# (D) tier-2 move with the fixed basis: LS projection on the first r2 columns, no rotation
rep('                    if kb_t > kb:\n'
    '                        assert kb_t == kb + 1, (li, kb, kb_t)\n'
    '                        jb = kb                             # stack slot of the mover\n',
    '                    if kb_t > kb and fija:\n'
    '                        assert kb_t == kb + 1, (li, kb, kb_t)\n'
    '                        Fcat = fnp.concatenate([FAo[0], FPo[0]], axis=1)                # (r1, 2n)\n'
    '                        F2 = fnp.linalg.solve(R0[:r2, :r2], R0[:r2, :] @ Fcat)          # (r2, 2n)\n'
    '                        fap2_cur, fap24_cur = pool.get_pair(("fap2", f2_side), (L - 1, 2, r2, n))\n'
    '                        fnp.copyto(fap2_cur[kb, 0], F2[:, :n])\n'
    '                        fnp.copyto(fap2_cur[kb, 1], F2[:, n:])\n'
    '                        FAP2 = fap2_cur[:kb + 1]\n'
    '                        fap24 = fap24_cur\n'
    '                        FA2 = FAP2[:, 0]\n'
    '                        FP2 = FAP2[:, 1]\n'
    '                        FAP = FAP[1:]\n'
    '                        fa_off += 1\n'
    '                        FAo = FAP[:, 0]\n'
    '                        FPo = FAP[:, 1]\n'
    '                        kb = kb_t\n'
    '                    elif kb_t > kb:\n'
    '                        assert kb_t == kb + 1, (li, kb, kb_t)\n'
    '                        jb = kb                             # stack slot of the mover\n')

src = src.replace('"""K3-simple', '"""v39 (фіксований базис старих, X_FIJA з e30_a, 18.10.2026) поверх v38_age.\nK3-simple', 1)
open('/home/trim/wb2lean/v39fija/estimator_v39_fija.py', 'w').write(src)
print('ok', len(src))
