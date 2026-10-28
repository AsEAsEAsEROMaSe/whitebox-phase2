"""sb1 (19.10.2026) = ланцюг B Susupro (#336537, github.com/logxio/fourth-cumulant, MIT, sha256 2537…a669)
+ ФІКСОВАНИЙ БАЗИС СТАРИХ ДЖЕРЕЛ (наш v39 = X_FIJA з публічного e30_a, MIT): після першого уніону базис лише
переноситься Qc <- W diag(w1) Qc; новий член проєктується МНК (QR Qp = Q0 R0, F = R0^-1 Q0^T [Aj|Pj]),
старі фактори не обертаються; tier 2 = перші r2 стовпців (U = [I; 0]), базис упорядковується один раз на
першому уніоні за Gram першого члена; перехід у tier 2: F2 = R0[:r2,:r2]^-1 R0[:r2,:] F.
У нас (v38 -> v39) це дало бал x1.041 (10/10), флопи -7.5%, MSE +3.9%; у e30_a MSE -4.5%, флопи -2.8%.
Передбачення (до прогону, q4 мережі 3+4 проти q4_sb): флопи x0.93…0.96, MSE x0.99…1.05, бал x1.01…1.05;
шанс 60% що бал > x1.01. SB1_FIJA=0 = ланцюг B біт у біт."""
src = open('/home/trim/wb2lean/v39fija/estimator_sb.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('QPASS = int(_os.environ.get("V21_QPASS", "1"))  # V21: subspace-iteration passes of the basis range finder\n',
    'QPASS = int(_os.environ.get("V21_QPASS", "1"))  # V21: subspace-iteration passes of the basis range finder\n'
    'FIJA = _os.environ.get("SB1_FIJA", "1") == "1"   # sb1 (white-box-89, from e30_a X_FIJA): fixed old basis after the first join\n'
    'FIJA_PASS = int(_os.environ.get("SB1_FIJA_PASS", "8"))   # sb1: subspace passes ordering the fixed basis at the first join\n')

# (A) no range finder on later joins; least-squares projector onto the transported basis
rep('                    for _pass in range(QPASS):\n'
    '                        # Yq = G Om with G = Aj dA Aj^T + Pj dP Pj^T + Qp Sg Qp^T\n',
    '                    fija = FIJA and ka > 0\n'
    '                    if fija:\n'
    '                        # sb1: the basis is the transported one; joiner = LS projection Qp^+ [Aj|Pj]\n'
    '                        Q0, R0 = fnp.linalg.qr(Qp)\n'
    '                        Mt = fnp.linalg.solve(R0, Q0.T)     # (r, nj) = R0^-1 Q0^T\n'
    '                        Qn = Qp\n'
    '                    for _pass in range(0 if fija else QPASS):\n'
    '                        # Yq = G Om with G = Aj dA Aj^T + Pj dP Pj^T + Qp Sg Qp^T\n')

# (B) joiner factors into the slot after the untouched old factors; no rotation
rep('                    fa_side ^= 1\n'
    '                    fap_new, fap4_new = pool.get_pair(("fap", fa_side), (L - 1, 2, r_old, n))\n'
    '                    m1 = ka - kb\n'
    '                    FAj = join_mm(Qn.T, Aj, fap_new[m1, 0])\n'
    '                    FPj = join_mm(Qn.T, Pj, fap_new[m1, 1])\n'
    '                    if ka > 0:\n',
    '                    if not fija:\n'
    '                        fa_side ^= 1\n'
    '                    fap_new, fap4_new = pool.get_pair(("fap", fa_side), (L - 1, 2, r_old, n))\n'
    '                    m1 = ka - kb\n'
    '                    sj_ = fa_off + m1 if fija else m1   # sb1: joiner slot\n'
    '                    Pm = Mt if fija else Qn.T\n'
    '                    FAj = join_mm(Pm, Aj, fap_new[sj_, 0])\n'
    '                    FPj = join_mm(Pm, Pj, fap_new[sj_, 1])\n'
    '                    if ka > 0 and not fija:\n')

# (C) slab bookkeeping; Gram core only on the rebuild path; first-join ordering of the fixed basis
rep('                    FAP = fap_new[:m1 + 1]\n'
    '                    fap4 = fap4_new\n'
    '                    fa_off = 0\n'
    '                    FAo = FAP[:, 0]\n'
    '                    FPo = FAP[:, 1]\n'
    '                    # cmp5: Sj = FA dA FA^T + FP dP FP^T as ONE same-array Gram (billed half)\n'
    '                    gj = pool.get("gram_j", (r_old, 2 * n))\n'
    '                    fnp.multiply(FAj, fnp.sqrt(dA_list[j])[None, :], out=gj[:, :n])\n'
    '                    fnp.multiply(FPj, fnp.sqrt(dP_list[j])[None, :], out=gj[:, n:])\n'
    '                    Sj = fnp.einsum("ia,ja->ij", gj, gj)\n'
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
    '                        # cmp5: Sj = FA dA FA^T + FP dP FP^T as ONE same-array Gram (billed half)\n'
    '                        gj = pool.get("gram_j", (r_old, 2 * n))\n'
    '                        fnp.multiply(FAj, fnp.sqrt(dA_list[j])[None, :], out=gj[:, :n])\n'
    '                        fnp.multiply(FPj, fnp.sqrt(dP_list[j])[None, :], out=gj[:, n:])\n'
    '                        Sj = fnp.einsum("ia,ja->ij", gj, gj)\n'
    '                        Sg = Sj if Sg is None else Sg + Sj\n'
    '                    if FIJA and nest and ka == 0:\n'
    '                        # sb1: order the fixed basis once: the r2 leading directions of the first\n'
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

src = src.replace('"""K3-simple', '"""sb1 (white-box-89, 19.10.2026): ланцюг B Susupro #336537 (MIT) + фіксований базис старих (X_FIJA з e30_a, MIT).\nK3-simple', 1)
open('/home/trim/wb2lean/v39fija/estimator_sb1.py', 'w').write(src)
print('ok', len(src))
