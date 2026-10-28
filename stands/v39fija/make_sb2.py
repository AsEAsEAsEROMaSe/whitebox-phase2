"""sb2 (19.10.2026) = sb1 + ТОЧНІ ЗАОЩАДЖЕННЯ під фіксованим базисом (наш v46, ідеї e30_a E28/E29, MIT):
  tri  : Mt = R0^-1 Q0^T трикутним розв'язком бісекцією (SB2_TRI рівнів) замість LU;
  nestg: F2 = F_top + (R11^-1 R12) F_bot;
  ueye : U = [I; 0] ⟹ Qc @ U = перші r2 стовпців (копія), підйом inner2 @ U^T = додавання в перші r2 стовпці.
У нас (v45 -> v46) разом із gsum ×1.025, ueye біт у біт. gsum у ланцюгу B уже є (PK2 range sums).
Передбачення (до прогону, проти sb1): флопи x0.985…0.99, MSE x1.000 (±0.2%), бал x1.01…1.015.
SB2_TRI=0 SB2_NESTG=0 SB2_UEYE=0 = sb1 біт у біт."""
src = open('/home/trim/wb2lean/v39fija/estimator_sb1.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('FIJA_PASS = int(_os.environ.get("SB1_FIJA_PASS", "8"))   # sb1: subspace passes ordering the fixed basis at the first join\n',
    'FIJA_PASS = int(_os.environ.get("SB1_FIJA_PASS", "8"))   # sb1: subspace passes ordering the fixed basis at the first join\n'
    'E2_TRI = int(_os.environ.get("SB2_TRI", "3"))           # sb2: triangular solve by block bisection (0: LU solve)\n'
    'E2_NESTG = _os.environ.get("SB2_NESTG", "1") == "1"    # sb2: F2 = F_top + (R11^-1 R12) F_bot\n'
    'E2_UEYE = _os.environ.get("SB2_UEYE", "1") == "1"      # sb2: U = [I; 0] under FIJA: products by U are column copies\n'
    '\n'
    '\n'
    'def _tri_solve(R, B, d):\n'
    '    """sb2: X = R^-1 B for upper-triangular R: X2 = R22^-1 B2, X1 = R11^-1 (B1 - R12 X2), d levels, LU solve at the leaves."""\n'
    '    r = R.shape[0]\n'
    '    if d <= 0 or r < 32:\n'
    '        return fnp.linalg.solve(R, B)\n'
    '    h = r // 2\n'
    '    X2 = _tri_solve(R[h:, h:], B[h:], d - 1)\n'
    '    X1 = _tri_solve(R[:h, :h], fnp.subtract(B[:h], fnp.matmul(R[:h, h:], X2)), d - 1)\n'
    '    return fnp.concatenate([X1, X2], axis=0)\n')

rep('                        Mt = fnp.linalg.solve(R0, Q0.T)     # (r, nj) = R0^-1 Q0^T\n',
    '                        Mt = _tri_solve(R0, Q0.T, E2_TRI) if E2_TRI > 0 else fnp.linalg.solve(R0, Q0.T)     # (r, nj) = R0^-1 Q0^T\n')

rep('                        F2 = fnp.linalg.solve(R0[:r2, :r2], R0[:r2, :] @ Fcat)          # (r2, 2n)\n',
    '                        if E2_NESTG:\n'
    '                            _G = _tri_solve(R0[:r2, :r2], R0[:r2, r2:], E2_TRI) if E2_TRI > 0 else fnp.linalg.solve(R0[:r2, :r2], R0[:r2, r2:])\n'
    '                            F2 = fnp.add(Fcat[:r2], fnp.matmul(_G, Fcat[r2:]))\n'
    '                        else:\n'
    '                            F2 = fnp.linalg.solve(R0[:r2, :r2], R0[:r2, :] @ Fcat)          # (r2, 2n)\n')

rep('                    self._mmS(Qc[lr0:ns], U, qub[lr0:ns])   # cmp7: Strassen\n',
    '                    if E2_UEYE and FIJA:\n'
    '                        fnp.copyto(qub[lr0:ns], Qc[lr0:ns, :r2])   # sb2: U = [I; 0]\n'
    '                    else:\n'
    '                        self._mmS(Qc[lr0:ns], U, qub[lr0:ns])   # cmp7: Strassen\n')

rep('                lift = self._mmS(inner2[0], U2.T, self._pool.get("lift", (mr, U2.shape[0])))   # cmp7: Strassen\n'
    '                inner = lift if inner is None else fnp.add(inner, lift, out=inner)\n',
    '                if E2_UEYE and FIJA:\n'
    '                    # sb2: U2 = [I; 0]: the lift is a column copy\n'
    '                    if inner is None:\n'
    '                        lift = self._pool.get("lift", (mr, U2.shape[0]))\n'
    '                        fnp.copyto(lift, 0.0)\n'
    '                        fnp.copyto(lift[:, :r2_], inner2[0])\n'
    '                        inner = lift\n'
    '                    else:\n'
    '                        fnp.add(inner[:, :r2_], inner2[0], out=inner[:, :r2_])\n'
    '                else:\n'
    '                    lift = self._mmS(inner2[0], U2.T, self._pool.get("lift", (mr, U2.shape[0])))   # cmp7: Strassen\n'
    '                    inner = lift if inner is None else fnp.add(inner, lift, out=inner)\n')

src = src.replace('"""sb1 (white-box-89, 19.10.2026)', '"""sb2 (white-box-89, 19.10.2026): sb1 + точні заощадження tri/nestg/ueye. sb1', 1)
open('/home/trim/wb2lean/v39fija/estimator_sb2.py', 'w').write(src)
print('ok', len(src))
