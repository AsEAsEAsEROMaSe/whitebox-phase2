"""v45 (18.10.2026) = v43 + РАНГИ ТОНКИХ НІГ ЯК У e30_a: R_FB 16 -> 4 (D21 feedback, V18), R_RES 16 -> 8 (залишок S21, V17).
Профіль v43 (prof43, s0, 227 u): частина, що лінійно залежить від rfb, 14.4 u (Zf перенос 3.3, Xt/Yt 3.6, GYR/GXR 3.2,
Xf 3.2, range finder 1.1); від rres ~7.4 u (Z перенос 1.9, MP 2.0, PPL 1.8, PZ 1.8). e30_a: обидва ранги вниз,
тонкі ноги 6.6 u проти наших ~20. Ми міряли лише вгору (R_FB 64: MSE -2% за +18% флопів, вересень).
Передбачення (до прогону, (4, 8)): флопи -5.5…-6.5%, MSE x0.96…1.00, бал x1.02…1.06; шанс 65% що бал > x1.01.
V45_R_FB=16 V45_R_RES=16 = v43 біт у біт."""
src = open('/home/trim/wb2lean/v39fija/estimator_v43.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('    R_FB = 16    # V18:', '    R_FB = int(_os.environ.get("V45_R_FB", "4"))    # V45 (e30_a: 4); V18:')
rep('    R_RES = 16   # rank', '    R_RES = int(_os.environ.get("V45_R_RES", "8"))   # V45 (e30_a: 8); rank')
src = src.replace('"""v44', '"""v44', 1)
src = src.replace('"""v40 (старий рівень', '"""v45 (ранги тонких ніг 4/8, 18.10.2026) поверх v43 = v40 (старий рівень', 1)
open('/home/trim/wb2lean/v39fija/estimator_v45.py', 'w').write(src)
print('ok', len(src))
