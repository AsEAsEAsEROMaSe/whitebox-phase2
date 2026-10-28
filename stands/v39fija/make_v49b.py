"""v49 (19.10.2026) = v46 + ОБРІЗАННЯ СТОВПЦІВ ХАБУ МОЛОДИХ ДЖЕРЕЛ ЗА ВНЕСКОМ (young-atom cap з публічного
#336537 Susupro, github.com/logxio/fourth-cumulant, MIT; у них ablation x1.044 adj, крок s17 разом з
active-set C/B 0.200 -> 0.169).
У v46 стовпці хабу новонародженого джерела впорядковані за alpha, живі = alpha >= -DEAD_TH (на ранніх шарах
майже всі 1024). Тут: оцінка внеску стовпця q: sc_q = 3|w2_q| |A_q|^2 + sqrt(S3c_q^2 + 9 e_q^2 |A_q|^2 + m_q)
(+ feed sqrt(c1_q^2 |A_q|^2 + dgw_q^2) |y| у режимі regen), m_q = (L (Rr^T Rr) L^T)_qq; порядок = sc за спаданням,
лишаємо найменше стовпців з 1 - V49_Y_EPS суми sc, округлено вгору до 32 (KALIGN дає те саме).
Рядки переносу (V37 DEADROW) лишаються за alpha: окрема перестановка argsort(-alpha) і лічильник alpha >= -DEAD_TH.
Передбачення (до прогону): молоді стовпці ~0.88 n на ранніх шарах, флопи x0.95…0.97 (перенос + хаб молодих),
MSE x1.00…1.02, бал x1.02…1.05; шанс 60% що бал > x1.01.
V49_Y_EPS=0 = v46 біт у біт."""
src = open('/home/trim/wb2lean/v39fija/estimator_v47.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('E46_GSUM = _os.environ.get("V46_GSUM", "1") == "1"',
    'Y49_EPS = float(_os.environ.get("V49_Y_EPS", "0.005"))   # V49: young-atom cap (Susupro #336537): keep 1 - eps of the hub score (0: off)\n'
    'E46_GSUM = _os.environ.get("V46_GSUM", "1") == "1"')

# rows of the transport keep their own alpha order and count
rep('        nb_perm = nb_mrow = None   # V35: newborn hub permutation and live-column row mask\n',
    '        nb_perm = nb_mrow = None   # V35: newborn hub permutation and live-column row mask\n'
    '        nb_rperm = None   # V49: newborn row order (alpha descending) for the V37 row cut\n'
    '        rows_alive = []   # V49: per source, alpha >= -DEAD_TH count (rows)\n')
rep('                    if DEADROW and nb_perm is not None and alive_list:\n'
    '                        nR = min(n, -(-alive_list[-1] // ROWALIGN) * ROWALIGN)\n',
    '                    if DEADROW and nb_rperm is not None and rows_alive:\n'
    '                        nR = min(n, -(-rows_alive[-1] // ROWALIGN) * ROWALIGN)\n')
rep('                        rsel = nb_perm[:nR]\n',
    '                        rsel = nb_rperm[:nR]\n')

# birth: Rr_full before the permutation (the score reads it), score order for the hub columns
rep('            if dead_on:\n'
    '                # V35: hub order of this source = alpha descending; live = alpha >= -DEAD_TH\n'
    '                nb_perm = fnp.argsort(-alpha)\n'
    '                nA = int(fnp.sum(alpha >= -DEAD_TH))\n'
    '                nA = max(nA, 1)\n',
    '            Rr_full = pool.get("rr", (n, r + 2))\n'
    '            fnp.multiply(Bm.T, 3.0, out=Rr_full[:, :r])\n'
    '            fnp.copyto(Rr_full[:, r], u_b)\n'
    '            fnp.copyto(Rr_full[:, r + 1], y_b)\n'
    '            if dead_on:\n'
    '                # V35: hub order of this source = alpha descending; live = alpha >= -DEAD_TH\n'
    '                nb_rperm = fnp.argsort(-alpha)\n'
    '                nA = int(fnp.sum(alpha >= -DEAD_TH))\n'
    '                nA = max(nA, 1)\n'
    '                rows_alive.append(nA)\n'
    '                nb_perm = nb_rperm\n'
    '                if Y49_EPS > 0.0:\n'
    '                    # V49: young-atom cap: score of hub atom q (birth unit), keep 1 - eps of the sum\n'
    '                    _ab = legs["AP0"][k_b, 0]\n'
    '                    _aa = fnp.sum(fnp.multiply(_ab, _ab, out=T1), axis=0)\n'
    '                    _sc = _aa * (3.0 * fnp.abs(w2))\n'
    '                    _Ln = pool.get("l49", (n, r + 2))\n'
    '                    fnp.copyto(_Ln[:, :r], Q)\n'
    '                    fnp.copyto(_Ln[:, r], w1sq)\n'
    '                    fnp.copyto(_Ln[:, r + 1], zeros_n)\n'
    '                    _mq = fnp.sum(fnp.multiply(_Ln @ (Rr_full.T @ Rr_full), _Ln), axis=1)\n'
    '                    _sc = _sc + fnp.sqrt(S3c * S3c + 9.0 * (e_b * e_b) * _aa + _mq)\n'
    '                    if regen and mode == 1 and not NO_FEED:\n'
    '                        _sc = _sc + fnp.sqrt(c1_b * c1_b * _aa + dgw * dgw) * fnp.sqrt(fnp.sum(y_b * y_b))\n'
    '                    nb_perm = fnp.argsort(-_sc)\n'
    '                    _cs = fnp.cumsum(fnp.take(_sc, nb_perm))\n'
    '                    nA = int(fnp.sum(_cs < _cs[n - 1] * (1.0 - Y49_EPS))) + 1\n'
    '                    nA = max(1, min(n, nA))\n')
rep('            Rr_full = pool.get("rr", (n, r + 2))\n'
    '            fnp.multiply(Bm.T, 3.0, out=Rr_full[:, :r])\n'
    '            fnp.copyto(Rr_full[:, r], u_b)\n'
    '            fnp.copyto(Rr_full[:, r + 1], y_b)\n'
    '            # slot of the source born here',
    '            # slot of the source born here')

src = src.replace('"""v47', '"""v49 (young-atom cap Susupro, 19.10.2026) поверх v46 (точні заощадження', 1)
open('/home/trim/wb2lean/v39fija/estimator_v49b.py', 'w').write(src)
print('ok', len(src))
