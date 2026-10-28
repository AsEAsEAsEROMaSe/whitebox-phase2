"""v36 = v35_dead + закон зрізу κ31 (форум 18246, kai_leal_gonzalez; наш вимір 16.10.2026 у lean regen, 8 дампів, LOO):
wk431_ij = b1_l · v_i · C_ij + b2_l · κ3_i · T_iij / v_i   (замість λ·C_off), T_iij = D21[i,j] (D21[a,c] = T_aac в обох кодах).
lean: 2.1994e-8 → 2.1118e-8 (×1.042, 8/8 мереж), транспонований зріз (контроль) ×0.957; b2 1.24–1.65 по шарах (у kai 1.17–1.41).
Ціна: 3 поелементні n² операції на шар (≈5e7 флопів разом). Таблиця b1,b2 фітована на всіх 8 дампах (v35/k31law_table.json).
V36_K31LAW=0 вимикає (точний шлях v35)."""
import json
src = open('/home/trim/wb2lean/v35/estimator_v35_dead.py').read()
def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:80])
    src = src.replace(a, b)

tab = json.load(open('/home/trim/wb2lean/v35/k31law_table.json'))
rows = [[0.0, 0.0]] + [tab[str(l)] for l in range(1, 16)]
tab_src = 'K31TAB = [' + ', '.join('(%.6e, %.6e)' % (a, b) for a, b in rows) + ']   # V36: (b1, b2) за шаром li (li=0 не вживається)\n'

rep('KALIGN = int(_os.environ.get("V35_KALIGN", "32"))        # V35: young hub width rounded up to this (Strassen block divisibility)\n',
    'KALIGN = int(_os.environ.get("V35_KALIGN", "32"))        # V35: young hub width rounded up to this (Strassen block divisibility)\n'
    'K31LAW = _os.environ.get("V36_K31LAW", "1") == "1"      # V36: (3,1) slice law b1 v_i C_ij + b2 D3_i D21_ij / v_i instead of lam C_off\n'
    + tab_src)

rep('                    wk431 = None if trim else fnp.multiply(C_off, (0.5 * METRIC_C * lam_prev), out=NN("wk431"))\n',
    '                    if K31LAW and not trim and D21 is not None:\n'
    '                        _b1, _b2 = K31TAB[min(li, len(K31TAB) - 1)]   # V36\n'
    '                        wk431 = fnp.multiply(C_off, (var * _b1)[:, None], out=NN("wk431"))\n'
    '                        _k31t = _zero_diag(fnp.multiply(D21, (D3 * _b2 / var)[:, None]))\n'
    '                        wk431 = fnp.add(wk431, _k31t, out=wk431)\n'
    '                    else:\n'
    '                        wk431 = None if trim else fnp.multiply(C_off, (0.5 * METRIC_C * lam_prev), out=NN("wk431"))\n')

src = src.replace('"""v35', '"""v36 (закон κ31, 16.10.2026) поверх v35', 1) if src.startswith('"""v35') else src
open('/home/trim/wb2lean/v36/estimator_v36_k31.py', 'w').write(src)
print('ok', len(src))
