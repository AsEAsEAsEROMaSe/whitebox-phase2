"""sbs (19.10.2026, white-box-8c) = sb2 + множник сили LT2-коректора на виході: row_ = row_ + SBS_LT * u_15.
Гіпотеза white-box-89: на грейдері FIJA погіршила MSE на +3.2% (локально +0.6%), бо LT2 вчився на траєкторіях
ланцюга B без FIJA ⟹ корекція частково не туди. SBS_LT=1.0 (дефолт) = sb2 біт у біт.
ПЕРЕДБАЧЕННЯ (до прогону, q4, проти sb2): оптимум 0.85…1.0; 0.8 ⟹ MSE ×0.99…1.02; 1.2 ⟹ MSE ×0.97…1.00 гірше;
шанс, що 0.8 дасть > ×1.005: 25%."""
src = open('/home/trim/wb2lean/v39fija/estimator_sb2.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('LT_OFF = _os.environ.get("LT_OFF", "0") == "1"\n',
    'LT_OFF = _os.environ.get("LT_OFF", "0") == "1"\n'
    'SBS_LT = float(_os.environ.get("SBS_LT", "1.0"))   # sbs: strength of the LT2 correction at the output\n')
rep('                        row_ = row_ + lt_u\n',
    '                        row_ = row_ + (lt_u if SBS_LT == 1.0 else lt_u * SBS_LT)\n')
open('/home/trim/wb2lean/v39fija/estimator_sbs.py', 'w').write(src)
print('ok', len(src))
