"""v39dump = v39 + дамп ознак корекції середнього на кожному шарі (E-A, 18.10.2026). ЛИШЕ ЛОКАЛЬНО, не пакувати
(імпортує numpy). V39_CORRDUMP=<dir> пише <dir>/net_NNN.npz: на шар (16, n) mu, var, pk1 (передбачене пост-середнє
без корекції), d3, d21n (рядкова сума |D21|), d21s (рядкова сума D21), k4s (скаляр). Значення ланцюга не міняються."""
src = open('/home/trim/wb2lean/v39fija/estimator_v39_fija.py').read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('NO_CORR = _os.environ.get("V17_NO_CORR", "1") == "1"\n',
    'NO_CORR = _os.environ.get("V17_NO_CORR", "1") == "1"\n'
    'CORRDUMP = _os.environ.get("V39_CORRDUMP", "")   # V39dump: local feature dump dir\n'
    'if CORRDUMP:\n'
    '    import numpy as _np\n'
    '    import time as _time\n')

rep('            # ---- online mean correction: delta = feats @ beta[l] (13 dots/neuron);\n',
    '            if CORRDUMP:\n'
    '                _z = ones_n * 0.0\n'
    '                _cd.append(dict(mu=_np.asarray(mu, dtype="float32"), var=_np.asarray(var, dtype="float32"),\n'
    '                                pk1=_np.asarray(pk1v, dtype="float32"),\n'
    '                                d3=_np.asarray(D3 if (mode == 1 and D3 is not None) else _z, dtype="float32"),\n'
    '                                d21n=_np.asarray(fnp.abs(D21) @ ones_n if (mode == 1 and D21 is not None) else _z, dtype="float32"),\n'
    '                                d21s=_np.asarray(D21 @ ones_n if (mode == 1 and D21 is not None) else _z, dtype="float32"),\n'
    '                                pk2=_np.asarray(pk2v, dtype="float32"), pk3=_np.asarray(pk3v, dtype="float32"),\n'
    '                                pk4=_np.asarray(pk4v, dtype="float32"),\n'
    '                                g4=_np.asarray(g4row if (mode == 1 and g4row is not None) else _z, dtype="float32"),\n'
    '                                crs=_np.asarray(C_off @ ones_n if C_off is not None else _z, dtype="float32"),\n'
    '                                cr2=_np.asarray((C_off * C_off) @ ones_n if C_off is not None else _z, dtype="float32"),\n'
    '                                k4s=float(K4_sigma) if K4_sigma is not None else 0.0))\n'
    '            # ---- online mean correction: delta = feats @ beta[l] (13 dots/neuron);\n')

rep('            if last:\n'
    '                rows.append(pk1v if delta is None else pk1v + delta)\n'
    '                break\n',
    '            if last:\n'
    '                rows.append(pk1v if delta is None else pk1v + delta)\n'
    '                if CORRDUMP:\n'
    '                    _np.savez(CORRDUMP + "/net_%d_%03d.npz" % (_time.time_ns(), ncall),\n'
    '                              **{k: _np.stack([d[k] for d in _cd]) for k in ("mu", "var", "pk1", "d3", "d21n", "d21s", "pk2", "pk3", "pk4", "g4", "crs", "cr2")},\n'
    '                              k4s=_np.array([d["k4s"] for d in _cd]))\n'
    '                break\n')

rep('        rows = []\n', '        rows = []\n        _cd = []   # V39dump\n')
open('/home/trim/wb2lean/v39fija/estimator_v39_dump.py', 'w').write(src)
print('ok')
