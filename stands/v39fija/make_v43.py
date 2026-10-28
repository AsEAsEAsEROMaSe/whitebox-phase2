"""v43 (18.10.2026) = кандидат: v40 з порогом мертвих 99 (тобто лише розпечені рядки, E-D; v40h ×1.013 3/4)
+ корекція середнього step_F87 (v41corr, ×1.095 10/10) + R_OLD 416 (×1.014). Будується з виправленого estimator_v40_oldrows.py."""
import subprocess, sys
S = '/home/trim/wb2lean/v39fija/'
src = open(S + 'estimator_v40_oldrows.py').read()
a = 'OLD_DEAD = float(_os.environ.get("V40_OLD_DEAD", "2.5"))'
b = '    R_OLD = int(_os.environ.get("V21_R_OLD", "384"))'
assert src.count(a) == 1 and src.count(b) == 1
src = src.replace(a, 'OLD_DEAD = float(_os.environ.get("V40_OLD_DEAD", "99"))').replace(b, b.replace('"384"', '"416"'))
open(S + '_v43_pre.py', 'w').write(src)
subprocess.check_call([sys.executable, S + 'make_v41corr.py', S + '_v43_pre.py', S + 'fit87c_tables.json', 'step_F87', S + 'estimator_v43.py'])
print('ok')
