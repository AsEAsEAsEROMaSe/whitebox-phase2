"""sbh (19.10.2026, white-box-8c) = база ланцюга B (sb1 або sb2) + АДАПТИВНА межа розпечених рядків (наш V40 OLD_HOT).
У ланцюгу B на обрізаних шарах (prune, ns < n, не trim) одиниці впорядковані за спаданням a_apx, і старі ноги
формуються лише на рядках lr0..ns; lr0 = S_SAT[li] з таблиці AW_S_SAT, а вона вся нульова ⟹ механізм вимкнений.
Тут lr0 = кількість рядків з a_apx > SBH_HOT цієї мережі, вниз до кратного SBH_ALIGN; той самий r0 іде в _dslices
(ті самі два місця, що читають S_SAT). SBH_HOT=0 (дефолт ВИМК.) = база біт у біт.
У нас (v40, точний alpha > 3.0) старі ноги лише на живих нерозпечених рядках.
ПЕРЕДБАЧЕННЯ (до прогону, q4 мережі 3+4 проти бази): HOT 3.0 ⟹ флопи ×0.96…0.99, MSE ×1.00…1.02, бал ×1.00…1.03;
шанс 45% що бал > ×1.01. Ризик: a_apx наближений (діагональ C попереднього шару), гарячі за a_apx ≈ гарячі за alpha.
Запуск: BASE=sb2 python3 make_sbh.py -> estimator_sbh.py"""
import os

E = '/home/trim/wb2lean/v39fija/'
BASE = os.environ.get('BASE', 'sb1')
src = open(E + 'estimator_%s.py' % BASE).read()


def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)


rep('S_SAT = _sched("AW_S_SAT", "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")\n',
    'S_SAT = _sched("AW_S_SAT", "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")\n'
    'SBH_HOT = float(_os.environ.get("SBH_HOT", "0"))     # sbh: adaptive saturated head, rows with a_apx > SBH_HOT (0: off)\n'
    'SBH_ALIGN = int(_os.environ.get("SBH_ALIGN", "32"))  # sbh: head size rounded down to this\n')

rep('                perm = fnp.argsort(-a_apx)\n',
    '                perm = fnp.argsort(-a_apx)\n'
    '                if SBH_HOT > 0.0:\n'
    '                    sat_li = (int(fnp.sum(a_apx > SBH_HOT)) // SBH_ALIGN) * SBH_ALIGN\n'
    '                else:\n'
    '                    sat_li = int(S_SAT[li])\n')

rep('                    lr0 = int(S_SAT[li])\n', '                    lr0 = min(sat_li, ns - SBH_ALIGN) if SBH_HOT > 0.0 else sat_li\n')
rep('r0=(int(S_SAT[li]) if (prune and ns < n and not trim) else 0),',
    'r0=(lr0 if (prune and ns < n and not trim) else 0),')
# Stale-row guard (as our V40: rows outside the kept block are ZERO, not left over from the ping-pong buffer):
rep("""                           legs4["AP1"][2 * kb:2 * ka][..., lr0:ns, :], smm.level(ns - lr0, r_old, n, s_sb))\n""",
    """                           legs4["AP1"][2 * kb:2 * ka][..., lr0:ns, :], smm.level(ns - lr0, r_old, n, s_sb))\n"""
    """                    if SBH_HOT > 0.0 and lr0 > 0:\n"""
    """                        fnp.copyto(legs4["AP1"][2 * kb:2 * ka][..., :lr0, :], 0.0)\n""")
rep("""                    smm.mm(QU[None, None][..., lr0:ns, :], fap24[:2 * kb], legs4["AP1"][:2 * kb][..., lr0:ns, :],\n"""
    """                           smm.level(ns - lr0, r2, n, s_sb))\n""",
    """                    smm.mm(QU[None, None][..., lr0:ns, :], fap24[:2 * kb], legs4["AP1"][:2 * kb][..., lr0:ns, :],\n"""
    """                           smm.level(ns - lr0, r2, n, s_sb))\n"""
    """                    if SBH_HOT > 0.0 and lr0 > 0:\n"""
    """                        fnp.copyto(legs4["AP1"][:2 * kb][..., :lr0, :], 0.0)\n""")
open(E + 'estimator_sbh.py', 'w').write(src)
print('estimator_sbh.py from', BASE, len(src))
