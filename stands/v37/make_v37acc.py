"""v37acc (16.10.2026, юзер: «варіант на найбільшу точність, скор ігнорувати»): v36 з іншими значеннями за замовчуванням.
- V21_NO_CONFINE 1: старі джерела без спільного базису (lean +8.4% raw за конфайнмент, v29 noconfine MSE x0.956);
- V35_DEAD_TH 0: точний шлях v34 без вирізання мертвих (v35 MSE x1.004);
- R_FB 64: тонкі ноги D21 (продакшн v32: MSE x0.977/0.984, флопи x1.18);
- V26_STRASSEN_MIN 4096: Штрассен ніде не вмикається, усі добутки BLAS (менше округлення f32; стіна v34 = додавання Штрассена ~55%).
- V36_K31LAW: як задано аргументом (1 або 0).
Обмеження грейдера: бюджет 2.2e12 на мережу (util<=1), стіна 120 с на мережу (безпечна медіана <=100 с)."""
import sys
k31 = sys.argv[1] if len(sys.argv) > 1 else "1"
src = open('/home/trim/wb2lean/v36/estimator_v36_k31.py').read()
def rep(a, b, cnt=1):
    global src
    c = src.count(a)
    assert c == cnt, (c, a[:90])
    src = src.replace(a, b)
rep('NO_CONFINE = _os.environ.get("V21_NO_CONFINE", "0") == "1"', 'NO_CONFINE = _os.environ.get("V21_NO_CONFINE", "1") == "1"')
rep('DEAD_TH = float(_os.environ.get("V35_DEAD_TH", "2.5"))', 'DEAD_TH = float(_os.environ.get("V35_DEAD_TH", "0"))')
rep('STRASSEN_MIN = int(_os.environ.get("V26_STRASSEN_MIN", "12"))', 'STRASSEN_MIN = int(_os.environ.get("V26_STRASSEN_MIN", "4096"))')
rep('    R_FB = 16    # V18', '    R_FB = int(_os.environ.get("V37_R_FB", "64"))    # V37acc (було 16); V18')
rep('K31LAW = _os.environ.get("V36_K31LAW", "1") == "1"', 'K31LAW = _os.environ.get("V36_K31LAW", "%s") == "1"' % k31)
src = src.replace('"""v36', '"""v37acc (найбільша точність, 16.10.2026) поверх v36', 1) if src.startswith('"""v36') else src
out = '/home/trim/wb2lean/v37/estimator_v37acc_k%s.py' % k31
open(out, 'w').write(src); print('ok', out, len(src))
