"""Генератор варіантів (_REP, _LAM_SCALE) — обидва живляться з ТІЄЇ САМОЇ розбіжності."""
import sys, re
src = open("/mnt/win_d/llm/White-Box-P2/estimator_p2_rep0.py").read()
assert "_REP = 0.0" in src and "_LAM_SCALE" in src
for spec in sys.argv[1:]:
    rep, lam, tag = spec.split(":")
    out = re.sub(r"^_REP = [0-9.]+", "_REP = %s" % rep, src, count=1, flags=re.M)
    out2 = re.sub(r"^_LAM_SCALE = [0-9.]+", "_LAM_SCALE = %s" % lam, out, count=1, flags=re.M)
    assert out2 != out or lam == re.search(r"^_LAM_SCALE = ([0-9.]+)", src, re.M).group(1)
    open("/mnt/win_d/llm/White-Box-P2/estimator_p2_%s.py" % tag, "w").write(out2)
    print(tag, "_REP=%s _LAM_SCALE=%s" % (rep, lam))
