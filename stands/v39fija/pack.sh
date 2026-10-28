#!/bin/bash
# 18.10.2026 a2: пакування кандидата: чиста тека /home/trim/wb2sub/<tag>/, validate, package, розпакувати,
# AST без numpy/scipy/torch на файлі З АРХІВУ, збіг з вихідником (md5), версії маніфесту flopscope 0.12.3 / whestbench 0.16.2.
# validate запускає оцінювач (важко) ⟹ лише коли немає whest run. Сабміт НЕ робить (лише з дозволу юзера).
# usage: pack.sh <estimator.py> <tag>
set -e
SRC=$1; TAG=$2; WB=/home/trim/fs123
if pgrep -f "bin/whest run" >/dev/null; then echo "іде whest run: не пакую"; exit 2; fi
D=/home/trim/wb2sub/$TAG; mkdir -p $D; cp "$SRC" $D/estimator_$TAG.py; cd $D
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
$WB/bin/whest validate --estimator estimator_$TAG.py --format plain 2>&1 | tail -15
$WB/bin/whest package --estimator estimator_$TAG.py --output sub_$TAG.tar.gz --yes 2>&1 | tail -5
rm -rf unpack; mkdir unpack; tar xzf sub_$TAG.tar.gz -C unpack; echo "архів:"; find unpack -type f | sed 's/^/  /'
$WB/bin/python3 -I - "$D" "$TAG" <<'PY'
import ast, glob, hashlib, json, sys
D, TAG = sys.argv[1], sys.argv[2]
src = open('%s/estimator_%s.py' % (D, TAG), 'rb').read()
ok = True
for f in glob.glob(D + '/unpack/**/*.py', recursive=True):
    b = open(f, 'rb').read(); bad = []
    for n in ast.walk(ast.parse(b)):
        if isinstance(n, ast.Import):
            bad += [a.name for a in n.names if a.name.split('.')[0] in ('numpy', 'scipy', 'torch')]
        if isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0] in ('numpy', 'scipy', 'torch'):
            bad.append(n.module)
    same = hashlib.md5(b).hexdigest() == hashlib.md5(src).hexdigest()
    print('AST %s: заборонені імпорти %s · збіг з вихідником %s' % (f.split('/unpack/')[1], bad or 'нема', same))
    ok &= not bad
for f in glob.glob(D + '/unpack/**/*.json', recursive=True):
    s = open(f).read()
    print('маніфест %s: flopscope 0.12.3 %s · whestbench 0.16.2 %s' % (f.split('/unpack/')[1], '0.12.3' in s, '0.16.2' in s))
    ok &= ('0.12.3' in s and '0.16.2' in s)
print('ПАКЕТ OK' if ok else 'ПАКЕТ НЕ OK')
PY
