#!/bin/bash
echo "ЗАСТАРІЛО 21.10.2026: файловий режим не кладе nn_asset.json в архів. Брати pack_s32f.sh (c2)"; exit 3
# 21.10.2026 f0: пакет s32lt = s32refire (koushik_rudra, MIT) + LT2 (Susupro, MIT), ваги вбудовуються lt2_embed.
# Як v39fija/pack.sh, плюс nn_asset.json і ліцензії в теці (грейдер бере файли теки). Сабміт НЕ робить (лише з дозволу юзера).
# usage: pack_s32.sh <lt2_weights.npz> <tag>
set -e
NPZ=$1; TAG=$2; WB=/home/trim/fs123; S=/home/trim/wb2lean/s32lt; R=/home/trim/wblogs/lit/acco/repo/r/submissions/337532_s32refire
if pgrep -f "bin/whest run" >/dev/null; then echo "іде whest run: не пакую"; exit 2; fi
D=/home/trim/wb2sub/$TAG; rm -rf $D; mkdir -p $D
$WB/bin/python3.12 /home/trim/ext/fc/repo/lab/lt2_embed.py $S/estimator_s32lt.py "$NPZ" $D/estimator_$TAG.py
cp $S/nn_asset.json $R/LICENSE $R/ALG053_LICENSE $D/; cd $D
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
$WB/bin/whest validate --estimator estimator_$TAG.py --format plain 2>&1 | tail -15
$WB/bin/whest package --estimator estimator_$TAG.py --output sub_$TAG.tar.gz --yes 2>&1 | tail -5
rm -rf unpack; mkdir unpack; tar xzf sub_$TAG.tar.gz -C unpack; echo "архів:"; find unpack -type f | sed 's/^/  /'
$WB/bin/python3 -I - "$D" "$TAG" <<'PY'
import ast, glob, hashlib, sys
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
    ok &= not bad and same
na = glob.glob(D + '/unpack/**/nn_asset.json', recursive=True)
print('nn_asset.json в архіві:', bool(na)); ok &= bool(na)
for f in glob.glob(D + '/unpack/**/*.json', recursive=True):
    if f.endswith('nn_asset.json'): continue
    s = open(f).read()
    print('маніфест %s: flopscope 0.12.3 %s · whestbench 0.16.2 %s' % (f.split('/unpack/')[1], '0.12.3' in s, '0.16.2' in s))
    ok &= ('0.12.3' in s and '0.16.2' in s)
print('ПАКЕТ OK' if ok else 'ПАКЕТ НЕ OK')
PY
