#!/bin/bash
# 21.10.2026 c2: як pack_s32.sh, але шаблон estimator_s32lt_fast.py (швидкий шар LT2). f0: пакет s32lt = s32refire (koushik_rudra, MIT) + LT2 (Susupro, MIT), ваги вбудовуються lt2_embed.
# Як v39fija/pack.sh, плюс nn_asset.json і ліцензії в теці (грейдер бере файли теки). Сабміт НЕ робить (лише з дозволу юзера).
# c2: режим ТЕКИ (estimator.py + nn_asset.json + ліцензії), бо файловий режим пакує лише estimator.py.
# usage: pack_s32.sh <lt2_weights.npz> <tag>
set -e
NPZ=$1; TAG=$2; WB=/home/trim/fs123; S=/home/trim/wb2lean/s32lt; R=/home/trim/wblogs/lit/acco/repo/r/submissions/337532_s32refire
if pgrep -f "bin/whest run" >/dev/null; then echo "іде whest run: не пакую"; exit 2; fi
D=/home/trim/wb2sub/$TAG/sub; O=/home/trim/wb2sub/$TAG; rm -rf $O; mkdir -p $D
$WB/bin/python3.12 /home/trim/ext/fc/repo/lab/lt2_embed.py $S/estimator_s32lt_fast.py "$NPZ" $D/estimator.py
cp $S/nn_asset.json $R/LICENSE $R/ALG053_LICENSE $D/; cd $O
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
$WB/bin/whest validate --estimator $D/estimator.py --format plain 2>&1 | tail -15
$WB/bin/whest package --estimator $D --output $O/sub_$TAG.tar.gz --yes 2>&1 | tail -5
rm -rf $O/unpack; mkdir $O/unpack; tar xzf $O/sub_$TAG.tar.gz -C $O/unpack; echo "архів:"; find unpack -type f | sed 's/^/  /'
$WB/bin/python3 -I - "$O" "$TAG" <<'PY'
import ast, glob, hashlib, sys
D, TAG = sys.argv[1], sys.argv[2]
src = open('%s/sub/estimator.py' % D, 'rb').read()
ok = True
for f in glob.glob(D + '/unpack/**/*.py', recursive=True):
    b = open(f, 'rb').read(); bad = []
    for n in ast.walk(ast.parse(b)):
        if isinstance(n, ast.Import):
            bad += [a.name for a in n.names if a.name.split('.')[0] in ('numpy', 'scipy', 'torch')]
        if isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0] in ('numpy', 'scipy', 'torch'):
            bad.append(n.module)
    same = hashlib.md5(b).hexdigest() == hashlib.md5(src).hexdigest() if f.endswith('/estimator.py') else True
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
