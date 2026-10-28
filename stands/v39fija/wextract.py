"""18.10.2026 a2: одноразове вилучення з parquet mini (ВАЖКЕ: один файл = 16 мереж = ~1.2 ГБ у RAM;
запускати ЛИШЕ коли немає жодного whest run, правило юзера):
  (1) ваги мереж дампу corrdump + відкладених s0..s9 → /home/trim/wbcache/W/<mlp_name>.npy (float32, 16×1024×1024);
  (2) уся істина all_layer_means → /home/trim/wbcache/truth.npz;
  (3) одномережеві набори /home/trim/wb2data/dsub/m60..m99 для дампу (рядок N глобального порядку = mN, як m10..m59
      від a5; metadata.json копія з m10, вона однакова для всіх).
Пломби: перечитує записане і звіряє біт у біт (ваги) та ім'я мережі (набори). Рестарт підхоплює зроблене.
Запуск: /home/trim/fs123/bin/python3 -I wextract.py ; кінець → маркер WEXTRACTDONE у /home/trim/wblogs/v37_runs.log"""
import glob, json, os, shutil, subprocess, sys, time
import numpy as np
import pyarrow.parquet as pq


def busy():
    return subprocess.run(['pgrep', '-f', 'bin/whest run'], capture_output=True).returncode == 0


if busy():
    sys.exit('іде whest run: не запускаю (правило юзера: важке лише без whest)')
P = ('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/'
     'aa99830fdc09fad15407b10e8e3459d3e18bba0a/data')
OUT, DS = '/home/trim/wbcache', '/home/trim/wb2data/dsub'
os.makedirs(OUT + '/W', exist_ok=True)
want = set(json.load(open('/home/trim/wb2lean/v39fija/fit87c_tables.json'))['step_F87']['nets'])
for i in range(10):
    want.add(json.load(open('/home/trim/wblogs/v37_fija_s%d.json' % i))['results']['per_mlp'][0]['mlp_name'])
print('потрібно ваг мереж:', len(want))
t0 = time.time()
truth, order = {}, []
for f in sorted(glob.glob(P + '/mini-*.parquet')):
    t = pq.read_table(f, columns=['mlp_name', 'all_layer_means'])
    for nm, alm in zip(t.column('mlp_name').to_pylist(), t.column('all_layer_means').to_pylist()):
        truth[nm] = np.array(alm, dtype=np.float64); order.append(nm)
np.savez(OUT + '/truth.npz', **truth)
if os.environ.get('WX_ALL') == '1':   # 18.10.2026 white-box-21: ваги всіх 100 мереж mini (дофіт таблиці на 90)
    want = set(order)
    print('WX_ALL: потрібно ваг мереж:', len(want))
dpath = lambda k: '%s/m%d/data/mini-00000-of-00001.parquet' % (DS, k)
for f in sorted(glob.glob(P + '/mini-*.parquet')):
    nms = pq.read_table(f, columns=['mlp_name']).column('mlp_name').to_pylist()
    b0 = order.index(nms[0])
    wtodo = [j for j, nm in enumerate(nms) if nm in want and not os.path.exists('%s/W/%s.npy' % (OUT, nm))]
    dtodo = [j for j in range(len(nms)) if 60 <= b0 + j < 100 and not os.path.exists(dpath(b0 + j))]
    if not wtodo and not dtodo:
        continue
    if busy():
        sys.exit('з\'явився whest run: зупиняюсь (зроблене лишається, рестарт підхопить)')
    full = pq.read_table(f)
    if wtodo:
        vals = full.column('weights').combine_chunks().flatten().flatten().flatten()
        Wall = vals.to_numpy(zero_copy_only=False).reshape(len(nms), 16, 1024, 1024)
        assert Wall.dtype == np.float32, Wall.dtype
        for j in wtodo:
            tmp = '%s/W/%s.tmp.npy' % (OUT, nms[j])
            np.save(tmp, Wall[j]); os.replace(tmp, '%s/W/%s.npy' % (OUT, nms[j]))
        j = wtodo[0]
        assert np.array_equal(np.load('%s/W/%s.npy' % (OUT, nms[j])), Wall[j]), nms[j]
        del vals, Wall
    for j in dtodo:
        dd = '%s/m%d' % (DS, b0 + j)
        os.makedirs(dd + '/data', exist_ok=True)
        shutil.copy(DS + '/m10/metadata.json', dd + '/metadata.json')
        pq.write_table(full.slice(j, 1), dd + '/data/tmp.parquet', compression='snappy')
        os.replace(dd + '/data/tmp.parquet', dpath(b0 + j))
        chk = pq.read_table(dpath(b0 + j), columns=['mlp_name']).column('mlp_name').to_pylist()
        assert chk == [nms[j]], (chk, nms[j])
    del full
    print('%s: ваг %d, наборів %d ✓ (%.0f с)' % (f[-27:], len(wtodo), len(dtodo), time.time() - t0), flush=True)
have = sum(os.path.exists('%s/W/%s.npy' % (OUT, nm)) for nm in want)
dsub_ok = all(os.path.exists(dpath(k)) for k in range(60, 100))
print('ваг: %d з %d · набори m60..m99: %s · %.0f с' % (have, len(want), 'усі' if dsub_ok else 'НЕ всі', time.time() - t0))
if have == len(want) and dsub_ok:
    open('/home/trim/wblogs/v37_runs.log', 'a').write('WEXTRACT2DONE\n' if os.environ.get('WX_ALL') == '1' else 'WEXTRACTDONE\n')
