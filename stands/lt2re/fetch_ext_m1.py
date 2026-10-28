"""19.10.2026 white-box-16: докачати shard_ext01..07 корпусу і лишити лише m1 (moments[:, :1]) з тими самими ключами
(idx, moments (1500,1,16,1024)), атомарно на місці (диск). Лише числові масиви, без pickle."""
import os, sys, numpy as np
from huggingface_hub import hf_hub_download
R = 'keenanpepper/arc-whestbench-p2-d8b-corpus-14k'; SHA = 'cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8'
D = '/home/trim/wb2data/corpus14k'
for i in range(8):
    fn = 'shard_ext%02d.npz' % i; p = os.path.join(D, fn)
    if os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:
            if z['moments'].shape[1] == 1:
                continue
    else:
        hf_hub_download(R, fn, repo_type='dataset', revision=SHA, local_dir=D)
    with np.load(p, allow_pickle=False) as z:
        idx = np.asarray(z['idx']); m1 = np.asarray(z['moments'][:, :1], dtype=np.float32)
    tmp = p + '.tmp.npz'
    np.savez(tmp, idx=idx, moments=m1)
    os.replace(tmp, p)
    print(fn, idx.min(), idx.max(), m1.shape, flush=True)
