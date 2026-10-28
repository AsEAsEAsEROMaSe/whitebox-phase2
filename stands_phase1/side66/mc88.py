"""КЕШ МК-ІСТИНИ для гілки «вивчена поправка до замикання» (С88).

Рахує ОДИН раз на мережу і кладе на диск:
  mh[q]  = E[h_q]              (32,256) f64
  Ch[q]  = Cov(h_q)            (32,256,256) f32
  Ca[q]  = Cov(relu(h_q))      (32,256,256) f32
  ma[q]  = E[relu(h_q)]        (32,256) f64
  out    = E[relu(h_31)]       (256,) f64

🔑 ДВА ПРОХОДИ замість fp64-граму: пасс 1 — вибіркові середні (дешеві редукції),
пасс 2 — грам ЦЕНТРОВАНИХ даних у fp32 (той самий генератор ⟹ та сама вибірка).
Похибка fp32-граму на центрованих ~5e-6 віднос., МК-шум при N=2^19 ~1.4e-3 ⟹
запас 280×. Це дає ×32 швидкості проти fp64 на P104 (де fp64 = 1/32 fp32).
"""
import glob, sys, os, time
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
N0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NN = int(sys.argv[3]) if len(sys.argv) > 3 else 4
LOG2N = int(sys.argv[4]) if len(sys.argv) > 4 else 19
assert DEV not in ('cuda:0', 'cuda:4')
NS = 1 << LOG2N
BATCH = 1 << 13
L_, W_ = 32, 256
OUT = '/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/mc88cache'
os.makedirs(OUT, exist_ok=True)

# 🚨 pool3 НЕ ЧІПАЄМО: 0.24 s на відкриття npz × 2454 = 10 хв лише на скан сідів.
#    Ваги тих самих мереж лежать у `fs_w.npy` (100 шт), індекс = позиція в
#    `fs_w_seeds.npy`. Мережі обмінні, тож який саме підмножинний набір — байдуже.
WZ = np.load('/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy', mmap_mode='r')
SD = np.load('/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w_seeds.npy')
print(f'доступно мереж: {len(SD)} · беру [{N0}:{N0+NN}] · 2^{LOG2N} · {DEV}', flush=True)

for n in range(N0, min(N0 + NN, len(SD))):
    dst = f'{OUT}/n{n:03d}.npz'
    if os.path.exists(dst):
        print(f'  net{n} вже є', flush=True)
        continue
    t0 = time.time()
    sd0 = int(SD[n])
    W32 = [torch.tensor(np.asarray(WZ[n][q], np.float32), device=DEV) for q in range(L_)]
    with torch.no_grad():
        SH = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        SA = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(8800 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                SH[q] += h.sum(0, dtype=torch.float64)
                a = torch.relu(h)
                SA[q] += a.sum(0, dtype=torch.float64)
            d += b
        mh = SH / NS; ma = SA / NS
        mh32 = mh.float(); ma32 = ma.float()
        GH = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        GA = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(8800 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                hc = h - mh32[q]
                GH[q] += (hc.T @ hc).double()
                a = torch.relu(h)
                ac = a - ma32[q]
                GA[q] += (ac.T @ ac).double()
            d += b
        Ch = GH / NS; Ca = GA / NS
        Ch = 0.5 * (Ch + Ch.transpose(1, 2)); Ca = 0.5 * (Ca + Ca.transpose(1, 2))
    np.savez(dst, seed=sd0,
             mh=mh.cpu().numpy(), ma=ma.cpu().numpy(),
             Ch=Ch.float().cpu().numpy(), Ca=Ca.float().cpu().numpy(),
             out=ma[L_ - 1].cpu().numpy())
    print(f'  net{n} seed={sd0} {time.time()-t0:.1f}s', flush=True)
print('готово', flush=True)
