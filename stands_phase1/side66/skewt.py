"""ЧИ ВИРАЖАЄТЬСЯ ПЕРЕКІС ЧЕРЕЗ УЖЕ ВІДОМЕ `t = m/s` (С78).

`attractor.py` довів: вектор γ₃ (256 чисел) НЕ залежить від вхідного розподілу
(cos ≥ 0.9956 між гаусом, рівномірним, Бернуллі, χ²) ⟹ форму задають ВАГИ.

Якщо так, γ₃ мусить виражатись через величини, які ланцюг (m, C) вже несе.
Найприродніший кандидат — саме `t = m/s`: нейрон перекошений рівно тому, що він
«однобокий», а однобокість і є t. Тоді `κ₃` коштує НУЛЬ додаткових флопів, і
вирок `k3prop.py` (тензор 256³ = 10.1× понад бюджет) обходиться збоку.

Міряємо cos(γ₃, кандидат) по 256 нейронах, пошарово.
"""
import sys
import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 16
W_, D, CH = 256, 32, 1 << 13
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')

def _prof(W, seed, npairs):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = [torch.zeros(4, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]; pd = p.double(); q = pd * pd
            Z[L][0] += pd.sum(0); Z[L][1] += q.sum(0)
            Z[L][2] += (q * pd).sum(0); Z[L][3] += (q * q).sum(0)
            x = torch.relu(p)
        done += b
    N = float(2 * npairs)
    g3, tt = [], []
    for L in range(D):
        a1, a2, a3 = Z[L][0]/N, Z[L][1]/N, Z[L][2]/N
        s2 = (a2 - a1*a1).clamp_min(1e-30); s = s2.sqrt()
        mu3 = a3 - 3*a1*a2 + 2*a1**3
        g3.append((mu3/s**3).cpu().numpy()); tt.append((a1/s).cpu().numpy())
    return np.array(g3), np.array(tt)

def cs(a, b):
    return float((a*b).sum()/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-30))

CAND = {'t': lambda t: t, '−t': lambda t: -t, 't³': lambda t: t**3,
        '1/(1+t²)': lambda t: 1.0/(1.0+t*t),
        't/(1+t²)': lambda t: t/(1.0+t*t),
        'sign(t)': lambda t: np.sign(t)}
acc = {k: np.zeros(D) for k in CAND}
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        g3, tt = _prof(W, 33_000 + 7*n, 1 << LOGN)
    for L in range(D):
        for k, f in CAND.items():
            acc[k][L] += cs(g3[L], f(tt[L]))
    del W; torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}', flush=True)

for k in acc: acc[k] /= NETS
print(f'\n{NETS} мереж — cos(γ₃, кандидат) по 256 нейронах\n')
print(f'{"шар":>4s} ' + ' '.join(f'{k:>11s}' for k in CAND))
print('-' * (5 + 12*len(CAND)))
for L in (1, 2, 3, 5, 8, 12, 16, 24, 31):
    print(f'{L:4d} ' + ' '.join(f'{acc[k][L]:11.4f}' for k in CAND))
best = max(CAND, key=lambda k: abs(acc[k][8:]).mean())
v = abs(acc[best][8:]).mean()
print(f'\n🔑 НАЙКРАЩИЙ кандидат по шарах 8..31: **{best}**, |cos| = {v:.4f}')
print(f'ВИРОК: {"🟢 ПЕРЕКІС ЧИТАЄТЬСЯ З `t` — κ₃ БЕЗКОШТОВНИЙ" if v > 0.9 else "⛔ `t` не пояснює перекосу — потрібне щось іще"}')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
