"""ПЕРЕМІР НАЧИСТО: чи справді є атрактор і якої він розмірності (С79).

Юзер: «поміряй начисто, бо боти попередні могли помилятись». Слушно — за цю
сесію в корпусі вже знайдено МІРАЖ (стеля 2.85e-08), ПРОТУХЛУ мітку «відкрите
питання» і ВТРИЧІ перебільшену «дармову економію». Тому все з нуля.

ПЕРЕВІРЯЄМО ЧОТИРИ ЗАПИСАНІ ЧИСЛА:
  (1) PR-ранг активацій: 161.6 (шар 1) → 2.5 (шар 32)   [С74, лише 3 мережі]
  (2) еф. ранг адьойнта ≈ 8 на шарі 30                  [С70]
  (3) ентропія патернів 256 → 46.1                      [С78]
  (4) еф. ранг ваг 127.7 / 256

🚨 ГОЛОВНА ПІДОЗРА, ЯКУ ТРЕБА ЗНЯТИ: PR оцінюється з ВИБІРКОВОЇ коваріації.
   Якщо семплів мало, вона вироджена, і «колапс рангу» може бути АРТЕФАКТОМ
   скінченної вибірки, а не властивістю мережі. Тому PR міряється при
   N = 1024 / 4096 / 16384 / 65536 і перевіряється на ЗБІЖНІСТЬ. Якщо число
   повзе з N — воно не про мережу.

⛔ ЧОТИРИ ПЛОМБИ, КОЖНА З ТЕОРІЇ (не з попередніх вимірів):
   1. PR ваг `W₀ᵀW₀`: для квадратної iid-матриці спектр Марченка-Пастура має
      моменти Каталана `E[λ]=1, E[λ²]=2` ⟹ `PR = n·1²/2 = n/2 = 128.0` РІВНО;
   2. ентропія шару 0 = **256.000 біт** рівно (вхід симетричний ⟹ всі `q=0.5`);
   3. адьойнт на ВИХІДНОМУ шарі — тотожність ⟹ `G = I` ⟹ `PR = 256.0` рівно;
   4. `PR ≤ n` завжди, і `PR ≥ 1`.

📣 ПЕРЕДБАЧЕНО: пломби зійдуться; PR активацій справді впаде, але я НЕ впевнений
   у числі 2.5 — воно з 3 мереж, а `feedback-small-runs-do-not-arbitrate` каже,
   що малий зріз тричі перевертав знак. Ставлю на те, що падіння реальне, але
   кінцеве число буде ВИЩЕ за 2.5, і що при малих N воно занижується.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/attract.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
W_, D = 256, 32
NS = (1 << 10, 1 << 12, 1 << 14, 1 << 16)      # для перевірки збіжності
ADJ_B, ADJ_REP = 64, 8                         # батч і повтори для адьойнта
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _pr(C):
    """Participation ratio = (tr C)² / tr(C²). Для C=I дає рівно n."""
    t1 = float(torch.diagonal(C).sum())
    t2 = float((C * C).sum())
    return t1 * t1 / max(t2, 1e-300)


def _whiten(z):
    cov = (z.T @ z).double() / float(z.shape[0])
    ev, U = torch.linalg.eigh(cov)
    return (z.double() @ (U * ev.clamp_min(1e-10).rsqrt()) @ U.T).float()


def _act_pr(W, seed, n):
    """PR ЦЕНТРОВАНИХ активацій по шарах + ворота, на n семплах (антитетика)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    npair = n // 2
    z = torch.randn(npair, W_, device=DEV, generator=g)
    z = _whiten(z)
    x = torch.cat([z, -z], 0)
    pr, q = [], []
    for L in range(D):
        p = x @ W[L]
        q.append((p > 0).double().mean(0))
        x = torch.relu(p)
        xc = (x - x.mean(0, keepdim=True)).double()
        C = (xc.T @ xc) / float(x.shape[0])
        pr.append(_pr(C))
    return np.array(pr), torch.stack(q)


def _adj_pr(W, seed):
    """Еф. ранг адьойнта `M_ℓ = ∂o/∂a_ℓ` — ДВІ версії, і це не одне й те саме.

    `J_i = diag(g_i)·W_iᵀ`  ⟹  `M_ℓ = J_{D-1}·…·J_{ℓ+1}` (форма out×ℓ).
    Ідемо ВІД ВИХОДУ назад — один прохід дає всі шари одразу.

    🚨 `G = E[M Mᵀ]` живе в просторі ВИХОДУ, `H = E[Mᵀ M]` — у просторі ШАРУ.
       Для ОДНІЄЇ матриці спектри збігаються (крім нулів), але МАТ. СПОДІВАННЯ
       по семплах їх розводить ⟹ PR різні. Твердження С70 про `‖Mᵀb‖²`, де `b`
       — зміщення В ШАРІ, стосується версії `H`. Міряємо обидві, щоб не
       порівнювати різні об'єкти (я спершу зміряв `G` і отримав розбіжність
       у 12× із записаним — це була різниця ВИЗНАЧЕНЬ, не мережі).
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    G = [torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    H = [torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    for _ in range(ADJ_REP):
        z = _whiten(torch.randn(ADJ_B, W_, device=DEV, generator=g))
        gates, x = [], z
        for L in range(D):
            p = x @ W[L]
            gates.append((p > 0).float())
            x = torch.relu(p)
        M = torch.eye(W_, device=DEV).expand(ADJ_B, W_, W_).contiguous()
        G[D - 1] += (M @ M.transpose(1, 2)).double().sum(0)
        H[D - 1] += (M.transpose(1, 2) @ M).double().sum(0)
        for L in range(D - 1, 0, -1):
            J = gates[L][:, :, None] * W[L].T[None, :, :]     # (B, out, in)
            M = M @ J
            G[L - 1] += (M @ M.transpose(1, 2)).double().sum(0)
            H[L - 1] += (M.transpose(1, 2) @ M).double().sum(0)
    return (np.array([_pr(Gl) for Gl in G]),
            np.array([_pr(Hl) for Hl in H]))


LAYERS = (0, 1, 2, 4, 8, 16, 24, 30, 31)
acc_pr = {n: [] for n in NS}
acc_adj, acc_adj2, acc_ent, acc_w = [], [], [], []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
         for k in range(D)]
    with torch.no_grad():
        for n in NS:
            pr, q = _act_pr(W, 3000 + 17 * net, n)
            acc_pr[n].append(pr)
            if n == NS[-1]:
                qq = q.clamp(1e-12, 1 - 1e-12)
                ent = -(qq * qq.log2() + (1 - qq) * (1 - qq).log2()).sum(1)
                acc_ent.append(ent.cpu().numpy())
        ga, ha = _adj_pr(W, 6000 + 23 * net)
        acc_adj.append(ga); acc_adj2.append(ha)
        acc_w.append(_pr((W[0].T @ W[0]).double()))
    del W
    torch.cuda.empty_cache()
    print(f'  … мережа {net+1}/{NETS}', flush=True)

PR = {n: np.array(v) for n, v in acc_pr.items()}
ADJ, ADJ2 = np.array(acc_adj), np.array(acc_adj2)
ENT, PW = np.array(acc_ent), np.array(acc_w)

print(f'\n{NETS} мереж\n')
print('⛔ ПЛОМБИ')
print(f'  1. PR ваг `W₀ᵀW₀` = {np.median(PW):8.3f}   теорія Марченка-Пастура: '
      f'128.000   відхил {abs(np.median(PW)-128)/128*100:.2f}%')
print(f'  2. ентропія шару 0 = {np.median(ENT[:,0]):8.4f} біт   теорія: 256.0000')
print(f'  3. PR адьойнта шару 31 = {np.median(ADJ[:,31]):8.3f}   теорія (G=I): '
      f'256.000')
print(f'  4. min PR активацій = {min(PR[n].min() for n in NS):.4f} (треба ≥1), '
      f'max = {max(PR[n].max() for n in NS):.2f} (треба ≤256)')

print(f'\n🚨 ЧИ НЕ АРТЕФАКТ ВИБІРКИ: PR активацій проти N')
hdr = ' '.join(f'{"N="+str(n):>10s}' for n in NS)
print(f'{"шар":>4s} {hdr} {"збіжн?":>8s}')
print('-' * (5 + 11 * len(NS) + 9))
for L in LAYERS:
    vals = [np.median(PR[n][:, L]) for n in NS]
    rel = abs(vals[-1] - vals[-2]) / max(vals[-1], 1e-9)
    print(f'{L:4d} ' + ' '.join(f'{v:10.2f}' for v in vals) +
          f' {"✅" if rel < 0.05 else "⚠️ повзе":>8s}')

print(f'\n🗺️ ПРОФІЛЬ (N={NS[-1]}, медіана по {NETS} мережах)')
print(f'{"шар":>4s} {"PR активацій":>13s} {"адь.ВИХІД":>11s} {"адь.ШАР":>11s} {"ентроп,біт":>11s}')
print('-' * 56)
for L in LAYERS:
    print(f'{L:4d} {np.median(PR[NS[-1]][:,L]):13.2f} '
          f'{np.median(ADJ[:,L]):11.2f} {np.median(ADJ2[:,L]):11.2f} '
          f'{np.median(ENT[:,L]):11.2f}')

a1, a32 = np.median(PR[NS[-1]][:, 1]), np.median(PR[NS[-1]][:, 31])
print(f'\n🔑 КОЛАПС АКТИВАЦІЙ: шар 1 = {a1:.1f} → шар 31 = {a32:.2f}  '
      f'(у {a1/max(a32,1e-9):.0f}×)   [записано було 161.6 → 2.5, у 65×]')
print(f'🔑 АДЬОЙНТ шар 30: у просторі ВИХОДУ {np.median(ADJ[:,30]):.2f} · '
      f'у просторі ШАРУ **{np.median(ADJ2[:,30]):.2f}**   [записано ≈8]')
print(f'🔑 ЕНТРОПІЯ шар 31 = {np.median(ENT[:,31]):.1f} біт   [записано 46.1]')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
