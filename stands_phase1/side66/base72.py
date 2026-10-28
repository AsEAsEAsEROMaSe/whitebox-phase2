"""ВІРНИЙ ПОРТ `estimator_side1.py` У TORCH — база для всіх наступних стендів.

🚨 ЧОМУ ЦЕ ОКРЕМИЙ ФАЙЛ. Учора чотири стенди поспіль міряли спрощений
`ψ = x_L − G(x₀−m₀)` БЕЗ вибілювання, і гаусів нахил показав 1.10×, який
зник дощенту, щойно вибілювання повернули (воно само дає 1.48×).
Урок: **базова лінія = ДІЮЧИЙ САБМІТ, прочитаний у коді**, а не моя модель
того, що він робить. Тому тут — рядок у рядок те, що в
`whest-starterkit/estimator_side1.py`:
  пілот 128 пар → обрізання мертвих нейронів → бюджет → вибілювання
  (`cov=xᵀx/2n`, `w0=cov^{−1/2}·W₀`) → антитетика → якір шару 0 (`σ/√2π`)
  → перенос затвором `v ← (v·W)·⟨relu'⟩` → σ-поправка `λ_i·φ(m̂/ŝ)(ŝ−s_замик)`
  на шарах 1..8 із замикання Мелера K=2 і ТОЧНОЮ діагоналлю.

Експортує `run_side1()` з гачками, щоб надбудови міряти НА НЬОМУ, а не поруч:
  * `lam_s` — вимкнути/змінити σ-поправку;
  * `anchor` — додати пошаровий μ-якір (`λ_a·(x̄_ℓ − m_замик)`);
  * `bias_k` — ШТУЧНО помножити похибку замикання на `k` (потрібне для
    прайсингу: «що дасть замикання, точніше в 1/k разів»);
  * `cv4` — контрольна змінна 4-го степеня поверх вибілювання.

КОНТРОЛІ (у `main`)
  * ✅ `lam_s=0, anchor=0` мусить дати рівно sub66;
  * ✅ вимкнене вибілювання мусить бути помітно ГІРШЕ (С55: 1.48×);
  * ✅ `bias_k=1` мусить збігтися з базою до останньої цифри;
  * ✅ `bias_k=0` (оракульне замикання) — стеля.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')

W_ = 256
INV_SQRT_2PI = 0.3989422804014327
SQRT2 = math.sqrt(2.0)
GATE_N = 1024
PILOT_PAIRS = 128
TARGET_UTIL = 0.0942
B_BUDGET = 272_000_000_000
MIN_PAIRS = 128
SEED = 12345
KSIG = 8
LAM_S = (0.98305, 0.52716, 0.35370, 0.21122,
         0.14512, 0.11722, 0.10011, 0.07145)


def _phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQRT_2PI


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def cut_and_budget(W, dev):
    """Пілот + обрізання + n_pairs — рівно як у сабміті."""
    depth = len(W)
    rng = np.random.default_rng(SEED)
    zp = torch.tensor(rng.standard_normal((PILOT_PAIRS, W_)).astype(np.float32),
                      device=dev)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(depth):
        h = a @ W[i]
        if i < depth - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    wc = []
    for i in range(depth):
        wi = W[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < depth - 1:
            wi = wi[:, alive[i]]
        wc.append(wi.contiguous())
    per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * depth * W_ * W_)
    left = TARGET_UTIL * float(B_BUDGET) - used
    return wc, max(MIN_PAIRS, int(left / (2.0 * max(per_cut, 1.0))))


def closure_parts(wc, ksig, dev, extras=False):
    """`s` доактивацій і `m` активацій шарів 1..ksig. Мелер K=2, точна діагональ.

    Порт `_closure_parts` із сабміту, у fp64 (стенд може дозволити собі точність;
    у сабміті fp32, і різниця окремо перевірена як несуттєва)."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    s_o, m_o = [], []
    # extras — замикальні середні ДВОХ додаткових статистик патерну:
    #   частка активних `Φ(m/s)` і другий момент `E[relu²]=(m²+s²)Φ+m·s·φ`.
    # Обидві виходять із ТІЄЇ САМОЇ рекурсії, зайвих флопів нема.
    f_o, q_o = [], []
    for i in range(ksig + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        if i >= 1:
            s_o.append(s)
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(d - C.diagonal())
        if i >= 1:
            m_o.append(m)
            if extras:
                f_o.append(Ph)
                q_o.append(d + m * m)
    if extras:
        return s_o, m_o, f_o, q_o
    return s_o, m_o


def run_side1(wc, npair, s_cl, m_cl, dev, seed, *, white=True,
              lam_s=LAM_S, lam_a=None, lam_q=None, lam_g=None,
              bias_k=1.0, truth_layers=None,
              cv4_dirs=None, cv4_beta=None, z_ext=None,
              q_ext=None, p_ext=None,
              collect_anchors=False, collect_sigma=False,
              collect_means=False, anchor_w=None, anchor_multi=None,
              anchor_sec=False):
    """Один жереб сабміту. Повертає (прогноз останнього шару, ознаки КЗ).

    z_ext — подати ЗОВНІШНІ семпли (для побітової звірки зі справжнім
    `estimator_side1.py`: там `z` іде з того самого потоку numpy, що й пілот).

    collect_anchors — повернути МАТРИЦЮ внесків пошарових μ-якорів, уже
    перенесених до виходу. Ланцюг переносу ЛІНІЙНИЙ за `v`, тож усі `ksig`
    якорів несуться одним матричним множенням поруч із `v`. Тоді
        `pred(λ) = pred_база − λ·A`
    і λ можна фітувати LOO ПО МЕРЕЖАХ без повторних прогонів.

    collect_sigma — ТЕ САМЕ для σ-поправки: вимикає `lam_s` і повертає
    матрицю `S` (ksig × n_L), де рядок `i−1` — це `φ(m̂/ŝ)(ŝ−s_замик)` шару
    `i`, вже перенесений до виходу. Тоді
        `pred(λ) = pred₀ − λᵀS`,
    і 31 коефіцієнт λ фітуються LOO ПО МЕРЕЖАХ з ОДНОГО прогону, а зрізи
    `λ[:K]` дають глибинний свіп задарма (підматриці нормальних рівнянь).
    """
    depth = len(wc)
    if z_ext is not None:
        z = z_ext
    else:
        g = torch.Generator(device=dev).manual_seed(int(seed))
        z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    w0 = wc[0]
    if white:
        cov = (x.T @ x) / denom
        ev, V = torch.linalg.eigh(cov.double())
        wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
        w0 = wh @ wc[0]
    xa = torch.relu(x @ w0)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = xa.double().mean(0) - sg0 * INV_SQRT_2PI
    ksig = len(s_cl)
    if collect_sigma:
        lam_s = None                    # базу рахуємо БЕЗ σ-поправки
    # anchor_w — ПОЕЛЕМЕНТНІ ваги якоря, список по шарах, кожен (nvar, n_ℓ).
    # Перенос лінійний, тож усі варіанти ваг несуться ОДНИМ проходом: A має
    # `nvar·ksig` рядків, рядок `v·ksig + (i−1)` = якір шару `i` з вагами `v`.
    # anchor_multi = (kinds, refs): якорі на ДОВІЛЬНІ пошарові статистики
    # (не лише E[a]). refs[q][i-1] — ІСТИННЕ значення статистики `kinds[q]`
    # шару `i`. Потрібно, щоб спитати, чи 12× — це стеля ЗАМИКАННЯ, чи лише
    # стеля ПЕРШИХ ДВОХ МОМЕНТІВ.
    kinds = None if anchor_multi is None else anchor_multi[0]
    nvar = (len(kinds) if kinds is not None else
            (1 if anchor_w is None else int(anchor_w[0].shape[0])))
    A = (torch.zeros(nvar * ksig, wc[0].shape[1], dtype=torch.float64,
                     device=dev) if collect_anchors else None)
    S = (torch.zeros(ksig, wc[0].shape[1], dtype=torch.float64, device=dev)
         if collect_sigma else None)
    # anchor_sec — ДРУГИЙ ПОРЯДОК переносу похибки. Перший порядок дає
    # `Φ(t)·(δ_{ℓ-1} W)`; наступний член від кривини ReLU (дельта-функція,
    # згладжена гаусіаном) є `(δ_{ℓ-1} W)²·φ(t)/(2s)`. Має сенс лише коли
    # `m_cl` близьке до істини — інакше `δ` це не похибка вибірки, а зміщення.
    A2 = (torch.zeros(ksig, wc[0].shape[1], dtype=torch.float64, device=dev)
          if anchor_sec else None)
    prev_aw = None
    # collect_means — СИРІ пошарові середні `mean(a_ℓ)`, БЕЗ переносу. Потрібні,
    # щоб питати «скільки дає шар ℓ, якщо дозволити ДОВІЛЬНИЙ напрямок», а не
    # лише той, що вибирає ланцюг воріт.
    M = [] if collect_means else None
    for i in range(1, depth):
        p = xa @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        if A is not None:
            A = (A @ wd) * gate.unsqueeze(0)
        if A2 is not None:
            A2 = (A2 @ wd) * gate.unsqueeze(0)
        if S is not None:
            S = (S @ wd) * gate.unsqueeze(0)
        if i <= ksig:
            pd = p.double()
            mh = pd.mean(0)
            sh = (pd.pow(2).mean(0) - mh * mh).clamp_min(1e-30).sqrt()
            # bias_k: штучно масштабуємо ПОХИБКУ замикання (для прайсингу)
            scl = s_cl[i - 1]
            if truth_layers is not None and bias_k != 1.0:
                scl = truth_layers['s'][i - 1] + bias_k * (
                    scl - truth_layers['s'][i - 1])
            if lam_s is not None:
                v = v + (lam_s[i - 1] * _phi(mh / sh)) * (sh - scl)
            if S is not None:
                S[i - 1] = S[i - 1] + _phi(mh / sh) * (sh - scl)
            xa = torch.relu(p)
            if A is not None and kinds is not None:
                xd = xa.double()
                for q, kd in enumerate(kinds):
                    if kd == 'a':
                        st = xd.mean(0)
                    elif kd == 'ind':
                        st = (pd > 0).double().mean(0)
                    elif kd == 'a2':
                        st = xd.pow(2).mean(0)
                    elif kd == 'p3':
                        st = pd.pow(3).mean(0)
                    elif kd == 'p4':
                        st = pd.pow(4).mean(0)
                    else:
                        raise ValueError(kd)
                    # 🚨 ділити МОЖНА ЛИШЕ НА СКАЛЯР: `s_cl[i-1]` — ВЕКТОР, і
                    # поелементне ділення міняє НАПРЯМОК якоря, а не масштаб
                    # (спіймано звіркою: 3.09× замість 19.9×)
                    o = {'a': 1, 'ind': 0, 'a2': 2, 'p3': 3, 'p4': 4}[kd]
                    sc = float(s_cl[i - 1].pow(o).mean()) if o else 1.0
                    A[q * ksig + i - 1] = (A[q * ksig + i - 1]
                                           + (st - anchor_multi[1][q][i - 1])
                                           / max(sc, 1e-30))
            if A2 is not None:
                aw2 = xa.double().mean(0) - m_cl[i - 1]
                if prev_aw is not None:
                    qq = prev_aw @ wd
                    A2[i - 2] = A2[i - 2] + qq * qq * _phi(mh / sh) / (2.0 * sh)
                prev_aw = aw2
            if A is not None and kinds is not None:
                pass
            elif A is not None:
                aw = xa.double().mean(0) - m_cl[i - 1]
                if anchor_w is None:
                    A[i - 1] = A[i - 1] + aw
                else:
                    # 🚨 НЕ `v`: це ім'я вже зайняте вектором переносу
                    for q in range(nvar):
                        A[q * ksig + i - 1] = (A[q * ksig + i - 1]
                                               + aw * anchor_w[i - 1][q])
            if lam_q is not None:
                # δ E[x²] — канал `_LAM_Q` сабміту (xq − q_замик)
                v = v + lam_q[i - 1] * (xa.double().pow(2).mean(0)
                                        - q_ext[i - 1])
            if lam_g is not None:
                # δ самої МАСКИ — канал `_LAM_G` сабміту (gate − Φ_замик)
                v = v + lam_g[i - 1] * (gate - p_ext[i - 1])
            if lam_a is not None:
                mcl = m_cl[i - 1]
                if truth_layers is not None and bias_k != 1.0:
                    mcl = truth_layers['m'][i - 1] + bias_k * (
                        mcl - truth_layers['m'][i - 1])
                v = v + lam_a[i - 1] * (xa.double().mean(0) - mcl)
        else:
            xa = torch.relu(p)
        if M is not None:
            pd = p.double()
            M.append((xa.double().mean(0),                  # E[a_ℓ] (вектор)
                      (pd > 0).double().mean(0),            # частка активних
                      xa.double().pow(2).mean(0)))          # E[a_ℓ²]
    pred = xa.double().mean(0) - v
    if anchor_sec:
        return pred, (A, A2)
    if collect_means:
        return pred, M
    if collect_sigma:
        return pred, S
    if collect_anchors:
        return pred, A
    feats = None
    if cv4_dirs is not None:
        # ✅ He₄ у напрямках `u`: середнє РІВНО нуль під N(0,I) ⟹ незміщена КЗ
        pr = (z @ cv4_dirs).double()
        h4 = (pr ** 4 - 6.0 * pr ** 2 + 3.0).mean(0)
        feats = h4
        if cv4_beta is not None:
            pred = pred - cv4_beta @ h4
    return pred, feats
