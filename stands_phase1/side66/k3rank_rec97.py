"""k3rank_rec97 — ДВА ПИТАННЯ, ЩО ВИРІШУЮТЬ ГІЛКУ ПІСЛЯ ×1.614:
 (A) чи виживе виграш під РАНГ-r обрізанням несеного κ₃ щошару?
 (B) чи не з'їсть його ВІЛЬНИЙ МАСШТАБ (аналог `_ALPHA` у сабміті)?

(A) обрізання симетричне за Такером: P = топ-r власних векторів грам-матриці
    mode-1 розгортки; T_r = T ×₁PPᵀ ×₂PPᵀ ×₃PPᵀ. Зберігає симетрію ТОЧНО.
(B) для кожної гілки рахую похибку ПІСЛЯ підгонки одного скаляра α (МНК на
    істину). Якщо поправка суто масштабна, після α виграш зникає — рівно те,
    що сталося з `γ` на шарах 1..8 у С88.

ПЕРЕДБАЧЕННЯ:
 (A1) r=32 збереже >80% виграшу, r=8 — менше половини (бо на шарах 1-8
      залишок ранг-16 = 20-57%).                                  шанс 55%
 (B1) виграш ПЕРЕЖИВЕ вільний масштаб: поправка міняє ФОРМУ вектора m,
      а не лише його довжину.                                     шанс 70%
"""
import sys, numpy as np
sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")
import k3rec97 as K

def trunc_sym(T, r):
    n = T.shape[0]
    X = T.reshape(n, n * n)
    w, V = np.linalg.eigh(X @ X.T)
    P = V[:, -r:]
    Y = P.T @ X                                   # (r, n*n)
    Y = Y.reshape(r, n, n).transpose(1, 0, 2).reshape(n, r * n)
    Y = (P.T @ Y).reshape(r, r, n).transpose(2, 1, 0).reshape(n, r * r)
    Y = (P.T @ Y).reshape(r, r, r)                # ядро Такера
    Z = np.einsum("abc,ia,jb,kc->ijk", Y, P, P, P, optimize=True)
    return Z

def run(W, mode, rank=None, L=32, lam=1.0):
    n = 256
    m = np.zeros(n); C = W[0].T @ W[0]
    T3 = None if mode == "G" else np.zeros((n, n, n))
    for l in range(1, L):
        m_a, C_a, T3_a = K.relu_step(m, C, T3, mode, lam)
        m = m_a @ W[l]; C = W[l].T @ (C_a @ W[l])
        if T3_a is None:
            T3 = None
        else:
            T3 = K.mode_mult(T3_a, W[l])
            if rank is not None:
                T3 = trunc_sym(T3, rank)
    s = np.sqrt(np.maximum(np.diag(C), K.SFLOOR)); t = m / s
    A = K.a_table(s, t)
    if T3 is None:
        return A[0]
    i = np.arange(n)
    return A[0] + lam * (T3[i, i, i] / s ** 3 / 6.0) * A[3]

N, NETS = 400000, 3
Wall = np.load(K.W_PATH, mmap_mode="r")
RANKS = [4, 8, 16, 32, 64, None]
acc = {}
print("k3rank_rec97 — ранг-r обрізання несеного κ₃ + вільний масштаб", flush=True)
for net in range(NETS):
    W = np.array(Wall[net]).astype(np.float64)
    mt, _ = K.truth(W, N, 12345 + net)
    den = np.sqrt((mt ** 2).mean())
    def err(p):
        e = np.sqrt(((p - mt) ** 2).mean()) / den
        al = float(p @ mt / (p @ p))                  # вільний масштаб (МНК)
        ea = np.sqrt(((al * p - mt) ** 2).mean()) / den
        return e, ea, al
    pg = run(W, "G"); eg, ega, alg = err(pg)
    print("\nМЕРЕЖА %d  G = %.4f%%  (після вільного α: %.4f%%, α=%.4f)"
          % (net, 100*eg, 100*ega, alg), flush=True)
    print("   ранг |  похибка  | множник | після α  | множник α | α", flush=True)
    for r in RANKS:
        p = run(W, "K3", rank=r)
        e, ea, al = err(p)
        acc.setdefault(r, []).append((e/eg, ea/ega))
        print("   %4s | %8.4f%% |  x%.3f  | %7.4f%% |   x%.3f   | %.4f"
              % ("повн" if r is None else r, 100*e, eg/e, 100*ea, ega/ea, al), flush=True)
print("\nПІДСУМОК по %d мережах (середні множники)" % NETS, flush=True)
for r in RANKS:
    v = np.array(acc[r])
    print("   ранг %4s : множник x%.3f   після вільного α x%.3f"
          % ("повн" if r is None else r, 1/v[:,0].mean(), 1/v[:,1].mean()), flush=True)
