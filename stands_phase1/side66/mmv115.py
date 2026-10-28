"""BATCHED ШТРАССЕН: 7 добутків ОДНИМ batched matmul, комбінації — einsum.

ДІАГНОЗ (зміряно): 85% residual `_mm` дає 18 `add` на рівень, не `concatenate`.
residual ≈ СТАЛА НА ВИКЛИК (17 мкс) майже незалежно від розміру ⟹ бити треба
в КІЛЬКІСТЬ викликів, а не в обсяг даних.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
  H1: batched знижує кількість викликів у ~7× на рівень, флопи ростуть <2%   55%
  H2: einsum на (7,2,2)-патерні тарифікується дорожче, ніж економить          30%
  H3: fnp не підтримує batched matmul 3D ⟹ ідея мертва технічно               15%
"""
import time
import numpy as np
import flopscope
import flopscope.numpy as fnp

# Штрассен: M_r = (Σ_qp T[r,q,p]·A[q,p]) @ (Σ_qp S[r,q,p]·B[q,p])
# C[i,j] = Σ_r U[i,j,r]·M[r]
T = np.zeros((7, 2, 2)); S = np.zeros((7, 2, 2)); U = np.zeros((2, 2, 7))
T[0,0,0]=1; T[0,1,1]=1;  S[0,0,0]=1; S[0,1,1]=1     # M1=(A11+A22)(B11+B22)
T[1,1,0]=1; T[1,1,1]=1;  S[1,0,0]=1                  # M2=(A21+A22)B11
T[2,0,0]=1;              S[2,0,1]=1; S[2,1,1]=-1     # M3=A11(B12-B22)
T[3,1,1]=1;              S[3,1,0]=1; S[3,0,0]=-1     # M4=A22(B21-B11)
T[4,0,0]=1; T[4,0,1]=1;  S[4,1,1]=1                  # M5=(A11+A12)B22
T[5,1,0]=1; T[5,0,0]=-1; S[5,0,0]=1; S[5,0,1]=1      # M6=(A21-A11)(B11+B12)
T[6,0,1]=1; T[6,1,1]=-1; S[6,1,0]=1; S[6,1,1]=1      # M7=(A12-A22)(B21+B22)
U[0,0,0]=1; U[0,0,3]=1; U[0,0,4]=-1; U[0,0,6]=1      # C11=M1+M4-M5+M7
U[0,1,2]=1; U[0,1,4]=1                                # C12=M3+M5
U[1,0,1]=1; U[1,0,3]=1                                # C21=M2+M4
U[1,1,0]=1; U[1,1,1]=-1; U[1,1,2]=1; U[1,1,5]=1      # C22=M1-M2+M3+M6
_T = _S = _U = None


def mm_old(a, b, level):
    if level <= 0 or a.shape[0] % 2 or a.shape[1] % 2 or b.shape[1] % 2:
        return fnp.matmul(a, b)
    m, k, n = a.shape[0]//2, a.shape[1]//2, b.shape[1]//2
    A11,A12,A21,A22 = a[:m,:k],a[:m,k:],a[m:,:k],a[m:,k:]
    B11,B12,B21,B22 = b[:k,:n],b[:k,n:],b[k:,:n],b[k:,n:]
    M1=mm_old(A11+A22,B11+B22,level-1); M2=mm_old(A21+A22,B11,level-1)
    M3=mm_old(A11,B12-B22,level-1);     M4=mm_old(A22,B21-B11,level-1)
    M5=mm_old(A11+A12,B22,level-1);     M6=mm_old(A21-A11,B11+B12,level-1)
    M7=mm_old(A12-A22,B21+B22,level-1)
    return fnp.block([[M1+M4-M5+M7, M3+M5],[M2+M4, M1-M2+M3+M6]])


def _split(x):
    """(R,2m,2k) -> (R,2,2,m,k) через reshape+transpose (виду, без копії)."""
    R, M2, K2 = x.shape
    m, k = M2//2, K2//2
    return x.reshape(R, 2, m, 2, k).transpose(0, 1, 3, 2, 4)


def mm_batched(a, b, level):
    """Рівні як осі БАТЧУ: усі 7^l добутків одним matmul на рівень."""
    if level <= 0:
        return fnp.matmul(a, b)
    A = a[None]; B = b[None]                       # (1,M,K),(1,K,N)
    for _ in range(level):
        if A.shape[1] % 2 or A.shape[2] % 2 or B.shape[2] % 2:
            break
        Aq = _split(A)                             # (R,2,2,m,k)
        Bq = _split(B)                             # (R,2,2,k,n)
        A = fnp.einsum("rqp,Rqpmk->Rrmk", _T, Aq)
        A = A.reshape(-1, A.shape[2], A.shape[3])  # (R*7,m,k)
        B = fnp.einsum("rqp,Rqpkn->Rrkn", _S, Bq)
        B = B.reshape(-1, B.shape[2], B.shape[3])
    M = fnp.matmul(A, B)                           # ОДИН batched matmul
    lv = 0
    R = M.shape[0]
    while R % 7 == 0 and R > 1:
        R //= 7; lv += 1
    for _ in range(lv):
        r, mm_, nn = M.shape
        Mr = M.reshape(r//7, 7, mm_, nn)
        C = fnp.einsum("ijr,Rrmn->Rijmn", _U, Mr)  # (R/7,2,2,m,n)
        C = C.transpose(0, 1, 3, 2, 4).reshape(r//7, 2*mm_, 2*nn)
        M = C
    return M[0]


def mm_lvl(a, b, level):
    """18 ЗВИЧАЙНИХ add, але на БАТЧІ ⟹ 18 викликів на РІВЕНЬ, не на виклик.
    Флопи ті самі, що в старого; кількість викликів падає у 7^(l-1)."""
    if level <= 0:
        return fnp.matmul(a, b)
    A = a[None]; B = b[None]
    done = 0
    for _ in range(level):
        if A.shape[1] % 2 or A.shape[2] % 2 or B.shape[2] % 2:
            break
        R = A.shape[0]; m = A.shape[1]//2; k = A.shape[2]//2; n = B.shape[2]//2
        A11 = A[:, :m, :k]; A12 = A[:, :m, k:]; A21 = A[:, m:, :k]; A22 = A[:, m:, k:]
        B11 = B[:, :k, :n]; B12 = B[:, :k, n:]; B21 = B[:, k:, :n]; B22 = B[:, k:, n:]
        A = fnp.concatenate([A11 + A22, A21 + A22, A11, A22,
                             A11 + A12, A21 - A11, A12 - A22], axis=0)
        B = fnp.concatenate([B11 + B22, B11, B12 - B22, B21 - B11,
                             B22, B11 + B12, B21 + B22], axis=0)
        done += 1
    M = fnp.matmul(A, B)
    for _ in range(done):
        r = M.shape[0] // 7
        M1 = M[0*r:1*r]; M2 = M[1*r:2*r]; M3 = M[2*r:3*r]; M4 = M[3*r:4*r]
        M5 = M[4*r:5*r]; M6 = M[5*r:6*r]; M7 = M[6*r:7*r]
        top = fnp.concatenate([M1 + M4 - M5 + M7, M3 + M5], axis=2)
        bot = fnp.concatenate([M2 + M4, M1 - M2 + M3 + M6], axis=2)
        M = fnp.concatenate([top, bot], axis=1)
    return M[0]


def measure(fn, a, b, L):
    flopscope.budget_reset()
    t0 = time.perf_counter(); out = fn(a, b, L); w = time.perf_counter()-t0
    d = flopscope.budget_summary_dict()
    return out, float(d['flops_used']), w


if __name__ == "__main__":
    N, K, M_ = 4096, 1024, 1024
    rng = np.random.default_rng(0)
    with flopscope.budget(int(2**48), quiet=True):
        _T = fnp.asarray(T.astype('float32')); _S = fnp.asarray(S.astype('float32'))
        _U = fnp.asarray(U.astype('float32'))
        a = fnp.asarray(rng.standard_normal((N, K)).astype('float32'))
        b = fnp.asarray(rng.standard_normal((K, M_)).astype('float32'))
        ref, f0, w0 = measure(mm_old, a, b, 0)
        nrm = float(fnp.max(fnp.abs(ref)))
        print("форма %dx%d @ %dx%d" % (N, K, K, M_))
        print("  L=0  щільний   %8.4f Гфл  wall %8.1f мс  ×1.0000" % (f0/1e9, w0*1e3))
        for L in (1, 2, 3, 4):
            for nm, fn in (("старий", mm_old), ("BATCHED", mm_batched), ("LVL", mm_lvl)):
                try:
                    o, f, w = measure(fn, a, b, L)
                    e = float(fnp.max(fnp.abs(o - ref)))/nrm
                    print("  L=%d %-8s  %8.4f Гфл  wall %8.1f мс  ×%.4f  похибка %.1e"
                          % (L, nm, f/1e9, w*1e3, f/f0, e))
                except Exception as ex:
                    print("  L=%d %-8s  ПОМИЛКА %s: %s" % (L, nm, type(ex).__name__, str(ex)[:70]))
