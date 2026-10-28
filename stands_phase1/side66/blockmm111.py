# С111: БЛОКОВЕ ПРЕДСТАВЛЕННЯ СТАНУ — Штрассен БЕЗ ЖОДНОЇ КОНКАТЕНАЦІЇ.
# Блоки C11,C12,C21,C22 після Штрассена = рівно те розбиття, що потрібне
# наступному шару ⟹ тримаємо стан кортежем через усі 32 шари, relu поблочно.
# ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: флопи ≈ як у Штрассена з concat (×0.88),
# residual ≈ як у щільного (бо копій нема). Якщо так — гілка дає ×1.22, не ×1.05.
import numpy as np, flopscope, flopscope.numpy as fnp

def run(fn):
    flopscope.budget_reset()
    with flopscope.budget(10**16, quiet=True):
        fn(); d = flopscope.budget_summary_dict()
    return (d['flops_used'], d['residual_wall_time_s']*1e3, d['wall_time_s']*1e3,
            sum(v['calls'] for v in d['operations'].values()))

rng = np.random.default_rng(0); N = 6144
A = rng.standard_normal((N,256)).astype(np.float32)
W = [(rng.standard_normal((256,256))*0.0884).astype(np.float32) for _ in range(32)]
Z = np.array(0, dtype=np.float32)

def dense():
    h = fnp.asarray(A)
    for l in range(32): h = fnp.maximum(h @ fnp.asarray(W[l]), fnp.asarray(Z))
    return h

def strassen_concat():
    h = fnp.asarray(A)
    for l in range(32):
        b = fnp.asarray(W[l]); m,k,n = N//2,128,128
        a11,a12,a21,a22 = h[:m,:k],h[:m,k:],h[m:,:k],h[m:,k:]
        b11,b12,b21,b22 = b[:k,:n],b[:k,n:],b[k:,:n],b[k:,n:]
        M1=(a11+a22)@(b11+b22); M2=(a21+a22)@b11; M3=a11@(b12-b22)
        M4=a22@(b21-b11); M5=(a11+a12)@b22; M6=(a21-a11)@(b11+b12); M7=(a12-a22)@(b21+b22)
        top=fnp.concatenate([M1+M4-M5+M7, M3+M5],axis=1)
        bot=fnp.concatenate([M2+M4, M1-M2+M3+M6],axis=1)
        h = fnp.maximum(fnp.concatenate([top,bot],axis=0), fnp.asarray(Z))
    return h

def strassen_blocked():
    """Стан = (h11,h12,h21,h22), жодної конкатенації всередині циклу."""
    m,k = N//2,128
    h11,h12 = fnp.asarray(A[:m,:k]), fnp.asarray(A[:m,k:])
    h21,h22 = fnp.asarray(A[m:,:k]), fnp.asarray(A[m:,k:])
    z = fnp.asarray(Z)
    for l in range(32):
        b = fnp.asarray(W[l]); n=128
        b11,b12,b21,b22 = b[:k,:n],b[:k,n:],b[k:,:n],b[k:,n:]
        M1=(h11+h22)@(b11+b22); M2=(h21+h22)@b11; M3=h11@(b12-b22)
        M4=h22@(b21-b11); M5=(h11+h12)@b22; M6=(h21-h11)@(b11+b12); M7=(h12-h22)@(b21+b22)
        h11 = fnp.maximum(M1+M4-M5+M7, z); h12 = fnp.maximum(M3+M5, z)
        h21 = fnp.maximum(M2+M4, z);       h22 = fnp.maximum(M1-M2+M3+M6, z)
    return fnp.concatenate([fnp.concatenate([h11,h12],axis=1),
                            fnp.concatenate([h21,h22],axis=1)],axis=0)

ref = np.asarray(dense())
print(f"{'варіант':<22}{'флопів':>16}{'×F':>8}{'residual мс':>13}{'wall мс':>10}{'викликів':>10}{'відн.похибка':>14}")
base=None
for nm,fn in [('щільний', dense), ('Штрассен+concat', strassen_concat), ('Штрассен БЛОКОВИЙ', strassen_blocked)]:
    f,r,w,c = run(fn)
    if base is None: base=(f,r)
    err = np.abs(np.asarray(fn())-ref).max()/ (np.abs(ref).max()+1e-30)
    print(f"{nm:<22}{f:>16,}{f/base[0]:>8.4f}{r:>13.2f}{w:>10.1f}{c:>10,}{err:>14.2e}")

# --- ДОДАТОК: чи достатньо ОДНОГО fnp.block замість ТРЬОХ concatenate ---
def strassen_block1():
    h = fnp.asarray(A)
    for l in range(32):
        b = fnp.asarray(W[l]); m,k,n = N//2,128,128
        a11,a12,a21,a22 = h[:m,:k],h[:m,k:],h[m:,:k],h[m:,k:]
        b11,b12,b21,b22 = b[:k,:n],b[:k,n:],b[k:,:n],b[k:,n:]
        M1=(a11+a22)@(b11+b22); M2=(a21+a22)@b11; M3=a11@(b12-b22)
        M4=a22@(b21-b11); M5=(a11+a12)@b22; M6=(a21-a11)@(b11+b12); M7=(a12-a22)@(b21+b22)
        h = fnp.maximum(fnp.block([[M1+M4-M5+M7, M3+M5],[M2+M4, M1-M2+M3+M6]]), fnp.asarray(Z))
    return h

print()
print("=== ДОДАТОК: один fnp.block замість трьох concatenate ===")
for nm,fn in [('щільний', dense), ('Штрассен concat×3', strassen_concat),
              ('Штрассен block×1', strassen_block1), ('Штрассен БЛОКОВИЙ стан', strassen_blocked)]:
    f,r,w,c = run(fn)
    err = np.abs(np.asarray(fn())-ref).max()/(np.abs(ref).max()+1e-30)
    print(f"{nm:<26}{f:>16,}  ×F {f/25769803776:.4f}  residual {r:>7.2f} мс  wall {w:>7.1f}  викликів {c:>6,}  похибка {err:.2e}")
