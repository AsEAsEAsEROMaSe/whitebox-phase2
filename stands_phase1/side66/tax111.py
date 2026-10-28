# С111: ПРЯМИЙ ВИМІР ПОДАТКУ НА ВИКЛИК fnp через residual_wall_time_s.
# Ставка 3.3 мкс узята з С110; від неї залежить оптимальний рівень Штрассена.
# ПЕРЕДБАЧЕННЯ: 3.0-4.7 мкс (записаний коридор). Якщо НИЖЧЕ — L=3 оживає.
import numpy as np, flopscope, flopscope.numpy as fnp, time
exec(open('/mnt/win_d/llm/White-Box/side66/price111.py').read().split('rng = np.random')[0].split('import numpy')[1].replace('as np, flopscope, flopscope.numpy as fnp',''), globals()) if False else None
def mm(a,b,level):
    if level<=0: return a@b
    m,k=a.shape; k2,n=b.shape
    if m%2 or k%2 or n%2 or min(m,k,n)<64: return a@b
    m2,kk,n2=m//2,k//2,n//2
    a11,a12=a[:m2,:kk],a[:m2,kk:]; a21,a22=a[m2:,:kk],a[m2:,kk:]
    b11,b12=b[:kk,:n2],b[:kk,n2:]; b21,b22=b[kk:,:n2],b[kk:,n2:]
    s1=a21+a22; s2=s1-a11; s3=a11-a21; s4=a12-s2
    t1=b12-b11; t2=b22-t1; t3=b22-b12; t4=t2-b21
    p1=mm(a11,b11,level-1); p2=mm(a12,b21,level-1); p3=mm(s4,b22,level-1)
    p4=mm(a22,t4,level-1); p5=mm(s1,t1,level-1); p6=mm(s2,t2,level-1); p7=mm(s3,t3,level-1)
    u1=p1+p2; u2=p1+p6; u3=u2+p7; u4=u2+p5; u5=u4+p3; u6=u3-p4; u7=u3+p5
    return fnp.concatenate([fnp.concatenate([u1,u5],axis=1),fnp.concatenate([u6,u7],axis=1)],axis=0)
rng=np.random.default_rng(0); N=6144
A=rng.standard_normal((N,256)).astype(np.float32)
W=[(rng.standard_normal((256,256))*0.0884).astype(np.float32) for _ in range(32)]
print(f"{'L':>3}{'викликів':>10}{'wall мс':>10}{'backend':>10}{'overhead':>10}{'RESIDUAL мс':>13}{'мкс/виклик':>12}")
base=None
for L in [0,1,2,3]:
    flopscope.budget_reset()
    with flopscope.budget(10**16, quiet=True):
        h=fnp.asarray(A); z=np.array(0,dtype=np.float32)
        for l in range(32): h=fnp.maximum(mm(h,fnp.asarray(W[l]),L),fnp.asarray(z))
        d=flopscope.budget_summary_dict()
    calls=sum(v['calls'] for v in d['operations'].values())
    r=d['residual_wall_time_s']*1e3
    if base is None: base=(calls,r)
    dc=calls-base[0]; dr=r-base[1]
    per=(dr*1e3/dc) if dc>0 else float('nan')
    print(f"{L:>3}{calls:>10,}{d['wall_time_s']*1e3:>10.1f}{d['flopscope_backend_time_s']*1e3:>10.1f}"
          f"{d['flopscope_overhead_time_s']*1e3:>10.1f}{r:>13.2f}{per:>12.2f}")
