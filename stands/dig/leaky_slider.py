"""18.10.2026, ідея юзера «повзунок перелому»: leaky ReLU f_a(x)=max(x, a·x), a=1 лінійна (середнє 0 ТОЧНО, розподіл гаус ТОЧНО), a=0 наша ReLU.
Тест: ті самі Z для всіх a, фіт полінома по (1−a) на [a0,1], екстраполяція в a=0, порівняння з E_N(0) на тих самих Z.
Передбачення до запуску: не дотягне (шанс успіху ~10%), біля a=1 середнє ~0 на фоні σ~256. Легкий варіант: 21 a, N=65536, fp32."""
import os, numpy as np, torch
os.environ.setdefault("CUDA_DEVICE_ORDER","PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
dev='cuda'; Wn=np.load('/home/trim/wb2data/dumps_p2/mlp_0000.npz')['weights']; W=[torch.tensor(Wn[l],dtype=torch.float32,device=dev) for l in range(16)]
A=np.round(np.linspace(0,1,21),4); N=65536; B=8192; S=np.zeros((len(A),1024)); S2=np.zeros((len(A),1024))
g=torch.Generator(device=dev).manual_seed(777)
for b in range(N//B):
    z=torch.randn(B,1024,generator=g,device=dev)
    for i,a in enumerate(A):
        h=z
        for w in W: p=h@w; h=torch.maximum(p,a*p)
        h=h.double(); S[i]+=h.sum(0).cpu().numpy(); S2[i]+=(h*h).sum(0).cpu().numpy()
E=S/N; sd=np.sqrt(np.maximum(S2/N-E**2,0)); rms=lambda v: np.sqrt((v**2).mean())
print("a     |E(a)|    σ(a)     шум E    cos(E(a),E(0))")
for i in range(0,21,2): print("%.2f  %.3e %.3e %.2e %.4f"%(A[i],rms(E[i]),rms(sd[i]),rms(sd[i])/np.sqrt(N),E[i]@E[0]/np.linalg.norm(E[i])/np.linalg.norm(E[0])))
print("\nекстраполяція з [a0,1] у a=0 (ціль E_N(0) на тих самих Z, |E(0)|=%.3f):"%rms(E[0]))
for a0 in (0.5,0.3,0.2,0.1,0.05):
    sel=A>=a0-1e-9; out=[]
    for d in (2,4,6,8,10):
        if sel.sum()<=d+1: continue
        c=np.polynomial.polynomial.polyfit(1-A[sel],E[sel],d); pred=np.polynomial.polynomial.polyval(1.0,c); out.append("ст.%d %.2e"%(d,rms(pred-E[0])/rms(E[0])))
    print("a0=%.2f  "%a0+" · ".join(out))
np.savez('/home/trim/wblogs/leaky_slider_d0.npz',A=A,E=E,sd=sd)
