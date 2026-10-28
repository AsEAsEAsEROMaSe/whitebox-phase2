"""18.10.2026, ідея юзера: багато середніх з випадково вимкненими нейронами (частка q), усереднити по масках і екстраполювати q→0.
Ціль на тих самих точках Z = повний ReLU на Z (МК), тож різниця = лише зміщення методу, без шуму МК. Друга ціль = банк.
Передбачення: зміщення не зникне, бо вимкнений нейрон = дискретний стрибок, крива по q не гладка біля 0 (relu_on: 128 вимкнених дають 0.21)."""
import sys, os, numpy as np, torch
os.environ.setdefault("CUDA_DEVICE_ORDER","PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
d=0; N=32768; K=16; dev='cuda'
Wn=np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%d)['weights']; W=[torch.tensor(Wn[l],dtype=torch.float32,device=dev) for l in range(16)]
mb=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%d)['mean'][15].astype(np.float64)
Z=torch.randn(N,1024,generator=torch.Generator(device=dev).manual_seed(777),device=dev)
def run(off):
    h=Z
    for l in range(16):
        p=h@W[l]; h=torch.where(off[l][None,:],p,torch.relu(p))
    return h.double().mean(0).cpu().numpy()
rng=np.random.RandomState(5)
full=run([torch.zeros(1024,dtype=torch.bool,device=dev)]*16)
rms=lambda a,b: np.sqrt(((a-b)**2).mean()); cos=lambda a,b: a@b/np.linalg.norm(a)/np.linalg.norm(b)
print("МК повний на Z проти банку: rms %.2e"%rms(full,mb))
qs=[1/1024,1/512,1/256,1/128,1/64,1/32]; M={}
for q in qs:
    ms=[run([torch.tensor(rng.rand(1024)<q,device=dev) for _ in range(16)]) for _ in range(K)]
    M[q]=np.mean(ms,0); sd=np.std([rms(m,full) for m in ms])
    print("q=1/%-5d (~%5.1f вимкн.) одна маска rms %.3f±%.3f · середнє %d масок rms %.3f cos %.4f"%(round(1/q),q*16384,np.mean([rms(m,full) for m in ms]),sd,K,rms(M[q],full),cos(M[q],full)))
x=np.array(qs); Y=np.array([M[q] for q in qs])
for deg,sl in [(1,slice(0,2)),(1,slice(0,3)),(2,slice(0,3)),(2,slice(0,4)),(3,slice(0,4)),(3,slice(0,6))]:
    c=np.polyfit(x[sl],Y[sl],deg); pred=c[-1]
    print("екстраполяція ст.%d по %d точках: rms до МК-повного %.3f · до банку %.3f · cos %.4f"%(deg,len(x[sl]),rms(pred,full),rms(pred,mb),cos(pred,full)))
