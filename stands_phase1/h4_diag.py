import torch, numpy as np
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))
w=torch.randn(12500,D,device=dev,generator=torch.Generator(device=dev).manual_seed(9000))
x=w
for li in range(L): x=relu(x@W[li])
print('a^L: nan?',torch.isnan(x).any().item(),'max',float(x.max()),'mean',float(x.mean()),'frac_zero',float((x==0).float().mean()))
s=(w@W[0])/std0
he4=s**4-6*s*s+3.0
print('he4 nan?',torch.isnan(he4).any().item(),'max',float(he4.abs().max()))
r2=(w*w).sum(1)
print('r2 mean',float(r2.mean()),'r2^2 max',float((r2*r2).max()))
Y=x  # even part ~ x here (single chunk, not paired) just for scale
print('Var(Y) per neuron mean',float(x.var(0).mean()))
