"""Асиміляція даних (юзер 15.10: «адаптивний механізм, що ізолює суму від даних»): ланцюг = прогноз з ізотропною похибкою σ_e²,
МК з N семплів = вимір з шумом Cov(h15)/N (анізотропний). Вінер/Калман у власному базисі Cov: k_i = σ_e²/(σ_e² + λ_i/N).
Ідеальна MSE = (1/n) Σ_i σ_e² (λ_i/N)/(σ_e² + λ_i/N) (якщо похибка ланцюга ізотропна в цьому базисі — перевіряємо).
Бал ∝ MSE · util, util = 0.234 + N·3.36e7/2^41. Дампи 0..3, Cov(h15) з 2^18 семплів.
Передбачення до запуску: оптимум бала ≤ x1.1 (шум МК надто великий); шанс > x1.5: 10%."""
import numpy as np, torch
dev='cuda'; NS=2**18; CH=2**15
for i in range(4):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],dtype=torch.float32,device=dev)
    S1=torch.zeros(1024,dtype=torch.float64,device=dev); S2=torch.zeros(1024,1024,dtype=torch.float64,device=dev); g=torch.Generator(device=dev).manual_seed(3)
    for _ in range(NS//CH):
        h=torch.randn(CH,1024,generator=g,device=dev)
        for l in range(16): h=torch.relu(h@W[l])
        hd=h.double(); S1+=hd.sum(0); S2+=hd.T@hd
    m=S1/NS; Cv=S2/NS-torch.outer(m,m); lam,U=torch.linalg.eigh(Cv); lam=lam.clamp_min(0).flip(0); U=U.flip(1)
    gt=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)['gt_mean'][15].astype(float); pr=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred'][15].astype(float)
    e=torch.tensor(pr-gt,device=dev); se2=float((e*e).mean())
    pe=(U.T@e)**2                                        # енергія похибки ланцюга по власних напрямках
    cum=lambda k: float(pe[:k].sum()/pe.sum())
    lamn=lam.cpu().numpy(); pen=pe.cpu().numpy()
    print('\nдамп %d: MSE ланцюга %.3e; слід Cov %.2f; ефект. ранг %.1f; λ: топ1 %.3g, медіана %.3g, мін %.3g'%(i,se2,float(lam.sum()),float(lam.sum()**2/(lam**2).sum()),lamn[0],np.median(lamn),lamn[-1]))
    print('   ізотропія похибки: частка в топ-16 %.3f (ізотр. 0.016), топ-128 %.3f (0.125)'%(cum(16),cum(128)))
    for frac in [0.05,0.2,0.5,0.766]:
        N=frac*2**41/3.36e7; noise=lamn/N
        mse_iso=np.mean(se2*noise/(se2+noise)); mse_real=np.sum(pen*noise/(pen+noise))/1024   # з реальною енергією похибки по напрямках
        util=0.234+frac
        print('   МК %.0f%% бюджету (N=%.0f): MSE ізотр. %.3e, з реальною похибкою %.3e · бал x%.2f (ізотр.) x%.2f (реальна)'%(frac*100,N,mse_iso,mse_real,se2*0.234/(mse_iso*util),se2*0.234/(mse_real*util)))
