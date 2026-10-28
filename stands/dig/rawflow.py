"""17.10.2026, ідея юзера: дивитись СИРИЙ рух накопичення середнього під час семплювання, на нейроні і на кожній вазі.
Один дамп, шар L, нейрон j. Семплюємо N точок через мережу (GPU), для нейрона j на шарі L рахуємо:
  вклад кожної ваги v_i(s) = h_i(s) * W[i,j] (1024 на семпл), їх суму p_j(s), ReLU(p_j(s)).
Записуємо: накопичене середнє (біжучий середній) p_j і ReLU(p_j) після 2^k семплів; накопичене середнє вкладів по вагах;
порівнюємо з ланцюгом (mu_j, і ланцюговий вклад E[h_i]·W[i,j]) і з істиною банку (1e9). Вихід: npz для артефакту."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo
dev='cuda'; i=int(sys.argv[1]); L=int(sys.argv[2]); N=int(sys.argv[3]); seed=777
d=np.load(lo.DUMP.format(i)); Wn=d['weights']; kp=np.load(lo.BANK.format(i))
W=[torch.tensor(Wn[l],dtype=torch.float32,device=dev) for l in range(16)]
g=torch.Generator(device=dev).manual_seed(seed)
lv=np.load('/home/trim/wblogs/live/d%d_l%02d.npz'%(i,L)); mu_c=lv['mu']; lvp=np.load('/home/trim/wblogs/live/d%d_l%02d.npz'%(i,L-1))
mu_true=kp['pre_mean'][L]; h_true_prev=kp['mean'][L-1]
# вибрати 3 нейрони: найбільша |похибка ланцюга|, медіанна, біля щілини (|t| мін)
e=mu_c-mu_true; s=np.sqrt(np.diag(lv['C'])); t=mu_c/s
js=[int(np.argmax(np.abs(e))), int(np.argsort(np.abs(e))[512]), int(np.argmin(np.abs(t)))]
B=4096; ks=[2**k for k in range(0,int(np.log2(N))+1)]
acc_p=torch.zeros(3,dtype=torch.float64,device=dev); acc_r=torch.zeros(3,dtype=torch.float64,device=dev); acc_v=torch.zeros(1024,3,dtype=torch.float64,device=dev)
acc_h=torch.zeros(1024,dtype=torch.float64,device=dev); n=0; traj=[]; traj_v=[]; first=None
Wj=W[L][:,js]   # (1024,3)
while n<N:
    z=torch.randn(B,1024,generator=g,device=dev); h=z
    for l in range(L): h=torch.relu(h@W[l])
    p=h@Wj; r=torch.relu(p); v=h[:,:,None]*Wj[None]   # (B,1024,3)
    if first is None: first=(h[0].double().cpu().numpy(), p[0].double().cpu().numpy())
    # покрокове накопичення всередині пакета для дрібних k
    for k in ks:
        if n<k<=n+B:
            m=k-n; traj.append((k,((acc_p+p[:m].double().sum(0))/k).cpu().numpy(),((acc_r+r[:m].double().sum(0))/k).cpu().numpy()))
            traj_v.append((k,((acc_v+v[:m].double().sum(0))/k).cpu().numpy()))
    acc_p+=p.double().sum(0); acc_r+=r.double().sum(0); acc_v+=v.double().sum(0); acc_h+=h.double().sum(0); n+=B
chain_v=(lvp['mu']*0+h_true_prev)[:,None]*Wj.cpu().numpy()   # вклад ваг за ІСТИННИМ E[h] попер. шару
Ph_prev=0.5*(1+torch.erf(torch.tensor(lvp['mu']/np.sqrt(np.diag(lvp['C'])))/np.sqrt(2))).numpy(); ph_prev=np.exp(-(lvp['mu']/np.sqrt(np.diag(lvp['C'])))**2/2)/np.sqrt(2*np.pi)
Eh_chain=lvp['mu']*Ph_prev+np.sqrt(np.diag(lvp['C']))*ph_prev
chain_v2=Eh_chain[:,None]*Wj.cpu().numpy()
np.savez('/home/trim/wblogs/rawflow_d%d_L%d.npz'%(i,L), js=js, ks=[x[0] for x in traj], p=np.array([x[1] for x in traj]), r=np.array([x[2] for x in traj]),
         v=np.array([x[1] for x in traj_v]), mu_chain=mu_c[js], mu_true=mu_true[js], t=t[js], s=s[js], Wj=Wj.cpu().numpy(), v_chain=chain_v2, v_truth=chain_v,
         h_mean_mc=(acc_h/n).cpu().numpy(), h_mean_true=h_true_prev, h_mean_chain=Eh_chain, h0=first[0], p0=first[1], e_all=e, t_all=t)
for a,j in enumerate(js):
    print("нейрон %4d (t=%.2f): істина μ %.6f · ланцюг %.6f (похибка %.2e) · МК після %d: %.6f (шум %.1e)"%(j,t[j],mu_true[j],mu_c[j],e[j],n,float(acc_p[a]/n),s[j]/np.sqrt(n)))
