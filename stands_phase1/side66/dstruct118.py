"""структура залишку Δ_l = P3f_l − Kab_l (продакшен-рекурсія, регіна): де сидить те, що важить"""
import numpy as np
OUT="/home/trim/wblogs"; NET="regina-johnston"
kab=np.load("%s/rec118h_kab_%s.npz"%(OUT,NET)); bk=np.load("%s/sub118_%s.npz"%(OUT,NET)); cb=np.load("%s/covbank118_%s.npz"%(OUT,NET))
print(" l  ‖Δ‖/‖Kt‖  діаг%   уздовж Kt  ампл.деф  топ8-SV%  топ64-SV%  Δ у топ64 Σ(h)%  Kt у топ64 Σ(h)%")
for l in range(1,16):
    Kt=bk["P3f_%d"%l]; K=kab["K_%d"%l]; D=Kt-K
    diag_share=(np.diag(D)**2).sum()/(D**2).sum()
    along=(D*Kt).sum()/np.linalg.norm(Kt)**2          # компонента Δ уздовж Kt (частка амплітуди, якої бракує)
    sv=np.linalg.svd(D,compute_uv=False); top8=(sv[:8]**2).sum()/(sv**2).sum(); top64=(sv[:64]**2).sum()/(sv**2).sum()
    S=cb["S_%d"%(l+1)] if "S_%d"%(l+1) in cb else None
    if S is not None:
        ev,U=np.linalg.eigh(S); U=U[:,::-1][:,:64]; P=U@U.T
        # Δ живе в p_l просторі (індекси a,b — нейрони p_{l}); Σ(h_{l+1}) — простір h після ReLU того ж p: індекси ті самі
        dsh=np.linalg.norm(P@D@P)**2/np.linalg.norm(D)**2; ksh=np.linalg.norm(P@Kt@P)**2/np.linalg.norm(Kt)**2
    else: dsh=ksh=float('nan')
    print("%2d   %.3f    %5.1f    %+.3f     %.3f     %5.1f     %5.1f       %5.1f          %5.1f"%(l,np.linalg.norm(D)/np.linalg.norm(Kt),100*diag_share,along,1-(K*Kt).sum()/np.linalg.norm(Kt)**2,100*top8,100*top64,100*dsh,100*ksh))
print("DONE_DSTRUCT")
