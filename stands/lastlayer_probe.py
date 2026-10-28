"""13.10.2026: що насправді відрізняє e-8 від e-11 на одній мережі (дамп 0, останній шар, істина gt = E[ReLU(p15)] з 1e9).
Драбина 1-D кроку при ІСТИННИХ понейронних кумулянтах p15 (банк 1e8): гаус(a,v) → +γ₃ → +γ₄ (Грам-Шарльє через квадратуру)
проти gt; наш pred; оракульний pred (усе істинне у use). Плюс ознаки похибки по нейронах."""
import numpy as np
d = np.load("/home/trim/wb2data/viz_0.npz"); kp = np.load("/home/trim/wb2data/kp_p2/mlp_00000.npz"); gt = d["gt"]; n = len(gt)
pb = d["pred_base"]; po = d["pred_oracle"]; l = 15
a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l]; E3 = kp["pre_m3"][l]; E4 = kp["pre_m4"][l]
v = E2 - a*a; s = np.sqrt(v); t = a/s; k3 = E3 - 3*a*E2 + 2*a**3; k4 = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v; g3 = k3/s**3; g4 = k4/v**2
x = np.linspace(-12, 12, 6001); wq = np.exp(-x*x/2); wq = wq/wq.sum()   # E над N(0,1), трапеції
Z = x[None, :]; R = np.maximum(a[:, None] + s[:, None]*Z, 0.0)
He3 = Z**3 - 3*Z; He4 = Z**4 - 6*Z**2 + 3
Eg = (R*wq).sum(1); E3t = (R*He3*wq).sum(1); E4t = (R*He4*wq).sum(1)
pg = Eg; p3 = Eg + g3/6*E3t; p4 = p3 + g4/24*E4t
mse = lambda p: float(((p-gt)**2).mean())
print(f"MSE: наш pred {mse(pb):.3e} · оракул (усе істинне у use) {mse(po):.3e} · підлога істини 7.9e-11")
print(f"1-D крок при ІСТИННИХ кумулянтах p15: гаус(a,v) {mse(pg):.3e} · +γ₃ {mse(p3):.3e} · +γ₄ {mse(p4):.3e}   ⟹ решта (γ₅+ і шум банку) {mse(p4):.3e}")
# шум банку на a, v: sd(a) = s/√1e8 ⟹ похибка E[ReLU] ≈ Φ(t)·s/1e4
print(f"шум банку 1e8 у (a,v) переноситься у MSE ≈ {float(np.mean((s/1e4)**2)):.2e}")
eb = pb - gt; eo = po - gt; e4 = p4 - gt
cs = lambda u, w: float(u@w/np.linalg.norm(u)/np.linalg.norm(w))
print(f"cos(e_наш, e_оракул) {cs(eb,eo):+.3f} · cos(e_наш, e_4кум) {cs(eb,e4):+.3f} · cos(e_оракул, e_4кум) {cs(eo,e4):+.3f}")
print(f"зсув: mean(e)/rms(e): наш {eb.mean()/np.sqrt((eb**2).mean()):+.3f} · оракул {eo.mean()/np.sqrt((eo**2).mean()):+.3f} · 4кум {e4.mean()/np.sqrt((e4**2).mean()):+.3f}")
print(f"ексцес e: наш {((eb-eb.mean())**4).mean()/eb.var()**2-3:+.2f} · оракул {((eo-eo.mean())**4).mean()/eo.var()**2-3:+.2f}; частка ‖e‖² у топ-10% нейронів: наш {np.sort(eb**2)[::-1][:102].sum()/(eb**2).sum():.2f} · оракул {np.sort(eo**2)[::-1][:102].sum()/(eo**2).sum():.2f} (ізотропно 0.10)")
feats = {"|t|": np.abs(t), "t": t, "sd": s, "gt": gt, "γ₃": g3, "γ₄": g4, "Φ(t)": 0.5*(1+np.vectorize(__import__("math").erf)(t/np.sqrt(2)))}
print("corr(|e|, ознака):", " · ".join(f"{k} {np.corrcoef(np.abs(eb),f)[0,1]:+.2f}/{np.corrcoef(np.abs(eo),f)[0,1]:+.2f}" for k,f in feats.items()), "(наш/оракул)")
print("corr(e, ознака):  ", " · ".join(f"{k} {np.corrcoef(eb,f)[0,1]:+.2f}/{np.corrcoef(eo,f)[0,1]:+.2f}" for k,f in feats.items()))
# внесок стану проти кроку: наш 1-D крок при НАШОМУ стані (mu, var his) гаусом
mu_h = d["mu_15"]; v_h = np.diag(d["C_15"]); s_h = np.sqrt(v_h); Rh = np.maximum(mu_h[:, None] + s_h[:, None]*Z, 0.0); pgh = (Rh*wq).sum(1)
print(f"гаусів 1-D крок при НАШОМУ стані (mu,var ш.15 ланцюга): MSE {mse(pgh):.3e}; cos з нашим pred−gt {cs(pgh-gt, eb):+.3f}; похибка стану: ‖mu−a‖/‖a‖ {np.linalg.norm(mu_h-a)/np.linalg.norm(a):.2e}, ‖s−s_true‖/‖s‖ {np.linalg.norm(s_h-s)/np.linalg.norm(s):.2e}")

Phi = 0.5*(1+np.vectorize(__import__("math").erf)(t/np.sqrt(2))); phi = np.exp(-t*t/2)/np.sqrt(2*np.pi)
em = Phi*(mu_h-a); esd = phi*(s_h-s); lin = em+esd
print(f"лінійна модель похибки e ≈ Φ(t)·Δmu + φ(t)·Δs: cos(e, модель) {cs(eb, lin):+.3f}, ‖модель‖/‖e‖ {np.linalg.norm(lin)/np.linalg.norm(eb):.3f}; частки: Δmu {np.linalg.norm(em)/np.linalg.norm(eb):.3f}, Δs {np.linalg.norm(esd)/np.linalg.norm(eb):.3f}; cos(e, Δmu-член) {cs(eb, em):+.3f}")
print(f"відносна похибка середніх p15 по нейронах: медіана |Δmu/a| {np.median(np.abs((mu_h-a)/a)):.2e}; щоб мати e-11 треба ≈ {np.sqrt(7.9e-11/2.19e-8)*1.5e-4:.1e}")
