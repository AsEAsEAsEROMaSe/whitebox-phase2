"""С116: ЧИ ПОКРИВАЮТЬ ПОНЕЙРОННІ γ₃,γ₄ ЩІЛИНУ, ЩО ЛИШАЄТЬСЯ ПРИ ТОЧНИХ (mp,C)?

[ceil116b] точні (mp,C) на шарі 15 ⟹ 0.0481%; для #1 треба 0.0266% ⟹ ×1.81.
Ця щілина — ЧИСТА ціна гаусової гіпотези на ОДНОМУ кроці від ТОЧНИХ моментів.

🔑 КЛЮЧОВЕ СПОСТЕРЕЖЕННЯ: `E[ReLU(p_j)]` залежить ЛИШЕ від МАРГІНАЛУ `p_j`.
Спільний розподіл не потрібен ВЗАГАЛІ. ⟹ стеля «понейронного маргіналу» = 0,
і питання суто в тому, скільки моментів цього маргіналу треба.

  гаус     E[ReLU]/s = ψ(t) = φ(t) + tΦ(t)
  +Еджворт          − (γ₃/6)·t·φ + (γ₄/24)·(t²−1)·φ

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 Еджворт із ТОЧНИМИ γ₃,γ₄ знімає >60% щілини                   60%
  P2 γ₃ важить більше за γ₄ (ReLU чутлива до асиметрії)            55%
  P3 залишок після 4 моментів >0.02% ⟹ 4 моментів МАЛО для #1      50%
ГЕЙТ: якщо 4 точні моменти дають <0.0266%, ціль стає інженерною —
«порахувати γ дешево». Якщо ні — потрібен увесь маргінал, і це інший клас.
"""
import sys, glob, numpy as np
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
NW = 1024; TRUE2 = 0.836447
V_MC = 3.3478454e-06; MULT = 0.100131; E2_GRADER = 1.4355576e-06

HI = {}
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True); HI[str(d["name"])] = d["m"][15].astype(np.float64)

B = []
for f in sorted(glob.glob("/home/trim/wblogs/gam116_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    if nm not in HI: continue
    B.append((nm, d, HI[nm]))
print("банків %d · N=%d" % (len(B), int(B[0][1]["n"])))

def psi(t): return phi(t) + t*Phi(t)

def modes(d, L=15):
    mp, sd = d["mp_%d" % L], d["sd_%d" % L]
    g3, g4 = d["g3_%d" % L], d["g4_%d" % L]
    t = mp/sd; ph = phi(t)
    gau = sd*psi(t)
    c3  = -sd*(g3/6.0)*t*ph
    c4  =  sd*(g4/24.0)*(t*t - 1.0)*ph
    return dict(gau=gau, g3=gau+c3, g4=gau+c4, both=gau+c3+c4,
                bank=d["relu_%d" % L], c3=c3, c4=c4)

def err(x, T): return float(np.mean((x-T)**2))
def cerr(x, T):
    a = (x@T)/max((x*x).sum(), 1e-300); return float(np.mean((a*x-T)**2))

ACC = {}
for nm, d, T in B:
    M = modes(d)
    for k in ["gau", "g3", "g4", "both", "bank"]:
        ACC.setdefault(k, []).append((err(M[k], T), cerr(M[k], T)))
    ACC.setdefault("_c3", []).append(float(np.linalg.norm(M["c3"])/np.linalg.norm(M["gau"])))
    ACC.setdefault("_c4", []).append(float(np.linalg.norm(M["c4"])/np.linalg.norm(M["gau"])))

Cb = np.mean([a[1] for a in ACC["gau"]])
ANCH = E2_GRADER/1.1417e-06        # прив'язка з ceil116b (косинусна база оцінювача)
def adj(c): 
    e2 = c*ANCH; return (1.0/(1.0/V_MC + 1.0/e2))*MULT

print("\n%-8s %11s %9s %8s | %11s %9s | %11s %s" %
      ("режим","косинусна","похибка%","×гаус","сира","похибка%","adj(мод.)","×до #1"))
LB = {"gau":"гаус","g3":"+γ₃","g4":"+γ₄","both":"+γ₃γ₄","bank":"🔒банк"}
for k in ["gau","g3","g4","both","bank"]:
    r = np.mean([a[0] for a in ACC[k]]); c = np.mean([a[1] for a in ACC[k]])
    print("%-8s %11.4e %8.4f%% %8.3f | %11.4e %8.4f%% | %11.4e %6.2f"
          % (LB[k], c, 100*np.sqrt(c/TRUE2), Cb/c, r, 100*np.sqrt(r/TRUE2), adj(c), adj(c)/7.30e-09))

print("\n🔒 ПЛОМБИ")
print("  🔒 гаус від ТОЧНИХ (mp,sd): %.4f%%   [ceil116b mpC@15 = 0.0481%%]"
      % (100*np.sqrt(Cb/TRUE2)))
bk = np.mean([a[1] for a in ACC["bank"]])
print("  🔒 банк проти цілі 2²³ (тавтологія): %.4f%%  = шум банку N=2²², має бути ≪ решти"
      % (100*np.sqrt(bk/TRUE2)))
print("  ‖поправка γ₃‖/‖гаус‖ = %.4f · ‖γ₄‖/‖гаус‖ = %.4f  ⟹ гейт збіжності ряду"
      % (np.mean(ACC["_c3"]), np.mean(ACC["_c4"])))

print("\n── ШАР 0: ПЕРЕДАКТИВАЦІЯ ГАУСОВА ТОЧНО ⟹ НАЙЖОРСТКІША ПЛОМБА ──")
for nm, d, T in B[:3]:
    mp, sd = d["mp_0"], d["sd_0"]
    g = sd*psi(mp/sd)
    print("  %-18s γ₃ %+.5f  γ₄ %+.5f  ·  ‖гаус − банк‖/‖банк‖ = %.3e"
          % (nm, d["g3_0"].mean(), d["g4_0"].mean(),
             np.linalg.norm(g-d["relu_0"])/np.linalg.norm(d["relu_0"])))

print("\n🎯 ВИРОК")
cb2 = np.mean([a[1] for a in ACC["both"]])
print("  потрібно для #1                 0.0266%%")
print("  гаус від точних моментів        %.4f%%" % (100*np.sqrt(Cb/TRUE2)))
print("  +точні понейронні γ₃,γ₄         %.4f%%" % (100*np.sqrt(cb2/TRUE2)))
print("  ⟹ щілину знято на %.0f%% (у похибці)"
      % (100*(np.sqrt(Cb)-np.sqrt(cb2))/(np.sqrt(Cb)-np.sqrt(0.0266e-2**2*TRUE2))))
