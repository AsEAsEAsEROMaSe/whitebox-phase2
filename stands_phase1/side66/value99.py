"""Гейт цінності: скільки коштує ЗНАННЯ величини похибки замикання на мережу.
Модель бленду (та сама, що в CLAUDE.md): MSE_i(w) = w²E_i² + (1-w)²V_i.
Питання: скільки дає w_i = V/(V+Ê_i²) замість глобального w, якщо предиктор
має ПОЗАВИБІРКОВИЙ R² = r2 у log E²."""
import math, sys, numpy as np
V   = 1.9227e-06          # [ЗМІРЯНО прилад _W_BLEND=0.0, 50 мереж]
E2  = 2.3385e-05          # [ЗМІРЯНО прилад _W_BLEND=1.0, 50 мереж]
SDL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50   # sd(log e) між мережами
R2  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
rng = np.random.default_rng(0)
M = 200000
# E_i² логнормальна з тим самим середнім і зміряним розкидом (sd log E² = 2*SDL)
s = 2.0 * SDL
g = rng.standard_normal(M)
Ei2 = E2 * np.exp(s * g - 0.5 * s * s)
wg = V / (V + Ei2.mean())                    # глобальний оптимум
base = (wg**2 * Ei2 + (1-wg)**2 * V).mean()
orac = (V * Ei2 / (V + Ei2)).mean()           # ідеальний предиктор
rho = math.sqrt(max(R2, 0.0))
# ЧЕСНИЙ предиктор: регресія на істину ⟹ ŷ = ρ²·(log E² - μ) + шум, але
# оптимальний план бере УМОВНЕ середнє ⟹ стиснення рівно ρ² (закон «розкид
# проксі мусить збігатись із розкидом цілі»)
lg = np.log(Ei2)
mu = lg.mean()
pred = np.exp(mu + R2 * (lg - mu))
wp = V / (V + pred)
prox = (wp**2 * Ei2 + (1-wp)**2 * V).mean()
print(f"sd(log e) = {SDL:.3f} · R²(позавибірковий) = {R2:.3f}")
print(f"  глобальний w = {wg:.5f}   MSE = {base:.6e}")
print(f"  ОРАКУЛ w_i (ідеальне знання E_i²)  MSE = {orac:.6e}  ⟹ ×{base/orac:.4f}")
print(f"  предиктор із R²={R2:.2f}            MSE = {prox:.6e}  ⟹ ×{base/prox:.4f}")
