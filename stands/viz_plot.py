import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
d = np.load("/home/trim/wb2data/viz_0.npz"); n = int(d["n"]); gt = d["gt"]; O = "/mnt/win_d/llm/White-Box-P2/docs/2026-10-13/viz/"
kp = np.load("/home/trim/wb2data/kp_p2/mlp_00000.npz")
eb = d["pred_base"] - gt; eo = d["pred_oracle"] - gt; floor_sd = np.sqrt(d["avg_var"] / 1e9); rng = np.random.default_rng(0); ef = rng.standard_normal(n) * floor_sd
rms = lambda x: float(np.sqrt((x**2).mean()))
# ---- Fig 1: вихід 32×32 ----
fig, ax = plt.subplots(1, 4, figsize=(20, 5.2)); vm = np.abs(eb).max()
for a, e, t in zip(ax[:3], [eb, eo, ef], ["наш best v34: pred − істина", "стеля класу (усі парні кумулянти істинні)", "підлога істини (шум МК 1e9)"]):
    im = a.imshow(e.reshape(32, 32), cmap="RdBu_r", vmin=-vm, vmax=vm); a.set_title(f"{t}\nRMS {rms(e):.2e}  MSE {rms(e)**2:.2e}", fontsize=10); a.axis("off")
fig.colorbar(im, ax=ax[:3], shrink=0.8, label="похибка виходу (одна шкала)")
gtm = kp["gt_mean"]; Lb = [float(((d["all_base"][l]-gtm[l])**2).mean()) for l in range(16)]; Lo = [float(((d["all_oracle"][l]-gtm[l])**2).mean()) for l in range(16)]
ax[3].semilogy(range(16), Lb, "o-", label="наш best"); ax[3].semilogy(range(16), Lo, "s-", label="стеля класу"); ax[3].axhline(floor_sd**2, color="k", ls="--", label="підлога істини 7.9e-11")
ax[3].set_xlabel("шар"); ax[3].set_ylabel("MSE середніх шару"); ax[3].set_title("похибка по шарах"); ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)
fig.suptitle("Вихід мережі (1024 нейрони як 32×32), дамп 0 (logan-fitzgerald): наш best проти межі точності", fontsize=12)
fig.savefig(O + "01_output_best_vs_ceiling.png", dpi=110, bbox_inches="tight"); plt.close(fig)
# ---- Fig 2: D21 як карти ознак ----
layers = [1, 4, 8, 12, 15]; crop = 96
fig, ax = plt.subplots(3, len(layers), figsize=(3.4*len(layers), 10.5))
for j, l in enumerate(layers):
    his = d[f"D21_{l}"]; tru = d[f"tD21_{l}"]; np.fill_diagonal(his, 0); np.fill_diagonal(tru, 0); res = tru - his
    vm = np.percentile(np.abs(tru), 99.5); norm = SymLogNorm(linthresh=vm/30, vmin=-vm, vmax=vm)
    for i, (M, t) in enumerate([(his, "наш D21 (зріз κ₃)"), (tru, "істинний D21 (банк 1e8)"), (res, "нев'язка істина − наш")]):
        a = ax[i, j]; a.imshow(M[:crop, :crop], cmap="RdBu_r", norm=norm); a.set_xticks([]); a.set_yticks([])
        eps = np.linalg.norm(res)/np.linalg.norm(tru)
        a.set_title(f"шар {l}: {t}" + (f"\nε {eps:.3f}" if i == 2 else ""), fontsize=9)
fig.suptitle(f"(2,1)-зріз κ₃ передактивацій, кут {crop}×{crop} з 1024×1024, симлог-шкала по істині шару", fontsize=12)
fig.savefig(O + "02_D21_maps.png", dpi=110, bbox_inches="tight"); plt.close(fig)
# ---- Fig 2b: повні матриці, блоки 8×8 (середнє модуля) ----
fig, ax = plt.subplots(2, len(layers), figsize=(3.4*len(layers), 7))
for j, l in enumerate(layers):
    tru = d[f"tD21_{l}"].copy(); his = d[f"D21_{l}"].copy(); np.fill_diagonal(tru, 0); np.fill_diagonal(his, 0); res = tru - his
    blk = lambda M: np.abs(M).reshape(128, 8, 128, 8).mean((1, 3))
    T = blk(tru); R = blk(res); vmx = np.percentile(T, 99)
    ax[0, j].imshow(T, cmap="magma", vmin=0, vmax=vmx); ax[0, j].set_title(f"шар {l}: |істина| блоки 8×8", fontsize=9)
    ax[1, j].imshow(R, cmap="magma", vmin=0, vmax=vmx); ax[1, j].set_title(f"|нев'язка|, та сама шкала", fontsize=9)
    for a in ax[:, j]: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Повний D21 1024×1024 (модуль, усереднений блоками 8×8): де сидить нев'язка", fontsize=12)
fig.savefig(O + "03_D21_full_blocks.png", dpi=110, bbox_inches="tight"); plt.close(fig)
# ---- Fig 3: діагоналі по шарах (κ₄, D3, var) як смуги 15×1024 ----
fig, ax = plt.subplots(3, 3, figsize=(18, 8))
for i, (key, tkey, name) in enumerate([("k4", "tk4", "κ₄ діагональ"), ("D3", "tD3", "κ₃ діагональ (D3)"), ("C", "tC", "дисперсія (diag C)")]):
    H = []; T = []
    for l in range(1, 16):
        h = d[f"{key}_{l}"]; t = d[f"{tkey}_{l}"]
        if key == "C": h = np.diag(h); t = np.diag(t)
        H.append(h); T.append(t)
    H = np.array(H); T = np.array(T); R = T - H; sc = np.linalg.norm(T, axis=1, keepdims=True)
    order = np.argsort(T[14])   # сортуємо нейрони за істиною останнього шару, щоб смуги читались
    vm = np.percentile(np.abs(T/sc), 99.5)
    for k, (M, t) in enumerate([(H/sc, f"наш {name}"), (T/sc, f"істинний {name}"), (R/sc, "нев'язка")]):
        a = ax[i, k]; im = a.imshow(M[:, order], aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm); a.set_yticks(range(0, 15, 2)); a.set_yticklabels(range(1, 16, 2)); a.set_xlabel("нейрон (відсортовано за істиною ш.15)"); a.set_ylabel("шар")
        eps = np.linalg.norm(R, axis=1)/np.linalg.norm(T, axis=1)
        a.set_title(t + (f"   ε по шарах {eps[0]:.2f}→{eps[7]:.2f}→{eps[14]:.2f}" if k == 2 else ""), fontsize=9)
fig.suptitle("Понейронні величини по шарах (нормовано нормою істини шару): наш ланцюг, істина, нев'язка", fontsize=12)
fig.savefig(O + "04_diagonals_layers.png", dpi=110, bbox_inches="tight"); plt.close(fig)
# ---- Fig 4: ε по шарах для чотирьох об'єктів ----
fig, a = plt.subplots(figsize=(8, 4.5))
for key, tkey, name, m in [("D21", "tD21", "D21 (зріз κ₃)", "o"), ("D3", "tD3", "κ₃ діаг", "s"), ("k4", "tk4", "κ₄ діаг", "^"), ("C", "tC", "C (коваріація)", "x")]:
    eps = []
    for l in range(1, 16):
        h = d[f"{key}_{l}"].copy(); t = d[f"{tkey}_{l}"].copy()
        if key == "D21": np.fill_diagonal(h, 0); np.fill_diagonal(t, 0)
        eps.append(np.linalg.norm(t - h)/np.linalg.norm(t))
    a.plot(range(1, 16), eps, marker=m, label=name)
a.set_xlabel("шар"); a.set_ylabel("відносна похибка ‖істина − наш‖/‖істина‖"); a.set_yscale("log"); a.grid(alpha=.3); a.legend(); a.set_title("Скільки ми втрачаємо на кожному об'єкті по шарах (дамп 0)")
fig.savefig(O + "05_eps_per_layer.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("MSE base", rms(eb)**2, "oracle", rms(eo)**2, "floor", floor_sd**2)
