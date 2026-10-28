"""newalg95 — ВИВІД З НУЛЯ: чи мережа працює в РЕЖИМІ ДВОХ ПОПУЛЯЦІЙ?

════════════════ ВИВІД (без опори на існуючий алгоритм) ════════════════
1) ТОЧНА ТОТОЖНІСТЬ, що нічого не припускає про розподіл:
       ReLU(x) = x + neg(x),   neg(x) = max(-x,0) >= 0
   ⟹ h_ℓ = W_ℓ·h_{ℓ-1} + n_ℓ ,  n_ℓ = neg(W_ℓ·h_{ℓ-1}) >= 0
   Лінійна частина переноситься ТОЧНО. Уся невідомість — у `n_ℓ`, і вона
   входить АДИТИВНО по шарах (а не мультиплікативно, як спотворення стану).

2) РЕЖИМ МЕРЕЖІ (два незалежні виводи, що сходяться):
   (а) He-init без біасів: ‖W h‖² = 2‖h‖² ⟹ щоб норма зберігалась, РІВНО
       половина нейронів мусить бути вимкнена. Самоузгодженість
       `t² = ‖m^h‖²/tr C^h` по увімкнених ⟹ фіксована точка `t = ±t*`.
   (б) кут між двома входами для ReLU: θ_{ℓ+1} ≈ θ_ℓ − θ_ℓ²/(3π)
       ⟹ θ_ℓ ≈ 3π/ℓ ⟹ при ℓ=31: θ=0.304, cosθ=0.954, SNR=20.7, t*≈4.5

3) НАСЛІДОК — ДВІ ПОПУЛЯЦІЇ З РІЗНОЮ ФІЗИКОЮ:
       A (t>0): h_j ≈ p_j        ГАУСОВА, несе всю масу
       B (t<0): h_j ≈ 0          АТОМ У НУЛІ, різко негаусова, але МАЛА
   ⟹ якщо це так, то ВЕСЬ негаусів дефект локалізований у B, і алгоритм є:
      точна лінійна пропагація по A + окрема дорога обробка малого B.

════════════════ ЩО МІРЯЄМО ════════════════
На кожному шарі ℓ (передактивація p_ℓ = h_{ℓ-1}·W_ℓ):
  T1  розподіл t_j = m_j/s_j: квантилі |t|, частка |t|<1, частка t>0
  T2  SNR_ℓ = ‖m‖²/tr(C)     ← має дати t*² якщо гіпотеза 2 вірна
  T3  cos між h_ℓ(z) і h_ℓ(z') для НЕЗАЛЕЖНИХ z  ← пломба θ_ℓ ≈ 3π/ℓ
  T4  розклад A/B: частка ‖m‖ і tr(C) наступної передактивації від A і від B
  T5  негаусовість (γ₃, γ₄) наступної передактивації ПОВНОЇ / від A / від B
      🔑 ЦЕ РІШУЧИЙ ВИМІР: якщо γ(A) ≈ 0, а γ(повна) ≈ γ(B) — гіпотеза жива
  T6  якість «маскової лінійної моделі» на ОДИН крок:
      ‖E[relu(p)] − 1[t>0]⊙m‖ / ‖E[relu(p)]‖

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску, щоб не тлумачити заднім числом) ═══
 (а) медіана |t| на шарах 16..31 буде 2.0…5.0                      шанс 70%
 (б) частка t>0 буде ≈0.5 ± 0.05 на глибині                        шанс 75%
 (в) cos(h(z),h(z')) на шарі 31 буде 0.90…0.98 (пломба 3π/ℓ)       шанс 55%
 (г) частка |t|<1 на глибині буде <0.25                            шанс 60%
 (д) 🔑 γ₃(від A) буде ВТРИЧІ менша за γ₃(повної)                  шанс 40%
 (е) 🔑 tr C(від B) / tr C(повної) < 0.15 на глибині               шанс 50%
 (ж) маскова модель T6 дасть похибку 5…30% (тобто ГІРШЕ за гаусову
     формулу — маска сама по собі НЕ алгоритм, лише розклад)       шанс 80%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 17
NSAMP = 1 << LOGN
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
AB_LAYERS = [1, 3, 7, 11, 15, 19, 23, 27, 30]   # де робимо дорогий A/B розклад


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


def moments(p):
    """(m, s, γ₃, γ₄) покоординатно, акумуляція у fp64."""
    pd = p.double()
    m = pd.mean(0)
    c = pd - m
    v = (c * c).mean(0)
    s = v.clamp_min(1e-300).sqrt()
    g3 = (c ** 3).mean(0) / s ** 3
    g4 = (c ** 4).mean(0) / v ** 2 - 3.0
    return m, s, g3, g4


print(f"newalg95 · {DEV} · {NNET} мереж · N=2^{LOGN}", flush=True)
t0 = time.time()
ACC = {}          # шар -> список рядків по мережах
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        g = torch.Generator(device=DEV).manual_seed(950000 + 131 * q)
        X = torch.randn(NSAMP, 256, generator=g, device=DEV)
        half = NSAMP // 2
        for L in range(32):
            p = X @ W[L]
            m, s, g3, g4 = moments(p)
            t = m / s
            at = t.abs()
            snr = float((m * m).sum() / (s * s).sum())
            frac_pos = float((t > 0).double().mean())
            frac_small = float((at < 1.0).double().mean())
            qs = torch.quantile(at, torch.tensor(
                [0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64, device=DEV))
            h = torch.relu(p)
            # T3 — кут між образами незалежних входів
            h1, h2 = h[:half].double(), h[half:2 * half].double()
            cosv = float(((h1 * h2).sum(1)
                          / (h1.norm(dim=1) * h2.norm(dim=1)).clamp_min(1e-300)
                          ).mean())
            # T6 — маскова лінійна модель на один крок
            mh = h.double().mean(0)
            mask = (t > 0).double()
            t6 = float((mh - mask * m).norm() / mh.norm().clamp_min(1e-300))
            row = [snr, frac_pos, frac_small, cosv, t6,
                   float(qs[0]), float(qs[1]), float(qs[2]),
                   float(qs[3]), float(qs[4]),
                   float(g3.abs().mean()), float(g4.abs().mean())]
            # T4/T5 — розклад наступної передактивації на A і B
            if L in AB_LAYERS and L + 1 < 32:
                Wn = W[L + 1]
                mk = (t > 0).to(p.dtype)
                pa_ = (h * mk) @ Wn
                pb_ = (h * (1.0 - mk)) @ Wn
                pn = pa_ + pb_
                _, sa, g3a, g4a = moments(pa_)
                _, sb, g3b, g4b = moments(pb_)
                mn, sn, g3n, g4n = moments(pn)
                va, vb, vn = (sa * sa).sum(), (sb * sb).sum(), (sn * sn).sum()
                row += [float(va / vn), float(vb / vn),
                        float(g3n.abs().mean()), float(g3a.abs().mean()),
                        float(g3b.abs().mean()),
                        float(g4n.abs().mean()), float(g4a.abs().mean()),
                        float(g4b.abs().mean())]
            else:
                row += [float('nan')] * 8
            ACC.setdefault(L, []).append(row)
            X = h
        print(f"  net{q} готово ({time.time()-t0:.0f} с)", flush=True)

M = np.array([np.mean(ACC[L], axis=0) for L in range(32)])
np.save("/mnt/win_d/llm/White-Box/arena62/logs/newalg95.npy", M)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 78)
print("T1/T2/T3 — РЕЖИМ МЕРЕЖІ: чи є фіксована точка t=±t* і чи злипає напрямок")
print("=" * 78)
print(f"{'шар':>4s}{'SNR':>8s}{'t*=√SNR':>9s}{'частка t>0':>11s}"
      f"{'|t|<1':>8s}{'|t| мед':>9s}{'|t| 10%':>9s}{'|t| 90%':>9s}"
      f"{'cos(z,z′)':>11s}{'3π/ℓ→cos':>10s}")
for L in range(32):
    r = M[L]
    th = 3.0 * np.pi / max(L, 1)
    pred = np.cos(min(th, np.pi / 2))
    print(f"{L:>4d}{r[0]:>8.2f}{np.sqrt(max(r[0],0)):>9.2f}{r[1]:>11.3f}"
          f"{r[2]:>8.3f}{r[7]:>9.2f}{r[5]:>9.2f}{r[9]:>9.2f}"
          f"{r[3]:>11.4f}{pred:>10.4f}")

print("\n" + "=" * 78)
print("T4/T5 — РІШУЧИЙ ВИМІР: чи вся негаусовість сидить у популяції B (t<0)")
print("=" * 78)
print(f"{'шар':>4s}{'VarA/Var':>10s}{'VarB/Var':>10s}"
      f"{'|γ₃| повн':>10s}{'|γ₃| A':>9s}{'|γ₃| B':>9s}"
      f"{'|γ₄| повн':>10s}{'|γ₄| A':>9s}{'|γ₄| B':>9s}")
for L in AB_LAYERS:
    r = M[L]
    print(f"{L:>4d}{r[12]:>10.4f}{r[13]:>10.4f}"
          f"{r[14]:>10.4f}{r[15]:>9.4f}{r[16]:>9.4f}"
          f"{r[17]:>10.4f}{r[18]:>9.4f}{r[19]:>9.4f}")

print("\n" + "=" * 78)
print("T6 — маскова лінійна модель на ОДИН крок (контроль: мусить бути ГІРШЕ)")
print("=" * 78)
for L in (0, 1, 3, 7, 15, 23, 31):
    print(f"  шар {L:>2d}: віднос. похибка 1[t>0]⊙m проти E[relu(p)] = "
          f"{M[L][4]*100:>7.3f}%")

print("\n🔑 ЧИТАТИ:")
print("  · частка t>0 ≈ 0.5 і |t| мед ≳ 2 ⟹ РЕЖИМ ДВОХ ПОПУЛЯЦІЙ підтверджено")
print("  · γ₃(A) << γ₃(повна) ⟹ негаусовість локалізована в B ⟹ алгоритм живий")
print("  · VarB/Var мала ⟹ B дешева для дорогої точної обробки")
