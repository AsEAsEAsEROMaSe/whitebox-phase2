"""Скан рівня Штрассена. Вирок — ПРОЄКЦІЯ НА ГРЕЙДЕРНИЙ ТАРИФ, не локальний бал."""
import json, math
B = 2.72e11
SP = "/tmp/claude-1000/-mnt-win-d-llm-White-Box/14046632-ec1b-4ec7-b4b4-7e24ea058ea0/scratchpad"
runs = [("база   L=0", "out_base.json"), ("Штрассен L=1", "out_str1.json"),
        ("Штрассен L=2", "out_str.json")]
data = {}
for tag, f in runs:
    try:
        data[tag] = json.load(open(f"{SP}/{f}"))["results"]["per_mlp"]
    except Exception as e:
        print(f"{tag}: НЕМА ({e})")
base = data["база   L=0"]
n = len(base)


def stats(pm):
    F = sum(m["flops_used"] for m in pm) / len(pm) / 2.0        # → грейдерний тариф
    R = sum(m["effective_compute"] - m["flops_used"] for m in pm) / len(pm) / 1e11
    mse = sum(m["final_layer_mse"] for m in pm) / len(pm)
    score = sum(m["final_layer_mse"] * max(0.1, (m["flops_used"] / 2 +
                1e11 * (m["effective_compute"] - m["flops_used"]) / 1e11) / B)
                for m in pm) / len(pm)
    return F, R, mse, score


F0, R0, m0, s0 = stats(base)
print(f"панель: {n} мереж · F подано у ГРЕЙДЕРНОМУ тарифі (локальне ÷2)\n")
print(f"{'варіант':<14}{'MSE':>14}{'F/B':>9}{'R мс':>8}{'C/B':>9}{'бал':>13}{'×бал':>8}{'знак':>8}")
for tag in [t for t, _ in runs if t in data]:
    pm = data[tag]
    F, R, mse, sc = stats(pm)
    C = (F + 1e11 * R) / B
    w = sum(1 for x, y in zip(base, pm) if y["final_layer_mse"] < x["final_layer_mse"])
    p = min(1.0, sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n * 2)
    sg = "—" if tag.startswith("база") else f"{w}/{n}"
    print(f"{tag:<14}{mse:>14.6e}{F/B:>9.5f}{R*1000:>8.1f}{C:>9.5f}{sc:>13.5e}"
          f"{s0/sc:>8.4f}{sg:>8}" + (f"  p={p:.4f}" if sg != "—" else ""))
print(f"\nрекорд грейдера 1.855e-07 ⟹ проєкції:")
for tag in [t for t, _ in runs if t in data]:
    _, _, _, sc = stats(data[tag])
    print(f"   {tag:<14} {1.855e-07/(s0/sc):.4e}")
