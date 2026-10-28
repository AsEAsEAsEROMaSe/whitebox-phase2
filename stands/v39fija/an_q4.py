# 19.10.2026 white-box-8c: таблиця q4 (сталий стан, мережі 3 і 4 процесу) по тегах + пари.
import json, re, sys, os
L = "/home/trim/wblogs"
TAGS = sys.argv[1:] or ["q4_v46", "q4_v47", "q4_v47r128", "q4_v48t285", "q4_v48t32"]
rss = {}
for ln in open(f"{L}/v37_runs.log"):
    m = re.match(r"DONE (\S+) peakRSS=(\S+)", ln)
    if m: rss[m.group(1)] = m.group(2)
U = 2 * 1024**3
R = {}
for t in TAGS:
    p = f"{L}/v37_{t}.json"
    if not os.path.exists(p): print(t, "немає"); continue
    r = json.load(open(p))["results"]; pm = r["per_mlp"]
    R[t] = pm
    print(f"\n{t}  n={len(pm)}  RSS={rss.get(t,'?')}  agg score {r['adjusted_final_layer_score']:.4e}")
    for m in pm:
        print(f"  {m['mlp_index']} {m['mlp_name'][:16]:16s} mse {m['final_layer_mse']:.4e} "
              f"score {m['adjusted_final_layer_score']:.4e} F {m['flops_used']/U:7.1f}u "
              f"wall {m['wall_time_s']:6.1f} res {m['residual_wall_time_s']:.3f}")
def steady(t, k):  # мережі 3 і 4 (індекси 2, 3)
    return [m[k] for m in R[t][2:4]]
ref = "q4_v46" if "q4_v46" in R else None
pairs = [("q4_v47", "q4_v46"), ("q4_v47r128", "q4_v46"), ("q4_v48t285", "q4_v47"), ("q4_v48t32", "q4_v47")]
print("\nпари на мережах 3+4 (score b/a > 1 = a краща):")
for a, b in pairs:
    if a in R and b in R and len(R[a]) >= 4 and len(R[b]) >= 4:
        sa, sb = steady(a, "adjusted_final_layer_score"), steady(b, "adjusted_final_layer_score")
        ma, mb = steady(a, "final_layer_mse"), steady(b, "final_layer_mse")
        fa, fb = steady(a, "flops_used"), steady(b, "flops_used")
        print(f"  {a} проти {b}: score ×{sum(sb)/sum(sa):.4f}  mse ×{sum(mb)/sum(ma):.4f}  "
              f"F ×{sum(fa)/sum(fb):.4f}  по мережах " + " ".join(f"{y/x:.4f}" for x, y in zip(sa, sb)))
