"""18.10.2026 a2: парне зведення v41corr проти бази (за замовчуванням fija = v39) на s0..s9: MSE, флопи, бал = MSE·флопи."""
import json, math, sys
base = sys.argv[1] if len(sys.argv) > 1 else 'fija'
for tag in sys.argv[2:] or ['v41c', 'v41c75']:
    rows = []
    for i in range(10):
        try:
            a = json.load(open('/home/trim/wblogs/v37_%s_s%d.json' % (base, i)))['results']['per_mlp'][0]
            b = json.load(open('/home/trim/wblogs/v37_%s_s%d.json' % (tag, i)))['results']['per_mlp'][0]
        except Exception:
            continue
        ka = [k for k in a if 'mse' in k.lower() and 'final' in k.lower()] or [k for k in a if 'mse' in k.lower()]
        k = ka[0]
        rows.append((a[k] / b[k], a['flops_used'] / b['flops_used'], a[k] * a['flops_used'] / (b[k] * b['flops_used'])))
    if not rows:
        print(tag, 'немає даних'); continue
    g = [math.exp(sum(math.log(r[j]) for r in rows) / len(rows)) for j in range(3)]
    print('%-7s проти %s, мереж %d: MSE ×%.4f · флопи ×%.4f · бал ×%.4f · кращих за балом %d/%d'
          % (tag, base, len(rows), g[0], g[1], g[2], sum(r[2] > 1 for r in rows), len(rows)))
