import json, sys, re
def load(f):
    d = json.load(open(f)); r = d['results']; pm = r.get('per_mlp', [])
    names = [m.get('mlp_name') for m in pm]
    mse = [m.get('final_layer_mse') for m in pm]; fl = [m.get('flops_used', m.get('effective_flops')) for m in pm]
    res = [m.get('residual_wall_time_s') for m in pm]
    return names, mse, fl, res, d.get('run_config', {}).get('n_mlps')
def peak(t):
    try: return int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', open(t).read()).group(1))/1e6
    except Exception: return float('nan')
b = load('/home/trim/wblogs/v35_th25.json')
print('base L5: n_mlps', b[4], 'names', b[0][:3])
for tag in sys.argv[1:]:
    try: x = load('/home/trim/wblogs/v35s_%s.json' % tag)
    except Exception as e: print(tag, 'нема', e); continue
    assert x[0] == b[0], 'інші мережі!'
    rm = sum(x[1])/sum(b[1]); rf = sum(x[2])/sum(b[2])
    print('%-6s MSE ×%.5f  флопи ×%.4f  бал ×%.4f  residual макс %.3f с (база %.3f)  RSS пік %.2f ГБ' % (tag, rm, rf, 1/(rm*rf), max(x[3]), max(b[3]), peak('/home/trim/wblogs/v35s_%s.time' % tag)))
