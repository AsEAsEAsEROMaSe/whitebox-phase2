import json, sys
for t in sys.argv[1:]:
    try: d = json.load(open('/home/trim/wblogs/v37_%s.json' % t))
    except Exception as e: print(t, "ERR", e); continue
    r = d.get('results', d)
    for p in r['per_mlp']:
        print("%-8s %-18s MSE %.6e  flops %.5e  util %.5f  adj %.5e  wall %.1f" % (t, p['mlp_name'], p['final_layer_mse'], p['flops_used'], p['flops_used']/2199023255552, p['adjusted_final_layer_score'], p['wall_time_s']))
