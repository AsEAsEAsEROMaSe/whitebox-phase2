import json, math
rows = []
for s in ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9']:
    tq, tb = ('w_q2_s4', 'w_b_s4') if s == 's4' else ('v9q_' + s, 'v9b_' + s)
    try:
        q = json.load(open('/home/trim/wblogs/v37_%s.json' % tq))['results']['per_mlp'][0]
        b = json.load(open('/home/trim/wblogs/v37_%s.json' % tb))['results']['per_mlp'][0]
    except Exception: print(s, 'нема'); continue
    r = b['adjusted_final_layer_score'] / q['adjusted_final_layer_score']
    rows.append(r)
    print("%-3s %-20s бал ×%.4f  MSE ×%.4f  флопи ×%.4f  стіна %.1f / %.1f" % (s, q['mlp_name'], r, b['final_layer_mse']/q['final_layer_mse'], q['flops_used']/b['flops_used'], q['wall_time_s'], b['wall_time_s']))
if rows: print("СЕРЕДНЄ ×%.4f (геом. ×%.4f), кращих %d/%d" % (sum(rows)/len(rows), math.exp(sum(map(math.log, rows))/len(rows)), sum(r > 1 for r in rows), len(rows)))
