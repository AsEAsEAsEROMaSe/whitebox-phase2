import json, glob, math, re
for cfg in ("l12a6","l12a5","l10a6"):
    rs=[]
    for s in ("s0","s1","s2","s3","s4","s5","s6","s7","s8","s9"):
        try:
            q=json.load(open('/home/trim/wblogs/v37_age_%s_%s.json'%(cfg,s)))['results']['per_mlp'][0]; b=json.load(open(('/home/trim/wblogs/v37_w_q2_s4.json' if s=='s4' else '/home/trim/wblogs/v37_v9q_%s.json'%s)))['results']['per_mlp'][0]
        except Exception: continue
        r=b['adjusted_final_layer_score']/q['adjusted_final_layer_score']; rs.append(r)
        print("%s %s %-17s бал ×%.4f MSE ×%.4f флопи ×%.4f"%(cfg,s,q['mlp_name'],r,b['final_layer_mse']/q['final_layer_mse'],q['flops_used']/b['flops_used']))
    if rs: print("  %s геом. ×%.4f (%d мереж)"%(cfg,math.exp(sum(map(math.log,rs))/len(rs)),len(rs)))
