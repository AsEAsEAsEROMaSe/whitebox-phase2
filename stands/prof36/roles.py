import json, glob, collections
rows = []
for p in glob.glob('/home/trim/wblogs/prof36.json.*'): rows += json.load(open(p))
tot = sum(r[6] for r in rows); B = 2199023255552; N3 = 2 * 1024**3
R = {"молоді: хаб D21": [1769], "молоді: перенос ніг W@[A,P] (+новонароджене)": [1063, 1071, 1088, 1093],
     "старі: ноги з базису (tier1/tier2)": [1039, 1044, 1043], "старі: хаб через базис": [1758, 1763, 1773, 1764],
     "перебудова базису (range finder, QR, ротації)": [926, 931, 955, 966, 945, 980, 947, 1023, 1007, 995, 963, 970, 999, 1003, 1004],
     "коваріація: W C, C_pre": [1067, 1091, 1620, 880], "тонкі ноги D21 зворотний зв'язок, Z": [1074, 1076, 1120, 1376]}
lr = {l: k for k, ls in R.items() for l in ls}
agg = collections.Counter()
for site, fn, op, inS, ch, c, f in rows:
    agg[lr.get(site, "_grp35 (feed, D3, PZ)" if fn == "_grp35" else "решта: програма членів, nonlin, інше")] += f
for k, f in agg.most_common(): print("%5.1f%%  %.3e  %.4f B  %6.1f n³  | %s" % (100*f/tot, f, f/B, f/N3, k))
print("TOTAL %.4e = %.4f B" % (tot, tot/B))
