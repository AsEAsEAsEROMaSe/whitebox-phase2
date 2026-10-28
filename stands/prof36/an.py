import json, glob, collections, sys
rows = []
for p in glob.glob('/home/trim/wblogs/prof36.json.*'): rows += json.load(open(p))
tot = sum(r[6] for r in rows); print("TOTAL %.4e flops = %.4f B" % (tot, tot / 2199023255552))
src = open('/home/trim/wb2lean/prof36/est36_prof.py').read().split('\n')
by = collections.defaultdict(lambda: [0, 0, set(), False])
for site, fn, op, inS, ch, c, f in rows:
    e = by[(site, fn)]; e[0] += c; e[1] += f; e[2].add(op); e[3] |= inS
cum = 0
for (site, fn), (c, f, ops, inS) in sorted(by.items(), key=lambda x: -x[1][1])[:int(sys.argv[1]) if len(sys.argv) > 1 else 40]:
    cum += f
    print("%5.1f%% %5.1f%% %-14s L%-5s n=%-5d %s %s | %s" % (100*f/tot, 100*cum/tot, fn, site, c, "S" if inS else " ", ",".join(sorted(ops))[:30], src[site-1].strip()[:110] if site else ""))
print("--- по функціях")
byf = collections.Counter()
for site, fn, op, inS, ch, c, f in rows: byf[fn] += f
for fn, f in byf.most_common(): print("%5.1f%% %s" % (100*f/tot, fn))
print("--- по операціях")
byo = collections.Counter()
for site, fn, op, inS, ch, c, f in rows: byo[op] += f
for op, f in byo.most_common(12): print("%5.1f%% %s" % (100*f/tot, op))
