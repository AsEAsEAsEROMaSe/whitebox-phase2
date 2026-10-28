import json, time, urllib.request, sys
D = open("/home/trim/wb2lean/dossier_full.md").read()
Q = ("You now have everything we know. SYNTHESIZE something NEW: a representation of the propagated state, or a different computational object, "
     "that is not in any closed class and not a relabeling of the current chain. Be bold, combine fragments from different findings, exploit the measured facts "
     "(isotropy in neuron space, orthogonal commits, non-additivity 23-25%, flow ranks, zero conditional mean of the sea, leaders at 44% price with the same MSE). "
     "Give 5 candidates. For each: (1) the object and how it is stored; (2) the exact mechanism by which it carries information the current chain lacks OR carries the same information cheaper; "
     "(3) which measured fact would have killed it and why it survives that fact; (4) flop cost per layer; (5) the first cheap test on our stand (we can inject truth from a 1e9-sample bank into any channel). "
     "Rank them by expected value. Do not propose anything from the closed-class list.")
runs = [(1.0, "medium", 11), (1.3, "medium", 12), (1.6, "medium", 13), (1.0, "high", 14)]
out = open("/home/trim/wblogs/ideas8080/ideas_full.jsonl", "a")
for T, eff, s in runs:
    body = {"model": "gpt-oss-20b", "messages": [{"role": "system", "content": "You are a bold, inventive applied mathematician. Answer in English, terse, numbered. Never restate the dossier."}, {"role": "user", "content": D + "\n\n# TASK\n" + Q}],
            "temperature": T, "top_p": 1.0, "max_tokens": 40000, "seed": s, "reasoning_effort": eff}
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200)
        j = json.loads(r.read()); ch = j["choices"][0]; txt = ch["message"].get("content", ""); fin = ch.get("finish_reason"); rl = len(ch["message"].get("reasoning_content", "") or ""); us = j.get("usage", {})
    except Exception as e: txt = "ERR " + repr(e); fin = "err"; rl = 0; us = {}
    out.write(json.dumps({"T": T, "effort": eff, "seed": s, "t": round(time.time()-t0), "finish": fin, "reasoning_len": rl, "usage": us, "text": txt}, ensure_ascii=False) + "\n"); out.flush()
    print("T", T, eff, "%.0fs" % (time.time()-t0), fin, "reason", rl, "text", len(txt), us, flush=True)
print("DONE_FULL", flush=True)
