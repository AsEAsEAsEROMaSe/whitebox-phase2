import json, time, urllib.request, sys
sys.path.insert(0, "/home/trim/wb2lean")
src = open("/home/trim/wb2lean/ideas8080.py").read(); CTX = src.split('CTX = """')[1].split('"""')[0]
ANG = {"E_flop_arithmetic": "Propose 8 legitimate algorithmic tricks (not accounting games) that reduce tariffed flops of the described pipeline: structured or batched contractions, reuse across sources, sketching with provable error, symmetric structure, precomputation reuse across layers, reordering of contractions. For each give the exact saving factor and the error introduced.",
       "D_randomness_structure": "The weights are random He-init matrices, independent across layers. Propose 8 ways to exploit this randomness itself (self-averaging, typicality, free probability, random matrix theory, cavity tricks) to shortcut the per-network computation while keeping per-network accuracy at the 1e-8 MSE level. Be concrete.",
       "F_leaders_guess": "Competitors reach the same MSE at 44% of our flops. Guess 8 concrete representations of the third-order state they might use that are cheaper than n x n factor matrices per source transported by W every layer. For each: data structure, transport cost per layer, what accuracy it keeps and why."}
out = open("/home/trim/wblogs/ideas8080/ideas.jsonl", "a")
for tag, q in ANG.items():
    for s in (3, 4):
        body = {"model": "gpt-oss-20b", "messages": [{"role": "system", "content": "You are a creative applied mathematician. Answer in English, terse, numbered, no preamble."}, {"role": "user", "content": CTX + "\n\nTask: " + q}],
                "temperature": 1.0, "top_p": 0.95, "max_tokens": 3500, "seed": s, "reasoning_effort": "low"}
        t0 = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800)
            j = json.loads(r.read()); ch = j["choices"][0]; txt = ch["message"].get("content", ""); fin = ch.get("finish_reason"); rl = len(ch["message"].get("reasoning_content", "") or "")
        except Exception as e: txt = "ERR " + repr(e); fin = "err"; rl = 0
        out.write(json.dumps({"tag": tag, "seed": s, "t": round(time.time()-t0), "finish": fin, "reasoning_len": rl, "text": txt}, ensure_ascii=False) + "\n"); out.flush()
        print(tag, s, "%.0fs" % (time.time()-t0), "finish", fin, "reason_len", rl, "text_len", len(txt), flush=True)
print("DONE_IDEAS_B", flush=True)
