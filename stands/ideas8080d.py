import json, time, urllib.request, sys
D = open("/home/trim/wb2lean/dossier.md").read()
Q = {"G_new_only": "You have the complete dossier. Propose up to 6 ideas that are NOT in any closed class and NOT equivalent to a measured dead end. For each: (a) the concrete object/representation, (b) why it escapes the specific measured fact that killed the nearest closed class (cite the number), (c) flop cost per layer, (d) expected MSE effect with confidence. Then name the single best bet. If you believe nothing new exists, say so and explain which measured fact is decisive.",
     "H_price_only": "Accuracy is fine; the gap to top-10 is price (F/B 0.234 vs 0.10-0.13 at similar MSE). Using the dossier, propose up to 6 ways to carry the SAME information (the 15 kappa_3 sources, full-rank young legs, n^3 transport per source per layer) at <= half the flops without touching accuracy, that are not equivalent to pre-summation across births, rank truncation, or output-flow projection (all measured). Give the exact contraction order and flop count. If impossible, prove it in two lines.",
     "I_sea_cheap": "The sea (fourth-order cumulant with four distinct indices) feeding kappa_3 births is the only missing information (ceiling x20). Its conditional mean given the pair state is zero (fresh W). Propose up to 6 ways to extract information about the sea for a SPECIFIC network below n^4 flops that do not reduce to 'kappa from covariance' (zero) or to transported own kurtosis (R^2 0.01-0.11). Think about which cheap quantity of the specific W correlates with the sea's effect on the output. Cite why it is not zero-mean given the pair state."}
out = open("/home/trim/wblogs/ideas8080/ideas_dossier_high.jsonl", "a")
for tag, q in Q.items():
    for s, eff in ((7, "high"),):
        body = {"model": "gpt-oss-20b", "messages": [{"role": "system", "content": "You are a sharp applied mathematician reviewing a research dossier. Answer in English, terse, numbered. Do not restate the dossier. Do not propose closed classes."}, {"role": "user", "content": D + "\n\nTask: " + q}],
                "temperature": 0.9, "top_p": 0.95, "max_tokens": 40000, "seed": s, "reasoning_effort": eff}
        t0 = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600)
            j = json.loads(r.read()); ch = j["choices"][0]; txt = ch["message"].get("content", ""); fin = ch.get("finish_reason"); rl = len(ch["message"].get("reasoning_content", "") or "")
        except Exception as e: txt = "ERR " + repr(e); fin = "err"; rl = 0
        out.write(json.dumps({"tag": tag, "seed": s, "effort": eff, "t": round(time.time()-t0), "finish": fin, "reasoning_len": rl, "text": txt}, ensure_ascii=False) + "\n"); out.flush()
        print(tag, s, eff, "%.0fs" % (time.time()-t0), fin, "reason", rl, "text", len(txt), flush=True)
print("DONE_DOSSIER_HIGH", flush=True)
