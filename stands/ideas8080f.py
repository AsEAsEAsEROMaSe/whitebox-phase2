"""Схема стискання думок (юзер 14.10): досьє → 10 інваріантів → 3 речення → 1 формула → аналогії в інших галузях → розгортання назад."""
import json, time, urllib.request
D = open("/home/trim/wb2lean/dossier_full.md").read()
def ask(msgs, T=1.0, eff="medium", seed=21, mx=40000):
    body = {"model": "gpt-oss-20b", "messages": msgs, "temperature": T, "top_p": 1.0, "max_tokens": mx, "seed": seed, "reasoning_effort": eff}
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200)
    j = json.loads(r.read()); return j["choices"][0]["message"].get("content", "")
SYS = {"role": "system", "content": "You are a precise mathematician. Answer in English, terse. Never restate the dossier."}
out = open("/home/trim/wblogs/ideas8080/compress.jsonl", "a"); log = {}
def step(name, prompt, T=0.7, eff="medium", seed=21):
    t0 = time.time(); txt = ask([SYS, {"role": "user", "content": prompt}], T=T, eff=eff, seed=seed); log[name] = txt
    out.write(json.dumps({"stage": name, "t": round(time.time()-t0), "text": txt}, ensure_ascii=False) + "\n"); out.flush(); print(name, "%.0fs" % (time.time()-t0), len(txt), flush=True); return txt
s1 = step("S1_invariants", D + "\n\n# TASK\nCompress everything above into exactly 10 invariants: statements that any method beating the current one MUST satisfy, each with the measured number that forces it. No methods, no advice.")
s2 = step("S2_three_sentences", "Invariants of a research problem:\n" + s1 + "\n\n# TASK\nCompress these 10 invariants into 3 sentences that state the core obstacle as a mathematical fact. No jargon of the original domain if avoidable; state it as pure structure (what is known, what is unknown, what the cost of the unknown is).")
s3 = step("S3_one_formula", "Core obstacle:\n" + s2 + "\n\n# TASK\nCompress to ONE sentence or ONE formula: the abstract structure of the obstacle (an estimator, an information constraint, a cost scaling). Then give its name if this structure has a name in mathematics.")
s4 = step("S4_analogies", "Abstract obstacle:\n" + s3 + "\n\n# TASK\nForget the original domain. List 12 problems in OTHER fields (physics, statistics, signal processing, coding theory, numerical linear algebra, combinatorics, control, economics, biology) that have exactly this abstract structure, and for each: how that field circumvents it (the trick), and what the trick needs as input. Be concrete and terse.", T=1.0)
s5 = step("S5_map_back", D + "\n\n# ABSTRACT OBSTACLE (compressed by you earlier)\n" + s3 + "\n\n# TRICKS FROM OTHER FIELDS (listed by you earlier)\n" + s4 + "\n\n# TASK\nMap each trick back to our problem. For each: the concrete object (what is stored, shape), the operation per layer with flop count, which measured fact in the dossier it must survive and whether it does (cite the number), and the first test with the truth bank. Reject tricks honestly if a measured fact kills them. End with the 3 survivors ranked, or state that none survive and which invariant is decisive.", T=0.8, eff="high")
print("DONE_COMPRESS", flush=True)
