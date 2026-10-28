import json, time, urllib.request
D = open("/home/trim/wb2lean/dossier_full.md").read()
def ask(msgs, T=1.0, eff="medium", seed=31, mx=40000):
    body = {"model": "gpt-oss-20b", "messages": msgs, "temperature": T, "top_p": 1.0, "max_tokens": mx, "seed": seed, "reasoning_effort": eff}
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200)
    return json.loads(r.read())["choices"][0]["message"].get("content", "")
SYS = {"role": "system", "content": "You are a precise, inventive mathematician. Answer in English, terse."}
S3 = ("ABSTRACT OBSTACLE (exact). A deterministic recursion over 16 steps carries a state whose expensive part is a set of ~15 'sources', each a symmetric 3-tensor of size n^3 (n=1024) "
      "stored exactly as n rank-1 terms with three n x n factor matrices; every step applies the same dense random linear map W to all three factors (cost 3 n^3 per source per step) "
      "and then reads two slices of each tensor (the (j,j,j) diagonal and the (j,j,k) matrix) in the frame of the NEXT random map. Facts: the tensors are isotropic in the n-dimensional index space "
      "(no sparsity, no clusters, no low rank: rank 512 of 1024 already loses 26%); contributions of different sources to the final answer are mutually orthogonal and of equal importance "
      "(dropping any one source costs x10); the parts of a source read at different steps are mutually orthogonal and unpredictable from each other; projecting the factors on the subspace "
      "that reaches the output destroys the answer (x25) although that subspace holds 99.99% of the linear flow, because the slices are cubic forms in every neuron's own frame. "
      "A competitor computes the SAME answer to the same accuracy with 44% of the arithmetic. The question: what representation or contraction order can carry these n^3 objects through "
      "16 random dense maps and deliver their diagonal and (j,j,k) slices in each new frame at less than 3 n^3 per source per step, exactly or to 1e-4 relative accuracy?")
out = open("/home/trim/wblogs/ideas8080/compress2.jsonl", "a")
def step(name, prompt, T=1.0, eff="medium", seed=31):
    t0 = time.time(); txt = ask([SYS, {"role": "user", "content": prompt}], T=T, eff=eff, seed=seed)
    out.write(json.dumps({"stage": name, "t": round(time.time()-t0), "text": txt}, ensure_ascii=False) + "\n"); out.flush(); print(name, "%.0fs" % (time.time()-t0), len(txt), flush=True); return txt
s4 = step("S4b_analogies", S3 + "\n\n# TASK\nList 12 problems in other fields with exactly this structure (a multilinear object pushed through a chain of dense linear maps, where only certain slices/contractions in each new frame are needed) and the trick each field uses to avoid re-transporting the whole object: e.g. transporting the slices themselves, dual/adjoint contraction, Khatri-Rao or Kronecker identities, generating functions, symmetrization tricks, reading the slice through the map instead of mapping then slicing, precomputing products of the maps, amortizing across the 15 sources. For each: the identity, what it requires, the cost.", T=1.0)
s5 = step("S5b_map_back", D + "\n\n# " + S3 + "\n\n# TRICKS (listed by you earlier)\n" + s4 + "\n\n# TASK\nFor each trick: write the exact contraction it proposes for OUR objects (factors X,Y,M,P of size n x n, maps W_l, needed slices D3[j]=T[j,j,j], D21[j,k]=T[j,j,k] in the frame after W), count the flops per source per layer honestly (a dense n x n x n x n product is 2n^3), say whether the identity is exact, and whether it beats 3n^3. Reject honestly. End with: the best exact candidate (if any) with its flop count, the best approximate candidate with its error mechanism, and the one-line reason if nothing beats 3n^3.", T=0.7, eff="high")
print("DONE_COMPRESS2", flush=True)
