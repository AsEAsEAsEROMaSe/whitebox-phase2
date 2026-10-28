"""Локальна модель на :8080 як генератор неточних ідей (прохання юзера 14.10). П'ять ракурсів × 2 семпли, T=1.0."""
import json, sys, time, urllib.request
URL = "http://127.0.0.1:8080/v1/chat/completions"
CTX = """Problem (a public benchmark): an MLP with 16 layers, width n=1024, ReLU, no biases, weights i.i.d. N(0, 2/n) (He init), input z ~ N(0, I).
We must output the exact expectation E[h_16] (1024 numbers) of the final-layer activations over z, analytically (no sampling of the truth at all),
and we are scored by MSE x max(0.1, flops/2^41). Only tariffed array ops count (matmul n^3 = 2.1e9 flops, fp32 and fp16 cost the same, fp64 double).
Current best method: propagate cumulants layer by layer: mean (n), covariance (n x n, exact through linear layers), third-order cumulant tensor
stored as a sum of 'sources' (one per layer where ReLU created it), each source = sum_j X[:,j] (x) P[:,j] (x) Y[:,j] with n x n factor matrices,
transported by multiplying every factor by W (n^3 per source per layer). Fourth order: only per-neuron diagonal, pair part regenerated from covariance.
ReLU step: per-neuron Edgeworth/Hermite formulas, pair covariance via Mehler series in the correlation. Old sources (age>=4) compressed to a shared rank-384 basis, age>=7 rank 224.
Result: MSE 2.1e-8 at 23% of the flop budget. Competitors reach MSE 1.2e-8 at 10-13% of the budget. Our MSE floor with exact pair cumulants would be 1e-9 but no cheap carrier exists.
Measured dead ends (do not repeat): more Mehler/Hermite terms (0.1% effect); higher-order Edgeworth from the same inputs; truncating young sources (rank 512 already x1.26 worse);
Monte Carlo with the budget (38% error); projecting sources onto the output flow subspace (x25 worse); per-network scalar knobs; using only covariance diagonal."""
ANGLES = {
 "A_cheaper_third_order": "Propose 8 concrete, unconventional ways to represent or transport the third-order state far cheaper than n^3 per source per layer, keeping accuracy. Be specific about data structures and operation counts. Wild ideas welcome, mark your confidence.",
 "B_other_formulations": "Forget cumulant propagation. Propose 8 entirely different mathematical formulations to compute E[h_16] for this random ReLU network with flop cost below 2e11 (e.g. operator/transfer methods, generating functions, path integrals, dual representations, exact identities for ReLU of Gaussians, recursion in a different variable). For each: the key identity, what is exact, what is approximated, estimated flops.",
 "C_fourth_order_cheap": "The missing accuracy is the fourth-order cumulant 'sea' (n^4 tensor) that feeds the birth of third-order cumulants at every ReLU. Propose 8 ways to obtain its effect on the per-neuron and pair statistics with n^3 or less flops, exploiting that weights are i.i.d. Gaussian and fresh at every layer. Give formulas.",
 "D_randomness_structure": "The weights are random He-init matrices, independent across layers. Propose 8 ways to exploit this randomness itself (self-averaging, typicality, free probability, random matrix theory, replica/cavity tricks, annealed vs quenched) to shortcut the per-network computation while keeping per-network accuracy at the 1e-8 MSE level. Be concrete.",
 "E_flop_arithmetic": "Propose 8 legitimate algorithmic tricks (not accounting games) that reduce tariffed flops of the described pipeline: structured or batched contractions, reuse across sources, sketching with provable error, symmetric or low-displacement-rank structure, precomputation reuse across layers, reordering of contractions. For each give the exact saving factor and the error introduced.",
}
out = open("/home/trim/wblogs/ideas8080/ideas.jsonl", "a")
for tag, q in ANGLES.items():
    for s in (1, 2):
        body = {"model": "gpt-oss-20b", "messages": [{"role": "system", "content": "You are a creative applied mathematician. Answer in English, terse, numbered, no preamble."}, {"role": "user", "content": CTX + "\n\nTask: " + q}],
                "temperature": 1.0, "top_p": 0.95, "max_tokens": 2200, "seed": s, "reasoning_effort": "low"}
        t0 = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800)
            j = json.loads(r.read()); txt = j["choices"][0]["message"].get("content", "")
        except Exception as e: txt = "ERR " + repr(e)
        out.write(json.dumps({"tag": tag, "seed": s, "t": round(time.time()-t0), "text": txt}, ensure_ascii=False) + "\n"); out.flush()
        print(tag, s, "%.0fs" % (time.time()-t0), len(txt), flush=True)
print("DONE_IDEAS", flush=True)
