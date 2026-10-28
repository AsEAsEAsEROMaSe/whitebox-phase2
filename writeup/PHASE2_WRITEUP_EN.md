# The cumulant chain as a balanced system of errors: what we measured on 1024×16, what it cannot see, and why exact pieces break it

*White-Box ARC Estimation Challenge, Phase 2 · team `trim_qewas` · submissions **337654** (2.866e-9) and **337568** (2.881e-9), the two best by score, selected by the default rule (no manual nomination)*

Version of 27.10.2026. The Ukrainian version is the authored original; this English version is translated from it.

---

## Abstract

We estimate `E[ReLU(W₁₅ᵀh₁₄)]` for random bias-free ReLU networks of size 1024×16 with a deterministic K=3 cumulant chain, a regenerated fourth cumulant and a learned terminal corrector. Our best score of 2.866e-9 (MSE 1.897e-8 at 15.12% of budget; the score is 395 times better than Monte Carlo at full budget, the MSE 60 times better) is built on public code from other participants: the s32refire engine (koushik_rudra) on ALG058 (TristanMiano), the V29 line of 504aldo and e30_a, all on the ARC reference; our second-best line, 337116, is built on the Susupro #336537 chain. Our contribution to these two scores is a retrained LT2 corrector with a fast layer (×1.05 in raw MSE) and a fixed transported basis for old sources, FIJA (×1.03); on our own line on top of V29 we added FIJA (×1.04), a ridge correction of the mean on the "fresh step" (×1.095), ranks of the thin legs (×1.036) and exact flop savings (×1.025), ×1.32 in total over two days, but that line stayed at 3.747e-9. This is not a new method, and we say so.

The contribution we ask the jury to assess this report for is a different one: **a map of what exactly the K=3 chain does not see, measured with oracles on the official networks, and a law that explains why almost every "more exact" correction to it does harm.**

**The mechanistic picture the rest follows from.** The report reads as one causal story about the network itself, not as a list of tricks. The first two steps are ideas of our team lead from Phase 1; the rest was measured in Phase 2.

- *The critical point.* A bias-free He-ReLU network sits exactly on the boundary: 50.003% of weights are negative, 50.2% of neurons are off on every layer; a bias of 0.15 makes the task six times easier. The output mean is therefore a small difference of large quantities: in the mirror decomposition `h = λ + m`, σ(m) ≈ 280 against 0.32 in h, and the tropical parts grow to 2e23 at |E[h]| 0.97 (§4.7), and the decompositions of the form "linear part plus correction" that we tried (mirror, leaky slider, tropical) lose to the chain by orders of magnitude.
- *The path tree.* The output is a sum over paths through the gates; the cumulants κ3, κ4 describe how paths couple on shared indices. The chain carries the diagrams whose indices pair up (pair statistics of κ3 and κ4); we wrote the κ4 birth diagrams in TT form and sealed them to 1e-15 (§4.5).
- *The sea.* Diagrams with all-distinct indices have zero mean over a fresh W (§4.1), so over the ensemble of W the chain is the conditional expectation of the truth given its state; for a fixed network the remaining error is deterministic, but carries no sign signal in pair quantities.
- *Where the error is born.* Every step adds an independent error, transport neither damps nor amplifies it (≈1), and 64% is born in the last four layers (§4.3).
- *Why exact pieces break it.* The chain holds an equilibrium between truncation error and calibrations; the output is 10 times more sensitive to four common modes of the state than to the true residual (§4.2, §4.4), so corrections can only be made consistently, all channels together.
- *Width.* At 1024, `W²` averages over 1024 inputs, and the `1/√n` fluctuations that carry the benefit of exact K=4 at width 128 are 2.8 times smaller (§4.5).

We did not derive the cumulant chain from this picture, we inherited it; but read this way, it is the computation the picture points to: it computes what has a nonzero conditional mean. We measured where and why it stops.

Six results, each with a number:

1. **The sea has zero conditional mean.** The contribution of distinct-index quadruples `Σ W_ji W_ki W_li W_mi κ_jklm` has zero expectation over a fresh `W` given any pair state, so over the ensemble of W the chain that discards it is the conditional expectation of the truth given its state: Bayes-optimal within the class of functions of the pair state. This explains why the sign of the error is unpredictable from the weights (R² 0.015) and why all leaders sit in one MSE band of 1.1…1.4e-8, and predicts that going below that band needs an n⁴ contraction or samples (§4.1).

2. **Balanced errors.** Substituting the exact value into one channel of the tuned chain makes the output worse, or neutral, or slightly better only when shrunk: the exact (2,2) slice of κ4 ×0.904; the exact (3,1) in the correct orientation ×1.009, and shrunk by a factor 0.75 ×1.055 (3 networks, the gain comes from one; the first version of this oracle had the (3,1) transposed and gave ×0.804, §6); correcting the underestimated K22 birth to the truth ×0.21; a scalar on D3 has its optimum at exactly 1.00 with a drop of ×0.5 at ±10%; a factor 0.75 on all κ4 sources in exact K=4 beats exact K=4 by 2.6…6.0 times; on the grader itself, exact transport of the (4) and (2,2) slices made the MSE worse monotonically with the blend, ×0.85…0.47 (§1). The shape of the exact slice at the ansatz level gives ×0.987, i.e. **at width 1024 the shape of a slice does not matter, only its level does, and the level is balanced against λ and the calibrations** (§4.2). The same mechanism was independently described by dogus_ozel (Ward defect, 18281), LuDoe (write-up for 335513 in thread 18251: imposing the true fourth-order level hurts by +29%) and megan_park (18282: λ 2…5 times larger than the measured one).

3. **Where the error lives.** The true `(m, C)` after ReLU on layer 14 give ×9.2 (range 7.8…12.6 over 18 networks), `m` alone ×5.0, `C` alone ×1.10; the curve in λ is quadratic. The true per-neuron c3/c4 of pre-activations on all layers give ×1.9; together with the true `C` ×4.5 and ×6.4 (product of solos 2.5 and 2.7): the channels **multiply** and have to be treated together. The rest of c4 that the chain does not see is the four-index abcd part (corr with g4row ≈ 0); κ5 gives ×1.00 (§4.3).

4. **Common modes ×10.** Derivatives of the output along 12 feature directions of the D3/g4 state are 2.5…6 times larger than the whole output error, while the true c3 residual is balanced in these modes by a factor of 10: the coherent part (93% of variance, R² 0.93) harms, and only the 7% incoherent part, which the features do not carry, helps. Any learned correction to the state from chain features (ridge, MLP, row contractions, per neuron or per output) gives ≤×1.02 or breaks (predicted c3 on layers 12…14 ×0.61…0.68 against the truth ×1.11…1.15). The effective rank of the sensitivity is 4 (§4.4).

5. **Anatomy of κ4.** K=4 diagrams cancel in pairs (star ≈ −(path + three2), cos −0.89…−0.95), on the exact stand any incomplete subset of the readouts is worse than no κ4 at all, atoms of young births do not compress (rank 16 is worse than K=3), the old state compresses to n/4 and acts as a regulariser. At 128×16, 70…81% of the value of K=4 comes from exact transport of the repeated (4) and (2,2) slices of κ4(h) over one or two layers with decay 0.5; **at 1024 this does not carry over**: the regeneration ansatz `(dG_i + dG_j)/6` already equals the exact transport `W²K22W²ᵀ` with corr 0.989…0.999 and scale 1.00…1.02, because `W²` averages 1024 inputs, and the `1/√n` fluctuations at 128 are 2.8 times larger. The TT readout formulas of all four birth slices (path, star, c⊗κ3, cross terms) are sealed to 1e-15 (§4.5).

6. **Cost.** Flop profile by role: young-source hub 23.6%, young-leg transport 22.2%, basis rebuild 15.7%, old sources through the basis 18.9%; the D21 slice must be dense (rank truncation ×9.5), basis entry age 4 is optimal from both sides, and there is no cheap place for deliberate error along the "age × layer" axis (1% of flops ≈ 4…10% of MSE). The Phase 2 residual gate acts per client operation (≈16 µs, 0.39 s of 0.40 s for 24 695 operations), not per flop: the sixth Strassen level pays off in flops and is zeroed by time (§4.6).

The report also carries the Phase 1 results these measurements stand on (§5), a list of our own errors with their mechanisms (§6), and an honest list of what other participants found and we did not: spherical input at zero flops, extraction of the scale mode, fast matrix multiplication, the oracle bias correction (§7).

---

## 0. Disclosure

All experiments, stands and numbers were produced in interactive sessions with an LLM (Claude, Anthropic; Claude Code) directed by the participant, who chose directions, approved runs and submissions, read the numbers and decided what to keep. No claim is an unverified statement of the model: every number comes from a log that the text refers to.

**Origin of the submission code.** Submission 337654 (and 337568, 337659) is the s32refire engine (koushik_rudra, github acco-cyber/arc-whitebox-estimation-2026, MIT; ALG058-V001 by TristanMiano with birth slicing and FUSE) plus our fast layer of the LT2 corrector (architecture by Susupro, github logxio/fourth-cumulant, MIT; weights retrained by us on 1000…1992 corpus networks). Submission 337116 is the Susupro #336537 chain (MIT) plus our fixed FIJA basis. Our own line (v29_504aldo → v34 → v46, submission 336991 = 3.747e-9) is built on V29 by 504aldo (github 504aldo/whest-p2-cumulant-k3, MIT), which in turn is built on the ARC reference `mlp_cumulant_propagation` (Wu, Lecomte, Winer, Robinson, Hilton, Christiano, arXiv:2605.05179). Our own third-order recursion written from scratch (329794, 3.744e-8) is described in §2.1. We ask the jury to weigh the measurements of §3…§4, not the base.

**What is ours, what is borrowed, what is shared.** Ours: the oracle protocol on the official networks and all numbers of §3…§4; FIJA; the ridge on the fresh step; the thin-leg ranks; the exact savings; the fast LT2 layer and its retraining; the TT algebra of the κ4 slices; the sea law; the common-mode law; the width law. Borrowed: the chains, the LT2 corrector as an architecture, the λ law of κ4 regeneration, Strassen-Winograd in flopscope operations. Shared with others, found independently: the balance of errors (dogus_ozel, megan_park, LuDoe), survival of the error along the mean (megan_park), "all or nothing" for κ4 (abhimanyu_singh). Reproduced after their reports: the correct orientation of the (3,1) slice (kai_leal_gonzalez, Nurlanggs).

---

## 1. Numbers first

| what | score | MSE | budget | where |
|---|---|---|---|---|
| **337654** midA: s32refire + LT2 mid, further-trained weights | **2.866e-9** | 1.897e-8 | 15.12% | grader, 50 public; IQR 2.61…3.06e-9 |
| **337568**: the same, Susupro weights retrained on 1000 networks | 2.881e-9 | 1.906e-8 | 15.12% | grader |
| 337659: LT2 full instead of mid | 2.887e-9 | 1.902e-8 | 15.18% | grader |
| 337116: Susupro chain #336537 + our FIJA | 3.120e-9 | 1.913e-8 | 16.32% | grader |
| 336991 v46: our line on V29 (FIJA, step_F87, R_FB 4/8, exact savings) | 3.747e-9 | 2.050e-8 | 18.29% | grader |
| 336478 v36 (our line before track E) | 4.929e-9 | 2.152e-8 | 22.9% | grader |
| 331514 V29 504aldo as is (base of the September measurements) | 5.418e-9 | 2.133e-8 | 25.4% | grader |
| 329794 our K=3 recursion from scratch (contraction to rank 64) | 3.744e-8 | 2.725e-7 | 13.8% | grader |
| Phase 1, 326725 (two-moment closure + MC) | 1.855e-7 | 1.849e-6 | 10.0% | grader |
| Monte Carlo at full budget (grader panel) | | 1.13e-6 | 100% | |
| top of the board 24.10.2026 (marius_binner, oliver_chen) | 1.3e-9 | 1.10e-8 | ≈12% | board |
| us on the board 24.10.2026 | #39 | | | 27 submissions |

Seven blind submissions in the deadline night (21.10 23:30…23:57 UTC) added κ4 pieces to the record; all are graded:

| submission | what was added to the record | score | MSE |
|---|---|---|---|
| 337665 | star term in g4row, λ 0.5 | 3.155e-9 | 1.903e-8 |
| 337662 | TT readout of (4), λ 0.5 | 3.434e-9 | 1.903e-8 |
| 337678 | exact coherent transport of the (2,2) slice, blend 0.5 | 3.452e-9 | 1.913e-8 |
| 337692 / 337688 / 337689 / 337681 | exact coherent transport of both (4) and (2,2), blend 0.3 / 0.5 / 0.7 / 1.0 | 4.034 / 4.856 / 5.834 / 7.340e-9 | 2.235 / 2.691 / 3.233 / 4.068e-8 |

The first three left the MSE unchanged and only raised the cost, as predicted (337662: +2 matmuls per layer = +19.5%, measured +19%). The last four are an oracle scan on the grader itself: adding the exactly transported (4) slice makes the MSE monotonically worse with the blend, ×0.85 / 0.70 / 0.59 / 0.47 at 0.3 / 0.5 / 0.7 / 1.0. Before submitting we predicted MSE ×1.00…1.025 from the 1024 stand, which tested the (2,2) slice only, and the 128 stand ranked blend 0.5 best; both missed. This is the balanced-errors law of §4.2 on the grader, and the width law of §4.5 explains why the 128 stand could not see it.

Units: B = the flop budget of one network (2⁴¹ ≈ 2.2e12); u = one dense 1024×1024 by 1024×1024 matmul, 2n³ ≈ 2.15e9 flops, so the budget is 1024 u. Convention: "×k" next to an MSE or a score means an improvement by a factor of k (>1 is better) unless stated otherwise; in §7 the numbers of other participants are given in their convention (an MSE factor, lower is better).

Comparison with the board: our gap of ×2.2 to the top consists of MSE ×0.58 and compute ×0.8. Per network the leaders are uniformly better (corr of log MSE us/Luna 0.68, ratio ×1.50 at sd log 0.084): consistent with the same class with a more exact step rather than a different space.

---

## 2. What we built

### 2.1 Our own K=3 recursion (329794) and why it lost

A deterministic recursion: each ReLU layer is closed by exact Gaussian moment formulas plus a third-order pair term: the slice `κ21[i,j] = κ(p_i, p_i, p_j)` of pre-activations enters the mean and covariance steps through the Hermite coefficients `c_k` of ReLU. `κ21` is carried as a sum of hub sources (hub leg = identity at birth, leg-arm = `c₁·ρ`, legs transported through `W`), and the slice is read as `2(H∘A)ᵀA + (A∘A)ᵀH`. To fit the budget, sources were contracted into a rank-64 kernel after 4 transports. Cost 0.138 B, fp32, 3 Strassen levels. Grader 2.725e-7.

The contraction is exactly where the loss is: the same construction without contraction (the 504aldo chain, D21 exact to ~2%) is 12 times more accurate. Our contraction sits below the rank cliff that 504aldo measured independently (rank 3n/8 at age 4). Self-consistent feedback of `κ21` into its own birth (SourceF21: `E[He₃He₁He₁] ∋ 3κ21_ij ρ_ik`) improved the relative error on the official networks from 0.0529% to 0.0389% (×1.85 in MSE) without the truth, but then the sources do not contract (rank 128/256 destroys the fine part at 97.5% of energy), and at 0.369 B the score does not move (5.7e-8 against 3.7e-8). **The fine high-rank part of κ21 is both where the accuracy and where the cost lives.**

### 2.2 The V29 line: what exactly we added (18…19.10.2026)

Numbers on 4…10 networks of the public mini split (s0…s9, count in brackets), score = MSE × max(0.1, C/B); the grader forecast from these networks matched the result to 0.25…0.6%.

| step | mechanism | score × | MSE × | flops × |
|---|---|---|---|---|
| v37 dead rows (α<−2.5 of the previous layer) in the transport of young legs and in the hub | a neuron with h≡0 has f′≈f″≈0; cutting it from κ3 is exact by construction (0.0000 of the norm) | 1.040 (6/6) | 0.991…1.008 | 1.029…1.043 |
| v38 young sources live to age 6 on layers ≥12 | birth map of the error: 64% on layers 12…15 | 1.014 (9/10) | 1.011 | 1.003 |
| v39 **FIJA**: fixed transported basis of old sources (range finder only at the first union, afterwards `Q ← W·diag(w₁)·Q` without re-orthonormalisation; a new old source = least-squares projection) | the basis rebuild cost 15.7% of flops per layer | 1.041 (10/10) | 0.962 | 1.081 |
| R_OLD 416 | | 1.014 (6/10) | 1.030 | 0.984 |
| v40h old level only in rows with α≤3 | | 1.013 (3/4) | 1.004 | 1.009 |
| **step_F87**: ridge table of 87 e30_a features, target "fresh step" `step_l = y_l − (y_{l−1}@W_l)·Φ_l`, c=1.0 | a linear emulation of the side stream predicted ×1.085, reality ×1.095 | 1.095 (10/10) | 1.095 | 0.9997 |
| v43 = all of the above | | 1.171 vs v38 | | |
| v45 thin-leg ranks R_FB 16→4, R_RES 16→8 (as in e30_a) | the gain comes from R_FB, R_RES is neutral; effects multiply | 1.036 (10/10) | 0.965 | 1.074 |
| v46 exact savings: ueye (U=[I;0] under FIJA, 1.26 u), triangular solve by bisection (0.187→0.094 u), nested level F2 = F_top + R11⁻¹R12·F_bot, block sums instead of (g,T)@(T,n²) | the flopscope solve tariff is 2n³/3 + 2n²·nrhs and ignores triangularity | 1.025 (10/10) | 1.0001 | 1.025 |
| Strassen pool arenas + ROWALIGN 32 | RSS 17→5 GB, wall 100→70 s; in steady state the score is ×1.00 ± 0.015 | 1.00 | | |

What did not work on this line, with numbers: v44 covariance on live inputs only (×0.993, flops −0.5%, because layers 0…7 have >896 live inputs); v40 dropping dead rows from the old level (×0.949…0.969, 0/4: the dead at 2.5 cannot be dropped, consistent with e30_a H1); the e30_a λ table (×1.0009); a joint fit of all layer tables to the output error (fitj ⛔) and features from the weights (fitx ≈0); R_OLD 448/384, R_OLD2 256; Strassen L6.

### 2.3 The line on the Susupro chain and s32refire (19…22.10.2026)

The Susupro chain as is gave ×1.26 over our v47 on our 4 q4 networks; our FIJA on top added ×1.03 (337116). Their LT2 corrector **does not port** into our chain (×2.30…2.56 worse): the corrector is tied to the state conventions of its own chain (`d21_rowsum` out of range, |z| 3.9…6.8 relative to training). Our pieces have to be ported into their chain, not the other way round.

The LT2 corrector (54k parameters, input of 51 state features per neuron per layer, seq training) gives 0.8749 of raw MSE on sb2 (0.8729 on the Susupro chain, measured by us on 100 mini networks); the factor is almost invariant to the engine (raw MSE as reported: pranay212 0.890 on their own K=3 chain, Susupro 0.887 on 3048 networks; ours 0.875 on sb2), but only 0.952 on ALG058, where its own nn_asset already takes part of the same residual (switching nn_asset off gives ×0.965 in score). The corrector sits at the ceiling of its features: further training (e2e, 363 seen networks, 1660 unseen), a wider network, output multipliers (×0.999…1.001), a multiplier on the correction (c\* 1.0062), a GP over graph aggregations (val −0.17%, the "foreign W" control the same), an MLP on the ridge residual (×0.990 vs ridge) all give zero. Doubling the data from 1000 to 1992 networks gave ×1.005 in score (337654 vs 337568: ×1.0052, exactly as a cross-check on 6 networks predicted). **The residual after LT2 is not predictable from chain quantities by any of the means we tried; gains come only from new information in the state.**

The fast LT2 layer (ours): fused matmuls `W@[prev8|zo|h|u]`, a single log1p transform on `[X0|Xv]`, copysign; the output equals the reference within fp32 (|Δ| 4.7e-10), operations 24 962 → 24 695. On the grader each operation costs ≈16 µs of residual against a 0.4 s gate, so this is insurance against zeroing, not score.

Code review of 337568 across all mathematical operations: the Wick table = exact `∂^p E[ReLU^q]`; the Edgeworth mean κ3/6, κ4/24, κ3²/72, κ3κ4/144; moments to cumulants via connected partitions; Strassen-Winograd (combos, leaf, assembly); FIJA: cond(Q) 9.1 (layer 6) → 9713 (layer 14), ×2.2 per layer, but fp64 re-orthonormalisation gives ×0.998 (fp32 does no harm). No mathematical errors were found; the chain sits on the trade-off line (all 12 public knobs within 0.1…5% of the optimum, all worse).

---

## 3. Instruments

**3.1 The oracle ladder on the official networks.** Replace one internal quantity of the chain with its true value (from MC with 2²²…2²⁶ samples or from the keenanpepper bank with N=1e8/1e9), let the rest run, measure the output. The result is an upper bound for every idea whose effect passes through that quantity. Seals: self-substitution bit for bit; layer 0 (where the input is exactly Gaussian) equals the noise of the truth; two halves of the MC give the same answer. This is the same protocol as in our Phase 1 report (18182), now on the K=3 chain and at 1024.

**3.2 Cost profiles.** A hook on `BudgetContext._charge_op` that attributes every operation record to a line of the estimator and a layer: the sum over lines equals `flops_used` bit for bit. It measured the profile by role (§4.6) and the fact that the local single-network cost is inflated ×1.07…1.08 (the first predict of a process picks a lower Strassen level and touches new pools): versions with different one-off costs must be compared only in a multi-network process.

**3.3 Seals of exact symmetries.** `z → −z ≡ W₀ → −W₀` holds bit for bit both in the lean chain and in production with fnp and Strassen; so does neuron permutation. This catches slice orientation errors (K21[a,b] = κ(a,a,b): the transposed one gives corr 0.61 instead of 0.998).

**3.4 Exact K=4 stand at 128×16** (Susupro engine, seal p=1 1.22356e-5 = README 1.22353e-5): each birth group as a separate object, per-slice readouts, age and lifetime of groups, rank of C in births, multipliers. §4.5 was obtained on it before checking at 1024.

**3.5 GPU shim flopscope→cupy** (3.8…4.2 s per network against 35 s on a CPU core, output within fp32): dumps of the chain state on 4096 corpus networks for training the corrector and for hooked oracles.

**3.6 What our instruments lacked.** Oracles built from a noisy MC truth are biased; michael_anderson7 (18272) showed that the exact σ² reads 0.776 without correction and 0.511 with a noise-ceiling correction. Our ceilings (×1.25 c4, ×1.9 c3/c4, ×4.5…6.4 synergy, ×9.2 pair state) are given without this correction, i.e. they are **underestimated**; the signs and the order of the verdicts do not change, the magnitudes may be larger.

---

## 4. What the protocol found

### 4.1 The sea has zero conditional mean, so the K=3 chain is Bayes-optimal in its class

The chain state on layer l: all pair statistics (m, C, κ3 slices, pair slices of κ4). The sea = `Σ_{j,k,l,m distinct} W_ji W_ki W_li W_mi κ_jklm(h_l)`. `W_{l+1}` is independent of `κ(h_l)` (fresh He initialisation), and `E[W_ji W_ki W_li W_mi] = 0` for distinct indices, so `E[sea | pair state] = 0`, while `Var[sea | state]` is finite and computable from pair data. Therefore the chain that discards the sea is the conditional expectation of the truth given its state. The same holds for the (1,1,1) sea of κ3 at birth.

What this explains (all measured before the synthesis):

| fact | number |
|---|---|
| the sign of the birth error is unpredictable from the weights | a feature grammar from W alone explains R² 0.015; the magnitude is predictable at R² 0.5…0.7 |
| the residual of the κ4 diagonal is not a function of n² objects of the previous layer | pair transport T22/T31/T4 with true pairs R² ≤0.13 = the shuffled control |
| the sign of the output mean error correlates with no neuron feature | corr(e_j, t) −0.05±0.12, ‖w_j‖ 0.00±0.04, Σw_j −0.02±0.09 across networks |
| the error is exchangeable across neurons | crowns by t = random groups (×1.165 against ×1.188 at k=16) |
| commits of κ3 sources are pairwise orthogonal | 15 sources × ages: mean |cos| 0.07, R² of the remaining deliveries of one source from the first ≤0.03 |
| all leaders in one MSE band | 1.1…1.4e-8 at different util |

Testable predictions: (a) the variance of the κ4 residual across neurons is predictable from the pair state (yes, R² of the magnitude 0.5…0.7); (b) no function of the pair state will give a correlation with the sign of the residual above noise (all probes of 12…22.10 agree); (c) any participant with MSE below ~1.1e-8 carries an n⁴ contraction or samples.

A refinement that matters: the sea has zero mean **over the ensemble of W**, but W is known, so for a specific network the sea is a deterministic quantity. Its tree diagrams (star `a₃·a₁³·ρρρ`, chain `a₁a₂a₂a₁·ρρρ`) are computable at n³ from the chain state (corr with the true sea 0.9999), but they catch only part of the per-neuron variation of γ4: output ×1.043…1.050 on networks 0/1 against a c4 ceiling of ×1.26, a constant β 0.5 gives ×1.014 on 8 networks. The rest of c4 = aabc (three distinct indices, 70%) + abcd (four distinct, 25%), mutually orthogonal; trees with three edges do not explain abcd (corr 0.00…0.12 on two networks), because the median t of pre-activations in Phase 2 is ≈0 (−0.08…0.26), and the premise "c₁≈1, higher c small" is false here.

### 4.2 Balanced errors: exactness in one channel does harm

This is the most repeated measurement of both phases, now with a mechanism.

| substitution in the s32 / sb2 chain (1024; 2…6 networks, MC 2²²…2²⁴) | MSE × (>1 is better) |
|---|---|
| exact (2,2) of pre-activation κ4 in wk4m | 0.904 (MC halves 0.899 / 0.885) |
| exact (2,2) ×0.5 / ×1.25 / ×1.6 / ×2 | 0.22 / 0.40 / 0.13 / 0.06 |
| shape of the exact (2,2) at the ansatz level | 0.987 |
| shape of the ansatz at the exact level | 0.894 |
| exact (3,1), correct orientation `k31[a,b] = κ(a,a,a,b)ᵀ`: ×1 / ×0.75 / ×0.5 (networks 0, 2, 1) | 1.009 (1.176 / 0.931 / 0.938) / 1.055 (1.172 / 1.003 / 1.000) / 1.041 |
| symmetric part of the exact (3,1) | 1.034 |
| exact (3,1) in transposed orientation (first version of the oracle, §6): ×1 / ×0.5 / ×0.25 | 0.804 / 0.932 / 0.924 |
| both: exact (2,2) and correct (3,1) ×1 / ×0.75 (networks 0, 2, 1) | 0.977 (1.042 / 0.873 / 1.024) / 0.996 (1.002 / 0.923 / 1.068) |
| both, with transposed (3,1) (first version) | 0.773 |
| K22 birth of the chain × (truth/chain): fully / half / layer 12 only | 0.21 / 0.65 / 0.95 |
| the same with LAM_SCALE 0.5 / 0 | 0.16…0.22 |
| scalar on D3 (chain c3) 0.9 / 1.1 / 1.2 | 0.41…0.49 / 0.40…0.60 / 0.15…0.23 |
| scalar on g4row 0.8 / 1.2 | 0.73…0.85 / 0.68…0.79 |
| D3 := diag + pair(K21), realisable, corr 0.957 with c3 | 0.06…0.08 |
| g4row + β·aabb(K22), β −0.1 / +0.1 / +0.3 | 0.96 / 0.87 / 0.48 (optimum β≈0) |
| true c3 minus 0.34·pair(K21) / ±0.17·pair | 0.07…0.09 / 0.21…0.33 |
| incoherent part only (c3 − D3) | 0.09…0.11 |
| predicted c3 (R² 0.93 from 12 neuron statistics) on layers 12…14 | 0.61 / 0.64 / 0.68 against the truth 1.15 / 1.11 / 1.11 |
| predictor of the c3/c4 residual with corr ρ = 0.9 / 0.7 / 0.5 | 0.061 / 0.039 / 0.049 (seal ρ=1: 4.488) |

On the exact K=4 stand (128×16, 3 networks, K3a 5.10/6.27/5.10e-5, full K=4 1.22/3.13/1.60e-5): a factor c on all κ4 sources gives 0.5 → 7.7/8.7/9.9e-6, **0.75 → 4.61/5.23/5.46e-6**, 0.875 → 6.4/12.3/7.8e-6, i.e. a shrunk K=4 is 2.6…6.0 times more accurate than exact K=4. Source pruning makes it cheaper only by switching κ4 off (deep pruning = κ4=0: 2.50/3.75/2.97e-5).

Mechanism. A chain truncated at the third (or fourth) cumulant makes a systematic truncation error at every step; λ, METRIC_C, the tables and the corrector are tuned so that this error and the error of the slice levels cancel each other. Any correction that shifts one level breaks the cancellation. At 1024 the **shape** of a slice adds nothing (×0.987): width averages; only the level matters, and it is already balanced. The (3,1) slice is the exception that confirms the rule: in the correct orientation it is neutral, and its best value is again a shrunk one (×0.75), the Phase 1 law "the optimal correction is shrunk". This agrees with three independent reports: dogus_ozel (18281) measures a Ward defect of the scale symmetry and conjectures "the closure is calibrated against its own defect"; LuDoe (335513) sees that imposing the true fourth-order level hurts by +29% on 24/24 networks even without correctors; megan_park (18282) has λ 2…5 times larger than the one measured on the MC truth, and the measured λ gives +70% MSE (+40% without the other fitted parts). In Phase 1 we saw the same on the two-moment closure: the true diagonal of C ×0.118, the true full C only ×1.10.

The partial-correction curve depends on the channel: for C it is quadratic (λ 0.25/0.5/0.75/1 → ×1.65/3.07/6.33/9.6 on layer 14), for c3/c4 it is concave (×1.31/1.67/1.98/2.07): half of the c3/c4 correction gives 65% of the gain, but only when the error points exactly towards the truth; orthogonal noise with corr 0.9 breaks it by ×16.

### 4.3 Where the error lives

**Oracle ladder on the Susupro chain (24 corpus networks, truth N=1e8):** the true diagonal mean after ReLU on layer l: l=4 ×1.03 (20/24), 7 ×1.13 (24/24), 10 ×1.43, 12 ×2.05, 14 ×5.14 (24/24); the true variance ×1.00 at any l; together with the k3, k4 diagonals ×1.00. Only the mean carries error; the chain's own error in k4 grows to 11% (layer 14), in k3 to 2%, in k2 to 0.15%, and substituting them gives zero.

**Pair state (18 networks, MC 2²⁵):**

| substitution on layer K | 7 | 10 | 12 | 14 |
|---|---|---|---|---|
| m | | ×1.42 | ×2.00 | ×5.01 (3.97…6.12) |
| C off-diagonal | | | ×1.18 | ×1.10 |
| m and C | ×1.22 | ×1.77 | ×2.96 | **×9.20** (7.80…12.62) |

Rank of the C correction on layer 14: 8 ×5.7 · 32 ×7.3 · 128 ×9.1 · full ×9.6 (with m). C on a set of layers: 14 ×1.09, 12…14 ×1.25, 10…14 ×1.34, 8…14 ×1.48 (plateau); from half the MC ×1.26, i.e. noise in C quickly eats the gain. Critical neurons (0.1 < P(h>0) < 0.9, 300…370 of 1024) carry nothing on their own: their rows alone ×1.35…1.54, the rest alone ×2.44…2.87, together ×9.1…9.4; the gain exists only as a consistent state.

**Per-neuron c3/c4 of pre-activations (hooks D3set/G4set, networks 0…3, MC 2²⁴):** d3 on all layers ×1.44/1.43/1.25/1.27, g4 ×1.14/1.25/1.20/1.30, both ×1.94/2.07/1.73/1.86 (geomean ×1.90); superadditive (1.34·1.22 = 1.63 < 1.90). D3 on live neurons is exact (corr 0.998…0.999, rel 3.5…6.6%, slope 1.00→1.02 with depth); the error is born in the κ3 hubs from (m, C, D21) and does not depend on c4 or the previous D3; g4row corr 0.59…0.77, the chain barely distinguishes γ4 between neurons (corr with per-neuron γ4 0.17…0.25, common shift −0.2…−0.38 rms). A c3 oracle on one layer at a time: L1…L15 each ×0.99…1.08, Σlog = full, there is no bad layer.

**Accounting (networks 1/0):** c3 ×1.43/1.44, c4 ×1.25/1.14, D21 ×1.19/1.02, c3+c4 ×2.07/1.94, c3+D21 ×1.75/1.54, all three ×2.54/2.64 ≈ 43% of log(×9.2). **Synergy with C:** the true C after ReLU on layers 1…14 plus c3/c4 on 1…15: C ×1.33/1.34, c3c4 ×1.85/2.04, **together ×4.49/×6.43** (product of solos 2.46/2.73) = 69…84% of log(mC14). The whole gain of C is in the off-diagonal (off ×1.33, diag ×0.96…0.97); the non-Gaussian part of C is ≈6%, and the chain already carries it (corr with the true one 0.998); only ~0.6% needs fixing. D21 on top of C is exactly ×1.000 (D21 only feeds the K11 off-diagonal, which C already sets). Partial oracle of both channels: λ=0.5 for both ×3.17/×3.04 (prediction before the run ×1.9…2.4), λ=0.75 ×4.47/×5.07; c3/c4 fully + C half ×4.4/×5.5 ≫ C fully + c3/c4 half ×3.1/×3.3, so c3/c4 is the priority channel.

**What exactly c4 is missing.** Decomposition by direct MC (inclusion-exclusion through `E[(x²@W²)(x@W)²]`, Q22, Q31, C, x4; c4/rest against an independent MC corr 0.98…0.99): rest = c4 − diag − aabb − aaab = aabc (three distinct indices) 66…84% of the variance on layers 5…15 + abcd (four distinct) 19…31%, orthogonal (corr 0.02…0.07). corr(g4row, aabc) 0.75…0.78, corr(g4row, abcd) ≈ 0: **the chain partly sees aabc and is blind to abcd.** Gate: a c4 oracle without abcd takes only 19% (on our c3) / 23% (on the exact one) of the log gain of c4 on network 1 and −9% / 6% on network 0, so the benefit of c4 sits in abcd (n⁴), and the aabc diagram is dead before being written. κ4 from a low rank of the state (top-r eigenvectors of Cov(h), r 4…64) = the control with zero (×0.10) or random 32 (×0.32…0.34); the chain's own g4 replaced by zero is ×17 worse, so refine, do not replace.

**κ5 per neuron:** the true κ5 of pre-activations through a fifth-order Edgeworth gives ×1.005/1.002/0.987 alone, +2…5% on top of true c3, c4; median |γ5| 0.002 (layer 0) → 0.015 (layer 15). The series beyond fourth order is closed by its ceiling, each order ×1/3.

**Where it is born and how it is transported.** The mean error on layer 14 = the sum of independent step errors (~5.5e-5‖m‖ per step, transport ≈1): true (m, C) on layer 13 → 5.5e-5, 12 → 7.8e-5, 11 → 9.5e-5 against a base of 1.6…1.9e-4. The v34 birth map (transport linearised through the gates, least-squares coefficient 1.00±0.02 on every layer): share of the final MSE from births at L4 0.013 · L8 0.045 · L10 0.091 · L12 0.115 · L13 0.152 · L14 0.178 · L15 0.196; **layers 12…15 = 64%**. Births on neighbouring layers are independent (cos(b_{l+1}, T b_l) 0.023 against a control of 0.028). The system `MSE = Σ|b_l P_l|²` is sealed (0.99): contribution of layers 0…4 1.5%, 5…9 15.6%, 10…14 58.4%, 15 19.1%. The per-layer relative error of μ of the chain against the truth is flat: 0.015% (layer 1) · 0.008% (4) · 0.009% (8) · 0.012% (12) · 0.015% (15), the same on 8 networks to the third digit. megan_park (18282) gives this a closed form: survival `S(ℓ) = ∏_{k>ℓ}(1 − arccos(c_k)/π)`, error along the mean survives at 0.86 against 0.22 for a generic one; our numbers "transport ≈ 1 for the mean error" and "64% from the last four" agree with it.

**The Phase 2 regime differs from Phase 1.** Median |t| of pre-activations 0.45 (layer 1) → 0.96 (4) → 1.52 (8) → 2.33 (15); share of |t|>2 from 0 to 0.57; at the knee |t|<0.5 from 0.56 to 0.11. Not "everyone at 3.5σ" as in Phase 1 at 256×32: a third of the neurons at depth are still in the body, so shape matters. The PR rank of Cov(h) is 242 (layer 4) → 61 (layer 15), not ~3; γ4 at 1024 is tiny (0.010 at layer 1 … 0.061 at layer 15), the chain's own κ4 error 0.001…0.013.

### 4.4 Common modes: why every learned correction to the state dies

Stand: derivatives of the output along 12 feature directions `δD3/δg4 = ε·sd·X̃_j` on layers 12…14 (D3, g4, mu, var, t, row and column sums of D21, |D21|, C², D21·C, D21², C), a linearity seal (linear ×1.151/1.113/1.110 against the run ×1.151/1.111/1.108), least squares on the output with leave-one-out over networks 0/1/2.

| quantity | networks 0 / 1 / 2 |
|---|---|
| ‖e₀‖ of the output error | 4.96 / 5.33 / 4.82e-3 |
| ‖A3_j‖ of the response to a unit (1 sd) feature direction | 1.2…3.2e-2, i.e. 2.5…6 × ‖e₀‖ |
| ‖A3_true‖ of the response to the full true c3 residual | 1.94 / 1.83 / 1.77e-3 = 0.37 ‖e₀‖ |
| cos(e₀, A3_true), λ\*, gain | −0.36 / −0.32 / −0.32, 0.9, ×1.15 / 1.11 / 1.11 |
| share of A3_true in the span of the 12 directions | 0.20 / 0.16 / 0.09 |
| |cos(e₀, A3_j)| | ≤0.17, signs change between networks |
| response spectrum σ(A3)/σ₁ | 1, 0.6, 0.45, 0.38, 0.16 (effective rank ~4) |
| least squares along one direction: λ\*, gain | ≤0.02 sd, ≤×1.02 |
| common β, LOO: A3 / A4 / both | ×0.98…0.995 / ×1.00 / ×0.98…0.99 (in-sample ×1.07 = overfit) |

The truth is balanced in the sensitive modes by a factor of 10 (a unit direction gives a response 2.5…6 times larger than the whole error, the full true residual only 0.37). Any learned correction with R² 0.93 has a residual of 0.26 sd, which in the sensitive modes misses by more than the whole error. The truth helps only because its coherent and incoherent parts cancel each other in these modes. Row contractions (28 C/D21-weighted sums) on the output behave the same: |A| 2.3…6×|e₀|, |cos(e₀, A)| ≤0.19 with sign changes, gain ≤×1.04 on one network, cross R² 0.08…0.13.

One mechanism explains d3pb, d3orth, chpre6, pairfit, chinj2, c4fit (a cross corr of 0.80 with c4 gives ×0.91…0.99 in the chain) and two Phase 1 results: the ceiling of a linear correction in the closure basis ×1.027 and the per-neuron lever of 25.2% against random 24.8%. **This mechanism leaves room only for methods that change the representation of the pairs (m, C) consistently.**

### 4.5 Anatomy of the fourth cumulant

**On the exact K=4 stand (Susupro engine, 256×16 and 128×16).**

Diagram cancellation. Energy `|Σv|²/Σ|v|²` of the birth-type contributions to the κ4 readout: layer 1 0.32…0.40, layer 8 0.51…0.65, layer 15 0.19…0.25; stable cancelling pairs (old groups): star~path cos −0.89…−0.95, star~three2 −0.87…−0.95, three2~slice211 −0.75…−0.86, i.e. star ≈ −(path + three2). But the merged group path+star+three2 is not simpler (energy at k=16: layer 1 0.644, layer 4 0.876, layer 15 0.986; separately 0.71/0.65), so the cancellation is linear in the same basis and does not lower the rank.

Transfer function by readout part (network 1, 256×16; all = 2.965e-7, none = 3.876e-6, K3a 1.036e-6): (4) 8.16e-6 · (3,1) 8.74e-6 · (4)+(3,1) 8.00e-6 · (4)+(3,1)+(2,2) 2.26e-6 · (2,1,1) 6.38e-6 · (2,2)+(2,1,1) 7.45e-6. **Any incomplete subset is worse than having no κ4 at all.** A scan of 127 subsets of birth types: only slice31 can be dropped (2.920e-7 = full), without star ×1.23 worse, without path ×1.31, without all two-leg births ×2.8; the "cheap" Susupro mode 8.77e-7 (×1.18 over K3a against ×3.5 for the full one). abhimanyu_singh (18264) independently: augment is "all or nothing", the kernel without legs is 10³ times worse.

State. Old groups (age ≥2) in a covariance basis of rank n/4 beat full K=4 on 3/3 networks (2.19e-7 against 2.97e-7; 3.09e-7 against 4.37e-7): truncating the old state is regularisation. Young groups are full-rank at birth: truncating atoms to rank 1/4/16 gives 2.19e-6/1.46e-6/1.10e-6 against K3a 1.04e-6, i.e. a truncated κ4 does more harm than its absence; the first readout of the newborn carries almost all of the gain ("silent young" S=1 ×0.34). No global rotation (cov, HOSVD) compresses the young. The wall of making K=4 cheaper is the first readout of the young (n⁴).

Lifetime and rank (128×16, networks 3/4/5): groups living 1 transport keep 0.69/1.19/1.15 of the log gain of K4 for 73 u (full 367 u); lifetime 2 = 0.89/2.03/1.39 for 112 u; C of rank 8 in births keeps 0.91/0.97/0.94; rank 8 × lifetime 2 = 0.84/2.03/1.45 for 112 u. Without two-leg births at all (only the group of exact repeated slices of κ4(h), transported exactly): lifetime 1 = 58%, lifetime 2 = 72% of the log gain for 33/51 u; readout of (4) only = 42%, (4)+(2,2) = 71%, +(3,1) = 74%; lifetime ∞ = 5.80e-5 (the gain disappears), decay 0.5 per transport = 81%. So **at 128 two thirds of the value of K=4 come from exact transport of the repeated (4) and (2,2) slices over one or two layers with decay**, and two-leg birth adds ×1.49 on top.

Rank transfer rule: the spectrum of C_off at 128 (layers 11/13) holds 86…90% of the energy in the top 4 and 91…94% in the top 8; at 1024 (layers 12/14) the top 8 hold only 56…66%, the top 32 87…91%, the top 64 93…95%, so "rank 4 at 128" ≈ rank 32 at 1024.

**TT form of the birth readouts.** After transporting `Q` on all legs (U = Q d(w₁) c, Ũ = U⊙Q, M = d(w₂) c d(w₂), H = ŨM, F = Q d(w₁), K̃_jcd = Σ κ3_jkl F_ck F_dl; Wick multipliers w₁=Φ, w₂=φ/σ, w₃=−αφ/σ²):

- (4) = 12 Σ(H⊙Ũ) + 4 Σ Q w₃ U³ + 12 Σ_j Ũ_aj w₂_j K̃_jaa;
- (3,1) = 6[(H⊙Q)Uᵀ + (H⊙U)Qᵀ] + (U³w₃)Qᵀ + 3(Q U² w₃)Uᵀ + 3[(Q Kd w₂)Uᵀ + (U Kd w₂)Qᵀ + 2 Σ_j Ũw₂ K̃_jad];
- (2,2) = 2[HŨᵀ + ŨHᵀ + 2X + Y + Yᵀ] + 2[(QUw₃)(U²)ᵀ + (U²w₃)(QU)ᵀ] + c⊗κ3 terms; cross terms X_ac = Σ_jk (U_aj Q_ak) M_jk (Q_cj U_ck), Y_ac = Σ (U_aj U_ak) M_jk (Q_cj Q_ck), with c = VΛVᵀ: X = Σ_s λ_s B^s⊙B^sᵀ, Y = Σ_s λ_s B^s⊙B^s;
- (2,1,1) = 2[(H;Q;U)+(H;U;Q)] + 8 cross terms + star + c⊗κ3.

All four slices = `exact_k4_group.slices` to 1e-15 (n=6 against a direct Sym₄; n=32, 128 against the Susupro engine; lifetime 2 through Q = W₂ d(s) W₁); the control w₃×0.75 breaks it by 0.1…0.4. The first version had two swapped placements and doubled terms, caught by the seal. On the 128 stand (rank of C 8 × lifetime 2): without the cross terms ×0.78, without c⊗κ3 ×0.53 (the c⊗κ3 term carries half of the log gain and needs a young κ3 with K4→κ3 feedback: with it 1.28e-5 = the full value, C⊗C stars alone 2.09e-5 = 26%).

**Width law (1024, s32refire, mini 0…2, control β=0 bit for bit).** Exact transport `W² K22_{L−1} W²ᵀ` in wk4m with blend β 0.35/0.5/0.7/1.0 → ×0.9966/0.9953/0.9886/0.9758, monotonically worse. corr(regeneration ansatz `(dG_i+dG_j)/6`, exact transport) 0.999 (layer 2) … 0.989 (layer 15), scale 1.00…1.02. At 1024 the ansatz **already is the exact transport to within 1%**: `W²` averages 1024 inputs, the `1/√n` fluctuations at 128 are 2.8 times larger, and that is where the 71…81% gain at small width comes from. The chain's K22 birth is underestimated with depth (scale chain/truth 0.98 (layer 4) → 0.91 (8) → 0.84 (11) → 0.72 (14), corr 0.98…0.999; g4 underestimates c4 with a slope of up to 1.36; the true K22 removes 10…25% of the c4t−g4 residual), but a multiplicative correction does harm (×0.21), because the tables are fitted to the underestimate. The grader confirmed: 337662 with the TT term (4) at λ 0.5 gave MSE 1.903e-8 against 1.897e-8 for the base.

**Conclusion of §4.5.** The 128 stand is fit for birth diagrams but not for transport accuracy. At 1024 isolated exact pieces of κ4 (slice transport, K22 birth, trees, star) do not pay off or do harm; only the full κ4 together with the all-slice abcd part and with rebalancing of the chain stays alive, with a c4 ceiling of ×1.25 alone and ×4.5…6.4 together with C. K=4 within budget at 1024, by the Susupro numbers, is dead by mechanism: the first readout of the young costs n⁴ = 512 u per layer, and a late start from layer 8 gives raw 1.19e-8 for ~1112 u at a budget of 1024 u.

### 4.6 Cost

**Flop profile by role (v36, one network, sum over places = flops_used bit for bit, 0.2453 B):**

| role | share |
|---|---|
| young: D21 hub | 23.6% |
| young: transport W@[A,P] + newborn | 22.2% |
| basis rebuild (range finder 10.4, rotations 6.9, tier2 6.0, projection 5.2, QR 2.5 … n³) | 15.7% |
| old: legs from the tier1/tier2 basis | 9.6% |
| old: hub through the basis | 9.3% |
| feed, D3, PZ | 7.4% |
| covariance W C, C_pre | 6.5% |
| term program and the rest | 3.3% |
| thin legs D21, Z | 2.4% |

The Susupro chain (sb2, C/B 0.160): κ3 slices (young hub, n³ per source) 43.5%, young-leg transport 26.1%, old basis 9.6%, W·C and C_pre 8.7%. Cost grows with depth through the young hub (C/B by layer: 0 0.0007 · 4 0.009 · 8 0.012 · 14 0.016). 337568: matrix multiplication 85% of flops (Strassen-Winograd L5), subtract 3.1%, add 2.9%, QR 1.8%; float64 0.46%.

**What does not compress.** The D21 slice must be dense: rank truncation kills the chain (×9.5 on 3 dumps; the truth truncated by SVD to r=384/128 is worse than the base, 2.73e-8/1.14e-7, at r=128 = 75% of the energy on layer 1). The basis entry age AGE_OLD 4 is optimal from both sides (age 3 score ×0.75…0.80, age 2 ×0.41…0.55, age 5 ×0.96…1.00, age 6 ×0.91…0.96). Along the "age × layer" axis there is no cheap place for deliberate error: (age 3, up to layer 4) score ×0.970, (3, 8) ×0.923, (2, 8) ×0.634; a stable rate of 1% of flops ≈ 4…10% of MSE. The hub energy is flat on early layers (the top half of the columns = 53% on layer 0) and concentrates only with depth (92% on layer 13), the admissible added D21 error is ≲3%, so savings are ≤8%. The rank of the linearised flow from layer l to the output (r99: 99 from layer 0 → 525 from layer 14) follows the same trend as the production age truncation (384 from age 4 against r99 272, 224 from age 7 against 183), but the basis of the flow to the output destroys even old sources (×25 at 99.99% of energy), because κ3 is consumed by cubic forms in the frame of every subsequent neuron, not through the mean channel. All 12 public knobs of V29 and 8 switches of sb2 are within 0.1…5% of the optimum, all worse: the chains sit on the trade-off line.

**Dead and linear neurons.** Rows of dead neurons (t<−3) in the κ3 legs and P hold 0.0000 of the norm, so cutting them is exact; threshold 2.5 gives ×0.9996 MSE for 8% of flops (in production really +1.9% in score because of the multiple-of-32 alignment and fixed shapes). The linear side (t>3) cannot be cut: ×1.53 at TL=3, because a linear neuron has f′=Φ≈1 and takes part in the pair terms of the hub. Telemetry of the leaders: their flops vary from network to network (cv 2.95%, corr between Luna and J2W 0.94) and follow the share of saturated neurons |t|>3 (16.5%, model cv 3.04%).

**Residual gate.** In Phase 2 `C = F`, residual is not charged, there is only the 0.4 s gate that zeroes predictions. Grader 337568: 0.39 s at 24 695 operations (≈16 µs per operation); operation profile: add+subtract 10 637 (6% of flops, Strassen combos), multiply 4480, copyto 3591, matmul 1816 (85% of flops). Strassen L6 on the generic path gives CB +0.2% and +331 operations (the families do not divide by 64); with padding CB −1.07…−1.73% at +2716…3980 operations, so residual 0.43…0.45 s > 0.4 and the predictions are zeroed. **Flops and the gate are separate**: the sixth level opens only after cutting 3…4 thousand small operations. pranay212 (18273) measured the same from the other side (≈4 µs per call, up to 24 µs after long operations; two neighbouring builds at 0.438 and 0.440 s, one passed, the other did not).

**Non-obvious tariffs.** A truncated SVD is charged cheaply (0.4 u for 29 calls) while the backend computes a full one (17 s of wall): wall and flops diverge. `reshape` is charged numel of the input even for a view. `fnp.zeros((n,n))` returns a SymmetricTensor with a tag that the index `[None, None, :k, :]` does not remove, and Strassen fails at assembly; take output buffers from the pool or with a 4-D slice. In Phase 1: the `as_symmetric` tag on an operand plus a three-way einsum gives ×0.751 on `W C Wᵀ`, the einsum `'ia,ib->ab'` on the same object ×0.502, Cholesky instead of eigh for whitening ×0.212; together 0.26% of the budget for free.

### 4.7 The structure of the error, measured from other sides

- **Commits as in git.** The κ3 correction of the chain = a sum of ~120 pairwise orthogonal deliveries (15 sources × ages), each one new information (R² of the rest from the first 2…3 deliveries ≤0.06); sources are non-additive by 23…25% (the birth of a young source depends on the state built by the old ones); from age ~5 on, the change of a commit between layers is 98% free linear transport of the mean, and the remaining ≈20% of the norm is a new delivery, which is what we pay n³ for.
- **The ladder of worse rungs.** The error of the top lies in a direction orthogonal to all cheap degradations of the chain (cos(d, e_top) −0.15…+0.15); Richardson/Aitken down the ladder gives ×0.98…1.00; the only rung with a noticeable coefficient (birth scale 0.9/1.1, c = ∓0.27 stable) gives ×1.023 at the cost of two chains. The ladder exists only upwards: K=3 → +d4 ×1.03 → +sea3 ×1.07 → +mem31 ×1.115 → D21 oracle ×1.68 → +κ31 ×2.72, and its rungs are expensive (n⁴).
- **Mixtures of chains.** reg + γ(dense − reg) with γ≈0.65 gives ×1.223 (LOO ×1.236, foreign d ×0.841), 12 chains LOO ×1.433, but this is interpolation at the cost of ≥2 chains; multiplier knobs on 35 term groups inside one chain LOO ≤×1.012, so a mixture lives in structural simplifications, not in coefficients. The MSE frontier of the methods we know against MC by cost: reg 2.20e-8 (×136 over MC at that cost), g96 1.81e-8, mixture of all 1.54e-8, K=3 pair oracle 1.09e-9 (×20, but computing it would cost ~16 budgets, n⁴).
- **How Monte Carlo cancels.** On one layer: the error of the running mean is exactly 1/√N in σ; one weight at 10⁶ samples has a contribution error of 1.7e-5, the sum over 1024 = 1e-3 = RMS·32, so the signs are independent and the cancellation is ×32, not ×1024; the chain error per weight has RMS 2.6e-5, but the sum is 7.8e-4 with a coefficient |Σ|/√Σe² of 0.25…0.95 (anti-correlation with W). The N at which MC matches the chain on a neuron: 1.6e6 … 1.8e9; the budget gives ~65k, so MC is 5…25 times noisier and carries no information about the chain error at an attainable N.
- **The mirror network and the law of conserved complexity.** `h = λ + m` exactly (λ = the linear network, m receives what was cut off), E[λ]=0, so the whole mean is in m, but σ(m) ≈ 280 against 0.32 in h; layer contributions cancel ×90. A leaky slider from the linear point: the series in (1−a) has an error of ×100…700. Tropical decomposition `h = max(F,G) − G`: |E[F]| ≈ |E[G]| grow ×36 per layer, 2e23 at the output at |E[h]| 0.97. Three forms of one law: the mean is not computed as a difference of large quantities.
- **Switching ReLU on neuron by neuron.** The direction of the final mean is formed by the last 2…4 layers (cos with the full one 0.10 at 8 layers, 0.51 at 14); in random order cos ≈ the share of switched-on neurons, so every neuron contributes equally, regardless of layer; extrapolation of the partial-nonlinearity curve is worse than the chain by ≈2000 times (switching a neuron on is a discrete jump).
- **Crowns of the error tree.** The sum of errors of output groups grows faster than noise (R(1024) = 4.89), but the share of MSE in the crowns is ≤1.6%; groups by t = random groups, so the errors of the means are exchangeable across neurons; the oracle "remove the crown after every layer" gives k=16 ×1.17, k=4 ×1.85, full means ×4.94; summing the truth requires ~5e6 samples per layer. Refinement: magnitudes are exchangeable, but signs are aligned along the principal directions of Cov(h15): the share of the error in the top 16 directions 0.10…0.17 against a control of 0.023…0.027; the oracle removing the projection onto the top 128 gives ×2.50; a Kalman filter with MC in these directions ×0.83 at 5% of the budget.
- **Turbulence analogy.** Millionshchikov's quasi-normal closure = our Gaussian step, EDQNM = damping of κ3 by age, realisability = PSD of the covariance. Tests: DAMP 0.85/0.70 monotonically worse (0.0753% → 0.1210% → 0.1578%), KSCALE optimum exactly 1.0, PSD after Edgeworth: min eig ≥ −2.6e-6 at max 4.5 = numerical zero. Unlike turbulence, our κ3 transport is exact (R² 0.992), and damping is not needed.

---

## 5. What from Phase 1 underlies this report

The full Phase 1 report (forum 18182, submission 326725, 2100 lines, 60 seals) is built around the ceiling protocol: substitute the exact value of one quantity and measure the upper bound of the class. The results Phase 2 relies on:

| Phase 1 result (256×32) | number | what became of it in Phase 2 |
|---|---|---|
| the network sits exactly on the critical point: negative weights 50.003%, switched-off neurons 50.2% per layer; the same weights made positive make the network linear, exact formula 1.7e-6 | b=0 is a cusp: a bias of 0.15 makes the task six times easier | at 1024×16 the K22 underestimate grows with depth 0.98→0.72: the same accumulation without separation of scales |
| ceiling of the two-moment class with exact (m,C) | ×3.36, ≈5th place | the K=3 class with exact pair cumulants up to 4th order: 1.085e-9 (×20), and it too does not go below the top without n⁴ |
| ceiling of a "better step formula" | ×1.19 (all γ schemes, quadratures, learned steps together) | a step with exact per-neuron (a, v, γ3, γ4) on the last layer: Gauss 1.93e-7 → +γ3 2.15e-8 → +γ4 9.75e-10: each order ×8…10; shape is not the bottleneck, the accumulated Δmu is |
| 2/3 of the m error is created by transport, not by the step; ideal local m ×1.11, true m ×3.11 | | Phase 2: the error = a sum of independent step errors with transport ≈1, 64% from the last 4 layers |
| balanced errors: the true diagonal of C ×0.118, full C ×1.10, σ alone ×0.52 and Edgeworth alone ×1.34 give ×113.94 together | | §4.2 and §4.3: channels multiply (C ×1.33, c3c4 ×1.85, together ×4.5…6.4) |
| the closure error is a pure 1/n effect (p = 1.002±0.055), but the law governs the size, not the direction; a fitted α takes 93% of the oracle | ×2.572 LOO | width law §4.5: the gain of slice transport at 128 comes from 1/√n fluctuations and disappears at 1024 |
| the backward operator has effective rank 2.7, 68% of the error lies in a 6-dimensional Lyapunov subspace computable from the weights | | at 1024×16 the r99 of the flow is 99 from layer 0 (not 2.7), PR rank of the state 61…242: transport does not destroy the error, and this makes the Phase 2 task a different one |
| non-Gaussianity is 96% produced by dependence between neurons, not by per-neuron non-Gaussianity; the effective CLT width ≈30 | shuffled γ4 0.014 against the real 0.395 | §4.1: the sea = triples and quadruples of distinct indices |
| ReLU masks carry non-moment information (4.9% at a noise of 0.28%, not decaying with distance), but it does not vary across networks | | |
| the Euler + Stein identity `E[out] = E[Δout]` is exact (bitwise), but lives in the tail: the error of ρ(0) is 68 times larger than the error of E[ReLU] | | the "body or tail" gate before every idea |
| there is no per-neuron or per-sample lever: selection by |t| 25.2% against random 24.8%; there is no universal sample quality (corr between networks +0.012 ± 0.146 = control) | | exchangeability of the error §4.7 |
| flat-budget theorem: score = a·p/B does not depend on N above the floor; the mode of the shift of 33 estimates is exactly on the truth | | Phase 2: `C = F`, the score is linear in C/B above 0.1, cost and accuracy are one basket |
| the truth of the dataset is itself MC: the final_layer_mse floor = avg_variance/1e9 | 4.9e-11 (mini) | Phase 2: floor 7.8e-11, the leaders ×140 above it, us ×240 |
| a free 0.26% of the budget from symmetry bookkeeping (tag on the operand + einsum, Cholesky instead of eigh) | | §4.6 |

Honest odds recorded on 18.10 before the deadline: top 3 <1%, top 10 <3%, realistically 3.3…3.6e-9. Actual: 2.866e-9, #39.

---

## 6. Where we were wrong

Phase 1 had 51 refutations of our own results out of 396 records and a catalogue of 32 mechanisms by which a measurement lies. Phase 2 added the following, each with a number:

1. **A pruning artefact.** Regressions of D3 on all 1024 neurons gave "D3 ≈ 0.66·pair + 0.73·triple + ν" and corr 0.84; on live neurons corr 0.9988, pair 0.98, triple 1.00. The V30 pruning zeroes D3 in 96 rows (layers 12, 14), and the mixture of live and pruned neurons invented a mechanism that does not exist.
2. **"61% of the value of K=4 comes from the star".** The only4 mode on the 128 stand did not shadow the group of exact slices; caught by a flat scan of the star multiplier (λ 0.5/1.0/1.5 → 2.35/2.30/2.27e-5) and by a stand without two-leg births: the value came from the slice group.
3. **128 → 1024.** We read the 71…81% gain from slice transport at 128 as a property of transport; at 1024 the regeneration ansatz already is the exact transport (corr 0.99). The blind submissions paid for this: 337662 and 337678 raised the cost by ~19% at unchanged MSE, and the four that also transported the (4) slice worsened the MSE by ×0.85…0.47 (§1). Law: the 128 stand is fit for birth diagrams, not for transport accuracy.
4. **Crowns: the first result of ×1.05…1.09 was an artefact of the pre-activation covariance.** The Phase 2 output = ReLU(p15), Cov must be taken from M11, not pre_M11; with the correct one the share of MSE in nearly dead outputs = 0.000.
5. **The coincidence of the MSE of cheap K=4 (1.11e-8) with the leader's MSE (1.10e-8).** It remained a coincidence without a cost mechanism; we nearly built a plan on it twice.
6. **The prediction about the residual predictor.** We expected ×2.0…2.8 at corr 0.7; measured ×0.039. The chain is hypersensitive to incoherent noise in c3/c4 (ΔMSE ≈ 0.6σ² on D3, 0.9σ² on D21 by the law of 18264); only a nearly exact correction (ρ ≥ 0.95) helps.
7. **The first version of the TT formulas**, with two swapped placements and doubled terms, was caught only by the n=6 seal against a direct Sym₄.
8. **The basis of the flow to the output.** The prediction "projecting legs onto the input directions of the future flow ×1.0…1.1" missed by ×30 (×25 at 99.99% of energy): the legs lie in the image of the past flow, not in the input directions of the future one.
9. **The verdict on a learned corrector, "×1.02…1.03 at most" (27.09), stood as closed for three weeks**, until forum post 18254 showed an MLP with state at ×0.887 of raw MSE. Cause: 80 networks instead of thousands. Since then we keep a register of weakly closed verdicts (N ≤24, an intermediate target, a different base, "closed by cost").
10. **The local cost of a single-network run is inflated ×1.07…1.08** (Strassen level of the first call, touching new pools); versions with different one-off costs must be compared only in a multi-network process.
11. **Oracles without a correction for the noise of the truth.** See §3.6; our ceilings are underestimated, the verdicts stand.
12. **A submission package with a default that did not match the measured one** (DEADHUB=0 instead of 1): before packaging, run the source without environment variables and compare bit for bit.
13. **Orientation of (3,1) in the oracle.** MC gives `k31[i,j] = E[x̃_i³ x̃_j]` (cube on the row), while the chain expects the transpose; the chain's ansatz `C_off·λ` is symmetric, so the diagnostic correlation of ansatz and truth cannot distinguish orientation, and we did not check it. After the reports of Nurlanggs (18252) and kai_leal_gonzalez (18246) we checked it on 27.10.2026 on the same MC data (seals: self-substitution ×1.0000, the old configuration ×0.8040 bit for bit): transposed ×0.804, correct ×1.009, shrunk 0.75 ×1.055; both slices together ×0.977 instead of ×0.773. The §4.2 conclusion about (3,1) changed from "harms" to "neutral, shrunk helps slightly"; this confirms the law "the optimal correction is shrunk" once more. The "shape at the ansatz level" control is unusable for (3,1): the off-diagonal mean of the truth is ≈0, and normalising by it explodes.

Three rules bought with these errors: measure any verdict about compression or transport at the working width, not a convenient one; read a single-channel oracle as an upper bound of the shape, not as a forecast of a method; check any "matches the leader" through the cost mechanism, not the MSE.

---

## 7. What others found and we did not

After the deadline we read all Phase 2 write-ups on the forum. An honest list of what our code lacks and what explains the ×2.2 gap. Numbers in the "number" column are given as their authors give them: an MSE or compute factor, lower is better.

| what | who | number | our status |
|---|---|---|---|
| spherical input: homogeneity implies a chain on the sphere, input κ4 of −6/(n+2) exactly in isotropic form, rescaling of the corrector target | dogus_ozel 18281 (1.503e-9), il.miscusi 18248 | 2 lines, 0 flops, MSE ×0.967 without the corrector | in Phase 1 we measured shell sampling (angular oracle ×1.83 over the closure) and did not carry it into the Phase 2 chain |
| extraction of the global scale mode: ε from the chain's own (i,i,j) κ3 slice, removed from C, κ3, κ4 together; makes a carried κ4 profitable | il.miscusi 18248 | ×0.380 [0.365, 0.394], 50/50; harms in memoryless chains, because the λ law is that very mode | our common-mode law ×10 (§4.4) describes the same thing from the side where it is not turned into a gain |
| fast matrix multiplication: Strassen-Winograd + Karstadt-Schwartz basis + Smirnov ⟨3,3,6;40⟩ from metered operations | dogus_ozel | ×0.50 of compute | we only have pooled Strassen-Winograd |
| oracle bias correction (noise ceiling) | michael_anderson7 18272 | σ² 0.776 → 0.511 | §3.6 |
| orientation of (3,1): the oracle must enter as Uᵀ in the k4(a,b,b,b) table | Nurlanggs 18252, kai 18246 | correct MSE ×0.909, transposed ×1.020; scale 0.6…0.8 | checked on 27.10.2026: our first oracle had (3,1) transposed (MSE ×1.24 worse); in the correct orientation MSE ×0.991, with scale 0.75 ×0.948, i.e. their result reproduced in sign and scale (§6, item 13) |
| merging κ3 sources of age ≥7 into one rank-256 basis by exact weighted truncated SVD | dogus_ozel | | our FIJA (fixed transported basis) is a weaker relative |
| the survival law S(ℓ) in closed form | megan_park 18282 | agreement to 1…4% | our numbers agree (§4.3), we did not derive the form |

What of ours they confirm independently: the balance of errors (three reports), the benefit of the (3,1) slice (theirs MSE ×0.909; our own September measurement on V29 ×1.22 in our convention, i.e. MSE ×0.82), "all or nothing" for κ4, invariance of the corrector factor to the engine.

---

## 8. What remains open

**Closed by ceiling (do not return):** corrections to the state from chain features in every application we tried (§4.4); isolated exact (2,2) slices, slice transport and K22 birth at 1024 (§4.2, §4.5; for (3,1) see open item 4); κ4 from a low rank of the state; κ5 and higher per neuron; scalar knobs on D3, D21, g4row, λ (optimum exactly 1.0 everywhere); mixtures of mirror chains (×1.005); 1D corrections of the E[ReLU](t) break curve; FORM/SORM on the sphere (the error is not concentrated: top 1% of samples 0.037 = Gauss); abcd tree diagrams; hub columns and age as a place for cheap error; extrapolation of the ReLU switch-on curve; averaging of masks.

**Closed by circumstance (the reason to return is named):** an MLP corrector on the ridge residual (90 networks; thousands are needed); mixtures of chains (cost of two chains; a cheap second chain was not tried); the self-consistent F21 feedback (old k3c base, util 0.369; not measured on the V29 line with the current cost tricks); ROWALIGN 32 (2 networks).

**Open:**
1. The full κ4 together with the all-slice abcd part and rebalancing of the chain: c4 ceiling ×1.25 alone, ×4.5…6.4 together with C; this needs either an n⁴ representation cheaper than 512 u per layer, or an MC inclusion for abcd, or sample pairs for κ3⊗κ3.
2. Spherical input and extraction of the scale mode in our line (§7): both cheap, the first at zero flops.
3. Whether the law "68% of the error in a 6-dimensional Lyapunov subspace" holds at 1024×16 (the r99 of the flow of 99 against 2.7 at 256×32 says no, but the share is not measured).
4. Why the correct (3,1) partly compensates the harm of the exact (2,2) (together ×0.977 against a product of solos 0.904·1.009 = 0.912), and whether this holds on more than three networks. Also whether the gain of the shrunk (3,1), ×1.055, which comes from one network of three, holds.
5. Why FIJA gives MSE +3.2% on the grader against a local +0.6%.

---

## 9. Reproducibility

The code of the submissions and of all stands, the run logs and the journals will be provided to the jury on request (an email to arc-whestbench@aicrowd.com or a reply in this thread). Contents:

- submissions: 337654 and 337568 (the s32refire engine + our fast LT2 layer with further-trained weights, packaging script), 337116 (the Susupro chain + FIJA), 336991 (our line on V29);
- our line on V29: version generators v37…v46, the step_F87 ridge table, FIJA, exact savings;
- the LT2 corrector: training on the flopscope→cupy GPU shim, state dumps on the corpus, the fast layer;
- Phase 2 oracles with hooks K11, D3set, G4set, K5set, D21set, Crec; the orac22 κ4 slice oracle (MC 2²², including the (3,1) orientation check of 27.10.2026);
- the exact K=4 stand at 128×16 and 256×16 and the TT seal;
- cost profiles by line and role;
- the lean chain and the September ladders;
- data: dataset `aicrowd/arc-whestbench-public-2026` (v2-phase2), moments `keenanpepper/arc-whestbench-p2-higher-moments-2026`, corpus `keenanpepper/arc-whestbench-p2-d8b-corpus-14k`;
- work journals and the project memory (≈500 dated records with numbers).

All numbers in this report come from recorded runs; where a number is given from three networks or fewer, the text says so.
