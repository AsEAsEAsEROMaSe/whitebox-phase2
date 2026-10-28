# Closed branches: what we measured and why it died

One row per branch, from both phases of the ARC White-Box Estimation Challenge 2026.
Phase 1: 256 wide, 32 deep, our own Gaussian two-moment closure blended with whitened Monte Carlo.
Phase 2: 1024 wide, 16 deep, factorized K=3 cumulant chains.

Conventions:
- `×` is the factor on MSE (or on the closure error where marked), **greater than 1 is better**.
- **ceiling**: the branch was closed by substituting the exact quantity (an oracle), so no implementation can do better than the number shown.
- **mechanism**: closed by a measured cause, not by its price.
- **circumstance**: closed by price, sample size or another condition; the reason to return is named.
- Stands are under `stands_phase1/side66/` (Phase 1) and `stands/` (Phase 2). Every experiment with its numbers is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## The four laws that killed most branches

1. **Balanced errors.** A tuned closure is a system of errors that cancel. Putting the exact value into one channel breaks the cancellation: true diagonal of C ×0.118 (Phase 1), exact (2,2) slice of κ4 ×0.904 (Phase 2), K22 birth set to truth ×0.21. The best correction is a shrunk one, never the exact one.
2. **Channels multiply.** Gains from fixing two channels together far exceed the product of solo gains (σ ×0.52 alone, Edgeworth ×1.34 alone, together ×113.9 in Phase 1; C ×1.33, c3/c4 ×1.85, together ×4.5…6.4 in Phase 2). Any idea that fixes one channel is dead before it is written.
3. **Common modes.** The output is 2.5…6 times more sensitive to feature directions of the state than its whole error; the truth is balanced in those modes by a factor of 10. Any learned correction of the state with R² below about 0.95 misses by more than the whole error.
4. **The sea has zero conditional mean.** Over the ensemble of W, terms with all-distinct indices average to zero given the pair state, so the K=3 chain is Bayes-optimal in its class. Features of the pair state cannot predict the sign of the residual.

## Phase 1 (256×32)

| branch | best result | type | why it died | stand |
|---|---|---|---|---|
| ideal two-moment closure (true m and C) | score ×3.36, about 5th place | ceiling | no moment class reaches the top even with an oracle | `orac101.py` |
| better step formula (all γ schemes, quadrature, learned steps) | closure ×1.19, score +3.2% | ceiling | 84% of the error is accumulation after the state diverges, not the step | `step105.py` |
| exact 2-D quadrature instead of Mehler K=2 | ×1.001 | ceiling | the step formula is already exact; the Gaussian hypothesis in the recursion is what fails | `decomp99.py` |
| κ3 in the recursion (carried, rank-r CP, one matmul) | ×1.045…1.223 against a threshold of 1.575 | mechanism | more than half of what κ3 fixes is scale, which a free scalar α takes for nothing | `k3cost97.py`, `k3diag97.py`, `hmat_k3.py` |
| γ4 table in the recursion | ×1.027, but a flipped-sign control gives ×1.013 | mechanism | half the gain is a foreign table; with oracle γ4 on exact (m, C) ×22.8, on ours ×1.02 | `gam101.py`, `orac101e.py` |
| Edgeworth series in the recursion | ×0.80…0.83 | mechanism | where the series converges it is tiny, where it is large it diverges (a₂ = 2.07 at t = 3.5) | Phase 1 write-up (forum 18182) |
| scalar knobs on transport (kap, mu, lam, beta, tau) | closure ×1.046, LOO 50/50; grader ×0.9955 | mechanism | m_deep is used twice (prediction and fit anchor); stands that see one branch miss the other | `scan101.py` |
| linear correction in the closure basis (up to 8 terms) | ×1.027 oracle ceiling | ceiling | at t ≈ 3.5, E[ReLU] ≈ m and the other terms have weight 1/34 | `split101.py` |
| hybrid: closure to layer K, then MC from N(m_K, C_K) | ×0.067 with our (m, C) | mechanism | the hybrid keeps the closure bias instead of averaging it | `hyb101.py` |
| backward reconstruction of the state from the output | effective rank of the backward operator 2.74 | mechanism | condition number 10³¹: information is destroyed irreversibly | `back101.py` |
| exact early layers (true (m, C) after layer K = 1, 2) | score ×1.001…1.009 | ceiling | the error saturates with depth, early corrections are forgotten | `early100.py` |
| per-neuron targeting by \|t\| | 25.2% against random 24.8% | mechanism | the error is isotropic across neurons | `walk100.py` |
| ReLU masks (orthant probabilities) | real signal ×17.7 over noise, partial corr with error +0.02 | mechanism | masks live in the tail, the error lives in the body of the distribution | `mask100g.py` |
| Euler + Stein identity E[out] = E[Δout] | exact, but ρ(0) error ×68 larger than E[ReLU] error | mechanism | the identity lives in the tail (zero is 3.5σ away) | `euler101b.py` |
| width extrapolation (Richardson in 1/n) | ×1.22 raw, ×0.9999 after cosine | mechanism | the whole gain is scale, which α takes for free; LOO, cross and foreign controls all pass and still give zero | `width101.py`, `rich101.py` |
| spherical variance reduction (radial part exact) | ×0.74 on top of whitening | mechanism | whitening already fixes all second moments; sphering breaks Gaussianity of the input | `homog98.py` |
| RQMC lattice | ×1.0156, sign test 12/24 | mechanism | whitening already takes what RQMC gives | `rqmc101b.py` |
| fixing the sample (4th moments, quality of a draw) | corr between networks +0.012 ± 0.146 = control | mechanism | there is no universal sample quality, MC error is network-specific | `qual102.py` |
| robust estimators (winsorising) | ×0.874 | mechanism | the output tail is only 1.22 times heavier than Gauss | `extreme102.py` |
| blend weight as a function of the network | ceiling ×1.029, realisable ×0.967 | ceiling | the proxy pulls both V and E² equally, their ratio does not move | `blend101.py`, `varpred101.py` |
| spectral (Wiener) blend by direction | cross ×1.013 | circumstance | eigh costs 0.59% of the budget, the net is +0.1% | Phase 1 write-up (forum 18182) |
| non-uniform sample allocation across networks | in-sample ×1.064, cross-seed ×0.978 | mechanism | the score is separable, the optimum is uniform; 67% of the V spread is noise | Phase 1 write-up (forum 18182) |
| cheaper pruning threshold for dead neurons | K=2 ×0.955, K=8 ×0.365 | mechanism | the current criterion already cuts 17.4% against 11.2% truly dead | Phase 1 write-up (forum 18182) |
| float32 in the submission | one network ×3.87e6 worse | mechanism | loss of range on one network ruins the average | Phase 1 write-up (forum 18182) |

## Phase 2 (1024×16)

| branch | best result | type | why it died | stand |
|---|---|---|---|---|
| exact (2,2) slice of pre-activation κ4 | ×0.904 | ceiling | balanced errors; the slice shape adds nothing at 1024 (×0.987) | `lt2alg/orac22/` |
| exact (3,1) slice, correct orientation | ×1.009; shrunk 0.75 ×1.055 (gain from one network of three) | ceiling | neutral; again "the optimal correction is shrunk" | `lt2alg/orac22/run_orac31t.py` |
| exact (2,2) and (3,1) together | ×0.977 | ceiling | the pair does not rescue the (2,2) harm | `lt2alg/orac22/run_orac31t_both.py` |
| K22 birth set to truth | ×0.21 (half ×0.65) | mechanism | the tables are fitted to the underestimated birth | write-up §4.2 |
| exact transport of the (4) and (2,2) slices (blind grader scan) | ×0.85 / 0.70 / 0.59 / 0.47 at blend 0.3 / 0.5 / 0.7 / 1.0 | mechanism | at 1024 the regeneration ansatz already is the exact transport (corr 0.99) | `night_oracle_scan/` |
| TT readout of (4), star term | MSE unchanged, cost +19% | mechanism | width averages the fluctuations that carry the gain at 128 | `night_oracle_scan/337662_tt`, `337665_star` |
| learned corrections to the state from chain features | ≤×1.02, LOO ×0.98…1.00 | mechanism | common modes (law 3) | `dig/`, write-up §4.4 |
| residual predictor of c3/c4 with corr 0.9 / 0.7 / 0.5 | ×0.061 / 0.039 / 0.049 | mechanism | the chain is hypersensitive to incoherent noise; only ρ ≥ 0.95 helps | write-up §4.2 |
| scalars on D3, D21, g4row, λ | optimum exactly 1.0 everywhere | mechanism | the chain sits on its trade-off line | write-up §4.2, §4.6 |
| κ4 from a low rank of the state (top-r eigenvectors, r 4…64) | ×0.10 (zero) to ×0.34 (random 32) | ceiling | the missing c4 is the abcd part, invisible to low rank | write-up §4.3 |
| aabc tree diagrams for c4 | takes 19…23% of the c4 gain | ceiling | the benefit of c4 sits in abcd (n⁴) | write-up §4.3 |
| per-neuron κ5 | ×1.005 / 1.002 / 0.987 | ceiling | each order beyond the fourth is ×1/3 of the previous | write-up §4.3 |
| rank truncation of the D21 slice | ×9.5 worse | mechanism | the slice must be dense | write-up §4.6 |
| cheap deliberate error by age or layer | (age 3, layer 4) ×0.970 | mechanism | 1% of flops ≈ 4…10% of MSE everywhere | write-up §4.6 |
| projecting legs onto the input basis of the future flow | ×25 worse at 99.99% energy | mechanism | legs live in the image of the past flow | write-up §6.8 |
| Strassen level 6 | CB −1.1…−1.7%, but residual 0.43…0.45 s > 0.4 s | mechanism | the residual gate is per operation, not per flop | write-up §4.6 |
| LT2 corrector of another chain ported into ours | ×2.30…2.56 worse | mechanism | the corrector is tied to the state conventions of its chain | write-up §2.3 |
| mixtures of mirror chains, Richardson down the ladder | ×1.005, ×0.98…1.00 | ceiling | the top error is orthogonal to every cheap degradation | write-up §4.7 |
| FORM/SORM on the sphere | top 1% of samples 0.037 = Gauss | mechanism | the error is not concentrated in samples | write-up §8 |
| switching ReLU on neuron by neuron (extrapolation) | ≈2000 times worse than the chain | mechanism | switching a neuron on is a discrete jump | `dig/relu_on.py` |
| mirror network, tropical decomposition | contributions cancel ×90, components grow ×36 per layer | mechanism | the mean is never computed as a difference of large quantities | `dig/leaky_slider.py`, `dig/tropical.py` |
| Kalman filter with MC in the top directions | ×0.83 at 5% of budget | mechanism | MC at an attainable N carries no information about the chain error | `dig/kalman.py` |
| turbulence closures (EDQNM damping of κ3) | damping 0.85 / 0.70 monotonically worse | mechanism | our κ3 transport is exact (R² 0.992), no damping needed | write-up §4.7 |

## Closed by circumstance (worth returning to)

| branch | what stopped it | what would reopen it |
|---|---|---|
| MLP corrector on the ridge residual | 90 networks | thousands of labelled networks (neopok919 shows −0.5% MSE per 2000 at zero flops) |
| two-chain mixtures (×1.22…1.43 LOO) | cost of two chains | a cheap second chain |
| self-consistent F21 feedback | measured only on an old base at util 0.369 | measure on the current V29 line with its cost tricks |
| full κ4 with the abcd part (c4 ceiling ×1.25, with C ×4.5…6.4) | n⁴ = 512 u per layer | an n⁴ representation cheaper than that, MC inclusion for abcd, or sample pairs for κ3⊗κ3 |
| spherical input and scale-mode extraction in our line | not tried in our chain | both cheap; others measured ×0.967 and ×0.380 (§7 of the write-up) |
