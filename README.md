# The cumulant chain as a balanced system of errors (ARC White-Box, Phase 2)

Code and stands of team **trim_qewas** for the ARC White-Box Estimation Challenge 2026, Phase 2 (random bias-free ReLU MLPs, 1024 wide, 16 deep).

* Write-up: [`writeup/PHASE2_WRITEUP_EN.pdf`](writeup/PHASE2_WRITEUP_EN.pdf) (Ukrainian original: [`writeup/PHASE2_WRITEUP_UA.pdf`](writeup/PHASE2_WRITEUP_UA.pdf))
* Forum topic: https://discourse.aicrowd.com/t/18295
* Phase 2 submissions (two best by score, default rule): **337654** (2.866e-9) and **337568** (2.881e-9)

Our best estimators are built on public code of other participants; see [`NOTICE.md`](NOTICE.md). The contribution we claim is the measurement work described in the write-up, and the stands here are what produced its numbers.

## Layout

```
submissions/          estimators exactly as graded (estimator.py is byte-identical to the submitted archive)
  337654_midA/          s32refire + fast LT2 mid layer, further-trained weights   2.866e-9  MSE 1.897e-8
  337568_mid/           the same, LT2 weights retrained on 1000 networks          2.881e-9  MSE 1.906e-8
  337659_full/          LT2 full instead of mid                                   2.887e-9  MSE 1.902e-8
  337116_susupro_chain_fija/  Susupro chain #336537 + our FIJA                    3.120e-9  MSE 1.913e-8
  336991_v46_v29_line/  our own line on V29 (FIJA, step_F87, R_FB 4/8, exact savings)  3.747e-9  MSE 2.050e-8
night_oracle_scan/    seven blind submissions on top of 337568 (write-up §1, §6.3)
  337665_star  3.155e-9 · 337662_tt  3.434e-9 · 337678_k22_b0.5  3.452e-9
  337692_lead_b0.3  4.034e-9 · 337688_lead_b0.5  4.856e-9 · 337689_lead_b0.7  5.834e-9 · 337681_lead_b1.0  7.340e-9
  exact transport of the (4) and (2,2) slices of κ4 makes the MSE monotonically worse with the blend:
  ×0.85 / 0.70 / 0.59 / 0.47 at 0.3 / 0.5 / 0.7 / 1.0
stands/               research stands, oracles and analysis scripts behind §3…§5 of the write-up
  lt2alg/orac22/        oracles of κ4 slices (2,2) and (3,1), incl. the orientation check of §4.2
  lt2alg/, lt2re/       LT2 corrector training, evaluation, ablations
  v37 … v39fija/, v38age/   generators of our V29 line, FIJA, step_F87 ridge tables
  s32lt/                port of LT2 into s32refire, the fast layer
  dig/                  probes of the error map, common modes, sea and birth diagrams
  bench/                cost benchmarks (Strassen ladder, dtypes)
stands_phase1/        stands of Phase 1 (256×32) and of the start of Phase 2, the results that §8 of the write-up stands on
  side66/               main Phase 1 stand folder (oracles, closure ceilings, κ, masks, width extrapolation, …)
  s76_*/, viz/, wbstand/   smaller groups of probes
tools/pack_s32f.sh    how the s32f packages were built and checked (AST, archive equals source)
writeup/              the report, Markdown and PDF, English and Ukrainian
```

## Running a submission

The estimators run under the official toolkit (`whestbench`, `flopscope` 0.12.3 as recorded in each `manifest.json`):

```bash
whest run --estimator submissions/337654_midA --dataset <snapshot root of arc-whestbench-public-2026> --split mini --format json
```

## Notes on the stands

* They are research scripts, kept as they were run (helper scripts for local chat models were removed). Paths are absolute to our machine, GPU pinning is by device UUID, and comments and logs are mostly in Ukrainian.
* Monte Carlo data, trained weight files (`*.npz`) and run logs are not included because of size; they are available on request.
* Every number in the write-up comes from these stands or from the grader; the write-up names the stand next to the number where it matters.

## Licence

Our code is MIT ([`LICENSE`](LICENSE)). Upstream code keeps its own MIT licences ([`NOTICE.md`](NOTICE.md)).
