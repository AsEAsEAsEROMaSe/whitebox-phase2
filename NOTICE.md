# Upstream code and licences

The submission estimators are built on public code of other participants. All upstream code is MIT licensed; its licence files are kept next to each estimator (`LICENSE`, `ALG053_LICENSE`, `LICENSE_504aldo`) and its headers are kept in the source.

| upstream | author | where it is used |
|---|---|---|
| V29 cumulant chain (#330093) | 504aldo | base of our own line (336991 v46) and of every engine below |
| e30_a engine, mean-correction table feature grammar | LordProcciuto | inside ALG058; feature grammar in v46 |
| ALG058-V001 (birth slicing, FUSE) | TristanMiano | base of s32refire |
| s32refire engine, github acco-cyber/arc-whitebox-estimation-2026 | koushik_rudra (Nanoloom) | engine of 337654, 337568, 337659 and the night oracle scan |
| K=3 chain #336537 and the LT2 corrector, github logxio/fourth-cumulant | Susupro | chain of 337116; LT2 architecture in 337654, 337568, 337659 |
| ARC reference factorized K=3 chain | ARC | the common root of all of the above |

What is ours is listed in §1 of the write-up ("What is ours, what is borrowed, what is shared"). In short: the fast LT2 layer and its retrained weights (337654, 337568), FIJA (fixed transported basis of old κ3 sources, 337116 and v46), the step_F87 ridge, thin-leg ranks, exact flop savings, all oracle stands and the night oracle scan.
