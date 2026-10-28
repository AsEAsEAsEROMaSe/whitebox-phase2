# s32lt (21.10.2026, team trim_qewas): s32refire + LT2 per-layer corrector of Susupro (MIT, github.com/logxio/fourth-cumulant),
# ported from our sb2; LT_OFF=1 restores s32refire. Weights: lt2_weights.npz next to this file or embedded _LT2_BLOB.
# ARC White-Box Estimation Challenge 2026 - submission of team koushik_rudra (Nanoloom).
# Base: TristanMiano's ALG058-V001 (MIT; e30_a engine by LordProcciuto; V29 by 504aldo),
# verbatim, with ONE own change: the engine's built-in birth-column cut is enabled by
# default (X_DEAD_TAU_B 0 -> 2.2, i.e. keep newborn atoms with alpha > -2.2; the
# dead_fis=5 machinery then physically compacts the transport family to the live
# column prefix). Rationale: Susupro's grader-verified column cut (forum topic 18254,
# github.com/logxio/fourth-cumulant, MIT) uses the equivalent w1 = Phi(alpha) < 0.014
# threshold: -3.3% adjusted score, raw MSE flat [0.996, 1.000], 50/50 nets; the
# threshold sits at LordProcciuto's measured cliff edge (WRITEUP H1). Kill-switch:
# X_DEAD_TAU_B=0 restores the ALG058-V001 op stream bit-for-bit.
# BANCO DE TRABAJO (01/10/2026): copia de third_party/whest-p2-cumulant-k3/estimators/estimator_v29.py, commit 1a4083f.
# Copyright (c) 2026 504aldo, licencia MIT (texto en third_party/whest-p2-cumulant-k3/LICENSE). El original no se toca.
# Las mejoras de Ouroboros entran aqui como mandos de BANCO (abajo), apagados por defecto: con todos apagados, este
# fichero debe dar exactamente la salida y el coste del original.
# No es un envio: usar codigo de otro participante en un envio lo decide el usuario con la clausula del reglamento.
# 8.a-quater (03/10/2026): copia de scratchpad/criba/base_cand.py con las tablas de cierre reajustadas en 70 redes
# (correccion de medias de 87 rasgos y LAM reescalada) como valor por defecto; X_TAB_OLD=1 = base_cand.py bit a bit.
# s32 fuse1 (2026-10-20, team koushik_rudra / Nanoloom): own change #2 vs ALG058-V001
# (change #1 = X_DEAD_TAU_B 2.2 birth-column cut, graded 337079 = 2.9663e-9): the V28
# Strassen leaf-batch fusion threshold V28_STRASSEN_FUSE_P default 343 -> 1, i.e. every
# leaf batch fuses, not only deep-level ones >= 343 blocks. BILLING-ONLY change: the
# fused path runs the same per-element add-tree and matmul operands on batched slabs, so
# predictions are bit-identical to the s27 op stream (verified at 128x8 and 256x8 mini,
# seeds 1000/1001: identical SHA-256 of predictions; billed FLOPs -3.9%/-4.3%, billed
# ops +2%, wall -2%). Kill-switch: V28_STRASSEN_FUSE_P=343 restores s27 billing.
"""K3-simple factored cumulant propagation + MEMORYLESS KAPPA4 REGENERATION (V17)
+ D21 FEEDBACK THIN LEGS (V18, 2026-09-02, F69)
+ FINAL-LAYER TRIM (V19, 2026-09-03)
+ RESIDUAL-TIME FUSION (V20, 2026-09-03)
+ AGE-GATED SHARED BASIS FOR OLD SOURCES (V21, 2026-09-03, F66 option C / F72)
+ PRE-ACTIVATION LAMBDA TABLE (V22, 2026-09-03, F68 prefit)
+ NESTED SECOND TIER FOR THE OLDEST SOURCES (V24, 2026-09-03, F72 tiers)
+ ADAPTIVE PER-MLP LAMBDA (V25, 2026-09-03)
+ POOLED BATCHED STRASSEN-WINOGRAD PRODUCTS (V26-V28, 2026-09-05, F79/F80)
+ NEWBORN + COVARIANCE RIDING THE TRANSPORT FAMILY, BLOCK-SYMMETRIC C_PRE (V29, 2026-09-06).

V29: the dense young legs are pre-scaled by the wick w1 at the wick stage, so the
transport family multiplies by the raw W and the newborn's A leg (born as w1 * C_off) and
the post-ReLU covariance C ride along as two extra slots; C_pre = (W C) W^T is assembled
from the three block products of its 2x2 partition (one Strassen family), layer 0 uses the
aliased Gram w32^T w32, and the trimmed last layer reads var from (W C) * W.

V25: the regenerated kappa4 off-diagonal coefficient is no longer a frozen per-layer
table but lam_l = LAM[l] * ((mean(dG)/mean(var)) / REF_R[l])^BETA, evaluated online at the
layer from the transported kappa4 diagonal dG and the pre-activation variance (both already
computed there). Fitted offline on public dumps 0-7: the per-MLP LS lambdas track this
ratio with log-log slope ~1 at every layer and the unexplained spread drops from 3-10% to
1-2%. Kill-switch: V25_BETA=0 -> V24 values.

V24: sources older than AGE_OLD2 transports are re-confined to a rank-R_OLD2 sub-basis U
(r1 x r2) INSIDE the shared tier-1 basis Qc:  A_s = Qc U FA2_s.  Legs are formed from
QU = Qc U (r2/n units per leg), the dslice contraction of tier-2 sources is lifted by U^T
into the tier-1 inner product (one trailing Qc^T per layer, unchanged), U rotates with the
tier-1 factors at each rebuild, and the oldest tier-1 member moves into tier 2 in factor
space right after each rebuild (range finder on the r1 x r1 core, no n-space work). Lean
reference (nested tiers 4:384 + 7:256, dumps 0/1): 2.155e-8 vs 2.164e-8 = free.
Kill-switch: V24_AGE_OLD2=0 -> V22 op stream.

V21: sources older than AGE_OLD transports are confined to one shared rank-R_OLD basis
Q (n x r): A_s = Q FA_s, P_s = Q FP_s with static factors. Per old source-layer the
dense legs are formed from the factors (2 n^2 r) and the two dslice contractions go
through the factors (2 n^2 r) instead of 4 dense n^3 units. One source joins the old
group per layer; at a join the basis is rebuilt by a one-pass randomized range finder on
the weighted leg Gram of old + joiner (sketch = a slice of the layer weight), the old
factors are rotated into the new basis and the Gram core S (r x r) is updated. The lean
reference (g_regen_probe.py "...+agerank:3:384") measures +8.4% raw on dumps 0/1 for a
~25% cost cut. Kill-switch: V21_NO_CONFINE=1 -> exact V20 op stream.

V20: exact, bit-identical values; only the client-side op stream changes. The nonlin
term products are written with out= into one persistent (T, n, n) buffer instead of
~30 fresh (n,n) multiplies + a ~50-matrix fnp.stack per layer (measured 4 ms + 3 ms of
residual per layer), and the range-finder sketches are one contiguous copy of the
weight slice per layer instead of strided views (strided qr/matmul input costs ~0.8 ms
each). Kill-switch: none needed (values identical); V19 stays the reference.

V19: exact cost cut, zero accuracy change. The final layer emits only the mean, whose
(1,) nonlin terms need var = diag(C_pre), D3 and the K4 diagonal -- never C_off, the
(2,1) dslice D21, nor the (1,1)/(2,1)/(2,2) term program. V18 computed all of them at
the last layer and discarded them: the two dense dslice contractions (2k n^3 units,
k = 15 live sources) plus the full C_pre sandwich (2 units, only its diagonal is used).
V19 transports only diag(C_pre) = colsum(w32 * (C @ w32)) (1 unit) and runs the source
machinery in D3-only mode at the last layer (transports + n^2 contractions, no D21).
Expected C: 0.509xB -> ~0.48xB. Kill-switch: V19_FULL_LAST=1 -> exact V18 op stream.
Below this line the V18 docstring follows unchanged.


V18: the V1.6 "K3-dslice feedback into the new-block factors" (dropped since V1.6 as
DS-shadowed) is worth -10% raw on the regen base (F69) and is rank <= 16 in D21.
Birth factors of the B1 hub pair become
  X1 = 3 a_b + Xt,   Xt = 1.5 d(w2) D21            ~ F1 R1,  F1 = d(w2) Q,   R1 = 1.5 Bm
  Y1 = a_b d(w2) + Yt, Yt = 0.5 d(w1) D21^T d(w3)  ~ F2 R2,  F2 = d(w1) Bm^T, R2 = 0.5 Q^T d(w3)
with D21 ~ Q Bm from a rank-R_FB randomized range finder (same recipe as Rres). The
identity-born thin legs F1/F2 travel in their own transported stack Zf (like Z, but never
entering the M-leg einsums); in the dslice contraction the new terms either add to the fused left factors
(LA += Xt*P*w2/3 + P*Yt; LP += A*Yt + Xt*A*w2/3 + Xt*Yt/3; D3 += rowsum(P*3*LPadd)) or
contract against the thin right factors ((AP + Xt*P/3) R2^T F2^T and
(AP*w2 + P*Yt)/3 R1^T F1^T). Extra n^3: ZERO; extra ~16 n^2 r per source-layer.
Lean reference: g_regen_probe.py "regen:c+no31+feedc+usec+fitfull+rres:16+sb2+nowk22
+d21r:16": 2.195e-8 / 2.091e-8 on P2 dumps 0/1 (V17 port: 2.51 / 2.25e-8).
Kill-switch: V18_NO_FB=1 -> exact V17 behaviour (zero thin legs).
Below this line the V17 docstring follows unchanged.


V17 (2026-09-02, F68): the augmented-K3 kappa4 channel (F51/F64) without any
per-source kappa4 memory. The kappa4 matrix core G of the aug chain is, to
R^2 ~0.9-0.97, diag(g) + lambda_l * C_off with ONE frozen scalar per layer
(LAM table, fitted offline on public MLPs). Everything the channel needs then
rides on objects V16b already carries:
  - transported diagonal dG_pre = (W*W) g + lambda (diag(C_pre) - (W*W) var)  [n^2]
  - use side: wk4_row = m dG, wk4_mat = (m/6)(dG_i + dG_j) (== V16b's vec+ww form),
    plus the (3,1) slice wk431 = (m/2) lambda C_pre_off feeding 4 extra n^2 terms
  - K4->K3 feed: hub pair X3 = d(w1) G_pre d(w1) = lambda A d(w1) + P d(w1^2 dG)
    (column scalings of the A/P legs), Y3 = (m/4) w2 1^T (rank-1, folds into the
    P-group + one rank-1 outer product), M_t1 = (m/4)(w2*dG)(w1^2)^T (rank-1 M-type
    -> one extra thin Z/L column).  Extra n^3 cost: ZERO.
Lean reference: scratch/lean_k3_aug.py regen chain (g_regen_probe.py
"regen:c+no31+feedc+usec"): 2.196e-8 / 2.161e-8 on P2 dumps 0/1 vs 4.1e-8 for V16b.
Below this line the V16b docstring follows unchanged.


V16b (2026-09-01, F63): exact cost restructuring of the source machinery.
  - The M leg (B3 = Sym(M x P x P)) is never transported: M_b = diag(S3c)
    + 3*S21^T and S21 = S_sep + Rres with S_sep = diag(e) C_off diag(w1) the
    exact leading (2,1) Wick term (a column-scaled copy of a_b) and Rres a
    D21-driven residual that is numerically low-rank (rank-64 keeps final MSE
    at parity, F63). Hence M = P diag(s) + 3 A diag(e) + 3 Z L^T with Z = P Rr
    a thin (n,r) transported leg and L static.
  - V27 (F80): every fresh (n,n)/(k,n,r) result of the layer body goes out= into a
    persistent named buffer (_Pool); thin stacks and shared-basis factors live in
    ping-pong slot buffers (no concatenates); slow einsum forms -> matmul(out=).
  - All dslice contractions collapse into TWO fused einsums per layer (right
    factors A and P) plus O(n^2 r) thin terms: per source-layer 2 dense
    transports + 2 dense contractions (was 3 + 4).
  - Rres range finder: randomized (Omega = a slice of the layer weight), one
    power iteration, flopscope-billed QR; ~4 n^2 r FLOPs per birth.

Port of the reference kprop k_max=3 SIMPLE factored algorithm (companion paper
arXiv:2605.05179) with cost engineering measured in F42-F45, plus two
accuracy riders measured in F47/F49 (combined 1.44x lower final MSE):
  - vec+ww (augmented-lite v0): the radial K4 channel keeps a per-neuron
    diagonal-content vector instead of a single scalar; born from K(4,) and
    the K(2,2) column sums, transported one layer by the true (W*W) row
    action in place of one average-metric cup. ~n^2/layer, cost-neutral.
  - online mean correction: per-layer ridge-fitted delta = feats @ beta
    (13 per-neuron features already computed in the loop; beta fitted
    offline on the public teacher-forced trajectory, embedded below).
    Applied inside the loop so it propagates.
Base cost engineering:
  - shared-P hub blocks per source layer: B1 = Sym(X1 x P x Y1) with X1 = 3*abar,
    B3 = Sym(M x P x P); identity-born legs all evolve by the same map
    leg <- W @ (w1 * leg), so one P per source serves every sub-block, and the
    wick scaling is folded into the next layer's weight (WD).
  - V1.6 ablations (each measured to cost <~1% final MSE): the B2 hub sub-block
    and the K3-dslice feedback into new-block factors are DS-shadowed at birth
    (their diagonal-slice content migrates into the exact M chain automatically);
    the WK22 struct term is dropped from Y1; nonlin terms pruned at relative
    contribution >= 1e-5 (62 of 90 kept).
  - residual-time engineering: per-source matrices held as batched (k,n,n)
    stacks; all wick vectors built at once in the unified form
    w(k,p) = const*sigma^e*(P1(a)phi + P2(a)Phi); nonlin terms evaluated as one
    stacked einsum per arity; dslice hub contractions fused into 3/4-operand
    einsums with no (k,n,n) intermediates; final layer computes only the mean.

Cost: ~1.83e12 flopscope FLOPs per MLP = 0.833x the 2^41 budget (deterministic,
data-independent). Local mini-dump accuracy: mse_final ~3.2e-8 test-set mean
(V1.6 base alone: ~6.2e-8; exact K3-simple reference: 4.8e-8 at an over-budget
2.24x; cov-prop baseline: 4.4e-6).
Prediction: per-layer post-ReLU means, shape (depth, width). No randomness.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# ---- embedded coefficient tables (generated by dump_k3_tables2.py) ----
# v2 K3-simple tables (dump_k3_tables2.py; pruned at rel>=1e-5)
WICK_PAIRS = [(0, 1),
 (0, 2),
 (0, 3),
 (0, 4),
 (1, 1),
 (1, 2),
 (2, 1),
 (2, 2),
 (3, 1),
 (3, 2),
 (3, 3),
 (3, 4),
 (4, 1),
 (4, 2),
 (4, 3),
 (4, 4),
 (5, 1),
 (5, 2),
 (6, 1),
 (6, 2),
 (7, 1)]

WICK_UNIFIED = [(1.0, 1, [1.0], [0.0, 1.0]),
 (1.0, 2, [0.0, 1.0], [1.0, 0.0, 1.0]),
 (1.0, 3, [2.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0]),
 (1.0, 4, [0.0, 5.0, 0.0, 1.0], [3.0, 0.0, 6.0, 0.0, 1.0]),
 (1.0, 0, [], [1.0]),
 (1.0, 1, [2.0], [0.0, 2.0]),
 (1.0, -1, [1.0], []),
 (2.0, 0, [], [1.0]),
 (1.0, -2, [-0.0, -1.0], []),
 (2.0, -1, [1.0], []),
 (6.0, 0, [], [1.0]),
 (1.0, 1, [24.0], [0.0, 24.0]),
 (1.0, -3, [-1.0, 0.0, 1.0], []),
 (2.0, -2, [-0.0, -1.0], []),
 (6.0, -1, [1.0], []),
 (24.0, 0, [], [1.0]),
 (1.0, -4, [-0.0, 3.0, -0.0, -1.0], []),
 (2.0, -3, [-1.0, 0.0, 1.0], []),
 (1.0, -5, [3.0, 0.0, -6.0, 0.0, 1.0], []),
 (2.0, -4, [-0.0, 3.0, -0.0, -1.0], []),
 (1.0, -6, [-0.0, -15.0, -0.0, 10.0, -0.0, -1.0], [])]

TERM_SPECS = {(1,): [('ones1', 'ones1', 0, None, 1.0),
        ('d3', 'ones1', 8, None, 0.16666666666666666),
        ('g4', 'ones1', 12, None, 0.041666666666666664),
        ('d3', 'd3', 18, None, 0.013888888888888888),
        ('d3', 'g4', 20, None, 0.006944444444444444)],
 (1, 1): [('c_off', 'ones2', 4, 4, 1.0),
          ('d21T', 'ones2', 4, 6, 1.0),
          ('c_off', 'c_off', 6, 6, 0.5),
          ('wk4m', 'ones2', 6, 6, 0.25),
          ('d3row', 'c_off', 4, 12, 0.3333333333333333),
          ('c_off', 'd21T', 6, 8, 1.0),
          ('d3row', 'd21T', 4, 16, 0.16666666666666666),
          ('g4row', 'c_off', 4, 16, 0.08333333333333333),
          ('d3row', 'd21', 6, 12, 0.16666666666666666),
          ('d21T', 'd21T', 6, 12, 0.25),
          ('c_off', 'wk4m', 8, 8, 0.25),
          ('d21T', 'd21', 8, 8, 0.25),
          ('g4row', 'd21T', 4, 18, 0.041666666666666664),
          ('d3row', 'wk4m', 6, 16, 0.08333333333333333),
          ('g4row', 'd21', 6, 16, 0.041666666666666664),
          ('d21T', 'wk4m', 8, 12, 0.25),
          ('g4row', 'wk4m', 6, 18, 0.020833333333333332)],
 (2,): [('ones1', 'ones1', 1, None, 1.0),
        ('d3', 'ones1', 9, None, 0.16666666666666666),
        ('g4', 'ones1', 13, None, 0.041666666666666664)],
 (2, 1): [('c_off', 'ones2', 5, 4, 1.0),
          ('d21T', 'ones2', 5, 6, 0.5),
          ('d21', 'ones2', 7, 4, 0.5),
          ('c_off', 'c_off', 7, 6, 0.5),
          ('wk4m', 'ones2', 7, 6, 0.25),
          ('d3row', 'c_off', 5, 12, 0.16666666666666666),
          ('c_off', 'd21T', 7, 8, 0.5),
          ('c_off', 'd21', 9, 6, 0.5),
          ('c_off', 'd3col', 13, 4, 0.16666666666666666),
          ('d3row', 'd21T', 5, 16, 0.08333333333333333),
          ('g4row', 'c_off', 5, 16, 0.041666666666666664),
          ('d3row', 'd21', 7, 12, 0.08333333333333333),
          ('d21T', 'd21T', 7, 12, 0.125),
          ('c_off', 'wk4m', 9, 8, 0.25),
          ('d21T', 'd21', 9, 8, 0.25),
          ('c_off', 'g4col', 17, 4, 0.041666666666666664),
          ('d21', 'd3col', 17, 4, 0.08333333333333333),
          ('g4row', 'd21T', 5, 18, 0.020833333333333332),
          ('g4row', 'd21', 7, 16, 0.020833333333333332),
          ('d21', 'g4col', 19, 4, 0.020833333333333332)],
 (2, 2): [('c_off', 'ones2', 5, 5, 1.0),
          ('d21T', 'ones2', 5, 7, 1.0),
          ('c_off', 'c_off', 7, 7, 0.5),
          ('wk4m', 'ones2', 7, 7, 0.25),
          ('d3row', 'c_off', 5, 13, 0.3333333333333333),
          ('c_off', 'd21T', 7, 9, 1.0),
          ('d3row', 'd21T', 5, 17, 0.16666666666666666),
          ('g4row', 'c_off', 5, 17, 0.08333333333333333),
          ('d21T', 'd21T', 7, 13, 0.25),
          ('c_off', 'wk4m', 9, 9, 0.25),
          ('g4row', 'd21T', 5, 19, 0.041666666666666664)],
 (3,): [('ones1', 'ones1', 2, None, 1.0),
        ('d3', 'ones1', 10, None, 0.16666666666666666),
        ('g4', 'ones1', 14, None, 0.041666666666666664)],
 (4,): [('ones1', 'ones1', 3, None, 1.0),
        ('d3', 'ones1', 11, None, 0.16666666666666666),
        ('g4', 'ones1', 15, None, 0.041666666666666664)]}

PK2K_TABLE = {(1,): [(((1,),), 1.0)],
 (1, 1): [(((1, 1),), 1.0)],
 (2,): [(((1,), (1,)), -1.0), (((2,),), 1.0)],
 (2, 1): [(((1, 0), (1, 1)), -2.0), (((2, 1),), 1.0)],
 (2, 2): [(((0, 1), (1, 0), (1, 1)), 4.0),
          (((0, 1), (2, 1)), -2.0),
          (((1, 0), (1, 2)), -2.0),
          (((1, 1), (1, 1)), -2.0),
          (((2, 2),), 1.0)],
 (3,): [(((1,), (1,), (1,)), 2.0), (((1,), (2,)), -3.0), (((3,),), 1.0)],
 (4,): [(((1,), (1,), (1,), (1,)), -6.0),
        (((1,), (1,), (2,)), 12.0),
        (((1,), (3,)), -4.0),
        (((2,), (2,)), -3.0),
        (((4,),), 1.0)]}

# Online mean-correction coefficients, (depth, n_features) = (16, 13); ridge-fitted
# offline (lam=1e-3) on the teacher-forced trajectory of 5 public v2-phase2 MLPs
# (k3_aug_corr.py, F47/F49). Feature order:
# 1, mu_pre, var_pre, sigma, alpha, |alpha|, phi, Phi, pred, D3, D21n, K4, sigma*phi
CORR_BETA = [
    [9.047924070408065e-06, 0.0, 3.1739399220558377e-06, -1.001890992962831e-05, 0.0, 0.0, 3.6095995822303605e-06, 4.523962037022356e-06, -4.1041365645909405e-06, 0.0, 0.0, 0.0, -4.104136563730213e-06],
    [-0.00021144661221045457, 2.9433696948174986e-06, 0.0003462675119684794, -8.927513941814157e-05, 3.2654450218955826e-06, -8.189452443772023e-07, 0.002237273864096366, 4.5955073102474984e-07, -1.1369974176193333e-05, -1.2997350491331133e-05, -2.537863394055266e-05, 0.0003733622375142006, -0.0019425202266636081],
    [-0.00020229139215584537, 1.000271315202233e-06, 0.0003238113820833415, 8.373236357018683e-06, 1.2098327507577007e-07, -1.9842791831952044e-06, 0.001976409037296056, 2.925565819012483e-06, -3.867841123257548e-06, 2.4290742674219072e-06, -2.4382096695485462e-05, 0.0009811554470205933, -0.0019828027194425195],
    [-0.00014337052165527383, -1.0930326089860847e-06, 0.0003217370290078323, -3.784401364288422e-05, 8.23278122333106e-06, 6.443269069146449e-06, 0.001643053753866545, 3.2898305742100015e-06, -1.6916246970775518e-05, -2.5002427677016677e-05, -1.5617249322051004e-05, 0.000721678068551959, -0.0018383483577257551],
    [-4.816989647892704e-05, 1.6744817911505923e-05, 0.0003924873520894163, -0.00019859318221267057, -6.206278737945743e-06, 9.114056522390382e-06, 0.0012907947282726098, 3.4188305934364313e-06, -2.0475334608078227e-05, 5.379830618136773e-05, -1.1405512979660586e-05, 0.0010070559186732254, -0.001569410267218006],
    [-0.00031901248473474055, 2.1505940899000224e-05, -1.030150132207361e-05, 0.0005192818210010263, 2.866659642444627e-06, 2.0444859277633333e-05, 0.0013701201322794546, -3.0218634832627234e-06, -4.934377232712823e-05, 1.7240818176102207e-05, -1.4371192314941062e-05, 0.00016398142695286006, -0.0017929463544335428],
    [-0.0002712334021131924, 1.8241572117937817e-05, -0.00013911689190045331, 0.0005261870290256655, 2.721277590115469e-06, 1.601949313013222e-05, 0.0009039266887176779, -4.218753601219429e-06, -4.286494834844438e-05, 1.5737791156115456e-05, -6.005811397428032e-06, -3.241204641548729e-05, -0.0012772016685319988],
    [-0.0004219412714856214, 6.373475413982465e-05, -0.00036133128747675757, 0.0009193783536355404, -2.8904635275133456e-06, 3.974378631151497e-05, 0.00111444116240623, 1.2094393021420443e-06, -0.00011656360320196783, -0.00011595929045631306, -6.306817678565869e-06, -0.00012411849303491868, -0.0016450793958699928],
    [-0.00015002735693833415, 5.075834607818153e-05, 3.9176599328349595e-05, 0.00019779502913078056, -2.377961035353491e-06, 2.8156681168687034e-05, 0.0008465817108788675, -4.509401316536409e-06, -8.69488144455512e-05, -0.00025452830562789567, 2.898123372324238e-06, -0.00012084244449459633, -0.0013171971447038728],
    [-0.00015433261058161755, 2.2045131669193608e-05, -0.00018095909302443154, 0.0003612896887359225, -9.647765558094461e-07, 1.107795528889642e-05, 0.0005376568748808842, -4.627353400197264e-06, -3.421029737450039e-05, -0.00026235837339376396, 1.4327695442654706e-06, -0.0003921000768991082, -0.0008764294890685028],
    [-0.00034017392198268175, 5.994785125152261e-05, -0.0006748381529166745, 0.0009836290520334938, -4.728579817951294e-06, 2.792551870747016e-05, 0.0005494983315233547, 4.501360307797776e-06, -9.838689354321231e-05, -0.0003717415157969712, -1.0492386756090072e-07, 0.00020270276477037187, -0.0009020111820971813],
    [-0.00017954578643124088, 6.986586132854883e-05, -3.412491626004671e-05, 0.0003325337792994089, -3.4046841229619147e-06, 3.377768794565494e-05, 0.0007196961290187987, 8.321635885540252e-06, -0.00012752078483613784, -0.0001868492414749541, 2.994579579153689e-06, -0.0005361471841543824, -0.0013000625602190119],
    [-8.113708371395053e-05, 6.626203708650396e-05, 0.0004628741284420011, -5.3587999495247974e-05, 2.7733824677303546e-06, 3.73131112613667e-05, 0.00046273970909515314, -6.139549868152084e-06, -0.0001416798324581666, 4.513760948578568e-05, -3.586201414373387e-06, -0.00010788999438329517, -0.0007396573644445799],
    [-0.00020274495747313578, 2.852020705959683e-05, -0.0006185058436784183, 0.0006751480499007955, -1.2867282684497467e-06, 1.4429521889536009e-05, 0.00021253867100942168, 2.0920107425898062e-06, -5.002242284889489e-05, -0.00017804555318281108, 3.1901969570195794e-06, -0.0003670581320939956, -0.00018841918390899277],
    [-0.00022186393313835127, 2.480935092758624e-05, -0.0009038276703732042, 0.000856110133719225, -4.0912078616958074e-06, 8.431289115779983e-06, -5.295935263051177e-05, -2.3757504692123435e-06, -2.2757691803695326e-05, -0.0005974422655620646, 2.162145197622491e-06, 0.0003631271990780135, 0.00040564733239909894],
    [-0.00013875239635447486, 2.8652351444287666e-05, -0.0002455812521803567, 0.00040608571696212935, 3.5408135795708984e-06, 1.9019144517277667e-05, 0.0001276467621706209, -6.325209118378654e-06, -7.523684069227385e-05, 0.000432314287562562, 0.0, -0.0001947182798308246, -1.7453628718729287e-05],
]

# ---- end embedded tables ----

# Regenerated kappa4 off-diagonal coefficient per layer, fitted at the PRE-activation
# (transported G_pre_off = LAM[l] * C_pre_off after layer l's ReLU); LS fits on the dense
# aug chain, mean over public P2 dumps 0-7 (runs/lam_prefit_d0-7.log, V22). The older
# post-ReLU table (runs/g_regen_10.log) differs mainly at layer 0 (1.95e-3 -> 4.99e-3).
# Suite shape only (gated like CORR_BETA).
LAM = [4.9895e-03, 8.0876e-03, 9.8291e-03, 1.0549e-02, 1.0851e-02, 1.0828e-02, 1.0589e-02, 1.0048e-02, 9.6483e-03, 9.1720e-03, 8.7730e-03, 8.3938e-03, 8.0555e-03, 7.6770e-03, 7.2588e-03, 7.2588e-03]
METRIC_C = 2.0
# debug kill-switches (parity_v17.py): set from the environment before import
import os as _os
import gc as _gc
LT_RES = []   # f0 21.10.2026: dump of the ALG058 nn_asset carried residual res_e per layer (li, res_e) after its update
LT_FE = ["mu_pre", "var", "alpha", "phi", "Phi", "D3", "g4row", "d21_abs", "d21_sq", "d21T_abs",
         "d21_colsum", "d21_rowsum", "coff_sq", "coff_sum", "pred", "K2v", "K3v", "K4v", "k22row",
         "k21row", "k21col", "e_b", "w1", "w2", "s3c", "g_post", "lam"]
LT_XF = ["var", "phi", "Phi", "K3v", "K4v", "pred", "g_post", "e_b"]
LT_F = len(LT_FE) + 2 * len(LT_XF)
# LT2 dump extras (appended after the transported columns in the dump only)
LT_FE2 = ["coff_abs", "cn2", "delta", "k22abs", "k22sq", "k21sq", "k11sq", "k11sum"]
LT_FILE = "lt_weights.npz"
LT2_FILE = "lt2_weights.npz"
LT2_FAST = True   # c2 21.10.2026: _lt2_layer_fast (same values within fp32, fewer fnp calls)
# LT2 weights embedded by lab/lt2_embed.py (name -> (shape, array typecode, base64 little-endian)): the
# grader's setup cannot read a shipped data file, so the weights travel inside this file.
import array as _array
import base64 as _b64
_LT2_BLOB = {
    "W1": ((78, 128), "f", "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"),
    "b1": ((16, 128), "f", "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"),
    "usc": ((16, 3), "f", "/iU9Rv4lPUYAAAAAM/j1RjP49UYfwQtNUhcAR1IXAEcCVWxO5sfoRubH6EaoLoBOMKvRRjCr0UbtqlNOaVy9RmlcvUb3uCtO6q6xRuqusUaVEQxO5ZCmRuWQpkavpvZNOM2cRjjNnEZ1wNhN+gqVRvoKlUaBFcBNe2iORntojkaQi61NInWJRiJ1iUYmcJ5NR6yERkeshEYonZNNA/N/RgPzf0Y6hIlNgTx5RoE8eUYG5n9NKyZpRismaUbBpnJN"),
    "Wo": ((128, 17), "f", "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"),
    "bo": ((17,), "f", "8dLvu8SJEL32smS9yY3cvPFJPr2qrRi9pU2aPKF/FjwP758462PkPaClKbzRMxo+/fiVvNoSMj0oFLi9/GqGvIrLS70="),
    "es": ((16,), "f", "WD2tOEI4BThl0f83gcQMON9IHDiPCy04BWs4OBK6RDg4+lA4L9tbOGcZZjjmYm44t/t2OH8GgDg+eYM4louMOA=="),
    "mz": ((16, 43), "f", "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"),
    "iz": ((16, 43), "f", "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"),
    "m2": ((16, 54), "f", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    "i2": ((16, 54), "f", "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"),
    "flags": ((7,), "q", "GwAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
    "Wm0": ((128, 128), "f", "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"),
    "bm0": ((128,), "f", "pOPovVjgM7x0ReM8ILiqvTGQh70TnpW9G5RAvPX9ujxSdvU8oDoYPNxDGT16/MU8ee2EvWwRjL0taKo912/QurHxlrwv8qe9Ce2JvdMOHL22ZpS9Slu/veHNHj3oZx09Pi3qPDm4pD1jGO4807uvvRBvtLzCT5q7WByrvYWiDr1xQrm9ueS4vbgvkL1iwB09IAQ4PYTb5L1ZCgi9JF7yvQvIWr162xi8w3AyPafnVjxeBBe97LwpO+UDJT2TZaU9pDqSPUj5C70ikxO9GQa1PAV0lr14aau8GFEQPK4hDz0qY1U9gfrOPLJKBL1rx1y9pwuTu+Dccrwb8aw8ldoHvgVWm7069PS8TKIyvX9KUDy7+hK8r6U3vYiwy7y1fsm9spFLvame/rzFJVA9pvxcPc+8QD2MG0+98R5avakbsL2ynei83gMFvXawyjsN0BU9ImtzO2IoZ7240oq9XXvVvRoMh7wHXq29IHHhvbI7b71KxSA9TYLWvY+oJb0AfNS8WfiYPB5bOD0GOwa7i34VvSRE4rwGJ5i9iqupvYEV2juAdjY8KstavT/8qb1xW3u9hqTSO6hSpzz0h829uyEFvPme070oi7S9phpBvS6Az7zoU449dLQKvfrJ4r0pM5U9egrkPJlXjr3KZUC9+CymPQg2gzzN0ec86FmnuuXHTbw="),
    "Wm1": ((128, 128), "f", "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"),
    "bm1": ((128,), "f", "hmj7vCGb3bxqOZQ9YgZ/PX2GZzyRS1i958NAvINhv7qmnaU824DePMinmb0muw29KtqZvJ0wGD3l90k8nJoJvaLbOjsAEk69z8xTvY+WTj3nu5s9e5ccPRB6qL3O7Lm8nx0DPZZFPD2GanY81fwovXmUCby+axc9403iPCLIij1M7428ftYZvUgM4zz33G09/KI0PQ/tEz3Nrqs9E9lpvZ/Kjr3s4XW9AUUmuiaTDrxOgbc8qLw7PbEuLDx2TN28y2VhvUI2jDwK+M+8bSWcPGILMr2uOua8up3UvFePOb3t4hQ8EBicvcB8Wr1eNrC822qsvaDNwDqK7jS9IORwvAEwLTwhn729rXIvPQj7IDx6Dou9v/KHvDp4NL0lt9O81iHvvGY4C739AA89gFKYuRiEvTsmhOy8moVrvXHi27x6Vku9to3gu2cVWTlDRfo7aNMmPWKDPTznLKY82jN2vd3N6DvyijQ9C5SzvaSORjvswUg9DlLpvOvi1bu1qNS8Ov21PBsVkb2UD0k9MzCVvbpfDb0fQDs8nOqju043QT2j4ig99cBePZwBNrwFwHu72Zq0vMbcGD1e/0u9EK/3vGbid72k4yw9n9CVvZSqJzrVoJI9wYxXOivBr7trgDy8bSntvHDZrTuC/Jy9KOEDPdzol70+dlM9/WDVPMNfND0="),
}


def _lt2_fast_prep(lt, d, fl, nl):
    """c2 21.10.2026: setup-time tables for Estimator._lt2_layer_fast: new column order of x =
    [own Fo | W@T1 | WW@T2 | Wu, Phi Wu (uin) | ones], the W1 rows permuted to it, the sign-log
    column mask and the standardisation (folded into W1 and a bias row unless zo is carried)."""
    f32 = fnp.float32
    Fo, xall, D, uin, embw = fl[0], bool(fl[1]), fl[2], bool(fl[3]), len(fl) > 6 and bool(fl[6])
    nx = len(LT_XF)
    a = Fo + 2 * nx
    b = a + (2 * Fo if xall else 0)
    c = b + (D if embw else 2 * D) if D else b
    perm, kind = [], []   # kind: (type, index into its mean/inv vector)
    perm += list(range(Fo)); kind += [("z", j) for j in range(Fo)]
    perm += [Fo + j for j in range(nx)]; kind += [("v", j) for j in range(nx)]
    if xall:
        perm += [a + j for j in range(Fo)]; kind += [("2", j) for j in range(Fo)]
    if D:
        perm += [b + j for j in range(D)]; kind += [("r", 0)] * D
    perm += [Fo + nx + j for j in range(nx)]; kind += [("v", nx + j) for j in range(nx)]
    if xall:
        perm += [a + Fo + j for j in range(Fo)]; kind += [("2", Fo + j) for j in range(Fo)]
    if D and not embw:
        perm += [b + D + j for j in range(D)]; kind += [("r", 0)] * D
    if uin:
        perm += [c + 2, c, c + 1]; kind += [("r", 0)] * 3
    fin = int(d["W1"].shape[0])
    assert sorted(perm) == list(range(fin)), (len(perm), fin)
    mask = [k[0] in ("z", "v") for k in kind] + [False]
    fold = not xall
    W1F, mL, iL, mL0, iL0 = [], [], [], [], []
    for i in range(nl):
        mz = [float(x) for x in d["mz"][i].tolist()]; iz = [float(x) for x in d["iz"][i].tolist()]
        m2 = [float(x) for x in d["m2"][i].tolist()]; i2 = [float(x) for x in d["i2"][i].tolist()]
        mv, iv = [], []
        for t, j in kind:
            if t == "z":
                mv.append(mz[j]); iv.append(iz[j])
            elif t == "v":
                mv.append(mz[Fo + j]); iv.append(iz[Fo + j])
            elif t == "2":
                mv.append(m2[j]); iv.append(i2[j])
            else:
                mv.append(0.0); iv.append(1.0)
        Wp = fnp.take(lt["W1L"][i], fnp.array(perm), axis=0).astype(fnp.float64)
        b1 = lt["b1L"][i].astype(fnp.float64)
        if fold:
            ivA = fnp.array(iv, dtype=fnp.float64); mvA = fnp.array(mv, dtype=fnp.float64)
            Wf = Wp * ivA[:, None]
            bf = b1 - (mvA * ivA) @ Wp
            W1F.append(fnp.concatenate([Wf, bf[None, :]], axis=0).astype(f32))
        else:
            W1F.append(fnp.concatenate([Wp, b1[None, :]], axis=0).astype(f32))
            mL.append(fnp.array(mv + [0.0], dtype=f32)); iL.append(fnp.array(iv + [1.0], dtype=f32))
            # layer 0 of the reference feeds zeros (not standardised zeros) in the zo columns
            mL0.append(fnp.array([0.0 if t == "2" else m for (t, _), m in zip(kind, mv)] + [0.0], dtype=f32))
            iL0.append(fnp.array([1.0 if t == "2" else v for (t, _), v in zip(kind, iv)] + [1.0], dtype=f32))
    lt["fast"] = {"names": LT_FE + (LT_FE2 if Fo > len(LT_FE) else []), "nrest": fin - Fo,
                  "mask": fnp.array(mask), "fold": fold, "W1L": W1F, "mL": mL, "iL": iL,
                  "mL0": mL0, "iL0": iL0}


def _lt2_embedded():
    return {k: fnp.array(_array.array(tc, _b64.b64decode(b)).tolist(),
                         dtype=fnp.float32 if tc == "f" else fnp.int64).reshape(sh)
            for k, (sh, tc, b) in _LT2_BLOB.items()}
LT_OFF = _os.environ.get("LT_OFF", "0") == "1"
LT_DUMP = []   # dump harness (LT_DUMP=1): per layer (X, Phi) in natural neuron order
LT_DUMP_ON = _os.environ.get("LT_DUMP", "0") == "1"
NO_FEED = _os.environ.get("V17_NO_FEED", "0") == "1"
NO_WK431 = _os.environ.get("V17_NO_WK431", "0") == "1"
NO_REGEN = _os.environ.get("V17_NO_REGEN", "0") == "1"
NO_FB = _os.environ.get("V18_NO_FB", "0") == "1"  # V18: D21 feedback thin legs off
FULL_LAST = _os.environ.get("V19_FULL_LAST", "0") == "1"  # V19: 1 -> V18 op stream
NO_CONFINE = _os.environ.get("V21_NO_CONFINE", "0") == "1"  # V21: 1 -> V20 op stream
QPASS = int(_os.environ.get("V21_QPASS", "1"))  # V21: subspace-iteration passes of the basis range finder
NO_SRC_LAST = _os.environ.get("V19_NO_SRC_LAST", "0") == "1"  # V19 probe: D3(last) := 0, skip source transports
# The V16b CORR_BETA mean rider was ridge-fitted on V16b's OWN trajectory and is
# anti-correlated with V16b's base error; on the regen base it HURTS (dump0: 3.57e-8
# with vs 2.51e-8 without). Off by default until refitted on the V17 trajectory.
NO_CORR = _os.environ.get("V17_NO_CORR", "0") == "1"
# V25: table scale 0.95 (8-dump scan 0.9/0.92/0.95/1.0 -> 2.2752/2.2727/2.2704/2.2782e-8)
LAM = [c * float(_os.environ.get("V17_LAM_SCALE", "0.95")) for c in LAM]
# 8.g: inclinacion de la tabla con la profundidad (0 = apagado): LAM[l] *= 1 + t * max(0, (l - 3) / 12) ** 1.5
LAM_TILT = float(_os.environ.get("X_LAM_TILT", "-0.1"))
LAM = [c * (1.0 + LAM_TILT * (max(0.0, (i - 3) / 12.0) ** 1.5)) for i, c in enumerate(LAM)]
# V25: reference ratio mean(dG)/mean(var) at layer l+1 (8-dump mean, scratch/lam_obs_d0-7.npz)
# and the log-log exponent of the adaptive rule (pooled fit 1.08; 1.0 shipped).
REF_R = [6.58815e-03, 8.18414e-03, 8.53136e-03, 8.38859e-03, 8.10153e-03, 7.74287e-03,
         7.35951e-03, 6.91093e-03, 6.55752e-03, 6.17892e-03, 5.83953e-03, 5.56824e-03,
         5.32459e-03, 5.05165e-03, 4.77181e-03]
BETA = float(_os.environ.get("V25_BETA", "1.0"))
AUD_ZS = _os.environ.get("AUD_ZS", "1") == "1"
AUD_IND2 = _os.environ.get("AUD_IND2", "1") == "1"
KVEC = _os.environ.get("X_KVEC", "1") == "1"   # K de las partes vectoriales (1,)..(4,) por monomios + un matmul (9 operaciones en vez de 29 por capa)
PWR = _os.environ.get("X_PWR", "1") == "1"   # potencias de alpha, 1/sigma y sigma con fnp.power (5 operaciones en vez de 15 por capa)
FOLD2 = _os.environ.get("X_FOLD2", "1") == "1"   # constantes -2 y 4 de K21/K22 dentro del vector (potencias de 2: mismo resultado exacto) y sin zero_diag redundantes
RELOJ = _os.environ.get("X_RELOJ", "1") == "1"   # 8.a-sexvicies: einsum 'tij,ti,tj->ij' como multiply*multiply*sum (misma aritmetica), en el sitio
DEBUG = []  # parity_v17.py: per-layer dict of diagnostics when V17_DEBUG=1

# ---- 8.a-quater (02/10/2026): tablas de cierre reajustadas en las redes 0-69 del Mini --------------------------------
#   Por defecto: correccion de medias de 87 rasgos (familia sigma*phi*He_k(alpha), k = 0..6, por moduladores, mas
#   tres sumas por fila de pares, n^2 cada una) y LAM reescalada. Ajustadas sobre la ruta oficial (retirada fisica
#   de neuronas, volcado con V26_STRASSEN=1 en el evaluador de la GPU).
#   X_TAB_OLD=1       : vuelve a las tablas de v29 (CORR_BETA de 13 rasgos y LAM de 8 volcados), bit a bit
#   X_TAB_SIN_PARES=1 : correccion de 59 rasgos sin las sumas por fila (solo operaciones de n por rasgo)
#   X_TAB_DUMP=1      : diagnostico; guarda por capa los rasgos en _TAB_REC (lo lee el conductor tablas/extrae.py)
#   X_TAB_FORCE=1     : diagnostico; fuerza la media propagada a la verdadera (_TAB_TRUTH, la pone el conductor)
TAB_OLD = _os.environ.get("X_TAB_OLD", "0") == "1"
TAB_SIN_PARES = _os.environ.get("X_TAB_SIN_PARES", "0") == "1"
TAB_DUMP = _os.environ.get("X_TAB_DUMP", "0") == "1"
TAB_FORCE = _os.environ.get("X_TAB_FORCE", "0") == "1"
_TAB_TRUTH = None
_TAB_REC = None
CORR_BETA_OLD = CORR_BETA
LAM_OLD = list(LAM)
# rasgos de la correccion nueva (nombres de _tab_feats) y su tabla (16, len(CORR_FEATS)); None = la de v29
CORR_FEATS = [
    '1',
    'mu',
    'var',
    'sigma',
    'alpha',
    '|alpha|',
    'phi',
    'Phi',
    'pred',
    'D3',
    'D21n',
    'K4',
    'sigma*phi',
    'pk2',
    'pk3',
    'pk4',
    'g4',
    'h0',
    'h0*s3',
    'h0*s4',
    'h0*s33',
    'h0*dn',
    'h0*v',
    'h0*ds',
    'h0*cr',
    'h0*c2',
    'h0*k4',
    'h1',
    'h1*s3',
    'h1*s4',
    'h1*s33',
    'h1*dn',
    'h1*v',
    'h1*ds',
    'h1*cr',
    'h1*c2',
    'h1*k4',
    'h2',
    'h2*s3',
    'h2*s4',
    'h2*s33',
    'h2*dn',
    'h2*v',
    'h2*ds',
    'h2*cr',
    'h2*c2',
    'h2*k4',
    'h3',
    'h3*s3',
    'h3*s4',
    'h3*s33',
    'h3*dn',
    'h3*v',
    'h3*ds',
    'h3*cr',
    'h3*c2',
    'h3*k4',
    'h4',
    'h4*s3',
    'h4*s4',
    'h4*s33',
    'h4*dn',
    'h4*v',
    'h4*ds',
    'h4*cr',
    'h4*c2',
    'h4*k4',
    'h5',
    'h5*s3',
    'h5*s4',
    'h5*s33',
    'h5*dn',
    'h5*v',
    'h5*ds',
    'h5*cr',
    'h5*c2',
    'h5*k4',
    'h6',
    'h6*s3',
    'h6*s4',
    'h6*s33',
    'h6*dn',
    'h6*v',
    'h6*ds',
    'h6*cr',
    'h6*c2',
    'h6*k4',
]
# 87 rasgos: ridge por capa (columnas estandarizadas, lambda por capa por validacion cruzada en 5 bloques
# de redes) del error fresco con la media de entrada forzada a la verdadera, redes 0-69 del Mini
# (tablas/ajusta.py sobre tablas/fis/v_forz; fichero beta_ext2_fis.json)
CORR_BETA_NEW = [
    [-4.261144979758156e-06, 0.0, -6.705452350262268e-09, 2.828524302311513e-07, 0.0, 0.0, 2.701808027101265e-08, 0.0, 7.086394612238364e-07, 0.0, 0.0, 0.0, 7.090058991638483e-07, -1.3011288280913495e-08, -3.293404530048712e-08, -8.074884473352455e-09, 0.0, 7.090058991637967e-07, 0.0, 0.0, 0.0, 0.0, -6.592306252065919e-08, 0.0, -6.234180002616521e-08, 8.221886169880119e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7.09005899163818e-07, 0.0, 0.0, 0.0, 0.0, 6.592306252066248e-08, 0.0, 6.234180002616147e-08, -8.221886169880079e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3633529972183288e-07, 0.0, 0.0, 0.0, 0.0, -2.197435417355468e-08, 0.0, -2.0780600008726875e-08, 2.740628723297785e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.726705994416413e-08, 0.0, 0.0, 0.0, 0.0, 4.394870834729561e-09, 0.0, 4.1561200017441255e-09, -5.4812574465852285e-08, 0.0],
    [-0.0005497861636007268, -4.4881686282889654e-05, -0.0003643196465626319, 0.0007762243374728116, -2.5297622310110068e-05, 2.7725997929742393e-07, 0.001237014043879092, 0.0002663585047750696, -8.806084269660673e-05, -7.101286732002827e-05, 5.481592123363902e-06, -0.024530047314287356, -0.0007705076775280276, 2.0669242197348773e-05, 1.4395261579756989e-05, -2.3258057769854424e-06, -0.00222691089385739, -0.00077050767268164, 0.0006388433536211053, 0.02042263851428284, 0.004414848805241711, -3.9842041772670754e-05, 0.00034190279084273943, 8.166681394330799e-06, 3.2429392062005436e-07, 6.673896187508641e-06, 0.11852331648584707, 9.198566723981004e-06, -0.00029574632724155124, -0.002033551737729275, 0.0107262885816427, -2.953937359528372e-06, -8.717203005890508e-05, -1.7839285080696256e-05, -5.783465323592171e-07, 2.2539818135471417e-05, 0.011208646464469703, -0.0003472654070025407, 0.0010279885883256756, 0.0027043777298973648, 0.03530706740460044, 0.00010221147257283557, -5.521141282543293e-05, 3.5522028710967634e-05, 1.2419742979611297e-06, 4.04212349286433e-06, 0.0883748354894245, 3.940759723220676e-05, -1.1648360776405754e-05, -0.001953801448543989, 0.00955803635149247, -4.954221978235622e-06, -8.070552566344059e-05, -1.665716987507499e-05, -5.896230481720031e-07, 2.032671513085648e-05, 0.015514454620405902, 2.7948185502725452e-05, 0.0002944738396602816, 0.004107213637611821, 0.002380250930803951, -6.158716755539017e-06, -5.2021567557092106e-05, 1.647539321559767e-05, 5.04406509477808e-07, 1.41039218606955e-05, 0.016822305603382336, 4.144618969936626e-06, 6.138934400826116e-05, -0.0003669247846732316, 0.0017017611001475708, -1.1871938877243163e-06, -9.905390134267264e-06, -2.394256699791207e-06, -1.320495601282792e-07, 2.7170238421888764e-06, 0.0028043507096266727, 1.933411901996806e-06, 2.049269275523545e-05, 0.00016194717760374794, 0.00028626374346339944, 1.7320766645923538e-07, -3.7159475630447802e-06, 1.3823025533692906e-06, 4.0376473147649654e-08, 1.3410594425622745e-06, 0.00035968938011942894],
    [-0.00017594914814554667, -2.963902335167814e-05, -4.495294975610801e-05, 0.00010506231952267301, -9.770806654427206e-06, -2.1268750233554477e-06, 0.000275804935678333, -6.998093882325058e-06, 8.764352440658797e-05, 0.0001781060267472959, 3.5632836663700806e-06, 0.006414630241164666, -3.8791627048282996e-05, 1.099168965285034e-05, -5.33217406051556e-06, 4.2835622750342676e-07, -0.003031073061861688, -3.8791626900185506e-05, -0.0009292468340931044, 0.019030571046640188, -0.004159605275319918, -3.8179946422317335e-05, 0.00012491701972493335, -2.6873292259891076e-06, 2.0239900030112144e-07, -2.0762150504781837e-05, -0.009766972853258981, 5.405060546897761e-07, -0.0011513237978630598, 0.004354612460571829, -0.005519802226142374, -1.3255406689624635e-05, 1.3800030560154932e-05, -4.038376656136299e-06, -1.267334174174609e-07, 3.8204455086486995e-06, -0.015775281096798974, 7.355355946081126e-05, -0.0016751922005410662, -0.0061111017759162105, 0.025980914821937186, 0.00015245521179810102, -4.351069882612098e-05, -2.5563833300008405e-05, 1.3776092401688174e-07, -7.629151650721768e-05, -0.08051939178236894, 1.7795039697838483e-05, -0.0011305217137598316, 0.003077230842437137, 0.004298937091026261, -1.3638002725012177e-05, 1.600751239401979e-05, -7.170220298496841e-06, -1.520962046381988e-07, 1.4371173629009473e-06, -0.007099049963614566, -2.4685052113382855e-05, -0.0006516538266154748, 0.002610510542372452, -0.008044489153663284, 2.710638547478676e-06, -4.8537761988520316e-06, -1.3838001105891199e-05, 2.05782247183418e-08, 8.750522182077617e-06, -0.000756212143115971, 4.889309410718183e-06, -4.592217711439509e-05, 0.00043718394621302477, 0.0011068622981941021, -2.25878391066945e-06, 2.7756664961039076e-06, -1.971540821118129e-06, -4.857496138083993e-08, 9.615797949187269e-08, -0.0009027184975914367, -1.1913629007927466e-06, -5.2100129843310385e-05, 5.494698402629759e-05, -0.00110718290691873, 7.884730580118566e-07, -3.060163756197375e-06, -1.4893670901024905e-06, 7.37943493766976e-11, 7.217274425610409e-07, 0.00010916234375577382],
    [-5.604899177005707e-05, -1.2217563017151057e-05, -4.6371147552033356e-05, 0.0001199775006791426, 4.695188695233916e-06, -4.666455148101125e-06, 0.0001189108861396539, 1.3842601972971323e-05, -4.5882302273281245e-06, 0.00019035874540310852, -4.286678911740285e-07, 0.004850939400379737, -0.00018728963740639374, 4.760542210114813e-06, -9.774830634457827e-07, 8.037448959120479e-08, -0.001393857484307504, -0.0001872896374242079, -0.0008309450163263044, 0.005119433349294271, -0.009994695062878449, -1.1776454135534671e-05, 0.0001987200759623972, -2.122090718650127e-06, 3.169547451261927e-07, -1.7395414853512858e-06, -0.004651069932875112, 0.0001622039260376485, -0.0012634318687121987, -0.0006774457346504798, 0.003709777508164308, -2.1764758286660565e-05, 5.6259498992621045e-05, 2.174124253595067e-06, -7.47220272415496e-07, -1.2501011444742953e-06, -0.0026754113188373636, -4.302697199256553e-05, -0.0009294622780165097, -0.02299152735032566, 0.025221072182780194, 0.0001932403074679779, -0.00010393667772657385, -7.711113005115434e-06, 1.9229867430291806e-06, -7.012181985762642e-05, -0.12015501373850224, 0.00018457258425520894, -0.0010291283741538358, -0.0010022003502574875, 0.0030688480641071817, -2.3526970512176576e-05, 4.96402189650071e-05, 3.3419730907750423e-06, -1.1371813219717878e-06, 3.3332212219579493e-06, -0.011145189235283199, -0.00014530807343632579, -8.450146589370706e-05, -0.00013923722387018054, -0.01226492963427785, 1.283564111888661e-05, 6.287555917085342e-05, 1.372343469698066e-06, 9.742001605892722e-07, 1.773797353750493e-05, -0.013010081304640833, 2.1298343352917023e-05, 4.8957272548655464e-05, 3.0466105396955436e-06, -0.0002733393630732134, -3.7161063257071597e-06, 1.09745957568765e-05, 1.1804640611920615e-06, -3.0308049497663267e-07, 1.0327375548425326e-06, -0.002149937843555406, -1.7143990652553884e-05, 1.1167774361340522e-05, 1.846164149027953e-05, -0.0011123890437063312, 8.902551142289634e-07, 7.282783514235137e-06, 3.562871885259161e-07, 1.1342011899387283e-07, 2.640905162024085e-06, -0.000892583793198049],
    [0.00023241381397323712, 3.4061823712327105e-05, 0.0003015328731139188, -0.0003708447573194124, 7.621476223480609e-06, 2.650584036049968e-06, -0.0004447478434907869, -0.00013270128763776565, 1.116239600894606e-05, 4.8482601140575436e-05, -6.450080295268105e-06, 0.002822861932382697, 0.00023801763210519092, -2.985528021683263e-05, 4.679990712207844e-06, -1.9097991647900697e-07, 0.0011089173436478967, 0.00023801763282501476, -0.001317968354067789, -0.004250294344242332, -0.016365714518901262, 2.5679232238070852e-05, -0.0005653916113097369, 2.135649454226736e-07, 9.949809149850314e-07, 7.74634916636966e-06, 0.00594325387500947, 0.0004524612296640117, 0.0005441702942267756, 0.006301812525761707, 0.03908225354821201, -4.9264630753351155e-05, 2.8044329281403005e-05, -3.1982465166477535e-06, -1.6113301030383322e-06, -1.8378340053321293e-05, 0.009418522785381255, 0.00011766634007068974, -0.00401249364498751, -0.041836465471493424, 0.013311369747623553, 0.00024530172668017455, -0.0009051568661153728, 1.3458421425527e-05, 4.470092812057403e-06, -7.435782162475657e-05, -0.08554668389095571, 0.0001683639311847755, 0.000676714258789145, 0.006934790724748085, 0.02970258528064486, -4.3031574837750834e-05, 0.00019340865302665147, 1.180649519444414e-06, -1.3668744725516907e-06, -1.8106385387697648e-05, 0.0012107688812721287, -0.0001843794609179818, -0.00119818309898684, -0.0022494000938435028, -0.014331897119241578, 2.8772341313211345e-05, -9.110363591029667e-05, 4.1448470823542605e-06, 2.849866756127054e-06, 3.5584724506799516e-06, -0.004031363694297382, -3.288199454687238e-06, 0.00031857804766299136, 0.001299535244138946, 0.002550391607888536, -5.625124245911916e-06, 4.409643225715909e-05, 1.3679211969561576e-06, -2.9824637179426224e-07, -2.2158728368768934e-06, -0.00037666302785920356, -1.5570294348535487e-05, -6.0907345169109574e-05, -6.274677686526782e-06, -0.0008942344349462155, 1.7331919168388613e-06, -5.447715599856778e-06, 4.3564737706037516e-07, 4.1281566060948213e-07, 7.432259600854147e-08, 3.7155847191420404e-06],
    [6.161773436978822e-05, 9.26845516753555e-06, 3.00499899627794e-05, -4.161495679528895e-05, 1.2560803351507264e-06, -4.382805143356612e-06, -0.0004314752320820379, 0.0002338523242964594, -0.00030098827214167717, 6.132721734411631e-05, -3.0129908689199935e-06, 0.002741842062828708, 5.9290088049473464e-05, 0.00011096092958861102, -1.779271702617586e-05, 1.0273755206923914e-06, 0.0009745222180501069, 5.9290093439384536e-05, -0.000681243006180898, -0.004685673066327666, -0.00895365959530765, -6.718932591005623e-07, 9.018267768716748e-05, 1.143556358856131e-06, -1.756393926665925e-08, 8.03585620679296e-06, 0.0044491465094052335, -0.00014807239603517345, -0.0007896241069582495, 2.859589121154541e-05, 0.013808245262339697, -2.833582605295999e-05, 0.0005441510286108297, -1.1118921660880851e-06, 7.997686726377704e-07, -1.1052317754263745e-05, 0.0022107986154840014, 0.00025079316735208626, -0.0021603839067422075, -0.04005973759927582, 0.03341957767490586, 0.00022087772815688314, -0.0006209146615341062, -7.130962267250479e-06, -7.863985339658433e-07, -8.150839709632592e-05, -0.16025254792180368, 0.0001264263069802946, -0.00019561927599271418, 0.00023382467222659442, 0.016261632740403244, -2.7763950917369e-05, 0.0002532212538378989, 1.210221213863566e-06, 6.091194716423781e-07, -8.363271266781115e-06, -0.003874854975029919, -6.955686862618664e-05, -0.0007674384326431574, -0.005030789716254281, -0.011126389127222748, 2.2780123447609403e-05, -8.300387299868901e-05, 6.044024810257994e-06, -6.615370818442049e-07, 5.179251184041081e-06, -0.004978562785640978, 3.0845962757482576e-05, 0.0002958717139060439, -0.00017728672795334352, 0.001860239008051289, -3.2430294620180207e-06, 2.373486197528995e-05, -2.926327663621263e-07, 5.997496912516102e-08, -8.478434873774928e-07, -0.0025922817389869273, 9.188043297569346e-07, -3.889752811506521e-05, -0.00043909271187029744, -0.0005169709914471112, 1.47257421828297e-06, -1.4715141158264387e-05, 9.487567325370749e-07, -9.177639050419914e-08, 5.270372446124222e-07, 0.0003231026113959096],
    [0.0001124931755532567, 2.4831768150737724e-06, 9.854790088202792e-05, -0.00016792569875295285, 2.976759126472426e-06, -6.602045223089606e-06, -0.0006506232453830426, 6.578743167686857e-05, -9.967623956693735e-05, 0.00018591142764374827, -3.0525649547242123e-06, 0.0005683457962969776, 0.00041624440913397395, 3.2407296061531624e-05, -4.344416163271618e-06, 1.9993756146847494e-07, 0.0016609449950523668, 0.00041624440902733974, -0.001373410428226406, -0.003044482004010318, -0.011997458609613374, 3.3001851605058143e-06, -0.00043737028445183825, 6.9478258057651775e-06, 4.000433472401825e-07, 1.5161567375583276e-05, -0.004797575649390797, 0.00021735115358557916, -0.0012558912960018783, 0.0005964975178907718, 0.024309560100178904, -3.8030199892572316e-05, 0.0004092274455320482, -6.205264810430023e-06, -2.3741029201021355e-06, -1.1158887418566124e-05, -0.013395246043799158, 0.0004328404858625673, -0.003350212914823549, -0.04068756713865295, 0.03929020765227835, 0.00022655547702083904, -0.0010832672648056897, -3.062890888735183e-05, 7.03323540040303e-07, -7.735289020325193e-05, -0.12045057368967674, 0.00010099589704396806, -0.0002775097591725623, 0.002987914003045627, 0.014890716566155567, -3.4634187021456184e-05, 0.0004000193039235195, -1.5711103905960966e-06, -9.150674984016483e-07, -6.370184545059065e-06, -0.011565731472954349, -0.00013109697757998913, -0.0010238938855568348, -0.003045950786118118, -0.01212510423790191, 2.454011415437713e-05, -6.933030248843621e-05, -3.050523971049357e-06, 1.0625124231442905e-07, -1.0986128075931377e-06, 0.008479420840982552, -2.1075631256791418e-07, 0.0003461925570635884, 0.0006718545051823334, 0.0015857406448167998, -4.387947443138627e-06, 7.349260334380125e-05, -8.405773632885743e-07, -2.6077986484216666e-07, -8.162979254090746e-07, -0.005113971881496061, -1.3204219742163299e-05, -7.438339235049851e-05, -9.356695774242667e-05, -0.0008085185599800057, 2.0197716046853403e-06, -4.953839720652842e-06, -2.9594497996770115e-07, 4.61320597005098e-09, -4.6205973729244765e-07, 0.000728725022019677],
    [8.230325324632402e-05, -6.110087860253512e-07, 0.0001563215869966033, -0.0001752994021738683, 1.3503960075398446e-06, -4.6255528068896035e-06, -0.0003911719102580852, 1.1106225445463952e-05, -2.400441315756559e-05, -1.3588725606018944e-05, -3.824882693510125e-07, 0.0025384730891867455, 0.0001572615751422796, 6.454001317006629e-06, -1.8705936046111602e-07, -4.371092853790003e-08, -0.0014056097038767478, 0.00015726157516479596, -0.00014471821772784555, 0.0026881259408804503, -0.004140169091660154, -1.655440872498693e-05, 0.00018963721572746385, 9.527424165509816e-06, 1.7340226318201867e-06, 2.284765992418031e-05, -0.04323074081879369, 2.7700354026918447e-05, -0.002350362724733325, 0.004217085286458963, 0.007056520720293073, -2.947053132774244e-05, 0.00041446973171548735, -1.3234016428807405e-05, -1.6301964718086915e-06, -9.014220451652385e-06, 0.011329434361410682, 0.00023432095241322632, -0.0004452407718424603, -0.028700578027677948, 0.04831045362045373, 0.0001949651150192181, -0.0008504258991090163, -3.852129304844112e-05, 2.0673843204976392e-06, -5.9559929710205344e-05, -0.2043297028379733, 0.00018147517252133766, -0.001192331551089273, 0.0018468475674517186, 0.005580065021176247, -2.9623526336990076e-05, 0.00029079197879015146, -1.1864182167214485e-05, -8.981424240151347e-07, -1.2478318001586054e-05, 0.005252964100393207, -0.00020161257702586008, -6.084899020735401e-06, -0.0003542758923668997, -0.0069087328187271025, 2.8621174660959694e-06, 0.00011475315298296372, -3.6867120398103907e-06, 9.322738283745231e-07, 1.4371234371900099e-05, 0.003687196535099791, 2.3066115471838365e-05, 0.00023384540285722273, 0.0003407803969634889, 0.0003078117208712714, -3.3025495999438316e-06, 4.605270490059516e-05, -2.2653516218566884e-06, -3.6204379579652854e-07, -2.375623315197633e-06, -0.0040830694734513015, -1.2098767333919266e-05, 1.0951782317635754e-05, -6.246737146484398e-05, -0.0005636963507093327, -1.7792862030382723e-07, -1.7584535503049e-06, -9.615722962345003e-07, 6.085300893533921e-08, 1.6122121451303606e-06, 0.002578768764795303],
    [4.559369085495595e-05, 1.450562783837517e-05, 0.0001662919570753606, -0.00013176966372739014, 1.6643678711617805e-06, 5.554667205497365e-06, -0.000281497575949744, -1.7132691675311594e-05, -2.3078349450920944e-05, 6.663028812596413e-05, -1.3050795577105908e-06, 2.8025100103898376e-05, 9.566411402570841e-05, -5.278569834645607e-06, 1.4575822636088718e-06, -1.1568769754107464e-07, -5.798538734505403e-05, 9.566411403043341e-05, -0.0007149488825033019, 0.0047014005164257066, -0.00948065016709198, -1.173482751247136e-05, 0.00017962208615814103, 1.1492026660474556e-05, 1.013078114988099e-06, 2.6037850897726148e-05, -0.1102231220509729, 0.00020634880936402672, -0.001952166541893616, 0.002462382558049187, 0.013547303616337037, -3.2259151596949245e-05, 0.00027963712199398225, -1.7897355966255944e-05, -2.548553804598834e-07, -7.084500336684598e-06, 0.04574985720409037, 0.0004816173821293753, -0.002610308446526616, -0.02890126095417246, 0.04832158037219471, 0.00021880911964869092, -0.0018674251548060844, -5.66515489615195e-05, 4.222934087564288e-06, -6.456450560866763e-05, -0.17669689661371166, 0.00018909172496799805, -0.0011426575964028165, 0.0015986158471129083, 0.017911021807134657, -3.188052193369404e-05, 0.000343490259410112, -1.0622091616397727e-05, -6.808269367532142e-07, -4.828710733263995e-06, 0.011377158160693875, -0.00017914534966504637, -0.0014380921509663171, 0.00021958036917502877, -0.0065204842984511995, 8.176209956393596e-06, -6.514714687379029e-05, 8.319364595765247e-06, 2.874096336442844e-06, 9.488763803165968e-06, 0.012041685122033715, 3.0045472151964533e-05, 0.0002566832809929635, -5.420485507245967e-06, 0.00400163335330505, -3.506274341712062e-06, 2.847692671389297e-05, -2.675351978785162e-06, -4.397531712644806e-07, -6.053038135639475e-07, 0.005015047403872636, -2.2187289152468506e-05, -0.00013669025094570686, 9.78364215058097e-05, -0.00024626775809647437, 7.409077192427577e-07, -6.356867935403974e-06, 9.293884613578118e-07, 2.787470251534198e-07, 1.1730218878525888e-06, 0.0017472963493187057],
    [8.299222913589063e-05, -8.447269605274008e-06, 0.0001231282392360623, -0.00018521474945721938, 3.622938876546277e-07, -7.838830901377118e-06, -0.0006137689098036688, 1.3629254481184278e-05, -6.711400610626901e-06, 4.5345359948921445e-05, -1.4313786884303446e-06, 0.00027022633594798556, 0.00044706368187637544, 8.836611764082302e-06, -8.39414579418136e-07, -1.367516496809384e-08, 0.0001658289868523036, 0.00044706368191459053, -0.0003782013620303779, 0.0033631429035998, -0.00991149801497614, -1.8111026240498025e-05, 2.089850289095784e-05, 7.997251575154843e-06, 1.846853474527458e-06, 2.9012213799365232e-05, -0.0974930712671194, 4.603743004527847e-05, -0.003372319047457997, 0.0030936545672534507, 0.011673939232323407, -3.405671353056188e-05, 0.0006044303124708534, -8.535875193034115e-06, 2.0894485967557343e-07, -6.4077477870880045e-06, 0.014415317506560804, 0.00028587106299525413, -0.00168774593763568, -0.030283657356927927, 0.05925375752364271, 0.00023189169109588234, -0.0014960182937704678, -5.68822043713868e-05, 3.994759924934582e-06, -6.874526137318025e-05, -0.0670247967099388, 0.0002080531944358319, -0.0024487853599039857, 0.002234005196387183, 0.010213030692475337, -2.4008283261613466e-05, 0.0003838160755821039, -2.327798976651914e-05, -2.8363635957275816e-07, -1.3121727040101449e-05, -0.046035852901171556, -0.0001947852846236793, -0.0006418389580845078, -0.00019291358679661056, -0.003647213674895519, 7.227318241477899e-06, 0.00011224791900181109, 1.5232939550292123e-05, 2.51853693754683e-06, 5.323335450647079e-06, 0.03149079759995329, 3.3892548161513224e-05, 9.584020735559769e-05, 0.00040703138099611706, 0.0011441456422775188, -2.22076912241323e-06, 5.5739059164598645e-05, -5.152633434663664e-06, -2.2951984844673347e-07, -2.814980259771225e-06, -0.01580171882579645, -1.891832158940675e-05, -3.9013265951815794e-05, 6.565000175365574e-05, -0.00030012797299837823, -1.1571637971449877e-07, 9.407657212180102e-06, 1.8681553423099285e-06, 1.9429354210333186e-07, 7.27740179811975e-07, 0.005325297466631824],
    [8.39119888134438e-05, -3.3648702309043466e-07, 0.00026938091950683233, -0.00025885266375632986, 3.0300547370580976e-06, -9.376197439384783e-07, -0.0006626702664892957, 1.501431002764794e-05, -4.893255329229002e-05, -0.00010036604424713718, 1.2679061933277684e-06, 0.005058901147746633, 0.0005394226712651629, 1.7884594488015835e-05, -2.6680664695743537e-06, 1.2097614818559486e-07, -0.0016613884550402789, 0.0005394226712783948, -0.0005011769660832131, 0.005703384144801092, -0.010541277357197133, -2.2369934541561178e-05, -0.0003416793448087571, 1.3918558803201214e-05, 1.9858754911428056e-06, 1.9394985782613603e-05, -0.16061944471745612, 0.00023724712593340766, -0.0026439022032563974, 0.0010009667300550779, 0.01426459906999472, -2.927174545641924e-05, 0.0007841467235360139, -2.853971036982843e-06, 7.37252738739599e-07, -1.378875534253642e-05, -0.028936196948161644, 0.0005305566272962495, -0.0012219314658112898, -0.030126882665497675, 0.05089927353852203, 0.00022622042827899217, -0.0022404609711183963, -7.220573832846506e-05, 3.085854278058111e-06, -6.978877697995449e-05, -0.2200894763173691, 0.00025975447776975107, -0.0008957099104921361, 0.0004052611839257131, 0.010343048535807585, -3.2694913977971454e-05, 0.0005479868582282554, -1.0924257923519706e-05, -8.059135756284535e-07, -5.488714500829171e-06, 0.006883508728987294, -0.0001953208490044048, -0.0003809608777228434, 0.0007193093316198325, -0.009066999655730514, -9.73156375030624e-08, 0.00016243373217326827, 6.367200624504456e-06, 1.869349949789062e-06, 4.2365767779057604e-06, -0.01683251286042515, 3.606145728915312e-05, 0.0004273893874153354, 9.504488589799422e-05, 0.0016461019731316745, -6.070268075887078e-06, 0.00011541178796541026, -1.540408730083067e-06, -2.815417896500862e-07, -4.970586006894676e-07, -0.0020358737540796677, -2.8511623346088734e-05, -9.688704819739242e-06, 0.0003126179506319224, -0.0006198290459534137, -1.3980454059688823e-08, 2.8238740376915493e-05, 1.0452145337799181e-06, 1.72823128022986e-07, 1.3949859284015499e-07, -0.005969806648317224],
    [7.236939310285167e-05, -1.7860357568766417e-05, 1.309571997464276e-05, -0.00012908407319255798, 4.948974696364124e-07, -1.122176230513439e-05, -0.00043552168346925723, 1.0548408601234488e-05, 7.074844201335063e-06, -3.119528115239612e-05, -6.08428970864276e-07, -0.0024633440467341974, 0.0003231778620317592, 1.5546483106279163e-05, -3.052217007337412e-06, 1.9198069922285908e-07, 0.0019782467311164845, 0.000323177862043553, -0.0005979820581972931, 0.0009826850829339556, -0.011839649715210131, -1.9225657625028567e-05, 0.0005535693465895999, 1.0707397013180478e-05, 1.2499875813580421e-06, 2.044873151538137e-05, -0.11122268914409915, 0.0002545144672710975, -0.002273328784555606, 0.0012159445886859674, 0.01389431794522889, -4.6714087373899376e-05, 0.0007756480948728218, 7.123268592432697e-06, 1.7666289531394615e-06, -1.5435895587472232e-06, -0.021636763510381192, 0.0005220417066265345, -0.0027019676666086997, -0.03240072981514961, 0.04701567149996318, 0.0002598523371790701, -0.002647576481808175, -0.00010971333385080513, 3.3281445457853766e-06, -7.514556296400983e-05, 0.03553682791918477, 0.00026282614190306967, -0.0014018533089044547, 0.0014537549841681255, 0.01574126520781408, -3.172359191909955e-05, 0.0006162289755598614, -1.1935252312938514e-05, 1.5384761246940126e-06, -6.714459784687241e-06, -0.06068524599666474, -0.00032352323846746153, -0.001191681325119642, 0.0013681075923650397, -0.007632620822553022, 3.0194193275824416e-06, 0.0005235179351702864, 5.310440324412441e-06, 2.136803287837425e-06, 5.484534895216375e-06, 0.03133456443262844, 4.291711789109983e-05, 0.00024033771427822603, 8.847321213096694e-05, 0.002541662205033301, -2.968047270037244e-06, 7.195485757530368e-05, -4.084861886304032e-06, -3.545296521963951e-07, -9.464718641547573e-07, -0.014703633226098259, -4.448254577103451e-05, -9.86773452712546e-05, 0.00042436320915920615, -0.0006606790489792383, 1.0075935584437799e-07, 7.797617911629059e-05, 5.407247761490848e-07, 2.1939304096656764e-07, 3.56414399149355e-07, 0.0010462576168719564],
    [2.7463340901043913e-05, 1.0532765211312844e-05, 0.00032738037248029905, -0.00016990322204530814, 2.1247068257648325e-06, 5.540433824670698e-06, -0.00015241564073483602, -3.658570210783812e-06, -3.351065654596527e-05, -8.121564869703503e-05, 2.2144762805482786e-06, 0.010923286125870969, -0.0001204078736580995, 3.221250237712536e-08, 6.291921377049757e-07, -7.850783622942969e-08, -0.007147976089081281, -0.00012040787363018636, -0.00029320566904110425, 0.006574644634147552, -0.016110725997552693, -1.374498194799468e-05, 0.0014569458905879414, 3.96322128038402e-06, 7.500566729626514e-07, 2.2874617939636863e-05, -0.41759132565856155, 0.0002524725300631072, -0.0021574991579352424, 0.0011621799961943973, 0.011300584304436426, -5.03241475107506e-05, 0.0008266876863001285, 1.243233986779503e-05, 3.302610532186678e-06, -1.1478777925757926e-06, 0.0846464499415056, 0.0008335759120809722, -0.0010164474697346073, -0.029511991345013786, 0.05250059665255271, 0.0002550166855036016, -0.004456091544978505, -0.0001390330800702142, 1.2445654178322748e-07, -7.205315461894277e-05, -0.02021815503148931, 0.0002483992500874662, -0.00109280785819182, 0.001186300395171893, 0.006878744644778086, -4.4437313757241124e-05, 0.0007570547360308487, 1.276141790698034e-05, 2.59404806405494e-06, -2.899872445525952e-06, -0.01732875717704074, -0.0001140005211910171, -4.4324600373164125e-05, 0.000925808683830212, -0.0060463765078928025, -3.6768879848688673e-06, -0.00015615695830440424, -2.9198723957788435e-06, 2.3261833529620808e-07, 1.292797270936045e-07, 0.09696543845256732, 5.388525911572326e-05, 0.00032485300285263886, 1.20419393707958e-05, 0.0003045063124356953, -5.793958305572152e-06, 6.730039022283683e-05, 1.2325091363407412e-06, 1.5669477039292842e-07, -1.1317723479770598e-06, -0.0032590820471666815, 1.4204513816592397e-06, 3.823977440365699e-05, 4.254531234083256e-06, -0.0001903190230193251, -8.172388685063086e-07, -5.1186444869219405e-05, -5.834703009730305e-08, -3.992888273291036e-08, -1.588394898426895e-07, 0.01849113195108925],
    [1.2134623052160133e-05, -4.251913966867108e-06, 1.3250052728802493e-05, -3.975891639322492e-06, 1.8978472439508538e-06, -1.7760685700916955e-06, -0.00019746300697426637, -9.4135340614604e-06, -3.0682081617128286e-07, -0.0003054775400198749, 4.618544742995434e-06, 0.0018276410910483872, -0.00019023288646541242, 4.248737805520833e-07, 1.6835581548520765e-07, -2.2051347892074075e-08, -0.0018919180512054104, -0.00019023288646521864, -0.00021743163692625086, 0.009022526415894547, -0.009347170075617119, -3.761302989066776e-05, 0.001976105655398546, 1.3176915946805961e-05, 2.0353652186611573e-06, 3.0445593936164594e-05, -0.296015199585113, 0.00030532740370551903, -0.00255868749410419, 0.000631251093695837, 0.007749660031928001, -3.221954013613903e-05, 0.0006927861417249127, -1.2868037768395238e-05, 1.4140076037180104e-06, -6.702381047858557e-06, 0.10512958756573311, 0.0003777620089285409, -0.00027899640660737954, -0.021173802482978527, 0.05595001673950924, 0.00021993827737001414, -0.003246362133001931, -0.00011475343140222753, 3.2689491652785266e-06, -5.6468822783105796e-05, -0.36878866284712253, 0.0001927007750937799, -0.0015107244913770402, 0.0017771003783298, 0.0064898077526994065, -3.0875533851306e-05, 0.0007505821360372439, -2.3546526578227797e-05, -8.28985349115777e-07, -4.431960955204352e-06, 0.08452844126108569, -0.00019573759193116394, -0.0003659510003601195, 0.0024941634792296813, -0.004516880500921562, -2.1935372629436184e-05, 0.00011079808179813957, 2.4670779083287747e-05, 2.0416599475801557e-06, 7.887733650516734e-06, 0.00013108066989932479, 4.164869185098102e-05, 0.00029033574239429054, 0.0003060292493449835, 0.0017082304073174404, -6.433322300936196e-06, 0.00012127299838681461, -4.724557786532995e-06, -9.623211089527778e-07, -1.0296771799131195e-07, 0.003097316980757353, 3.2777312675575743e-06, -4.992569377984614e-05, 0.0001651373985337135, -0.00016380101136873642, -1.7459211927447189e-06, -5.213166585885459e-05, 1.1617586252811213e-06, 2.750752281036708e-07, 1.9559672921053488e-07, 0.004280655898298841],
    [6.348821953195761e-05, -1.067649693736207e-05, 0.00013831696548827617, -0.0001830918700024828, 1.1029172012845896e-06, -5.232006842107872e-06, -0.0006265232845009117, 1.740366457965831e-05, -2.3543879946747552e-05, -8.589521292776848e-05, 1.4900264074168725e-06, -0.0015253396663238974, 0.0005823077919824393, 2.2449795651454523e-05, -4.608599098770826e-06, 3.1297878555066774e-07, -3.961340177686327e-06, 0.0005823077920052222, -0.0008154625062834491, 0.005028490739570679, -0.009203629605065048, -2.3508010028227556e-05, -0.0001313497700151821, 3.372480456231979e-05, 3.13447090384418e-07, 2.2062529639519372e-05, -0.42407747910891663, 8.82790573528613e-05, -0.0036710802371141007, 0.00209625072487916, 0.0120275218803641, -3.886679829540938e-05, 0.0012455469873134237, -1.806217463442056e-05, 7.004450938107457e-06, -2.828638651161891e-07, 0.14719592993364192, 0.0009034463149442651, -0.0011535375479808576, -0.03607321617593074, 0.056719565543361244, 0.00029907971913598064, -0.004935913259191906, -0.00018967917315052687, 2.141051621155193e-06, -7.089293127972034e-05, 0.38673166748161997, 3.820611072038232e-05, -0.0020874965793231826, 0.002339205415703378, 0.006733744360937709, -3.168636302174249e-05, 0.0010329081360256562, -3.0614699743261295e-05, 4.984508670156252e-06, 2.2636334099210668e-06, 0.19277378308068674, -0.000201181998380687, -0.0001970711654945016, -0.0008972744166536046, -0.005235027562483207, -4.421908314833066e-06, 0.0004576796009384435, -2.2296066893838285e-08, 5.392041708508576e-07, 3.0547998911660702e-06, 0.19920871790867423, -1.233858510881488e-05, 0.0002790581791470394, 0.0005177127522551763, 0.00048091843996341996, -3.0330355342824905e-06, 0.00014230647785885085, -9.095322933059864e-06, 6.46616785169276e-07, 4.009828508636351e-07, 0.0552709094330559, -2.0006114925250023e-05, 1.8082516061153277e-05, 3.952870179054859e-06, -0.000369932929104245, -1.2942923582466529e-06, 6.464415692101928e-05, -8.955836500867281e-07, -1.719418856074502e-08, 3.374067791262687e-07, 0.02727968260427468],
    [-2.0743347707560682e-05, 1.3671025319101745e-05, 0.00011364501511199332, 1.655855682710938e-05, -1.4268565256098358e-06, 4.231155836602725e-06, 0.0001350948314898996, 1.5224570385650475e-05, -2.462351086359067e-05, -0.0002530319560013957, 0.0, 0.004900093462338552, -0.0003684113388362567, -3.561873076272105e-06, 1.5154923622495958e-06, -1.3786855869679759e-07, -0.004059890049242272, -0.00036841133883474045, -0.0019725057129645197, 0.0005450466474259193, -0.01836222317338478, 0.0, 0.0017629657805762775, 0.0, 0.0, 0.0, -0.23350348603624646, 5.099024285727002e-05, 0.0021383474553907937, 0.00025041483493183446, 0.02882706110159701, 0.0, 0.0013010910702604133, 0.0, 0.0, 0.0, -0.3148621005392629, 0.00043475062137894875, -0.006678192521432975, -0.019498409084362123, 0.04942353028862447, 0.0, -0.0009462419887251454, 0.0, 0.0, 0.0, -0.2461973581271961, 8.188948607397675e-05, 0.003375966621343289, -0.0004269663281646406, 0.01929901454266391, 0.0, 0.0010169218246155538, 0.0, 0.0, 0.0, -0.061987894919019244, -0.00026830801811106563, -0.002200563381619892, 0.0005682310868860344, -0.013275060737899205, 0.0, -0.0006747466957079559, 0.0, 0.0, 0.0, 0.17732094041870938, 2.559820201166195e-05, 0.0010475687520143694, -0.00025720505832096415, 0.0021031363805301463, 0.0, 0.00010523011477431823, 0.0, 0.0, 0.0, -0.0009312041527143518, -2.788377509296971e-05, -0.0001331958316098067, 0.000158576705696682, -0.0005541718432294779, 0.0, -7.821141721622491e-05, 0.0, 0.0, 0.0, 0.022670408652947095],
]
CORR_FEATS_SP = [
    '1',
    'mu',
    'var',
    'sigma',
    'alpha',
    '|alpha|',
    'phi',
    'Phi',
    'pred',
    'D3',
    'D21n',
    'K4',
    'sigma*phi',
    'pk2',
    'pk3',
    'pk4',
    'g4',
    'h0',
    'h0*s3',
    'h0*s4',
    'h0*s33',
    'h0*dn',
    'h0*v',
    'h1',
    'h1*s3',
    'h1*s4',
    'h1*s33',
    'h1*dn',
    'h1*v',
    'h2',
    'h2*s3',
    'h2*s4',
    'h2*s33',
    'h2*dn',
    'h2*v',
    'h3',
    'h3*s3',
    'h3*s4',
    'h3*s33',
    'h3*dn',
    'h3*v',
    'h4',
    'h4*s3',
    'h4*s4',
    'h4*s33',
    'h4*dn',
    'h4*v',
    'h5',
    'h5*s3',
    'h5*s4',
    'h5*s33',
    'h5*dn',
    'h5*v',
    'h6',
    'h6*s3',
    'h6*s4',
    'h6*s33',
    'h6*dn',
    'h6*v',
]
# 59 rasgos, sin sumas por fila: ridge por capa (columnas estandarizadas, lambda por capa por validacion cruzada en 5 bloques
# de redes) del error fresco con la media de entrada forzada a la verdadera, redes 0-69 del Mini
# (tablas/ajusta.py sobre tablas/fis/v_forz; fichero beta_ext_fis.json)
CORR_BETA_SP = [
    [-7.146087885934449e-07, 0.0, -3.726778702824789e-08, 1.2881145667752743e-07, 0.0, 0.0, 4.221318576535407e-09, 0.0, 3.2251892746044714e-07, 0.0, 0.0, 0.0, 3.2288243940264023e-07, -7.413542882888294e-08, -3.731592642656358e-08, -5.591459366059789e-09, 0.0, 3.2288243940261207e-07, 0.0, 0.0, 0.0, 0.0, -7.468694229243222e-08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.2288243940263017e-07, 0.0, 0.0, 0.0, 0.0, 7.468694229243229e-08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0762747980140633e-07, 0.0, 0.0, 0.0, 0.0, -2.4895647430796355e-08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.1525495960112264e-08, 0.0, 0.0, 0.0, 0.0, 4.979129486179289e-09],
    [-0.00070284853141679, -2.0236622412170056e-05, -0.0004150888142920276, 0.0009302072933925334, -2.4514493718095594e-05, 5.53861407875597e-07, 0.0013821601672992267, 0.00024583796508508183, -0.00010054584526455401, -7.617548077192995e-05, 4.859235521771864e-06, 0.006249264670504794, -0.0007517634760197885, 1.4680401812653944e-05, 1.2228463263412632e-05, -1.9543396392259236e-06, -0.0017245403965312235, -0.0007517634764487318, 0.0005807606002429755, 0.01254100973740371, 0.001743082229299178, -3.483680327957267e-05, 0.0004401534928935356, -4.121887234250856e-06, -0.0003076499095317343, -0.00015803670077858822, 0.01680698843270667, -6.492124018959728e-07, -5.961946807200656e-05, -0.00021765312984830664, 0.000769023073571322, -0.008987985555052707, 0.0236435143288233, 0.00011027626503618068, 3.648685748828185e-05, 7.890944085101876e-05, 1.9688313242841474e-05, 0.00014945969964655882, 0.014024507798650965, -2.6887705356188096e-06, -6.55885309627933e-05, 6.677490987871532e-05, 0.00020528115944075007, 0.0014395821105930903, -0.0031894839687967385, -1.9674306698825916e-06, -3.4254666623877065e-05, 1.3479980977540218e-05, 7.004244427460252e-05, -2.468191530773645e-05, 0.0022041508579731567, -8.724918591153568e-07, -8.32194623034707e-06, 2.613134282994379e-06, 1.4578074694119813e-05, -4.4581649811243765e-06, -0.0001832646357051475, 5.070575752412775e-07, -2.917116881778216e-06],
    [-0.00017913960988829027, -2.9294133275486856e-05, -3.7260729746081595e-05, 6.194996393534399e-05, -8.132394033243457e-06, -1.8099880627667536e-06, 0.00013021392327026385, 4.6136420061010634e-05, 4.3721442092105806e-05, 0.0001343788456492633, 6.7535908299185595e-06, 0.016135387234490626, 6.866724220636273e-05, 8.685251377393286e-06, -8.935451289959807e-07, -1.184342763205086e-07, -0.0034215691806801975, 6.866724219724613e-05, -0.000419424050324552, 0.015631728392331616, -0.0018827888718262993, -7.053507133526799e-05, 0.00010609923058527993, -3.0933098425177104e-05, -0.0012405135572157431, 0.0019376671539649071, -0.0144995598971205, -7.0556857139616986e-06, 2.3667622560443822e-06, -8.9164837220024e-05, -0.00032854127037465475, -0.013294748801631132, 0.03054092652424433, 9.021010988771384e-05, -9.315068913652538e-06, 1.3005911466199506e-05, -0.0010581746669013565, 0.0016973142350898182, -0.0051612331569476375, -7.460688610929269e-06, -4.467688030584995e-06, 2.56773802585871e-05, -0.00010100156062887807, 0.002862675564751241, -0.00887514112991915, -8.599271219698915e-06, 4.043082397206488e-06, 3.6638972267655823e-06, -1.939122959861421e-05, 0.00025062063028799737, -0.00043477531300768106, -1.3111419880497758e-06, -4.854991571939148e-07, 3.228111385681771e-07, -5.146194070248291e-06, 0.0001455420532795999, -0.0011865175342196976, 1.4643582740833448e-07, -1.511009302031528e-06],
    [-9.79754413101399e-05, -1.4994094293055756e-05, -4.429041994787482e-05, 0.00013098816229690968, 3.986056853181399e-06, -4.983970237883983e-06, 0.00018066273663860073, 2.1672032380552644e-05, -2.4778491639042483e-06, 0.0001954269956885302, 2.65093915499624e-07, 0.015049299695894788, -0.00019638167482227356, 5.573989153650726e-06, -9.71156427029951e-07, 6.616872678922454e-08, -0.002652662290904103, -0.0001963816746434607, -0.000875708121213094, 0.004159841432216907, -0.00810811211681562, -2.202111380359401e-05, 0.00024072889781696439, 0.00013217217025142167, -0.0014001456177988177, -0.00040805171309249666, 0.003997266156248699, -2.2554909427214837e-05, 7.227327439349774e-05, -0.00023776179262000678, -0.0009527350363655142, -0.030265128327792715, 0.03230706045425271, 0.00015788875508736995, -0.00016926731501172, 0.00019578769559101225, -0.0010768755650044032, -0.0017963088676819262, 0.002881297963317773, -2.3236898520587038e-05, 2.316068986817012e-05, -8.565513289003806e-05, -5.105044088230708e-05, -0.0007127088084432327, -0.01195810049472687, 1.3915351708291358e-05, 1.5132686299630153e-05, 2.6082354812531083e-05, 5.2973255766357565e-05, -0.0001886741044826744, -0.0004810615822876055, -3.4668974040202603e-06, 3.7768717869904222e-06, -9.76209315234299e-06, 1.8081306079762168e-05, -8.432323944812587e-07, -0.0011204400565649351, 1.406115830234962e-06, 1.7427724067435998e-06],
    [0.0002723099855323672, 3.6066738946636163e-05, 0.0003889715058136504, -0.0005079236802743118, 6.365545717987632e-06, 1.5225080289772324e-06, -0.0003198384417392718, -0.0002413263542042209, 0.00019621946427170043, 5.146426879599281e-05, -7.682787291147567e-06, 0.011148032705294, 0.00031076066283847555, -0.00013102679482334022, 2.6823372416358762e-05, -1.8574339905767609e-06, 0.00094571821892737, 0.0003107606627800645, -0.0014347820324267845, -0.009718209657211702, -0.013639373039829102, 3.118387351116856e-05, -0.000698951570866861, 0.00037512609857758526, 3.269184826059901e-05, 0.007476701297008364, 0.04158355484684497, -5.2749746063513715e-05, 5.711130989769407e-05, 0.0002749926125937209, -0.004466108501345754, -0.05675543144435795, 0.022751840351364978, 0.0002344440353613844, -0.0012225281613266695, 6.312489764489892e-05, 0.0002569563990374901, 0.007295007178641674, 0.03242727856082917, -4.673051085440639e-05, 0.0002525084395110365, -3.3007722427749307e-05, -0.0014029942998833587, -0.006215450056466147, -0.01283008750117645, 3.62086642185266e-05, -0.0002274461621912686, -1.4525444476060264e-05, 0.00026764228513921824, 0.0012825611570687306, 0.003027198169191327, -6.079641090682682e-06, 5.030901175197849e-05, -6.357649462698932e-06, -7.928669201989526e-05, -0.00028420910533366524, -0.0008694838353202497, 2.6184896538556406e-06, -1.61160529237627e-05],
    [0.0001270133325497061, 1.1045022866112736e-05, 0.00019991987881622016, -0.0002648323871569141, 1.7495213215773534e-06, -2.7885401161524595e-06, -0.000686236193401059, 0.000246569538117384, -0.00033321438132641584, 6.594784116617832e-05, -2.960014131221456e-06, 0.011692469425406823, 0.0003864288740239843, 0.00012628929914160027, -2.1036920640706682e-05, 1.259228992938149e-06, -0.0003080715933011267, 0.0003864288743102394, -0.0006549820990999274, -0.006183806757016539, -0.00704527563690483, -5.475967310258257e-06, -0.0002596080785562, -0.0002629397810392608, -0.000986703329619257, 0.0016103081877422273, 0.013296967550177721, -3.339434625628437e-05, 0.0006641611112534744, 0.00035021244075784567, -0.0020955880746574692, -0.05233959479938437, 0.0414269156655752, 0.00017465054213968665, -0.0008633554187387689, 3.9428408233142924e-05, -0.00027325949853867815, 0.0015434703872216971, 0.016698702341095835, -3.321389910875133e-05, 0.00033848462569216166, 4.102967451001164e-05, -0.0007622326403880829, -0.007066580962698787, -0.010983274140171483, 2.4912718753864594e-05, -0.00017411809248075929, 1.996113255083824e-05, 0.0002958354567379472, -1.3396133465293903e-05, 0.002001725708954323, -4.254111212076453e-06, 3.15164982847635e-05, 9.602583734663307e-06, -3.87833232352704e-05, -0.0005999362468546227, -0.0005531106202770017, 2.038191634959815e-06, -2.2571590898263816e-05],
    [0.00014944776929591352, 4.9465692338651085e-06, 0.00020273445008426778, -0.0003008361401318174, 2.865218367158373e-06, -5.089458688002933e-06, -0.0008301298703388331, 0.00013092571536021168, -0.00020314271628478887, 0.0001804695058946705, -3.1653962180904384e-06, 0.005711208350150507, 0.0006673337343572578, 8.174019959518657e-05, -1.418599830898049e-05, 8.756653201681838e-07, 0.0011934607503586861, 0.0006673337341112503, -0.001329853661138052, -0.005654590087272164, -0.010552351500242146, 4.835749265100346e-06, -0.0007547817879556032, 8.573225245242589e-05, -0.0016339035265173737, 0.00021699492798292426, 0.023895943003883195, -4.241882083174072e-05, 0.0005397574681315038, 0.00041875324153821077, -0.003148365190975517, -0.0495285440588797, 0.04772569111960058, 0.0001839141055071381, -0.001143862773553596, 6.971574781225538e-05, -0.0005658666863447162, 0.0026106057466293215, 0.01408387795965243, -3.7135096911899806e-05, 0.0004130984805717805, -2.5532673449130997e-05, -0.0009360336988902401, -0.004756971345413463, -0.010714225698083309, 2.2734821673263334e-05, -0.0001291891764446603, 5.296567947265465e-07, 0.00031311972141684054, 0.000506886749076645, 0.0015659003369541378, -4.985483147284903e-06, 6.33582514988859e-05, -5.116842493976169e-06, -6.730766110590942e-05, -0.00026209221943282706, -0.0007283497598591535, 1.907021761416596e-06, -1.1256177939786515e-05],
    [9.905059952681858e-05, -6.872128132410094e-06, 0.00019308058710213531, -0.00022606092932324432, 1.0196068958842425e-06, -8.631675978772751e-06, -0.0003644077450646089, 1.6509255325375098e-05, -1.3979648479608305e-05, -1.55551096101056e-05, -4.6101668964593837e-07, 0.006221900630477914, 0.00018102090427450838, 7.702093857164045e-06, -4.2221150634757275e-07, -3.017991689879694e-08, -0.0024395944617163663, 0.00018102090430088607, -0.00019112586643327614, 0.0010878919253691957, -0.002785363514528661, -9.581461861935426e-06, 0.00014469279712715305, -3.546034404522806e-05, -0.0025062899088319907, 0.004424947216520104, 0.009053939206075905, -3.6988680714481756e-05, 0.0005344546729372868, 5.1635089825916056e-05, -0.000677310165086196, -0.03422228075382881, 0.05373414601690246, 0.00015189949108990918, -0.0007938211893947422, 0.0001269244710474307, -0.0013104351541443062, 0.0019416479395218626, 0.007628549800320484, -3.9843885691423385e-05, 0.00040797257093668247, -0.0001671111254405133, -0.00013825404406506893, -0.00020963053327317595, -0.006978449075590136, 9.884831669318812e-06, 7.942066898426165e-05, 1.8302546606562844e-05, 0.0002194165359932294, 0.0002352153834293354, 0.0008770593786869577, -5.547546822967899e-06, 5.532839371683816e-05, -1.0917607592000537e-05, -5.229468088729156e-06, 3.7240440655156206e-05, -0.0006711473149884798, 7.394609437113437e-07, -1.9887053827839522e-06],
    [5.718393285657409e-05, 1.3290665185332971e-05, 0.00019115759927135114, -0.000162122703424332, 1.3139535165434758e-06, 4.470208421316747e-06, -0.00021768425603842047, -2.0933579715771573e-05, -1.3892276394724662e-05, 4.078041905596485e-05, -1.3351435079220105e-06, -0.001575583427939778, 0.0001350892395678582, -7.169381457564952e-06, 1.6445594440595637e-06, -1.1558858555397161e-07, -0.00018692458967175526, 0.00013508923958097753, -0.0005430496258945547, 0.001958107482215361, -0.008688331514234987, -1.9171100909987003e-06, -0.00011299247779659831, 0.00011725730926617561, -0.0022027416651479665, 0.0038621518135516855, 0.011726866944526024, -4.0165552548937024e-05, 0.0004840335680386664, 0.0002673483777159768, -0.002022019764585507, -0.03205974586984922, 0.055921062667861426, 0.0001622428837441272, -0.0016899579803844312, 0.00016254148372709702, -0.0013440913222633403, 0.0017492139124828745, 0.016182423643287716, -3.620883733608058e-05, 0.00040208690689013657, -0.00016859063077206886, -0.0011528028232638246, 0.0006943465673313133, -0.005949219666301166, 1.5482099716376254e-05, -9.27622976111027e-05, 2.4244709827799092e-05, 0.00022976753177870597, 9.102453731962185e-05, 0.0035828394601956186, -4.161176780414726e-06, 4.396286344213619e-05, -2.18752859212475e-05, -0.00010562253058961485, 0.00017682714111691398, -0.0002231454247502607, 1.8029697834638373e-06, -1.1305562044635031e-05],
    [6.304019382907947e-05, -1.047947294018223e-05, 3.176662474643298e-05, -9.871177782716116e-05, 2.9766190940100003e-07, -9.236519627144238e-06, -0.0003277452175894402, 2.9086085241354207e-06, 8.080809729339237e-06, 1.3195895434150384e-05, -1.2651965217163514e-06, -0.0012849693915694773, 0.00024697682246824687, 5.917024736813538e-06, -6.686445977462607e-07, 3.176317269817603e-09, 0.0008209015216715423, 0.000246976822505017, -0.0001923529833863681, 9.083323334535931e-05, -0.008851621774572587, -6.790256427640656e-06, 0.00019032330454988832, -1.0078945906906305e-05, -0.00366494960838344, 0.004240073023744142, 0.01001571347907421, -3.786670907491737e-05, 0.0006571768371638826, 5.819676051250156e-05, -0.0012082793341902072, -0.03128199353733508, 0.06597760912117917, 0.0001670821555079605, -0.0010933868079760088, 0.00017594421907300326, -0.0026237840001327706, 0.0019229827956415452, 0.010061329192974231, -3.810776721757439e-05, 0.0003635748034191512, -0.00017148487413413998, -0.0004144084013123222, 0.0002525878417342583, -0.0038502254671698762, 1.3861800637976434e-05, 8.562371326628156e-05, 3.276275831649671e-05, 8.082012114225118e-05, 0.00020594395790815434, 0.0011202420635055992, -5.623296037747565e-06, 3.916027735213768e-05, -2.0897468866845634e-05, -1.3959894195440672e-05, 0.00018395820435217704, -0.0004082383439531177, 1.0905366595641761e-06, 1.0913903308775228e-05],
    [1.8267436525186366e-05, 4.853125243467652e-07, 0.00012172000355890254, -4.768973953498515e-05, 3.6318113774960485e-06, 1.3346151685588354e-06, -0.00021480947475344616, -7.214930823654195e-06, -2.0265926776061913e-05, -0.00020090353865913962, 4.456462197766283e-06, 0.001556516367345237, 4.4586699227328635e-05, 1.8764836353046932e-07, 7.88007431830876e-07, -9.913837950832938e-08, -0.0040687243213678, 4.458669922782761e-05, -0.0003018031557210379, 0.005716535698762439, -0.012465736207637986, -2.589550299634589e-05, 0.00039352223673343195, 0.00016410012158659485, -0.0019039276651539626, 0.001498597246196621, 0.0149014932684186, -3.6451585235467e-05, 0.0009123678789514627, 8.315980500436203e-05, -0.0006338065862686575, -0.025515936684596705, 0.05125800125074129, 0.00012220513025653157, -0.0018210542885229786, 0.00016592274011144683, -0.00013875237388506686, 0.0014212167011839387, 0.010099253803385496, -3.653494192913724e-05, 0.0007140199775047605, -0.00014749876077545089, -3.6696646959335675e-05, 0.0015126363617895793, -0.011694811360853229, -5.6472313829688325e-06, -2.9647141077158652e-05, 2.769061333316221e-05, 0.0005456416552905973, 0.000134559046433923, 0.0013954265974335752, -6.065090246003478e-06, 0.00011775529169538504, -1.0602319813686956e-05, 3.34711125761163e-05, 0.00013640936883640913, -0.0007794811854686049, -2.9629098826582784e-07, -1.766642389032569e-05],
    [3.567299911043301e-05, -1.5388314485598417e-05, -0.00011254505183054761, 4.608166417223498e-06, 3.733438659161463e-07, -1.0115137445088426e-05, -0.00012654047573235636, 5.869195132977563e-06, 1.2427443941856063e-05, -5.507807881776271e-05, -6.46088524957751e-07, -0.0030179710793363478, 5.945768244720071e-05, 9.93644323397983e-06, -1.8570921082269364e-06, 1.1192125523244212e-07, 0.002653034122883962, 5.945768246296249e-05, -0.0004635850258427704, -0.00259844498426627, -0.01004391186819695, -7.784820616968851e-06, 0.0008764977631691363, 0.00021563881657095263, -0.002513363606018072, 0.0012554175725948938, 0.012315720503044351, -4.334351989852532e-05, 0.0007901500844551462, 0.00016913259725418248, -0.002290653113897983, -0.03040805253116492, 0.055151533171038054, 0.00015525763567239272, -0.001592617289529538, 0.00025948847530141096, -0.0015680392112704935, 0.0007416883443608843, 0.01563354228501443, -4.130262879552061e-05, 0.0005975490039377398, -0.0002736073710079451, -0.001005366942353541, 0.0016078808794767112, -0.007183392069867348, 8.290624064871913e-06, 0.00038514979899799433, 5.383064701060434e-05, 0.0002055952400552171, -0.0001337363745895563, 0.0026067834565777943, -5.574178463915789e-06, 4.9337537131235905e-05, -3.9533287134771806e-05, -7.817984108683856e-05, 0.0004240581960365506, -0.0006256970178922256, 6.104780734116175e-07, 5.792155345602653e-05],
    [-4.572380159331895e-07, 1.3062343955522499e-05, 0.00012775008997943426, -2.09237407752805e-05, 2.573290092742034e-06, 6.906517966734165e-06, 0.00022959892499806243, -2.0771324118104004e-05, -1.998011969597569e-05, -0.00011449633118486038, 2.7872257592241174e-06, -0.005481359602260422, -0.0003742751707904281, -9.06634013224025e-06, 2.278550612601993e-06, -1.7476564178440988e-07, -0.0017945567287779092, -0.0003742751707959313, -0.0002670958854033625, -0.00020255932002444593, -0.015528826785861788, -6.443649668745348e-06, 0.0012092999950314965, 0.00023522206106557243, -0.0023921787673658276, 0.0025569869198491632, 0.010838133159788257, -4.361345805542567e-05, 0.0009668551410735539, 0.0005185165374820613, -0.0011088525620932836, -0.027404391263269472, 0.0606849460541545, 0.00013638684432149917, -0.003568453217580508, 0.0002784291742004395, -0.0012437466369172986, 0.000866120637381666, 0.0072473471074903685, -3.9665010564614105e-05, 0.0006839495359202084, -0.00012828006725429115, -0.00014222110392753, 0.0019328051681344594, -0.00623942207061957, -2.7234647250321577e-06, -0.00011393188936289723, 5.415608236062421e-05, 0.00030553977088991876, -3.855101854930088e-05, 0.0005542528679573577, -5.8797942104545644e-06, 6.705585302410304e-05, -9.09438358308518e-06, 2.455556676579505e-05, 0.0002040221831286192, -0.0002448399130613671, -3.651361766293507e-07, -2.0754757975311512e-05],
    [1.8086339644214005e-05, -5.5923251905203865e-06, 1.2077194509353224e-05, -1.0716306280965927e-05, 1.0792374598598261e-06, -3.4757647567500433e-06, -7.698098269135067e-05, -7.414563218322359e-06, 6.1453334035801324e-06, -0.0003319280424986369, 3.6163677316135896e-06, -0.003008447914173043, -0.00017137632506390846, 6.590432279881648e-07, 6.836950111394045e-08, -1.485901413305724e-08, -0.001377560354298542, -0.00017137632506516808, -1.940292447382363e-05, 0.00445017943070402, -0.009822948132834302, -1.3983805452904609e-05, 0.001317273150327159, 0.00021320723475897987, -0.002568470276517256, 0.0009524360783555302, 0.0058192592315347135, -3.81106032569071e-05, 0.0009264230945670194, 0.00019885110055279176, 0.0005047250899934856, -0.024070974328507046, 0.062425308886281394, 0.00011968987129594029, -0.002792215241459692, 0.00014273823297062963, -0.001452083797842558, 0.002087883171930653, 0.0048302821788642515, -4.299380392997756e-05, 0.0009475706565559804, -0.00018894662931012124, 5.3372040459492e-05, 0.0022951662511974846, -0.005571046424857707, -4.53014938154075e-06, -2.120064393047698e-05, 4.979185283314093e-05, 0.00030483128842195747, 0.00020493989921172655, 0.0014072053631830816, -8.37737046000703e-06, 0.0001016634550889891, -5.309372562982325e-06, -5.643555069199392e-06, 0.0002474508613062347, -0.00018460759882227312, -1.130738608263267e-06, -3.6091062849210025e-05],
    [3.536243491909846e-05, -9.824336976056121e-06, -3.7253536158377316e-05, -4.6267284272666234e-05, 1.5758200968481674e-06, -4.490336260778705e-06, -0.00022581938890650302, 5.583037990776974e-06, -1.1605399739422913e-05, -0.00010588614495738282, 2.1070593003208623e-06, -0.00751012579087396, 0.00021737683569426176, 1.4368006944056526e-05, -2.9426442312191804e-06, 2.0220426361967427e-07, 0.002825846467632349, 0.00021737683571861183, -0.000832042333042266, -0.0009091277223113116, -0.010242510635045998, 1.173465492680804e-06, 0.00021695408721820127, 2.432073202061182e-05, -0.0038783598412169256, 0.003965370437483652, 0.011574809811570523, -4.234978080015726e-05, 0.0015646906726149417, 0.0004891205569089513, -0.00100100035893172, -0.031491160617077504, 0.06584647025086889, 0.00014765575181000952, -0.0029589203465932626, -4.486527944902321e-05, -0.0022064039934799255, 0.004433474345237608, 0.0063205999356564695, -3.917308155533356e-05, 0.0014359078437331408, -0.0002209073351892513, -0.00012331305985382047, 0.0006582981022212586, -0.005837364759750615, 2.993852859283455e-06, 0.0005617760752268063, -4.355553758332236e-05, 0.0002754962562112098, 0.0010986893336259245, 0.0004756263922192565, -5.866131055903984e-06, 0.0002748888765422866, -2.8361746062195223e-05, 2.82601136957533e-05, 0.00024180363374575945, -0.000434277228930914, -5.3867259301787e-07, 8.964163470266548e-05],
    [-1.4285915301612962e-05, 1.283519738608496e-05, 8.998347000306871e-05, 9.220694952951063e-06, -1.4970401550267926e-06, 3.7124577534370986e-06, 0.00010934557535832671, 2.0534973897370637e-05, -2.875643344895553e-05, -0.00026160609322477914, 0.0, -0.004590961310318774, -0.0002829842537493632, -1.2485319412664458e-06, 1.2120863708635688e-06, -1.3060235770906975e-07, -0.0011843898871083203, -0.0002829842537502008, -0.0019867626146879425, -0.0013888789296071888, -0.018404674777017425, 0.0, 0.0010910124491985253, 0.00019846063425931564, 0.0021496036854314567, -0.0024460418849804482, 0.02933950841146972, 0.0, 0.0004845780197933717, 0.000628820503571295, -0.0067731354031832424, -0.022736799422408888, 0.04910936019208035, 0.0, -0.0016926374629251523, 7.198584006923588e-05, 0.0033858735151346695, -0.0006574274113316589, 0.019753286079392144, 0.0, 0.0009765002281550431, -0.0003229502899592566, -0.002250971038413668, 0.001370879842887968, -0.013612397658042985, 0.0, -0.00022749417891832055, 1.9699959651596567e-05, 0.0010487583576212517, -0.00021618357977910366, 0.002177241128511332, 0.0, 0.0001222262738017111, -3.125922531414968e-05, -0.00013799634691946834, 0.0002355565272314482, -0.0005964132596488974, 0.0, -3.240678231117499e-05],
]
# LAM de v29 por 0.96 * (1 + (-0.1) * (l - 7) / 7): escala e inclinacion por diferencias finitas en lazo cerrado,
# redes 0-29, con la correccion nueva puesta (tablas/lam_fd.py); despues se aplica V17_LAM_SCALE (0,95)
LAM_NEW = [0.005268912, 0.008429589942857143, 0.01010993142857143, 0.010705728, 0.010863401142857143, 0.01069187657142857, 0.01031066057142857, 0.00964608, 0.009130048457142858, 0.008553545142857142, 0.008061133714285714, 0.007597588114285714, 0.007180902857142857, 0.006738212571428571, 0.0062716032, 0.0061720539428571424]
if not TAB_OLD and TAB_SIN_PARES and CORR_BETA_SP is not None:
    CORR_FEATS, CORR_BETA_NEW = CORR_FEATS_SP, CORR_BETA_SP
if not TAB_OLD and CORR_BETA_NEW is not None:
    CORR_BETA = CORR_BETA_NEW
if not TAB_OLD and LAM_NEW is not None:
    LAM = [c * float(_os.environ.get("V17_LAM_SCALE", "0.95")) for c in LAM_NEW]
    LAM = [c * (1.0 + LAM_TILT * (max(0.0, (i - 3) / 12.0) ** 1.5)) for i, c in enumerate(LAM)]

# 8.a-quindecies (1): correccion de medias fusionada. Misma matematica que _tab_feats @ beta:
#   delta = F_base @ b_base + sigma*phi * sum_j alpha^j * (Mod @ Bm)_j
# con h<k>*m = sigma*phi*He_k(alpha)*m, Mod = num * var^(-E) (moduladores en una pila) y la base de Hermite
# pasada a monomios en las tablas (Bm = Bh @ T^T, T[j][k] = coef. de alpha^j en He_k). ~12 operaciones por capa
# en vez de ~170. X_CORR_FUSE=0 vuelve a _tab_feats (bit a bit el candidato).
CORR_FUSE = _os.environ.get("X_CORR_FUSE", "1") == "1"
CORR_F64 = _os.environ.get("X_CORR_F64", "0") == "1"   # diagnostico (con X_CORR_FUSE=0): delta sumada en float64
_FZ = None
if CORR_FUSE and not TAB_OLD and CORR_FEATS is not None:
    import re as _re
    _MODS = ["", "s3", "s4", "s33", "dn", "v", "ds", "cr", "c2", "k4"]
    _MOD_E = {"": 0.0, "s3": 1.5, "s4": 2.0, "s33": 3.0, "dn": 1.0, "v": 0.0, "ds": 1.0, "cr": 1.0, "c2": 2.0, "k4": 0.0}
    _hre = _re.compile(r"^h(\d+)(?:\*(\w+))?$")
    _base_i = [c for c, nm in enumerate(CORR_FEATS) if not _hre.match(nm)]
    _hk = {}
    for c, nm in enumerate(CORR_FEATS):
        m_ = _hre.match(nm)
        if m_:
            _hk[c] = (int(m_.group(1)), m_.group(2) or "")
    _KMAX = max(k for k, _ in _hk.values()) if _hk else 0
    _He = [[1.0], [0.0, 1.0]]   # coeficientes en monomios de He_k (Hermite probabilista)
    for k in range(2, _KMAX + 1):
        a_ = [0.0] + _He[k - 1]
        b_ = _He[k - 2] + [0.0, 0.0]
        _He.append([a_[j] - (k - 1) * b_[j] for j in range(k + 1)])
    _mods_used = [m for m in _MODS if any(md == m for _, md in _hk.values())]
    _BH = []
    for li_ in range(len(CORR_BETA)):
        row = CORR_BETA[li_]
        bh = [[0.0] * (_KMAX + 1) for _ in _mods_used]
        for c, (k, md) in _hk.items():
            mi = _mods_used.index(md)
            for j, t in enumerate(_He[k]):
                bh[mi][j] += row[c] * t
        _BH.append(bh)
    _FZ = {"base": [CORR_FEATS[c] for c in _base_i],
           "BB": [[CORR_BETA[l][c] for c in _base_i] for l in range(len(CORR_BETA))],
           "mods": _mods_used, "E": [-_MOD_E[m] for m in _mods_used], "BH": _BH,
           "J": [float(j) for j in range(_KMAX + 1)],
           "pares": any(m in ("ds", "cr", "c2") for m in _mods_used)}


def _tab_rec(li, **kw):
    """X_TAB_DUMP: copia de las cantidades por capa (tensores del calce o floats)."""
    d = _TAB_REC.setdefault(li, {})
    for k, v in kw.items():
        d[k] = v.clone() if hasattr(v, "clone") else v


def _tab_feats(names, ones_n, mu, var, sigma, alpha, phi, Phi, pk1v, D3f, D21n, k4f, g4f, PK1,
               d21s=None, crs=None, cr2=None):
    """8.a-quater: rasgos por neurona de la correccion nueva, (n, len(names)). Operaciones de n por rasgo.
    h<k>[*m] = sigma * phi * He_k(alpha) [* m], He_k de Hermite probabilista; moduladores adimensionales
    s3 = D3 / sigma^3, s4 = g4 / sigma^4 (g4 = cumulante 4 marginal transportado), s33 = s3^2, dn = D21n / var,
    v = var, k4 = K4 escalar; y los de pares (sumas por fila, n^2 cada una, las calcula quien llama):
    ds = (D21 1) / var, cr = (C_off 1) / var, c2 = ((C_off * C_off) 1) / var^2."""
    base = {"1": ones_n, "mu": mu, "var": var, "sigma": sigma, "alpha": alpha, "phi": phi, "Phi": Phi,
            "pred": pk1v, "D3": D3f, "D21n": D21n, "K4": k4f, "g4": g4f,
            "pk2": PK1[1], "pk3": PK1[2], "pk4": PK1[3], "d21s": d21s, "crs": crs, "cr2": cr2}
    cache = {}

    def get(nm):
        if nm in cache:
            return cache[nm]
        if nm in base:
            v = base[nm]
        elif nm == "|alpha|":
            v = fnp.abs(alpha)
        elif nm == "sigma*phi":
            v = sigma * phi
        elif nm.startswith("He"):
            k = int(nm[2:])
            v = ones_n if k == 0 else (alpha if k == 1 else alpha * get("He%d" % (k - 1)) - get("He%d" % (k - 2)) * float(k - 1))
        elif nm == "inv_var":
            v = 1.0 / var
        elif nm == "m:s3":
            v = D3f * (get("inv_var") / sigma)
        elif nm == "m:s4":
            v = g4f * (get("inv_var") * get("inv_var"))
        elif nm == "m:s33":
            v = get("m:s3") * get("m:s3")
        elif nm == "m:dn":
            v = D21n * get("inv_var")
        elif nm == "m:v":
            v = var
        elif nm == "m:k4":
            v = k4f
        elif nm == "m:ds":
            v = d21s * get("inv_var")
        elif nm == "m:cr":
            v = crs * get("inv_var")
        elif nm == "m:c2":
            v = cr2 * (get("inv_var") * get("inv_var"))
        elif nm.startswith("h"):
            hb, _, md = nm.partition("*")
            v = get("sigma*phi") * get("He" + hb[1:])
            if md:
                v = v * get("m:" + md)
        else:
            raise KeyError(nm)
        cache[nm] = v
        return v

    return fnp.stack([get(nm) for nm in names], axis=1)

# pruned V16b table + the 4 (3,1)-slice use-side terms
_I = WICK_PAIRS.index
TERM_SPECS[(1, 1)].append(('wk431', 'ones2', _I((1, 1)), _I((3, 1)), 1.0 / 3.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 2)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 2)), 1.0 / 3.0))
ALPHA_DEG = 5
SIG_EMIN, SIG_EMAX = -6, 4
D2_IPS = [(1, 1), (2, 1), (2, 2)]
D1_IPS = [(1,), (2,), (3,), (4,)]
D2_OUT = [(1, 1), (2, 1), (2, 2)]
D1_OUT = [(2,), (3,), (4,)]
MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4",
                 "wk431"}


def _statics(n: int) -> dict:
    c = 4 - 2 + n / 2.0 - 1.0
    P2 = (c - 2.0) / (16.0 * 2.0 * c * (1.0 - c) * (2.0 - c))
    return dict(P2=P2, k4_c4=24.0, k4_c22=24.0, wk4_c4=1.0, wk4_c22=1.0 / 3.0,
                cA=6.0 / (n + 4.0), cI=-3.0 / ((n + 2.0) * (n + 4.0)))


def _zero_diag(A):
    fnp.fill_diagonal(A, 0.0)
    return A


def _build_wick_consts():
    np_ = len(WICK_PAIRS)
    C1 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    C2 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    # sigma-power selection with the constant prefactor folded in
    SELC = [[0.0] * np_ for _ in range(SIG_EMAX - SIG_EMIN + 1)]
    for j, (const, e, p1, p2) in enumerate(WICK_UNIFIED):
        SELC[e - SIG_EMIN][j] = const
        for i, c in enumerate(p1):
            C1[i][j] = c
        for i, c in enumerate(p2):
            C2[i][j] = c
    return C1, C2, SELC


def _term_prog(mode: int):
    """Fused-einsum program for the nonlin terms of one availability mode."""
    missing = MODE0_MISSING if mode == 0 else ({"wk431"} if mode == 2 else set())
    np_ = len(WICK_PAIRS)
    d2 = []  # (group, a, b, wl, wr, coef)
    d1 = []
    for g, ip in enumerate(D2_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d2.append((g, a, b, wl, wr, coef))
    for g, ip in enumerate(D1_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d1.append((g, a, b, wl, coef))
    SL2 = [[0.0] * np_ for _ in d2]
    SR2 = [[0.0] * np_ for _ in d2]
    IND2 = [[0.0] * len(D2_IPS) for _ in d2]
    AB2 = []
    for t, (g, a, b, wl, wr, coef) in enumerate(d2):
        SL2[t][wl] = coef
        SR2[t][wr] = 1.0
        IND2[t][g] = 1.0
        AB2.append((a, b) if a <= b else (b, a))
    SL1 = [[0.0] * np_ for _ in d1]
    IND1 = [[0.0] * len(D1_IPS) for _ in d1]
    B1 = []
    for t, (g, a, b, wl, coef) in enumerate(d1):
        SL1[t][wl] = coef
        IND1[t][g] = 1.0
        B1.append((a, b) if a <= b else (b, a))
    return dict(SL2=SL2, SR2=SR2, IND2=IND2, AB2=AB2, SL1=SL1, IND1=IND1, B1=B1)

# ---- V26 (F79): Strassen-Winograd block products on pooled buffers ----------------
STRASSEN_LEVELS = int(_os.environ.get("V26_STRASSEN", "5"))   # V28 ladder (suite harness, C/B | residual): L3 0.3107 | 0.170 s, L4 0.2973 | 0.187, L5 0.2872 | 0.205, L5 + shared basis 0.2680 | 0.242   # dump-0 ladder: 1/2/3 -> C/B 0.344/0.327/0.3125; mini residual at L3 = 0.356 s mean with a 0.42 s tail (7/100 failed) -> ship L1 until the base residual is cut
STRASSEN_HUB = int(_os.environ.get("V26_STRASSEN_HUB", str(STRASSEN_LEVELS)))
STRASSEN_NEW = int(_os.environ.get("V26_STRASSEN_NEW", "0"))   # V29: unused (the newborn rides the family)
CPRE_LEV = int(_os.environ.get("V29_CPRE_LEV", _os.environ.get("V26_STRASSEN", "5")))  # V29: C_pre block family levels
STRASSEN_SB = int(_os.environ.get("V28_STRASSEN_SB", str(STRASSEN_LEVELS)))   # V28: shared-basis formings + contractions   # newborn transport levels (0: ~30 ms residual for 0.5% cost is a bad trade)
STRASSEN_MIN = int(_os.environ.get("V26_STRASSEN_MIN", "32"))  # smallest leaf block side (V28: 32; f32 error grows ~1.4x per level, MSE unchanged to 4 digits at L5)
STRASSEN_FIRST = int(_os.environ.get("V28_STRASSEN_FIRST", "4"))
STRASSEN_FUSE_P = int(_os.environ.get("V28_STRASSEN_FUSE_P", "1"))   # s32: 343 -> 1 (fuse every leaf batch; bit-exact, FLOPs -3..-4%)  # V28: fused per-product leaf when the leaf batch has >= this many blocks (deep levels: 7x smaller pools, +4 ops per leaf)  # V28: level cap of the first predict() of a process (see _predict_core)
# X_L6 (02/10/2026): sexto nivel de Strassen con la memoria acotada (ver _Strassen._mm6). Apagado = programa original.
WINO_LEAF = _os.environ.get("X_WINO_LEAF", "1") == "1"   # t014: Winograd leaf of mm (15 adds instead of 18)
L6 = _os.environ.get("X_L6", "0") == "1"
L6_MIN = int(_os.environ.get("X_L6_MIN", "12"))    # hoja minima del nivel extra (lado de bloque tras niveles+1 cortes)
L6_BMIN = int(_os.environ.get("X_L6_BMIN", "1"))   # lote minimo (huecos del operando derecho / k del hub)
L6_PAD = float(_os.environ.get("X_L6_PAD", "0"))   # >0: rellena con ceros hasta multiplos de 2^(niveles+1) si el volumen crece <= esto
WARM = _os.environ.get("V26_WARM", "1") == "1"
# ORACULO (estructura-v29, 02/10): neuronas "muertas" (alpha < -DEAD_TAU): se anulan sus pesos de Wick, sus filas y
# columnas de C_off y D21; su media se queda en el termino gaussiano. Apagado (0) = flujo original.
DEAD_TAU = float(_os.environ.get("X_DEAD_TAU", "2.85"))
# e30 (14/10/2026): X_DEAD_TAU_L = "a-b:t,c:t,..." umbral de retirada propio de las capas a..b (0-15); el resto usa X_DEAD_TAU. Vacio = e29coste bit a bit.
DEAD_TAU_L = {}
for _t in _os.environ.get("X_DEAD_TAU_L", "").split(","):
    if _t:
        _r, _, _v = _t.partition(":")
        _a, _, _b = _r.partition("-")
        for _l in range(int(_a), int(_b or _a) + 1):
            DEAD_TAU_L[_l] = float(_v)
DEAD_LOG = _os.environ.get("X_DEAD_LOG", "")
# modo "k3": la cadena de media/covarianza/kappa4 usa los pesos de Wick exactos; solo la maquinaria de fuentes K3
# (rebanadas D3/D21, nacimiento, patas y su transporte) pierde las neuronas muertas.
DEAD_MODE = _os.environ.get("X_DEAD_MODE", "todo")
# RETIRADA FISICA (muertas/impl, 02/10/2026). Con dead_tau > 0 la MASCARA de arriba fija los valores (oraculo); dead_fis
# decide que productos n^3 se hacen ya sin las apagadas (mismo valor salvo redondeo, menos FLOPs facturados):
#   0 = solo mascara (oraculo, sin ahorro);  1 = (a) dimension interna de los transportes de las jovenes y de W C / C_pre;
#   2 = 1 + (c) filas de salida: transportes, patas viejas re-formadas y la rebanada D21 (filas y columnas);
#   3 = 2 + (a) en la UNION de las viejas (buscador de rango, proyecciones y transporte de la base en filas conservadas);
#   4 = 3 + (b) columnas de la fuente: cada fuente guarda sus columnas permutadas (vivas de su capa de nacimiento
#       primero) y las familias cortan [:m]; la pata P pierde las columnas apagadas (unica diferencia con la mascara:
#       el buscador de rango de la base vieja ya no las ve);
#   5 = 4 + patas finas (Z, Zf) solo en filas conservadas y con la dimension interna reducida, y la union de la
#       base vieja solo con las columnas conservadas de la fuente que se une.
# Los tamanos vivos se redondean hacia arriba a multiplos de DEAD_MULT (rellenando con apagadas, que valen cero) y las
# familias reducidas usan hoja minima DEAD_SMIN para conservar los niveles de Strassen. dead_tau = 0 -> flujo original.
CONFIG = {"dead_tau": DEAD_TAU, "dead_fis": int(_os.environ.get("X_DEAD_FIS", "5")),
          "dead_round": _os.environ.get("X_DEAD_ROUND", "cerca")}


def _vivas(alpha, tau, n, f32, modo):
    """Mascara de vivas, numero de vivas por umbral, tamano conservado (multiplo de DEAD_MULT) y permutacion con las
    conservadas primero (None si se conservan todas). modo 'arriba': mascara = alfa > -tau y se rellena con apagadas;
    'cerca'/'abajo': si el multiplo elegido queda por debajo del numero de vivas, se apagan las vivas de menor alfa."""
    alive = fnp.greater(alpha, -tau).astype(f32)
    nal = int(fnp.sum(alive))
    if modo == "arriba" or n % DEAD_MULT:
        m = min(n, -(-nal // DEAD_MULT) * DEAD_MULT)
    else:
        m = (nal // DEAD_MULT) * DEAD_MULT if modo == "abajo" else int(round(nal / DEAD_MULT)) * DEAD_MULT
        m = max(DEAD_MULT, min(n, m))
        if m < nal:
            rango = fnp.argsort(fnp.argsort(-alpha, kind="stable"), kind="stable")   # 0 = la de mayor alfa
            alive = fnp.less(rango, m).astype(f32)
    perm = fnp.argsort(1.0 - alive, kind="stable") if m < n else None
    return alive, nal, m, perm
DEAD_MULT = int(_os.environ.get("X_DEAD_MULT", "32"))
DEAD_SMIN = int(_os.environ.get("X_DEAD_SMIN", "16"))
DEAD_FORCE = _os.environ.get("X_DEAD_FORCE", "0") == "1"   # solo pruebas: aplica la logica fuera de la forma 1024x16
DUMP_OUT = _os.environ.get("X_DUMP_OUT", "")
DEAD_PMASK = _os.environ.get("X_DEAD_PMASK", "0") == "1"
DEAD_STAT = {}   # diagnostico: por llamada, vivas y tamanos por capa
# COMPACTA (02/10/2026, tarde). Apagados, el flujo es el de la sonda 1 bit a bit.
#   X_DEAD_CENT=1: covarianza C entera. Las apagadas conservan sus filas y columnas de C y su var_prev (Wick sin
#     mascara solo en K11 y K2, con D3 y D21 de las apagadas a cero); todo lo demas sigue la mascara. La W C de la capa
#     siguiente usa la dimension interna completa. Excluye la etapa p.
#   X_DEAD_COMP (dead_fis >= 5), letras: p = C_pre solo en filas/columnas conservadas de su capa (var por filas);
#     q = transporte de la base Qc solo en filas conservadas; f = rebanadas D3/D21 enteras en filas conservadas.
#   X_DEAD_TAU_B=t (0 < t < dead_tau): umbral propio de las columnas de la fuente (etapa b); con dead_fis >= 4 se
#     quitan fisicamente, por debajo es mascara (columnas de a_b y de la pata P de la recien nacida).
DEAD_CENT = _os.environ.get("X_DEAD_CENT", "0") == "1"
DEAD_COMP = _os.environ.get("X_DEAD_COMP", "pqf")
DEAD_TAU_B = float(_os.environ.get("X_DEAD_TAU_B", "2.2"))
DEAD_ON_B = float(_os.environ.get("X_DEAD_ON_B", "0"))   # > 0: las de alfa > +t no dan columnas a la fuente nueva


# ---- X_STV (t025): combos y ensamblado de Strassen con operaciones apiladas sobre pares de bloques --------------------------
# Cada nivel de Strassen cuesta 22 operaciones facturadas (7 + 7 combos, 8 de ensamblado). Los pares de cuadrantes se leen
# como UNA vista de pila (as_strided con el salto entre cuadrantes; salto 0 = el mismo bloque dos veces) y los destinos
# como pares de ranuras: 5 + 4 + 5 = 14 operaciones, mismas sumas con otro orden (|dif| ~1e-7). Los programas salen de una
# busqueda exhaustiva (pares de origen y destino siempre con salto > 0; el signo de M3 y M7 va guardado cambiado).
# Orden de las ranuras (posicion de M1, M2, M4, M5, M6, M7, M3). Si la orientacion de memoria no es la prevista (fila-mayor
# de bloque: bc*sc < br*sr) el nivel usa las 22 operaciones de siempre.
STV = _os.environ.get("X_STV", "0") == "1"
TT22_LAM = float(_os.environ.get("X_TT22_LAM", "0.7"))   # c2 22.10.2026: exact coherent (2,2) transport W² (diag(K4v)+K22) W²ᵀ blended into wk4m
K4C1 = _os.environ.get("X_K4C1", "0") == "1"   # N1 (04/10): kappa4 marginal regenerado con forma por neurona, g4row = 0,99*dG*METRIC_C + 0,85*D3^2/var
C1_A = 0.99
C1_B = 0.85
_STV_POS = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 3: 6}
_STV_L = (((2,), "copy", (3,)), ((3,), "op", (0,), (1,), 1), ((6,), "copy", (0,)),
          ((0, 1), "op", (0, 2), (3, 3), 1), ((4, 5), "op", (2, 3), (0, 1), -1))
_STV_R = (((0,), "op", (0,), (3,), 1), ((1, 3), "copy", (0, 3)), ((2, 6), "op", (2, 3), (0, 1), -1),
          ((4, 5), "op", (0, 2), (1, 3), 1))
_STV_A = (((2,), "op", (2,), (4,), 1), ((3,), "op", (1,), (2,), -1), ((0, 1), "op", (1, 5), (5, 3), -1),
          ((0, 3), "acc", (4, 6), 1), ((0, 3), "acc", (7, 3), -1))


def _stv_str(T):
    """Pasos de T en elementos (tensor de la plataforma o, en local, ndarray)."""
    if hasattr(T, "stride"):
        return tuple(T.stride())
    return tuple(x // T.itemsize for x in T.strides)


def _stv_pair(T, delta):
    """Vista (b, 2, P, h, q) de T (b, P, h, q) con salto `delta` elementos entre el elemento 0 y el 1 de la pila."""
    st = _stv_str(T)
    shp = (T.shape[0], 2) + tuple(T.shape[1:])
    stt = (st[0], delta) + st[1:]
    if hasattr(T, "as_strided"):
        return T.as_strided(shp, stt)
    raise RuntimeError("as_strided no disponible en este array")   # en el servidor y el kit los arrays de flopscope lo tienen


def _stv_ok(X, Y, out, hub):
    """Orientacion prevista (fila-mayor de bloque) en X, Y y out: los tres con bc*sc < br*sr."""
    for T, sh in ((X, (X.shape[2], X.shape[3])), (Y, (Y.shape[2], Y.shape[3]) if hub else (Y.shape[2], Y.shape[3])),
                  (out, (out.shape[2], out.shape[3]))):
        st = _stv_str(T)
        if not (sh[1] // 2) * st[3] < (sh[0] // 2) * st[2]:
            return False
    return True


def _stv_run(prog, dst, src):
    for op in prog:
        D = dst(op[0])
        if op[1] == "copy":
            fnp.copyto(D, src(op[2]))
        elif op[1] == "op":
            (fnp.add if op[4] > 0 else fnp.subtract)(src(op[2]), src(op[3]), out=D)
        else:
            (fnp.add if op[3] > 0 else fnp.subtract)(D, src(op[2]), out=D)


def _stv_slots(buf):
    """Destino: ranuras de buf (b, 7, P, h, w) como una o como par."""
    sl = _stv_str(buf)[1]

    def dst(d):
        if len(d) == 1:
            return buf[:, d[0]]
        return _stv_pair(buf[:, d[0]], (d[1] - d[0]) * sl)
    return dst


def _stv_combos(buf, T, Q, prog, sw):
    """Combos de las 4 vistas Q (de T (b, P, m, kd)) a las ranuras de buf; False si la orientacion no es la prevista."""
    st = _stv_str(T)
    br, bc = T.shape[2] // 2, T.shape[3] // 2
    a_, b_ = br * st[2], bc * st[3]
    if not b_ < a_:
        return False
    off = (0, a_, b_, a_ + b_) if sw else (0, b_, a_, a_ + b_)

    def src(sy):
        if len(sy) == 1:
            return Q[sy[0]]
        if sy[0] == sy[1]:
            return Q[sy[0]][:, None]
        return _stv_pair(Q[sy[0]], off[sy[1]] - off[sy[0]])
    _stv_run(prog, _stv_slots(buf), src)
    return True


def _stv_assemble(M, out):
    """Ensamblado de M (b, 7, P, h, w) en out (b, P, m, w); False si la orientacion de out no es la prevista."""
    h, w = M.shape[3], M.shape[4]
    st = _stv_str(out)
    a_, b_ = h * st[2], w * st[3]
    if not b_ < a_:
        return False
    C = (out[..., :h, :w], out[..., :h, w:], out[..., h:, :w], out[..., h:, w:])
    oc = (0, b_, a_, a_ + b_)
    sl = _stv_str(M)[1]

    def dst(d):
        if len(d) == 1:
            return C[d[0]]
        return _stv_pair(C[d[0]], oc[d[1]] - oc[d[0]])

    def src(sy):
        if len(sy) == 1:
            return M[:, _STV_POS[sy[0]]]
        pa, pb = _STV_POS[sy[0]], _STV_POS[sy[1]]
        if pa == pb:
            return M[:, pa][:, None]
        return _stv_pair(M[:, pa], (pb - pa) * sl)
    _stv_run(_STV_A, dst, src)
    return True


class _Strassen:
    """Recursive Strassen on batched operands. Layouts (last two axes = the matrix):
      plain: X (bx, P, m, kd) @ Y (by, P, kd, w) -> out (by, P, m, w), bx in {1, by}
      hub:   X (k, P, m, kd), Y (k, P, w, kd)    -> out (P, m, w) = sum_k X_k @ Y_k^T
    Every combo / product is ONE flopscope op over the whole batch, written with out=
    into pooled buffers shared by role (F53: fresh result buffers dominate residual).
    The seven products: M1=(X11+X22)(Y11+Y22) M2=(X21+X22)Y11 M3=X11(Y12-Y22)
    M4=X22(Y21-Y11) M5=(X11+X12)Y22 M6=(X21-X11)(Y11+Y12) M7=(X12-X22)(Y21+Y22);
    C11=M1+M4-M5+M7 C12=M3+M5 C21=M2+M4 C22=M1-M2+M3+M6.  For the hub kernel the
    right operand is Y^T, whose blocks are (Y^T)11=Y[..,:h,:q], (Y^T)12=Y[..,h:,:q],
    (Y^T)21=Y[..,:h,q:], (Y^T)22=Y[..,h:,q:] (no transposes: the kernel contracts j)."""

    def __init__(self, dtype, bmax):
        self.dtype = dtype
        self.bmax = int(bmax)
        self.pool = {}

    def _buf(self, key, shape):
        """V27: pools grow to the largest batch actually seen (the suite's young-source
        batch is <= 2 (AGE_OLD + 1), a third of the 2 (L-1) capacity), so deeper levels
        stay within memory; a larger batch later simply reallocates once."""
        b = self.pool.get(key)
        if b is None or b.shape[0] < shape[0]:
            b = fnp.empty(shape, dtype=self.dtype)
            fnp.copyto(b, 0.0)   # metered page touch (first-MLP residual, F80)
            self.pool[key] = b
        return b

    def _buf2(self, key, shape5):
        """V27: a pooled (b, 7, P, h, q) buffer together with its (b, 7P, h, q) view,
        both made once at allocation (fnp.reshape is a logged op: 0.06-0.3 ms per call,
        462 calls per MLP at four levels)."""
        e = self.pool.get(key)
        b = int(shape5[0])
        if e is None or e[0].shape[0] < b:
            b5 = fnp.empty(shape5, dtype=self.dtype)
            fnp.copyto(b5, 0.0)   # metered page touch (first-MLP residual, F80)
            b4 = fnp.reshape(b5, (b, shape5[1] * shape5[2], shape5[3], shape5[4]))
            e = (b5, b4)
            self.pool[key] = e
        return e[0][:b], e[1][:b]

    def level(self, m, kd, w, lev):
        """V28: largest level <= lev at which every block side divides and stays
        >= STRASSEN_MIN (a family whose sides do not allow `lev` runs shallower
        instead of falling back to dense)."""
        while lev > 0 and not self._ok(m, kd, w, lev):
            lev -= 1
        return lev

    @staticmethod
    def _ok(m, kd, w, lev):
        d = 2 ** lev
        return (lev > 0 and m % d == 0 and kd % d == 0 and w % d == 0
                and min(m, kd, w) // d >= STRASSEN_MIN)

    def _combos(self, X, Q, kind, key, sw=False, stv=False):
        """Seven Strassen combos of the quadrant views Q=(X11,X12,X21,X22) of X
        (b, P, m, kd) into a pooled (bmax, 7, P, h, q) buffer; returns (b, 7P, h, q)."""
        b, P, m, kd = X.shape
        h, q = m // 2, kd // 2
        buf, buf4 = self._buf2(key, (b, 7, P, h, q))
        if stv and _stv_combos(buf, X, Q, _STV_L if kind == "L" else _STV_R, sw):
            return buf4
        X11, X12, X21, X22 = Q
        # Winograd order (slots P1..P7 = 0..6): 8 sums + 6 copies instead of 10 sums + 4 copies
        if kind == "L":
            fnp.add(X21, X22, out=buf[:, 4]); fnp.subtract(buf[:, 4], X11, out=buf[:, 5])
            fnp.subtract(X11, X21, out=buf[:, 6]); fnp.subtract(X12, buf[:, 5], out=buf[:, 2])
            fnp.copyto(buf[:, 0], X11);       fnp.copyto(buf[:, 1], X12)
            fnp.copyto(buf[:, 3], X22)
        else:
            fnp.subtract(X12, X11, out=buf[:, 4]); fnp.subtract(X22, buf[:, 4], out=buf[:, 5])
            fnp.subtract(X22, X12, out=buf[:, 6]); fnp.subtract(buf[:, 5], X21, out=buf[:, 3])
            fnp.copyto(buf[:, 0], X11);       fnp.copyto(buf[:, 1], X21)
            fnp.copyto(buf[:, 2], X22)
        return buf4

    @staticmethod
    def _assemble(M, out, stv=False):
        """M (b, 7, P, h, w) products -> out (b, P, m, w) quadrants (7 ops, Winograd)."""
        h, w = M.shape[3], M.shape[4]
        if stv and _stv_assemble(M, out):
            return
        C11, C12 = out[..., :h, :w], out[..., :h, w:]
        C21, C22 = out[..., h:, :w], out[..., h:, w:]
        fnp.add(M[:, 0], M[:, 1], out=C11)
        fnp.add(M[:, 0], M[:, 5], out=C21); fnp.add(C21, M[:, 4], out=C12)
        fnp.add(C12, M[:, 6], out=C22); fnp.add(C21, M[:, 6], out=C21)
        fnp.subtract(C21, M[:, 3], out=C21); fnp.add(C12, M[:, 2], out=C12)

    def mm(self, X, Y, out, lev):
        """plain: X (bx, P, m, kd), Y (by, P, kd, w) -> out (by, P, m, w)."""
        if L6 and X.shape[1] == 1 and lev == STRASSEN_LEVELS:
            pl = self._plan6(X.shape[2], X.shape[3], Y.shape[3], Y.shape[0])
            if pl is not None:
                if pl == (X.shape[2], X.shape[3], Y.shape[3]):
                    self._mm6(X, Y, out, lev + 1)
                else:
                    self._mm6_pad(X, Y, out, lev + 1, pl)
                return
        bx, P, m, kd = X.shape
        by, w = Y.shape[0], Y.shape[3]
        if not self._ok(m, kd, w, lev):
            fnp.matmul(X, Y, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        XQ = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        YQ = (Y[..., :q, :v], Y[..., :q, v:], Y[..., q:, :v], Y[..., q:, v:])
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            # V28: fused leaf (deep levels only) -- one product at a time into the out quadrants
            self._leaf_mm(XQ, YQ, (out[..., :h, :v], out[..., :h, v:], out[..., h:, :v], out[..., h:, v:]),
                          bx, by, P, h, q, v)
            return
        stv = STV and _stv_ok(X, Y, out, False)
        Xc = self._combos(X, XQ, "L", ("L", P, h, q), False, stv)
        Yc = self._combos(Y, YQ, "R", ("R", P, q, v), False, stv)
        # V28: the products reuse this level's left-combo buffer when the callee recurses
        # (its combos consume Xc before anything is written into out); a fused-leaf callee
        # reads X while writing out, so it keeps a separate M. Square blocks only.
        callee_recurses = self._ok(h // 2, q // 2, v // 2, lev - 2)
        mkey = ("L", P, h, q) if (callee_recurses and q == v) else ("M", P, h, v)
        Mb, Mb4 = self._buf2(mkey, (by, 7, P, h, v))
        self.mm(Xc, Yc, Mb4, lev - 1)
        self._assemble(Mb, out, stv)

    def _leaf_mm(self, XQ, YQ, OQ, bx, by, P, h, q, v, tag="1"):
        """Fused leaf of mm: M6->C22, M4->C11, M1 (+C11, +C22), M2->C21 (C22-=), M3->C12
        (C22+=), M5 (C11-=, C12+=), M7 (C11+=): 7 matmul writes + 8 adds, one combo
        pair alive at a time (copies for the raw-quadrant operands keep every matmul
        input contiguous)."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = self._buf(("L" + tag, P, h, q), (bx, P, h, q))[:bx]
        Rb = self._buf(("R" + tag, P, q, v), (by, P, q, v))[:by]
        Mb = self._buf(("M" + tag, P, h, v), (by, P, h, v))[:by]
        if WINO_LEAF:
            # Winograd form: 8 pre-adds + 7 post-adds (15 instead of 18), same 7 products
            Lb2 = self._buf(("L2" + tag, P, h, q), (bx, P, h, q))[:bx]
            Rb2 = self._buf(("R2" + tag, P, q, v), (by, P, q, v))[:by]
            fnp.add(X21, X22, out=Lb); fnp.subtract(Y12, Y11, out=Rb)                 # S1, T1
            fnp.matmul(Lb, Rb, out=C12)                                              # P5 -> C12
            fnp.subtract(Lb, X11, out=Lb2); fnp.subtract(Y22, Rb, out=Rb2)            # S2, T2
            fnp.matmul(Lb2, Rb2, out=C21)                                            # P6 -> C21
            fnp.matmul(X11, Y11, out=C11)                                            # P1 -> C11
            fnp.add(C21, C11, out=C21)                                               # U2 = P1+P6
            fnp.add(C12, C21, out=C12)                                               # U4 = U2+P5
            fnp.subtract(X11, X21, out=Lb); fnp.subtract(Y22, Y12, out=Rb)            # S3, T3
            fnp.matmul(Lb, Rb, out=Mb)                                               # P7
            fnp.add(C12, Mb, out=C22)                                                # U7 = U4+P7 -> C22
            fnp.add(C21, Mb, out=C21)                                                # U3 = U2+P7
            fnp.subtract(Rb2, Y21, out=Rb)                                           # T4
            fnp.matmul(X22, Rb, out=Mb)                                              # P4
            fnp.subtract(C21, Mb, out=C21)                                           # U6 = U3-P4 -> C21
            fnp.subtract(X12, Lb2, out=Lb)                                           # S4
            fnp.matmul(Lb, Y22, out=Mb)                                              # P3
            fnp.add(C12, Mb, out=C12)                                                # U5 = U4+P3 -> C12
            fnp.matmul(X12, Y21, out=Mb)                                             # P2
            fnp.add(C11, Mb, out=C11)                                                # C11 = P1+P2
            return
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb)
        fnp.matmul(Lb, Rb, out=C22)                                   # M6 -> C22
        fnp.add(X21, X22, out=Lb)
        fnp.matmul(Lb, Y11, out=C21)                                  # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                               # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb)
        fnp.matmul(X22, Rb, out=C11)                                  # M4 -> C11
        fnp.add(C21, C11, out=C21)                                    # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)                                    # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb)
        fnp.matmul(X11, Rb, out=C12)                                  # M3 -> C12
        fnp.add(C22, C12, out=C22)                                    # C22 += M3
        fnp.add(X11, X12, out=Lb)
        fnp.matmul(Lb, Y22, out=Mb)                                   # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)                                    # M7
        fnp.add(C11, Mb, out=C11)

    def hub(self, X, Y, out, lev):
        """hub: X (k, P, m, kd), Y (k, P, w, kd) -> out (P, m, w) = sum_k X_k Y_k^T."""
        if L6 and X.shape[1] == 1 and lev == STRASSEN_LEVELS:
            pl = self._plan6(X.shape[2], X.shape[3], Y.shape[2], X.shape[0])
            if pl is not None:
                if pl == (X.shape[2], X.shape[3], Y.shape[2]):
                    self._hub6(X, Y, out, lev + 1)
                else:
                    self._hub6_pad(X, Y, out, lev + 1, pl)
                return
        k, P, m, kd = X.shape
        w = Y.shape[2]
        if not self._ok(m, kd, w, lev):
            # dense: batched GEMM over (k, P) then a k-sum (the einsum form
            # 'kpij,kpcj->pic' takes a slow non-BLAS path: 0.5 s per call at 512^2)
            T = self._buf(("K", P, m, w), (k, P, m, w))[:k]
            fnp.matmul(X, fnp.swapaxes(Y, -1, -2), out=T)
            fnp.sum(T, axis=0, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        XQ = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        # (Y^T) blocks expressed on the un-transposed Y (rows c, cols j)
        YQ = (Y[..., :v, :q], Y[..., v:, :q], Y[..., :v, q:], Y[..., v:, q:])
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            self._leaf_hub(XQ, YQ, (out[..., :h, :v], out[..., :h, v:], out[..., h:, :v], out[..., h:, v:]),
                           k, P, h, q, v)
            return
        stv = STV and _stv_ok(X, Y, out[None], True)
        Xc = self._combos(X, XQ, "L", ("L", P, h, q), False, stv)
        Yc = self._combos(Y, YQ, "R", ("R", P, v, q), True, stv)
        Mb, Mb4 = self._buf2(("HM", P, h, v), (1, 7, P, h, v))
        self.hub(Xc, Yc, Mb4[0], lev - 1)
        self._assemble(Mb, out[None], stv)

    def _leaf_hub(self, XQ, YQ, OQ, k, P, h, q, v, tag="1"):
        """Fused leaf of hub (same product order as _leaf_mm); each product is a batched
        GEMM over (k, P) into T then a k-sum into its quadrant / the (P, h, v) scratch."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        sx = "" if tag == "1" else tag
        Lb = self._buf(("L" + tag, P, h, q), (k, P, h, q))[:k]
        Rb = self._buf(("R" + tag, P, v, q), (k, P, v, q))[:k]
        T = self._buf(("K" + sx, P, h, v), (k, P, h, v))[:k]
        Mb = self._buf(("KS" + sx, P, h, v), (1, P, h, v))[0]
        def prod(L_, R_, dst):
            fnp.matmul(L_, fnp.swapaxes(R_, -1, -2), out=T)
            fnp.sum(T, axis=0, out=dst)

        if WINO_LEAF:
            Lb2 = self._buf(("L2" + tag, P, h, q), (k, P, h, q))[:k]
            Rb2 = self._buf(("R2" + tag, P, v, q), (k, P, v, q))[:k]
            fnp.add(X21, X22, out=Lb); fnp.subtract(Y12, Y11, out=Rb); prod(Lb, Rb, C12)   # S1, T1, P5
            fnp.subtract(Lb, X11, out=Lb2); fnp.subtract(Y22, Rb, out=Rb2); prod(Lb2, Rb2, C21)   # S2, T2, P6
            prod(X11, Y11, C11)                                                      # P1
            fnp.add(C21, C11, out=C21); fnp.add(C12, C21, out=C12)                   # U2, U4
            fnp.subtract(X11, X21, out=Lb); fnp.subtract(Y22, Y12, out=Rb); prod(Lb, Rb, Mb)   # S3, T3, P7
            fnp.add(C12, Mb, out=C22); fnp.add(C21, Mb, out=C21)                     # U7, U3
            fnp.subtract(Rb2, Y21, out=Rb); prod(X22, Rb, Mb)                        # T4, P4
            fnp.subtract(C21, Mb, out=C21)                                           # U6
            fnp.subtract(X12, Lb2, out=Lb); prod(Lb, Y22, Mb)                        # S4, P3
            fnp.add(C12, Mb, out=C12)                                                # U5
            prod(X12, Y21, Mb); fnp.add(C11, Mb, out=C11)                            # P2, C11
            return
        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb); prod(Lb, Rb, C22)   # M6 -> C22
        fnp.add(X21, X22, out=Lb); prod(Lb, Y11, C21)                                # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                                              # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb); prod(X22, Rb, C11)                           # M4 -> C11
        fnp.add(C21, C11, out=C21)                                                   # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb); prod(Lb, Rb, Mb)       # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb); prod(X11, Rb, C12)                           # M3 -> C12
        fnp.add(C22, C12, out=C22)                                                   # C22 += M3
        fnp.add(X11, X12, out=Lb); prod(Lb, Y22, Mb)                                 # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb); prod(Lb, Rb, Mb)  # M7
        fnp.add(C11, Mb, out=C11)

    # ---- X_L6: un nivel mas con la memoria acotada ---------------------------------------------------------------
    def _plan6(self, m, kd, w, b):
        """Lados con los que la familia (m, kd, w; lote b) hace el nivel extra, o None. Lados multiplos de
        d = 2^(niveles+1) con hoja >= L6_MIN: los mismos. Si no, con L6_PAD > 0: los redondeados a d cuando el volumen
        m kd w crece a lo sumo un factor 1 + L6_PAD (un lado de ~1000 que sube 32: +3,2 %, frente a ~-7 % del nivel)."""
        d = 2 ** (STRASSEN_LEVELS + 1)
        if b < L6_BMIN:
            return None
        M_, K_, W_ = -(-m // d) * d, -(-kd // d) * d, -(-w // d) * d
        if min(M_, K_, W_) // d < L6_MIN:
            return None
        if (M_, K_, W_) == (m, kd, w):
            return (m, kd, w)
        if L6_PAD > 0 and M_ * K_ * W_ <= (1.0 + L6_PAD) * m * kd * w:
            return (M_, K_, W_)
        return None

    def _pad(self, role, A, R, C):
        """Copia de A (b, 1, r, c) en un fondo (b, 1, R, C) con el relleno a cero (cada vez: el fondo es compartido)."""
        b, _, r, c = A.shape
        B = self._buf((role, 1, R, C), (b, 1, R, C))[:b]
        if B.shape[2] != R or B.shape[3] != C:
            B = B[:, :, :R, :C]
        fnp.copyto(B[..., :r, :c], A)
        if R > r:
            fnp.copyto(B[..., r:, :], 0.0)
        if C > c:
            fnp.copyto(B[..., :r, c:], 0.0)
        return B

    def _mm6_pad(self, X, Y, out, lev, pl):
        M_, K_, W_ = pl
        m, w = X.shape[2], Y.shape[3]
        Xp = self._pad("PX", X, M_, K_)
        Yp = self._pad("PY", Y, K_, W_)
        by = Y.shape[0]
        Op = self._buf(("PO", 1, M_, W_), (by, 1, M_, W_))[:by]
        if Op.shape[2] != M_ or Op.shape[3] != W_:
            Op = Op[:, :, :M_, :W_]
        self._mm6(Xp, Yp, Op, lev)
        fnp.copyto(out, Op[..., :m, :w])

    def _hub6_pad(self, X, Y, out, lev, pl):
        M_, K_, W_ = pl
        m, w = X.shape[2], Y.shape[2]
        Xp = self._pad("PX", X, M_, K_)
        Yp = self._pad("PY", Y, W_, K_)
        Op = self._buf(("PH", 1, M_, W_), (1, 1, M_, W_))[0]
        if Op.shape[1] != M_ or Op.shape[2] != W_:
            Op = Op[:, :M_, :W_]
        self._hub6(Xp, Yp, Op, lev)
        fnp.copyto(out, Op[..., :m, :w])

    def _mm6(self, X, Y, out, lev):
        """mm con `lev` niveles (lev >= 2): los de arriba agrupados como en mm (mismas claves de fondo), los dos ultimos
        en la hoja fusionada doble. Lados ya comprobados por _plan6."""
        bx, P, m, kd = X.shape
        by, w = Y.shape[0], Y.shape[3]
        h, q, v = m // 2, kd // 2, w // 2
        XQ = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        YQ = (Y[..., :q, :v], Y[..., :q, v:], Y[..., q:, :v], Y[..., q:, v:])
        if lev <= 2:
            self._leaf_mm2(XQ, YQ, (out[..., :h, :v], out[..., :h, v:], out[..., h:, :v], out[..., h:, v:]),
                           bx, by, P, h, q, v)
            return
        Xc = self._combos(X, XQ, "L", ("L", P, h, q))
        Yc = self._combos(Y, YQ, "R", ("R", P, q, v))
        # el llamado consume Xc en sus combos antes de escribir en out solo si recursa agrupado (lev - 1 > 2)
        mkey = ("L", P, h, q) if (lev - 1 > 2 and q == v) else ("M", P, h, v)
        Mb, Mb4 = self._buf2(mkey, (by, 7, P, h, v))
        self._mm6(Xc, Yc, Mb4, lev - 1)
        self._assemble(Mb, out)

    def _hub6(self, X, Y, out, lev):
        """hub con `lev` niveles (lev >= 2), como _mm6."""
        k, P, m, kd = X.shape
        w = Y.shape[2]
        h, q, v = m // 2, kd // 2, w // 2
        XQ = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        YQ = (Y[..., :v, :q], Y[..., v:, :q], Y[..., :v, q:], Y[..., v:, q:])
        if lev <= 2:
            self._leaf_hub2(XQ, YQ, (out[..., :h, :v], out[..., :h, v:], out[..., h:, :v], out[..., h:, v:]),
                            k, P, h, q, v)
            return
        Xc = self._combos(X, XQ, "L", ("L", P, h, q))
        Yc = self._combos(Y, YQ, "R", ("R", P, v, q), True)
        Mb, Mb4 = self._buf2(("HM", P, h, v), (1, 7, P, h, v))
        self._hub6(Xc, Yc, Mb4[0], lev - 1)
        self._assemble(Mb, out[None])

    def _leaf_mm2(self, XQ, YQ, OQ, bx, by, P, h, q, v):
        """Hoja fusionada doble de mm: el orden y los fondos de _leaf_mm, pero cada uno de los siete productos es a su
        vez una hoja fusionada (_leaf_mm, fondos "2", bloques h/2) sobre el mismo lote P."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = self._buf(("L1", P, h, q), (bx, P, h, q))[:bx]
        Rb = self._buf(("R1", P, q, v), (by, P, q, v))[:by]
        Mb = self._buf(("M1", P, h, v), (by, P, h, v))[:by]
        a, c, e = h // 2, q // 2, v // 2

        def pr(L_, R_, D):
            self._leaf_mm((L_[..., :a, :c], L_[..., :a, c:], L_[..., a:, :c], L_[..., a:, c:]),
                          (R_[..., :c, :e], R_[..., :c, e:], R_[..., c:, :e], R_[..., c:, e:]),
                          (D[..., :a, :e], D[..., :a, e:], D[..., a:, :e], D[..., a:, e:]),
                          bx, by, P, a, c, e, "2")

        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb)
        pr(Lb, Rb, C22)                                               # M6 -> C22
        fnp.add(X21, X22, out=Lb)
        pr(Lb, Y11, C21)                                              # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                               # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb)
        pr(X22, Rb, C11)                                              # M4 -> C11
        fnp.add(C21, C11, out=C21)                                    # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb)
        pr(Lb, Rb, Mb)                                                # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb)
        pr(X11, Rb, C12)                                              # M3 -> C12
        fnp.add(C22, C12, out=C22)                                    # C22 += M3
        fnp.add(X11, X12, out=Lb)
        pr(Lb, Y22, Mb)                                               # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb)
        pr(Lb, Rb, Mb)                                                # M7
        fnp.add(C11, Mb, out=C11)

    def _leaf_hub2(self, XQ, YQ, OQ, k, P, h, q, v):
        """Hoja fusionada doble del hub (como _leaf_mm2): cada producto es una hoja _leaf_hub (fondos "2") que hace
        su suma en k abajo; el producto M1/M5/M7 cae en el fondo KS de la hoja de siempre."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = self._buf(("L1", P, h, q), (k, P, h, q))[:k]
        Rb = self._buf(("R1", P, v, q), (k, P, v, q))[:k]
        Mb = self._buf(("KS", P, h, v), (1, P, h, v))[0]
        a, c, e = h // 2, q // 2, v // 2

        def prod(L_, R_, D):
            self._leaf_hub((L_[..., :a, :c], L_[..., :a, c:], L_[..., a:, :c], L_[..., a:, c:]),
                           (R_[..., :e, :c], R_[..., e:, :c], R_[..., :e, c:], R_[..., e:, c:]),
                           (D[..., :a, :e], D[..., :a, e:], D[..., a:, :e], D[..., a:, e:]),
                           k, P, a, c, e, "2")

        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb); prod(Lb, Rb, C22)   # M6 -> C22
        fnp.add(X21, X22, out=Lb); prod(Lb, Y11, C21)                                # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                                              # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb); prod(X22, Rb, C11)                           # M4 -> C11
        fnp.add(C21, C11, out=C21)                                                   # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb); prod(Lb, Rb, Mb)       # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb); prod(X11, Rb, C12)                           # M3 -> C12
        fnp.add(C22, C12, out=C22)                                                   # C22 += M3
        fnp.add(X11, X12, out=Lb); prod(Lb, Y22, Mb)                                 # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb); prod(Lb, Rb, Mb)  # M7
        fnp.add(C11, Mb, out=C11)


class _StrassenR(_Strassen):
    """_Strassen con fondos independientes de la forma del bloque (muertas/impl): la clave es (papel, P) y el fondo se
    dimensiona al mayor bloque visto; las formas menores usan vistas [:h, :q]. Asi las familias reducidas (lados que
    cambian de capa a capa y de red a red) no crean fondos nuevos. `smin` = hoja minima vigente (por llamada)."""

    def __init__(self, dtype, bmax):
        super().__init__(dtype, bmax)
        self.smin = STRASSEN_MIN

    def _ok(self, m, kd, w, lev):
        d = 2 ** lev
        return (lev > 0 and m % d == 0 and kd % d == 0 and w % d == 0
                and min(m, kd, w) // d >= self.smin)

    def _buf(self, key, shape):
        k2 = (key[0], key[1])
        b = self.pool.get(k2)
        if b is None or b.shape[0] < shape[0] or b.shape[2] < shape[2] or b.shape[3] < shape[3]:
            shp = tuple(shape) if b is None else (max(b.shape[0], shape[0]), shape[1],
                                                  max(b.shape[2], shape[2]), max(b.shape[3], shape[3]))
            b = fnp.empty(shp, dtype=self.dtype)
            fnp.copyto(b, 0.0)
            self.pool[k2] = b
        if b.shape[2] != shape[2] or b.shape[3] != shape[3]:
            return b[:, :, :shape[2], :shape[3]]
        return b

    def _buf2(self, key, shape5):
        k2 = (key[0], key[1])
        e = self.pool.get(k2)
        b = int(shape5[0])
        if e is None or e[0].shape[0] < b or e[0].shape[3] < shape5[3] or e[0].shape[4] < shape5[4]:
            shp = tuple(shape5) if e is None else (max(e[0].shape[0], b), shape5[1], shape5[2],
                                                   max(e[0].shape[3], shape5[3]), max(e[0].shape[4], shape5[4]))
            b5 = fnp.empty(shp, dtype=self.dtype)
            fnp.copyto(b5, 0.0)
            b4 = fnp.reshape(b5, (shp[0], shp[1] * shp[2], shp[3], shp[4]))
            e = (b5, b4)
            self.pool[k2] = e
        if e[0].shape[3] == shape5[3] and e[0].shape[4] == shape5[4]:
            return e[0][:b], e[1][:b]
        return e[0][:b, :, :, :shape5[3], :shape5[4]], e[1][:b, :, :shape5[3], :shape5[4]]


# ---- P2 (03/10/2026): ejecutor de hoja ancha y cobertura de Strassen; mandos apagados = programa enviado -----------
# X_WIDE=1: las familias con operando izquierdo compartido (mm con bx = P = 1) y las del hub (P = 1, suma en k) se
#   multiplican como UN producto ancho: (m x kd) @ [Y_1 | ... | Y_b] (kd x b w) o [X_1 | ... | X_k] [Y_1 | ... | Y_k]^T
#   (m x k kd por w x k kd). Misma aritmetica de Strassen salvo el orden de las sumas; las hojas de BLAS pasan de
#   b 7^L llamadas a 7^L (NumPy llama a BLAS una vez por matriz de la pila). Precio: copiar operandos y salida
#   (b (kd + m) w elementos, ~0,2 % de la familia).
# X_L6W=1 (requiere X_WIDE): un nivel mas en las familias anchas de nivel maximo (hoja X_L6W_MIN).
# X_P2B=<grupos>: productos sueltos de la union y del nivel 2 por Strassen (ver _p2mm).
# X_SB16=1: las familias de la base compartida con hoja minima 16 (un nivel mas en las de r = 352).
WIDE = _os.environ.get("X_WIDE", "0") == "1"
L6W = _os.environ.get("X_L6W", "0") == "1"
L6W_MIN = int(_os.environ.get("X_L6W_MIN", "16"))
P2B = _os.environ.get("X_P2B", "rf,proj,rot,qc,inner,nestt")   # 8.a-quindecies: +qc,inner,nestt (X_P2B=rf,proj,rot = 333752)
P2B_MIN = int(_os.environ.get("X_P2B_MIN", "8"))
P2B_LEV = int(_os.environ.get("X_P2B_LEV", "6"))
SB16 = _os.environ.get("X_SB16", "1") == "1"
_P2S = set(t for t in P2B.split(",") if t)


def _g(smm, tok):
    """Grupo `tok` de X_P2B activo (y ejecutor envuelto, que es el que sabe hacer productos 2-D sueltos)."""
    return isinstance(smm, _Wide) and (tok in _P2S or "all" in _P2S)


def _sbon(smm, i=0):
    if SB16:
        smm.smin = (E29_SB7_MIN if i in E29_SB7 else SB_LO_MIN) if i in SB_LO else SB_MIN


def _sboff(smm):
    if SB16:
        smm.smin = STRASSEN_MIN
SB_MIN = int(_os.environ.get("X_SB_MIN", "16"))
SB_LO = set(int(t) for t in _os.environ.get("X_SB8", "1,2,3,4,5,6,7,8").split(",") if t)   # sitios (1-8) de la base compartida con hoja X_SB_LO_MIN
SB_LO_MIN = int(_os.environ.get("X_SB_LO_MIN", "8"))
# e20_viejo (13/10/2026), ORACULO X_RPROJ=r': la pata derecha (indice b) de la contraccion D21 del nivel viejo se proyecta
# sobre las r' direcciones de mas energia del Gram ponderado de las patas viejas en esta capa (Qc S Qc^T, S = Sg + U S2 U^T).
# La formacion (D3 y las patas a de D21) no se toca. 0 = e16 bit a bit. Solo mide precision (su coste no es el de una
# implementacion).
RPROJ = int(_os.environ.get("X_RPROJ", "0"))
RPROJ_PASS = int(_os.environ.get("X_RPROJ_PASS", "8"))
# e20_viejo X_BUG=l (0 = e16 bit a bit): fusion BUG en cada union del nivel viejo (ver la union en _predict_core).
BUG_L = int(_os.environ.get("X_BUG", "0"))
BUG_PASS = int(_os.environ.get("X_BUG_PASS", "2"))
# e20_viejo ORACULO X_APP=l (0 = e16 bit a bit): base del nivel viejo SIN truncar. La primera union fija r0 = X_APP_R0
# columnas (buscador de rango de e16); cada union siguiente conserva entera la base transportada (solo la ortonormaliza:
# factores viejos <- R0 F) y le anade l direcciones nuevas del residuo de la entrante. V21_R_OLD = tamano maximo del
# bufer (>= r0 + 9 l), un solo nivel (V24_AGE_OLD2=0). Mide precision; su coste (todo a V21_R_OLD) no es el de una
# implementacion por bloques.
APP_L = int(_os.environ.get("X_APP", "-1"))   # -1 = apagado; 0 = base transportada sin direcciones nuevas
APP_R0 = int(_os.environ.get("X_APP_R0", "352"))
# e20_viejo X_FIJA=1 (0 = e16 bit a bit; pide un solo nivel, V24_AGE_OLD2=0): BASE FIJA del nivel viejo. La primera union
# elige la base (buscador de rango de e16, rango V21_R_OLD); despues la base solo se transporta (Qc <- W diag(w1) Qc, sin
# reortonormalizar) y nunca se trunca ni se rota: cada entrante se proyecta sobre ella por minimos cuadrados (QR de la
# base transportada + solve con R) y entra con sus factores en ese marco. Sin buscador, sin rotacion de los factores
# viejos y sin nucleo de Gram en las uniones siguientes. Igual en aritmetica exacta que el oraculo X_APP=0.
FIJA = _os.environ.get("X_FIJA", "1") == "1"
# X_FIJA_RC=r' (con X_FIJA): la base fija se ordena una vez, en la primera union, por el Gram de la primera entrante
# (r' direcciones mayores primero; iteracion de subespacio de X_FIJA_PASS pasadas) y la pata derecha de la contraccion D21
# del nivel viejo usa solo sus r' primeras columnas. La formacion (D3 y patas a) sigue a rango completo.
FIJA_RC = int(_os.environ.get("X_FIJA_RC", "0"))
FIJA_PASS = int(_os.environ.get("X_FIJA_PASS", "8"))
# X_FIJA_PROJ: como se proyecta la entrante sobre la base fija. "qr" = QR de la base + solve(R, Q^T [Aj|Pj]);
# "qrm" = QR + M = solve(R, Q^T) (r x filas) y luego M [Aj|Pj]; "ne" = ecuaciones normales M = solve(Qp^T Qp, Qp^T).
FIJA_PROJ = _os.environ.get("X_FIJA_PROJ", "qrm")
# e24hot X_VB_HOT=t (0 = e20_viejo bit a bit): nivel viejo barato. Las filas de la capa se ordenan [vivas con alfa <= t |
# vivas con alfa > t (calientes) | muertas]; las patas densas del nivel viejo (formacion, contraccion D21 e inner Qc^T) se
# hacen solo en las primeras `mo` filas (no calientes, redondeado a DEAD_MULT); las demas quedan a cero. Nunca en las fuentes
# jovenes (x17). Ver tasks/data/foco/piezas/viejo_barato.md.
VB_HOT = float(_os.environ.get("X_VB_HOT", "3.0"))


def _eye_cols(m, r, dt):
    e = fnp.zeros((m, r), dtype=dt)
    fnp.fill_diagonal(e, 1.0)
    return e


# e28fijo (13/10/2026): ahorros EXACTOS del coste que no escala con las vivas; apagados (X_E28 vacio) = e20_viejo / e24hot bit a bit.
#   ueye: con base fija U = [I; 0] (seleccion de columnas): Qc U, y la elevacion inner2 U^T, son copias y no productos (mismos valores)
#   tri:  M = R^-1 Q^T (R triangular) por biseccion en bloques (d niveles): matmul + solve de bloques en vez de LU denso
#   nest: F2 de la saliente = R22^-1 R0[:r2,:] F (identica a G22^-1 G2, G = R^T R, R triangular superior), sin formar G, solo las
#         columnas vivas de la saliente y con la solve por bloques; nest2: F2 = F[:r2] + R22^-1 (R12 F[r2:]) (medio producto)
#   oldc: las patas formadas de las fuentes viejas (tier 1 y 2) se escriben ya compactas en un fondo propio y _dslices las lee de
#         ahi: se ahorran la devolucion al orden natural (take) y la recogida de esas fuentes (mismos valores)
E28 = set(t for t in _os.environ.get("X_E28", "ueye+tri+nest+nest2+oldc+fj").replace("+", ",").split(",") if t)   # separador "," o "+" (vara.py separa mandos por comas)
E28_UEYE = "ueye" in E28 and FIJA
E28_TRI = int(_os.environ.get("X_E28_TRI", "2")) if "tri" in E28 else 0
E28_NEST = "nest" in E28 and FIJA
E28_NEST2 = "nest2" in E28 and FIJA
E28_TRI2 = int(_os.environ.get("X_E28_TRI2", "1"))
E28_OLDC = "oldc" in E28
E28_FJ = "fj" in E28 and FIJA   # la proyeccion de la entrante se escribe directamente en su hueco de la base fija
# e29coste (13/10/2026): ahorros exactos sobre e28fijo; X_E29 vacio = e28fijo bit a bit.
#   nestg: F2 = F[:r2] + G F[r2:] con G = R0[:r2,:r2]^-1 R0[:r2,r2:] (r2 x r2, una solve de r2 columnas) en vez de R0[:r2,:r2]^-1 (R0[:r2,r2:] F[r2:]) (solve de 2 wj columnas)
#   sgdead: no se calcula Sg = Vr^T Sg Vr tras ordenar la base fija (nadie lo vuelve a leer con la base fija)
#   nosymc: sin la etiqueta as_symmetric de C (7 n^2 facturados por capa; ninguna operacion posterior la aprovecha)
E29 = set(t for t in _os.environ.get("X_E29", "nestg+sgdead+nosymc").replace("+", ",").split(",") if t)
E29_NESTG = "nestg" in E29 and FIJA
E29_SGDEAD = "sgdead" in E29 and FIJA
E29_NOSYMC = "nosymc" in E29
E29_SB7 = set(int(t) for t in _os.environ.get("X_E29_SB7", "").split(",") if t)   # sitios 1-8 de la base compartida con hoja minima X_E29_SB7_MIN (los de rango r2 = 224 caben a 5 niveles con hoja 7)
E29_SB7_MIN = int(_os.environ.get("X_E29_SB7_MIN", "7"))


def _tri_solve(R, B, d):
    """R X = B con R triangular superior (n x n): d niveles de biseccion en bloques (X2 = R22^-1 B2; X1 = R11^-1 (B1 - R12 X2));
    las hojas con la solve densa. Mismo resultado que fnp.linalg.solve(R, B) salvo redondeo; factura ~n^2 nrhs en vez de 2n^2 nrhs."""
    n = R.shape[0]
    if d <= 0 or n % 2:
        return fnp.linalg.solve(R, B)
    h = n // 2
    X2 = _tri_solve(R[h:, h:], B[h:], d - 1)
    X1 = _tri_solve(R[:h, :h], fnp.subtract(B[:h], fnp.matmul(R[:h, h:], X2)), d - 1)
    return fnp.concatenate([X1, X2], axis=0)


def _rproj_right(inner, Qc, S, rp, rows=None):
    """inner Qc^T Pi con Pi = proyector ortogonal sobre las rp direcciones mayores de Qc S Qc^T (iteracion de subespacio)."""
    Q, R = fnp.linalg.qr(Qc)
    M = R @ (S @ R.T)
    Wv = fnp.asarray(M[:, :rp])
    for _ in range(RPROJ_PASS):
        Wv, _r = fnp.linalg.qr(M @ Wv)
    X = inner @ (R.T @ Wv)                 # (n, rp)
    QW = Q @ Wv                            # (n, rp)
    if rows is not None:
        QW = fnp.take(QW, rows, axis=0)
    return X @ QW.T


class _StrassenK1(_StrassenR):
    """_StrassenR para las familias anchas: el hub con k = 1 escribe el producto de hoja directamente en su destino
    (sin el fondo T ni la suma en k, que con k = 1 es una copia facturada)."""

    def hub(self, X, Y, out, lev):
        k = X.shape[0]
        if k != 1:
            return _StrassenR.hub(self, X, Y, out, lev)
        _, P, m, kd = X.shape
        w = Y.shape[2]
        if not self._ok(m, kd, w, lev):
            fnp.matmul(X[0], fnp.swapaxes(Y[0], -1, -2), out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        XQ = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        YQ = (Y[..., :v, :q], Y[..., v:, :q], Y[..., :v, q:], Y[..., v:, q:])
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            self._leaf_hub1(XQ, YQ, (out[..., :h, :v], out[..., :h, v:], out[..., h:, :v], out[..., h:, v:]), P, h, q, v)
            return
        stv = STV and _stv_ok(X, Y, out[None], True)
        Xc = self._combos(X, XQ, "L", ("L", P, h, q), False, stv)
        Yc = self._combos(Y, YQ, "R", ("R", P, v, q), True, stv)
        Mb, Mb4 = self._buf2(("HM", P, h, v), (1, 7, P, h, v))
        self.hub(Xc, Yc, Mb4[0], lev - 1)
        self._assemble(Mb, out[None], stv)

    def _leaf_hub1(self, XQ, YQ, OQ, P, h, q, v):
        X11, X12, X21, X22 = (a[0] for a in XQ)
        Y11, Y12, Y21, Y22 = (a[0] for a in YQ)
        C11, C12, C21, C22 = OQ
        Lb = self._buf(("L1", P, h, q), (1, P, h, q))[0]
        Rb = self._buf(("R1", P, v, q), (1, P, v, q))[0]
        Mb = self._buf(("KS", P, h, v), (1, P, h, v))[0]

        def prod(L_, R_, dst):
            fnp.matmul(L_, fnp.swapaxes(R_, -1, -2), out=dst)

        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb); prod(Lb, Rb, C22)   # M6 -> C22
        fnp.add(X21, X22, out=Lb); prod(Lb, Y11, C21)                                # M2 -> C21
        fnp.subtract(C22, C21, out=C22)
        fnp.subtract(Y21, Y11, out=Rb); prod(X22, Rb, C11)                           # M4 -> C11
        fnp.add(C21, C11, out=C21)
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb); prod(Lb, Rb, Mb)       # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb); prod(X11, Rb, C12)                           # M3 -> C12
        fnp.add(C22, C12, out=C22)
        fnp.add(X11, X12, out=Lb); prod(Lb, Y22, Mb)                                 # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb); prod(Lb, Rb, Mb)  # M7
        fnp.add(C11, Mb, out=C11)


class _Wide(_StrassenR):
    """Envoltorio del ejecutor (X_WIDE / X_P2B): delega en `inner` salvo las familias que admiten la forma ancha,
    que copia a fondos 2-D (vistas basicas: gratis) y multiplica con su propio ejecutor `w` (fondos aparte: mezclar
    lotes de b y anchos de b w en las mismas claves inflaria los fondos al producto de los maximos)."""

    def __init__(self, inner):
        self.inner = inner
        self.w = _StrassenK1(inner.dtype, inner.bmax)
        self.dtype = inner.dtype
        self.bmax = inner.bmax
        self.pool = inner.pool
        self.p2 = {}
        self.top = STRASSEN_LEVELS

    @property
    def smin(self):
        return self.inner.smin

    @smin.setter
    def smin(self, v):
        self.inner.smin = v

    def level(self, m, kd, w, lev):
        return self.inner.level(m, kd, w, lev)

    def _ok(self, m, kd, w, lev):
        return self.inner._ok(m, kd, w, lev)

    def _b2(self, key, r, c):
        b = self.p2.get(key)
        if b is None or b.shape[0] < r or b.shape[1] < c:
            shp = (r, c) if b is None else (max(b.shape[0], r), max(b.shape[1], c))
            b = fnp.empty(shp, dtype=self.dtype)
            fnp.copyto(b, 0.0)
            self.p2[key] = b
        return b[:r, :c]

    def _lev(self, m, kd, w, lev):
        """Nivel de la familia ancha: el del llamador; con X_L6W uno mas en las de nivel maximo si los lados lo admiten."""
        self.w.smin = self.inner.smin
        if L6W and lev >= self.top:
            sm = self.w.smin
            self.w.smin = min(sm, L6W_MIN)
            if self.w._ok(m, kd, w, lev + 1):
                return lev + 1
            self.w.smin = sm
        return lev

    def mm(self, X, Y, out, lev):
        bx, P, m, kd = X.shape
        by, Py, _, w = Y.shape
        if not (WIDE and bx == 1 and P == 1 and Py == 1 and by > 1 and lev > 0):
            return self.inner.mm(X, Y, out, lev)
        Yw = self._b2("wy", kd, by * w)
        for i in range(by):
            fnp.copyto(Yw[:, i * w:(i + 1) * w], Y[i, 0])
        Ow = self._b2("wo", m, by * w)
        lw = self._lev(m, kd, by * w, lev)
        self.w.mm(X, Yw[None, None], Ow[None, None], lw)
        for i in range(by):
            fnp.copyto(out[i, 0], Ow[:, i * w:(i + 1) * w])

    def hub(self, X, Y, out, lev):
        k, P, m, kd = X.shape
        w = Y.shape[2]
        if not (WIDE and P == 1 and k > 1 and lev > 0):
            return self.inner.hub(X, Y, out, lev)
        Xw = self._b2("hx", m, k * kd)
        Yw = self._b2("hy", w, k * kd)
        for i in range(k):
            fnp.copyto(Xw[:, i * kd:(i + 1) * kd], X[i, 0])
            fnp.copyto(Yw[:, i * kd:(i + 1) * kd], Y[i, 0])
        lw = self._lev(m, k * kd, w, lev)
        self.w.hub(Xw[None, None], Yw[None, None], out, lw)

    def mm2(self, A, B, out, lev=None, smin=None):
        """Producto 2-D suelto A (m x kd) @ B (kd x w) -> out por el ejecutor ancho (X_P2B); nivel el mayor <= lev
        que admiten los lados con hoja >= smin; 0 = producto denso."""
        m, kd = A.shape
        w = B.shape[1]
        self.w.smin = P2B_MIN if smin is None else smin
        lv = self.w.level(m, kd, w, P2B_LEV if lev is None else lev)
        if lv == 0:
            return fnp.matmul(A, B, out=out)
        self.w.mm(A[None, None], B[None, None], out[None, None], lv)
        return out


# ---- BANCO (Ouroboros, 01/10/2026): mandos de las mejoras portadas; apagados, el fichero es el original -----------
BANCO = {
    "vistas_cache": False,   # _StrassenVC: las vistas de los fondos de Strassen se crean una vez y se reutilizan.
                             # Rebanar un array de flopscope cuesta ~3 us de tiempo residual y v29 hace ~14.600
                             # rebanados por prediccion (~0,05 de sus 0,32 s). Exacto: mismas operaciones y FLOPs.
}


class _StrassenVC(_Strassen):
    """_Strassen con cache de vistas. Solo se guardan vistas de objetos PROPIOS (los fondos y las vistas que esta
    clase crea de ellos): la cache guarda la referencia al objeto, asi que su id() no puede reutilizarse. Las vistas
    de los operandos del que llama (nivel 0) se crean en cada llamada, como en el original. Si un fondo crece, la
    cache entera se vacia."""

    def __init__(self, dtype, bmax):
        super().__init__(dtype, bmax)
        self.vc = {}      # (id(obj), clave) -> (obj, vistas)
        self.b2 = {}      # (clave del fondo, b) -> (entrada del fondo, v5, v4)
        self._propios = {}   # id(obj) -> obj

    def _prop(self, obj, clave, crea):
        """Vistas de `obj` bajo `clave`: de la cache si `obj` es propio; si no, recien creadas."""
        k = (id(obj), clave)
        e = self.vc.get(k)
        if e is not None and e[0] is obj:
            return e[1]
        v = crea()
        if self._propios.get(id(obj)) is obj:
            self.vc[k] = (obj, v)
        return v

    def _marca(self, obj):
        self._propios[id(obj)] = obj
        return obj

    def _buf2(self, key, shape5):
        b = int(shape5[0])
        c = self.b2.get((key, b))
        e = self.pool.get(key)
        if c is not None and c[0] is e:
            return c[1], c[2]
        if e is None or e[0].shape[0] < b:
            b5 = fnp.empty(shape5, dtype=self.dtype)
            fnp.copyto(b5, 0.0)
            b4 = fnp.reshape(b5, (b, shape5[1] * shape5[2], shape5[3], shape5[4]))
            e = (b5, b4)
            self.pool[key] = e
            self.vc.clear()
            self.b2.clear()
            self._propios = {}
        v5, v4 = self._marca(e[0][:b]), self._marca(e[1][:b])
        self.b2[(key, b)] = (e, v5, v4)
        return v5, v4

    def _cuartos(self, X, a, c, modo="xy"):
        """modo 'xy': (X11, X12, X21, X22) partiendo filas en a y columnas en c; modo 'yt': los bloques de Y^T
        expresados sobre Y sin trasponer (filas c, columnas j), como en hub."""
        if modo == "xy":
            return self._prop(X, ("q", a, c), lambda: (X[..., :a, :c], X[..., :a, c:], X[..., a:, :c], X[..., a:, c:]))
        return self._prop(X, ("t", a, c), lambda: (X[..., :a, :c], X[..., a:, :c], X[..., :a, c:], X[..., a:, c:]))

    def _siete(self, buf):
        return self._prop(buf, "7", lambda: tuple(buf[:, i] for i in range(7)))

    def _combos(self, X, Q, kind, key, sw=False):
        b, P, m, kd = X.shape
        h, q = m // 2, kd // 2
        buf, buf4 = self._buf2(key, (b, 7, P, h, q))
        X11, X12, X21, X22 = Q
        B = self._siete(buf)
        if kind == "L":
            fnp.add(X11, X22, out=B[0]); fnp.add(X21, X22, out=B[1])
            fnp.copyto(B[2], X11);       fnp.copyto(B[3], X22)
            fnp.add(X11, X12, out=B[4]); fnp.subtract(X21, X11, out=B[5])
            fnp.subtract(X12, X22, out=B[6])
        else:
            fnp.add(X11, X22, out=B[0]); fnp.copyto(B[1], X11)
            fnp.subtract(X12, X22, out=B[2]); fnp.subtract(X21, X11, out=B[3])
            fnp.copyto(B[4], X22);       fnp.add(X11, X12, out=B[5])
            fnp.add(X21, X22, out=B[6])
        return buf4

    def _monta(self, M, out):
        """_assemble con las vistas de M y de out en cache (8 operaciones, las mismas)."""
        h, w = M.shape[3], M.shape[4]
        C11, C12, C21, C22 = self._cuartos(out, h, w)
        m = self._siete(M)
        fnp.add(m[0], m[3], out=C11); fnp.subtract(C11, m[4], out=C11)
        fnp.add(C11, m[6], out=C11)
        fnp.add(m[2], m[4], out=C12)
        fnp.add(m[1], m[3], out=C21)
        fnp.subtract(m[0], m[1], out=C22); fnp.add(C22, m[2], out=C22)
        fnp.add(C22, m[5], out=C22)

    def mm(self, X, Y, out, lev):
        bx, P, m, kd = X.shape
        by, w = Y.shape[0], Y.shape[3]
        if not self._ok(m, kd, w, lev):
            fnp.matmul(X, Y, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        XQ = self._cuartos(X, h, q)
        YQ = self._cuartos(Y, q, v)
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            self._leaf_mm(XQ, YQ, self._cuartos(out, h, v), bx, by, P, h, q, v)
            return
        Xc = self._combos(X, XQ, "L", ("L", P, h, q))
        Yc = self._combos(Y, YQ, "R", ("R", P, q, v))
        callee_recurses = self._ok(h // 2, q // 2, v // 2, lev - 2)
        mkey = ("L", P, h, q) if (callee_recurses and q == v) else ("M", P, h, v)
        Mb, Mb4 = self._buf2(mkey, (by, 7, P, h, v))
        self.mm(Xc, Yc, Mb4, lev - 1)
        self._monta(Mb, out)

    def hub(self, X, Y, out, lev):
        k, P, m, kd = X.shape
        w = Y.shape[2]
        if not self._ok(m, kd, w, lev):
            T = self._buf(("K", P, m, w), (k, P, m, w))[:k]
            fnp.matmul(X, fnp.swapaxes(Y, -1, -2), out=T)
            fnp.sum(T, axis=0, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        XQ = self._cuartos(X, h, q)
        YQ = self._cuartos(Y, v, q, "yt")
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            self._leaf_hub(XQ, YQ, self._cuartos(out, h, v), k, P, h, q, v)
            return
        Xc = self._combos(X, XQ, "L", ("L", P, h, q))
        Yc = self._combos(Y, YQ, "R", ("R", P, v, q), True)
        Mb, Mb4 = self._buf2(("HM", P, h, v), (1, 7, P, h, v))
        sal = self._prop(Mb4, "0", lambda: self._marca(Mb4[0]))
        self.hub(Xc, Yc, sal, lev - 1)
        if self._propios.get(id(out)) is out:
            self._monta(Mb, self._prop(out, "N", lambda: self._marca(out[None])))
        else:
            self._monta(Mb, out[None])


class _Pool:
    """V27 (F80): persistent named scratch buffers. A fresh (n,n) result costs ~0.10 ms
    of residual (allocation + wrapper), the same op written with out= into a pooled
    buffer ~0.013 ms; the pool lives on the estimator across predict() calls (per-MLP
    allocation made the residual creep over a suite, F79). Keys carry the shape, so a
    different width simply adds buffers."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.bufs = {}

    def get(self, name, shape):
        key = (name, tuple(int(x) for x in shape))
        b = self.bufs.get(key)
        if b is None:
            b = fnp.empty(key[1], dtype=self.dtype)
            fnp.copyto(b, 0.0)   # metered page touch (see class note)
            self.bufs[key] = b
        return b

    def get_pair(self, name, shape):
        """V28: an (S, 2, a, b) slot buffer together with its one-time (2S, 1, a, b) view
        (fnp.reshape is a logged, per-element billed op; slabs of whole slots stay
        contiguous, so [2i:2j] of the view is slots i..j-1 with the pair axis folded
        into the Strassen batch)."""
        key = ("pair", name, tuple(int(x) for x in shape))
        e = self.bufs.get(key)
        if e is None:
            b5 = fnp.empty(key[2], dtype=self.dtype)
            fnp.copyto(b5, 0.0)
            b4 = fnp.reshape(b5, (key[2][0] * key[2][1], 1, key[2][2], key[2][3]))
            e = (b5, b4)
            self.bufs[key] = e
        return e


# 8.a-quindecies (3): reserva de fondos en setup() (no facturado). X_PREALLOC=1: a 1024x16 crea en setup el _Pool y
# el ejecutor de Strassen con los fondos ya dimensionados (formas maximas registradas en redes del Mini que no son las
# de prueba; ops/registra.py) con fnp.zeros (paginas perezosas: la reserva no toca memoria) y marca la primera llamada
# como no primera (sin el tope de STRASSEN_FIRST niveles). Un fondo que falte se crea en predict como antes.
PREALLOC = _os.environ.get("X_PREALLOC", "1") == "1"   # X_PREALLOC=0 = 333752
PREALLOC_LEV = _os.environ.get("X_PREALLOC_LEV", "1") == "1"
_PRE_TABLA = [
    ('k1', ('L', 1), 'T', (1, 7, 1, 496, 512)),
    ('k1', ('L', 49), 'T', (1, 7, 49, 124, 128)),
    ('k1', ('L', 7), 'T', (1, 7, 7, 248, 256)),
    ('k1', ('L1', 343), 'A', (1, 343, 62, 64)),
    ('k1', ('M', 1), 'T', (1, 7, 1, 496, 496)),
    ('k1', ('M', 49), 'T', (1, 7, 49, 124, 124)),
    ('k1', ('M', 7), 'T', (1, 7, 7, 248, 248)),
    ('k1', ('M1', 343), 'A', (1, 343, 62, 62)),
    ('k1', ('R', 1), 'T', (1, 7, 1, 512, 496)),
    ('k1', ('R', 49), 'T', (1, 7, 49, 128, 124)),
    ('k1', ('R', 7), 'T', (1, 7, 7, 256, 248)),
    ('k1', ('R1', 343), 'A', (1, 343, 64, 62)),
    ('pool', ('abbuf', (52, 1024, 1024)), 'A', (52, 1024, 1024)),
    ('pool', ('abnat', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('ap', (15, 1024, 1024)), 'A', (15, 1024, 1024)),
    ('pool', ('bm', (8, 1024)), 'A', (8, 1024)),
    ('pool', ('c', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('cc', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('cpc', (1025, 1025)), 'A', (1025, 1025)),
    ('pool', ('cpre', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('cpx', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('cpy', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('d21', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('d21fb', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('d21new', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('d21w', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('d3col', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('d3p', (1025,)), 'A', (1025,)),
    ('pool', ('g4col', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('gxr', (15, 1024, 4)), 'A', (15, 1024, 4)),
    ('pool', ('gy', (30, 1, 1024, 1024)), 'A', (30, 1, 1024, 1024)),
    ('pool', ('gy2', (30, 1, 1024, 1024)), 'A', (30, 1, 1024, 1024)),
    ('pool', ('apc', (30, 1, 1024, 1024)), 'A', (30, 1, 1024, 1024)),
    ('pool', ('gyr', (15, 1024, 4)), 'A', (15, 1024, 4)),
    ('pool', ('hub', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('hubc', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('inner', (1, 1024, 352)), 'A', (1, 1024, 352)),
    ('pool', ('inner2', (1, 1024, 224)), 'A', (1, 1024, 224)),
    ('pool', ('k21', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('k22', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('l', (15, 1024, 10)), 'A', (15, 1024, 10)),
    ('pool', ('lift', (1024, 352)), 'A', (1024, 352)),
    ('pool', ('mp', (15, 1024, 1024)), 'A', (15, 1024, 1024)),
    ('pool', ('om', (1024, 352)), 'A', (1024, 352)),
    ('pool', ('om_r', (1024, 8)), 'A', (1024, 8)),
    ('pool', ('omf', (1024, 4)), 'A', (1024, 4)),
    ('pool', ('p2y1', (2, 1, 352, 1024)), 'A', (2, 1, 352, 1024)),
    ('pool', ('pair', 'AP0', (15, 2, 1024, 1024)), 'T', (15, 2, 1024, 1024)),
    ('pool', ('pair', 'AP1', (15, 2, 1024, 1024)), 'T', (15, 2, 1024, 1024)),
    ('pool', ('pair', 'lap', (15, 2, 1024, 1024)), 'T', (15, 2, 1024, 1024)),
    ('pool', ('pair', ('fap', 0), (15, 2, 352, 1024)), 'T', (15, 2, 352, 1024)),
    ('pool', ('pair', ('fap', 1), (15, 2, 352, 1024)), 'T', (15, 2, 352, 1024)),
    ('pool', ('pair', ('fap2', 0), (15, 2, 224, 1024)), 'T', (15, 2, 224, 1024)),
    ('pool', ('pair', ('fap2', 1), (15, 2, 224, 1024)), 'T', (15, 2, 224, 1024)),
    ('pool', ('pk11', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('pk22', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('pk2aud', (3, 1024, 1024)), 'A', (3, 1024, 1024)),
    ('pool', ('pp', (15, 1024, 1024)), 'A', (15, 1024, 1024)),
    ('pool', ('ppl', (15, 1024, 10)), 'A', (15, 1024, 10)),
    ('pool', ('qp', (1024, 352)), 'A', (1024, 352)),
    ('pool', ('r1t', (15, 1024, 4)), 'A', (15, 1024, 4)),
    ('pool', ('r2t', (15, 1024, 4)), 'A', (15, 1024, 4)),
    ('pool', ('rep21', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('rr', (1024, 10)), 'A', (1024, 10)),
    ('pool', ('s21', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('ssep', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('sym_o', (3, 1, 512, 512)), 'A', (3, 1, 512, 512)),
    ('pool', ('sym_x', (3, 1, 512, 1024)), 'A', (3, 1, 512, 1024)),
    ('pool', ('sym_y', (3, 1, 1024, 512)), 'A', (3, 1, 1024, 512)),
    ('pool', ('t', (15, 1024, 1024)), 'A', (15, 1024, 1024)),
    ('pool', ('t1', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('u', (15, 1024, 1024)), 'A', (15, 1024, 1024)),
    ('pool', ('wcc', (1, 1, 1024, 1024)), 'A', (1, 1, 1024, 1024)),
    ('pool', ('wd', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('wdc', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('wg', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('wk431', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('wk4m', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('wq', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('wqj', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('wr', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('ww', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('x1b', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('xt', (15, 1024, 1024)), 'A', (15, 1024, 1024)),
    ('pool', ('xtb', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('y', (1024, 8)), 'A', (1024, 8)),
    ('pool', ('y1b', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('y2', (1024, 8)), 'A', (1024, 8)),
    ('pool', ('yq', (1024, 352)), 'A', (1024, 352)),
    ('pool', ('yq1', (1024, 352)), 'A', (1024, 352)),
    ('pool', ('yq2', (1024, 352)), 'A', (1024, 352)),
    ('pool', ('yt', (15, 1024, 1024)), 'A', (15, 1024, 1024)),
    ('pool', ('ytb', (1024, 1024)), 'A', (1024, 1024)),
    ('pool', ('zc', (15, 1025, 10)), 'A', (15, 1025, 10)),
    ('pool', ('zfc', (15, 1025, 8)), 'A', (15, 1025, 8)),
    ('pool', (('qc', 0), (1024, 352)), 'A', (1024, 352)),
    ('pool', (('qc', 1), (1024, 352)), 'A', (1024, 352)),
    ('pool', (('qcp', 0), (1025, 352)), 'A', (1025, 352)),
    ('pool', (('qcp', 1), (1025, 352)), 'A', (1025, 352)),
    ('pool', (('z', 0), (15, 1024, 10)), 'A', (15, 1024, 10)),
    ('pool', (('z', 1), (15, 1024, 10)), 'A', (15, 1024, 10)),
    ('pool', (('zf', 0), (15, 1024, 8)), 'A', (15, 1024, 8)),
    ('pool', (('zf', 1), (15, 1024, 8)), 'A', (15, 1024, 8)),
    ('sr', ('HM', 1), 'T', (1, 7, 1, 512, 512)),
    ('sr', ('HM', 343), 'T', (1, 7, 343, 64, 64)),
    ('sr', ('HM', 49), 'T', (1, 7, 49, 128, 128)),
    ('sr', ('HM', 7), 'T', (1, 7, 7, 256, 256)),
    ('sr', ('K', 2401), 'A', (8, 2401, 32, 32)),
    ('sr', ('K', 343), 'A', (14, 343, 120, 64)),
    ('sr', ('KS', 2401), 'A', (1, 2401, 32, 32)),
    ('sr', ('KS', 343), 'A', (1, 343, 64, 64)),
    ('sr', ('L', 1), 'T', (14, 7, 1, 512, 512)),
    ('sr', ('L', 343), 'T', (8, 7, 343, 64, 64)),
    ('sr', ('L', 49), 'T', (14, 7, 49, 128, 128)),
    ('sr', ('L', 7), 'T', (14, 7, 7, 256, 256)),
    ('sr', ('L1', 2401), 'A', (8, 2401, 32, 32)),
    ('sr', ('L1', 343), 'A', (8, 343, 64, 64)),
    ('sr', ('M', 1), 'T', (14, 7, 1, 496, 512)),
    ('sr', ('M', 343), 'T', (9, 7, 343, 64, 64)),
    ('sr', ('M', 49), 'T', (14, 7, 49, 128, 128)),
    ('sr', ('M', 7), 'T', (14, 7, 7, 248, 256)),
    ('sr', ('M1', 2401), 'A', (9, 2401, 32, 32)),
    ('sr', ('M1', 343), 'A', (9, 343, 64, 64)),
    ('sr', ('R', 1), 'T', (14, 7, 1, 512, 512)),
    ('sr', ('R', 343), 'T', (9, 7, 343, 64, 64)),
    ('sr', ('R', 49), 'T', (14, 7, 49, 128, 128)),
    ('sr', ('R', 7), 'T', (14, 7, 7, 256, 256)),
    ('sr', ('R1', 2401), 'A', (9, 2401, 32, 32)),
    ('sr', ('R1', 343), 'A', (9, 343, 64, 64)),
]


class Estimator(BaseEstimator):
    AGE_OLD = int(_os.environ.get("V21_AGE_OLD", "4"))   # V21: sources with age > AGE_OLD are confined
    R_OLD = int(_os.environ.get("V21_R_OLD", "416"))     # V21: shared basis rank (lean ladder F72)
    AGE_OLD2 = int(_os.environ.get("V24_AGE_OLD2", "8"))  # V24: age gate of the nested tier (0 = off)
    R_OLD2 = int(_os.environ.get("V24_R_OLD2", "224"))    # V24: rank of the nested sub-basis U (8-dump frontier runs/v24_7_*_d0-7.log: 224 best, 128 cliff)
    QPASS2 = int(_os.environ.get("V24_QPASS2", "2"))      # V24: passes of the r1-space range finder
    R_FB = int(_os.environ.get("V18_R_FB", "4"))    # V18: rank of the D21 feedback thin legs (F69 lean ladder: 8/16/32 -> 2.17/2.14/2.14e-8)
    R_RES = int(_os.environ.get("V17_R_RES", "8"))   # rank of the S21 residual leg (V17 ladder on dumps 0/1: 16/32/64 -> 2.35/2.39/2.41e-8 at 0.492/0.502/0.518xB)

    def __init__(self) -> None:
        self._setup_rng = None
        self._wick_consts = _build_wick_consts()
        self._progs = {0: _term_prog(0), 1: _term_prog(1), 2: _term_prog(2)}
        self._ind2b = {}
        for _m, _p in self._progs.items():
            _g = [max(range(len(r)), key=lambda c: r[c]) for r in _p["IND2"]]
            _b = []
            for _G in range(len(D2_IPS)):
                _idx = [t for t, x in enumerate(_g) if x == _G]
                if _idx:
                    assert _idx == list(range(_idx[0], _idx[-1] + 1))
                _b.append((_idx[0], _idx[-1] + 1) if _idx else (0, 0))
            self._ind2b[_m] = _b
        self._i11 = WICK_PAIRS.index((1, 1))
        self._i21 = WICK_PAIRS.index((2, 1))
        self._i12 = WICK_PAIRS.index((1, 2))
        self._i31 = WICK_PAIRS.index((3, 1))

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        self._lt = None
        if not LT_OFF:
            sub = getattr(ctx, "submission_dir", None) or _os.path.dirname(_os.path.abspath(__file__))
            path = _os.path.join(str(sub), LT_FILE)
            if _os.path.exists(path):
                # standardisation and the per-layer output scale are folded into W1/b1/w4/b4
                # offline; per-layer pieces are split once here (a slice of a server array is
                # a round trip, so predict() never slices them)
                d = fnp.load(path)
                nl = int(d["W1"].shape[0])
                b4 = [float(x) for x in d["b4"].tolist()]
                self._lt = {"W1L": [d["W1"][i] for i in range(nl)],
                            "b1L": [d["b1"][i] for i in range(nl)],
                            "w4L": [d["w4"][i] for i in range(nl)],
                            "b4L": b4,
                            "W2": d["W2"], "b2": d["b2"], "W3": d["W3"], "b3": d["b3"]}
            path2 = _os.path.join(str(sub), LT2_FILE)
            if _LT2_BLOB or _os.path.exists(path2):
                # LT2 (lt2_export.py, embedded by lt2_embed.py): shared W1/Wo; per layer only the bias,
                # the 3 uin row scalings and es differ. Per-layer arrays are built once here (setup
                # FLOPs are free; predict() never slices a server array)
                d = _lt2_embedded() if _LT2_BLOB else fnp.load(path2)
                fl = [int(x) for x in d["flags"].tolist()]
                nl = int(d["b1"].shape[0]); fin = int(d["W1"].shape[0])
                es_ = [float(x) for x in d["es"].tolist()]
                W1 = d["W1"]; Wo = d["Wo"]; bo = d["bo"]
                usc = d["usc"].tolist()
                W1L = []
                for i in range(nl):
                    if fl[3]:
                        rs = fnp.array([1.0] * (fin - 3) + [float(x) for x in usc[i]], dtype=fnp.float32)
                        W1L.append(W1 * rs[:, None])
                    else:
                        W1L.append(W1)
                wo0 = Wo[:, 0]; bo0 = float(bo.tolist()[0])
                lt2 = {"v2": True, "Fo": fl[0], "xall": bool(fl[1]), "D": fl[2], "uin": bool(fl[3]),
                       "relu": bool(fl[5]), "embw": len(fl) > 6 and bool(fl[6]),
                       "W1L": W1L, "b1L": [d["b1"][i] for i in range(nl)],
                       "woL": [wo0 * es_[i] for i in range(nl)], "boL": [bo0 * es_[i] for i in range(nl)],
                       "mzoL": [d["mz"][i][:fl[0]] for i in range(nl)], "izoL": [d["iz"][i][:fl[0]] for i in range(nl)],
                       "mzvL": [d["mz"][i][fl[0]:] for i in range(nl)], "izvL": [d["iz"][i][fl[0]:] for i in range(nl)],
                       "m2L": [d["m2"][i] for i in range(nl)], "i2L": [d["i2"][i] for i in range(nl)],
                       "mids": [(d[f"Wm{k}"], d[f"bm{k}"]) for k in range(fl[4])]}
                if fl[2] > 0:
                    lt2["WhL"] = [Wo[:, 1:]] * nl
                    lt2["bhL"] = [bo[1:]] * nl
                if LT2_FAST:
                    _lt2_fast_prep(lt2, d, fl, nl)
                self._lt = lt2
        if WARM:
            # V26: first-call warm-ups (the residual audit showed one-off 5-30 ms gaps
            # before the first qr / norm.pdf / sandwich of every MLP: library init).
            try:
                f32 = fnp.float32
                z = fnp.zeros((64, 16), dtype=f32)
                fnp.linalg.qr(z + 1.0)
                v = fnp.zeros(64, dtype=f32)
                flops.stats.norm.pdf(v)
                flops.stats.norm.cdf(v)
                e = flops.as_symmetric(fnp.eye(64, dtype=f32), symmetry=(0, 1))
                fnp.einsum("ij,ia,jb->ab", e, z[:, :16] + 1.0, z[:, :16] + 1.0)
                # one tiny call per op signature the chain uses (matmul / einsum path
                # and symmetry machinery initialize lazily: 24 ms before the first
                # small matmul, 1-2 ms before each first einsum subscript)
                a = fnp.zeros((4, 21), dtype=f32) + 1.0
                b = fnp.zeros((21, 64), dtype=f32) + 1.0
                m = fnp.zeros((64, 64), dtype=f32) + 1.0
                t3 = fnp.zeros((3, 64, 64), dtype=f32) + 1.0
                g3 = fnp.zeros((3, 3), dtype=f32) + 1.0
                _ = a @ b
                _ = m @ m
                _ = m @ v
                fnp.einsum("tij,ti,tj,tg->gij", t3, t3[:, :, 0], t3[:, 0, :], g3)
                fnp.einsum("ti,ti,tg->gi", t3[:, :, 0], t3[:, :, 1], g3)
                fnp.einsum("iq,kqn->kin", m[:, :16], t3[:, :16, :])
                fnp.einsum("ki,kc->ic", t3[:, :, 0], t3[:, :, 1])
                fnp.einsum("kij,kij->i", t3, t3)
                fnp.einsum("kij,kj->ki", t3, t3[:, 0, :])
                fnp.einsum("ki,ki->i", t3[:, :, 0], t3[:, :, 1])
                fnp.einsum("kij,kqj->iq", t3, t3[:, :16, :])
                fnp.einsum("kiq,kjq->kij", t3, t3)
                fnp.einsum("kij,kjq->kiq", t3, t3)
                fnp.einsum("kiq,kcq->ic", t3, t3)
                fnp.einsum("pq,kqn->kpn", m[:16, :16], t3[:, :16, :])
                fnp.einsum("cij,cjk->cik", t3[:1], t3)
                fnp.sum(t3, axis=(0, 2))
                fnp.sum(t3, axis=0)
                fnp.diag(m)
                fnp.diagflat(v)
                fnp.fill_diagonal(m, 0.0)
                fnp.matmul(t3, fnp.swapaxes(t3, -1, -2))
                fnp.concatenate([m, m], axis=1)
                fnp.stack([v, v], axis=1)
                fnp.maximum(v, 1e-10)
                fnp.sqrt(v + 1.0)
                fnp.power(v + 1.0, 1.0)
                fnp.clip(v, 0.5, 2.0)
                fnp.mean(v)
                fnp.copy(m)
                fnp.abs(v)
            except Exception:  # noqa: BLE001  warm-up must never fail a submission
                pass

        if PREALLOC and int(ctx.width) == 1024 and int(ctx.depth) == len(CORR_BETA):
            try:
                self._prealloc(int(ctx.depth))
            except Exception:  # noqa: BLE001  la reserva nunca debe tumbar el envio
                for _a in ("_pool", "_smm", "_ncall"):
                    if hasattr(self, _a):
                        delattr(self, _a)

    def _prealloc(self, L):
        f32 = fnp.float32
        pool = _Pool(f32)
        inner = _StrassenR(f32, 2 * (L - 1))
        smm = _Wide(inner) if (WIDE or P2B) else inner
        for tabla, key, tp, shp in _PRE_TABLA:
            b = fnp.reshape(fnp.zeros((math.prod(shp),), dtype=f32), shp)   # 1-D: fnp.zeros cuadrado sale simetrico
            if tabla == "pool":
                if tp == "T":
                    pool.bufs[key] = (b, fnp.reshape(b, (shp[0] * shp[1], 1, shp[2], shp[3])))
                else:
                    pool.bufs[key] = b
                continue
            if tabla == "p2":
                if isinstance(smm, _Wide):
                    smm.p2[key] = b
                continue
            if tabla == "sr":
                d = inner.pool
            elif tabla == "k1" and isinstance(smm, _Wide):
                d = smm.w.pool
            else:
                continue
            d[key] = (b, fnp.reshape(b, (shp[0], shp[1] * shp[2], shp[3], shp[4]))) if tp == "T" else b
        self._pool = pool
        self._smm = smm
        if PREALLOC_LEV:
            self._ncall = 1

    # ------------------------------------------------------------------
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        was = _gc.isenabled()
        _gc.disable()
        try:
            return self._predict_core(mlp, budget)
        finally:
            if was:
                _gc.enable()

    def _predict_core(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        n = mlp.width
        f32 = fnp.float32
        st = _statics(n)
        metric2 = 2.0 ** 2
        L = len(mlp.weights)
        # vec+ww K4 transport and the fitted mean correction are calibrated for the
        # Phase-2 suite shape ONLY; any other shape (e.g. the grader smoke MLP)
        # runs the shape-generic V1.6 base path. The correction destabilizes small
        # widths (non-finite blowup observed at 16x64) -- never apply it off-suite.
        riders = (n == 1024 and L == len(CORR_BETA)) or (DEAD_FORCE and L == len(CORR_BETA))
        dead_tau = float(CONFIG.get("dead_tau", 0.0))
        fis = int(CONFIG.get("dead_fis", 0)) if (dead_tau > 0.0 and riders and DEAD_MODE == "todo"
                                                   and n % DEAD_MULT == 0 and STRASSEN_HUB > 0) else 0
        dround = str(CONFIG.get("dead_round", "arriba"))
        cent = bool(int(CONFIG.get("dead_cent", DEAD_CENT))) and dead_tau > 0.0 and riders and DEAD_MODE == "todo"
        comp = str(CONFIG.get("dead_comp", DEAD_COMP)) if fis >= 5 else ""
        comp_p = "p" in comp and not cent
        comp_q = "q" in comp
        comp_f = "f" in comp
        tau_b = float(CONFIG.get("dead_tau_b", DEAD_TAU_B))
        on_b = float(CONFIG.get("dead_on_b", DEAD_ON_B))
        if not (riders and DEAD_MODE == "todo" and dead_tau > 0.0):
            on_b = 0.0
        if on_b > 0.0 and not (0.0 < tau_b < dead_tau):
            tau_b = dead_tau
        elif not (0.0 < tau_b < dead_tau and riders and DEAD_MODE == "todo"):
            tau_b = 0.0
        alive_b = alive_b_prev = perm_b_next = perm_b_prev = None
        mb_next = n
        alive_prev = alive_now = None
        dperm_prev = None   # permutacion de la capa anterior: sus vivas primero
        dKp = None          # sus mp primeras posiciones = conjunto conservado (vivas + relleno)
        mp = n
        perm_next = None
        mp_next = n
        mcol = []           # por fuente (hueco de la pila): columnas conservadas (n si no hay permutacion)
        jw = []             # e28: por fuente, ancho de columnas con que entro en la base fija (el resto de sus factores son ceros)
        stat = []

        C1l, C2l, SELCl = self._wick_consts
        C1 = fnp.asarray(C1l, dtype=f32)
        C2 = fnp.asarray(C2l, dtype=f32)
        SELC = fnp.asarray(SELCl, dtype=f32)
        FP = {}
        for mode, p in self._progs.items():
            FP[mode] = {k: fnp.asarray(v, dtype=f32)
                        for k, v in p.items() if k in ("SL2", "SR2", "IND2", "SL1", "IND1")}
        ones_n = fnp.ones(n, dtype=f32)
        if KVEC:
            _kc = [[0.0] * 4 for _ in range(11)]
            _kc[0][0] = 1.0
            _kc[1][1] = 1.0; _kc[4][1] = -1.0
            _kc[2][2] = 1.0; _kc[5][2] = -3.0; _kc[8][2] = 2.0
            _kc[3][3] = 1.0; _kc[6][3] = -4.0; _kc[7][3] = -3.0; _kc[9][3] = 12.0; _kc[10][3] = -6.0
            KC = fnp.asarray(_kc, dtype=f32)
        if FOLD2:
            CDD = fnp.asarray([[1.0, 0.0], [9.0, 0.0], [0.0, 1.0]], dtype=f32)
            C0DD = fnp.asarray([[9.0, 1.0]], dtype=f32)
        if PWR:
            EXA = fnp.asarray([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=f32)
            EXI = fnp.asarray([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], dtype=f32)
            EXS = fnp.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=f32)
        zeros_n = fnp.zeros(n, dtype=f32)
        ones_col = (ones_n)[:, None]
        ones_row = ones_col.T
        ones2 = ones_col * ones_row
        if _FZ is not None:
            beta_rows = None
            fz_BB = fnp.asarray(_FZ["BB"], dtype=f32)
            fz_BH = fnp.asarray(_FZ["BH"], dtype=f32)
            fz_E = fnp.asarray(_FZ["E"], dtype=f32)
            fz_J = fnp.asarray(_FZ["J"], dtype=f32)
        else:
            beta_rows = [fnp.asarray(b, dtype=f32) for b in CORR_BETA]
        # V20: one persistent buffer for the d=2 term products (max terms over modes)
        t2max = max(len(p["AB2"]) for p in self._progs.values())
        # V27 (F80): persistent pooled buffers (see _Pool)
        pool = getattr(self, "_pool", None)
        if pool is None or pool.dtype != f32:
            pool = _Pool(f32)
            self._pool = pool

        def NN(name):
            return pool.get(name, (n, n))

        abbuf = pool.get("abbuf", (t2max, n, n))
        T1 = NN("t1")   # layer-local (n,n) scratch

        mu = fnp.zeros(n, dtype=f32)
        C = fnp.eye(n, dtype=f32) if E29_NOSYMC else flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))   # e29 nosymc: C del capa 0 no se lee (C_pre = Gram)
        A_st = P_st = Z_st = L_st = None
        newborn = None
        w2b_list = []
        s_list = []
        e_list = []
        c1_list = []   # per source: lambda_b * w1_b   (X3 = A*c1 + P*c2, column scalings)
        c2_list = []   # per source: w1_b^2 * dG_pre_b
        y_list = []    # per source: (m/4) w2_b       (Y3 = y 1^T)
        g_prev = None  # post-ReLU kappa4 diagonal core of the previous layer
        var_prev = None
        lam_prev = 0.0
        regen = riders and not NO_REGEN  # memoryless kappa4 channel: suite shape only
        lt = getattr(self, "_lt", None) if riders else None
        lt_dump = LT_DUMP_ON and riders
        lt_on = lt is not None or lt_dump
        lt_ext = lt_dump or (lt is not None and lt.get("v2", False) and lt["Fo"] > len(LT_FE))
        lt_u = None
        lt_prev = None
        fd = None
        bufs = None
        legs = None
        # V26: pooled Strassen products (F79); the pools persist across predict() calls
        # (allocating/freeing ~2.5 GB per MLP made the residual clock creep up over a
        # 100-MLP suite). Pool keys carry the block shapes, so a different width simply
        # adds new buffers; a different depth changes the batch capacity -> rebuild.
        smm = getattr(self, "_smm", None)
        if (smm is None or smm.bmax != 2 * (L - 1) or smm.dtype != f32
                or (fis == 0 and isinstance(smm, _StrassenR))):
            smm = (_StrassenVC if BANCO["vistas_cache"] else _Strassen)(f32, 2 * (L - 1))
            self._smm = smm
        if fis > 0 and not isinstance(smm, _StrassenR):
            smm = _StrassenR(f32, 2 * (L - 1))
            self._smm = smm
        if (WIDE or P2B) and fis > 0 and not isinstance(smm, _Wide):
            smm = _Wide(smm)
            self._smm = smm
        gy = gy2 = None
        if fis > 0:
            gy = pool.get("gy", (2 * (L - 1), 1, n, n))
            gy2 = pool.get("gy2", (2 * (L - 1), 1, n, n))
        apc_buf = pool.get("apc", (2 * (L - 1), 1, n, n)) if (E28_OLDC and fis >= 5 and comp_f) else None
        ncall = getattr(self, "_ncall", 0)
        self._ncall = ncall + 1
        s_lev = STRASSEN_LEVELS if ncall > 0 else min(STRASSEN_LEVELS, STRASSEN_FIRST)
        s_sb = STRASSEN_SB
        self._s_hub = s_lev
        r = min(int(self.R_RES), n)  # smoke shapes can be narrower than the rank
        rfb = 0 if NO_FB else min(int(self.R_FB), n)  # V18 feedback rank
        ZK1 = 1 if (AUD_ZS and rfb > 0) else 0   # rebanadas iniciales sin patas finas (fuente de la capa 0)
        # V21: shared-basis state for old sources (suite-width only: r must be < n)
        r_old = int(self.R_OLD)
        rc_app = 0       # X_APP: rango activo de la base sin truncar
        confine = (not NO_CONFINE) and r_old < n
        age_old = int(self.AGE_OLD)
        ka = 0           # number of confined (old) sources = leading slots of the stacks
        Qc = None        # (n, r) current basis of the old legs (pre-wick of this layer)
        fa_side = f2_side = qc_side = z_side = 0   # V27: ping-pong buffer sides
        FAP = FAP2 = None    # V28: interleaved factor slabs (m, 2, r, n) [slot, {A, P}]
        fap4 = fap24 = None  # their one-time (2 (L-1), 1, r, n) views (current side)
        fa_off = 0           # slot offset of FAP inside its side buffer (1 after a nest)
        legs4 = None
        FAo = FPo = None   # (ka, r, n) static factors: A_s = Qc FA_s, P_s = Qc FP_s
        Sg = None        # (r, r) weighted Gram core of the TIER-1 legs in the factor basis
        # V24: nested tier 2 (slots [0:kb] of the stacks, the oldest sources)
        r2 = int(self.R_OLD2)
        age_old2 = int(self.AGE_OLD2)
        nest = confine and age_old2 > 0 and r2 < r_old
        kb = 0           # number of tier-2 sources (leading slots, subset of the ka confined)
        U = None         # (r1, r2) shared sub-basis of tier 2 inside Qc's factor space
        FA2 = FP2 = None   # (kb, r2, n) static factors: A_s = Qc U FA2_s
        S2 = None        # (r2, r2) weighted Gram core of tier 2 in the U basis
        QU = None        # (n, r2) = Qc U, formed once per layer
        dA_list = []     # per source: hub-column Gram weights of the A-type legs
        dP_list = []     # per source: hub-column Gram weights of the P-type legs
        R1T_st = R2T_st = None  # V18: static right factors of the thin legs, (k, n, rfb)
        Zf_st = None  # V18: transported thin legs [F1 | F2], (k, n, 2 rfb); separate from Z
                      # so the M-leg einsums (Z L^T, PPL Z^T) never see them
        K4_sigma = None
        K4_vec = None
        rows = []
        res_e=fnp.zeros(n,dtype=f32)

        w1_prev = None  # wick w(1) of the previous layer, folded into WD
        tt22_prev = None  # c2 22.10.2026: (K22, K4v) of the previous layer

        for li, w in enumerate(mlp.weights):
            last = li == L - 1
            trim = last and not FULL_LAST  # V19: mean-only final layer
            skip_src = trim and NO_SRC_LAST
            w32 = w if w.dtype == fnp.float32 else w.astype(f32)
            W = w32.T
            # ---- linear ----
            mu = W @ mu
            C_pre = None
            if li == 0:
                # V29: C = I at the input, so the sandwich is the plain Gram w32^T w32
                # (aliased 2-operand einsum: exact, 0.5 u, output tagged symmetric; F77)
                if trim:
                    var = fnp.maximum(fnp.sum(w32 * w32, axis=0), 1e-10)
                else:
                    C_pre = fnp.einsum("ia,ib->ab", w32, w32)
            elif skip_src:
                # probe path only (V19_NO_SRC_LAST): no family at this layer
                if trim:
                    var = fnp.maximum(fnp.sum(w32 * (C @ w32), axis=0), 1e-10)
                else:
                    C_pre = fnp.einsum("ij,ia,jb->ab", C, w32, w32)
            # li >= 1: C_pre is formed after the transport family below (W C rides in it)
            # Source stacks evolve by W @ diag(w1_prev); the newborn (added after
            # last layer's wick) evolves by the raw W. Fold the wick into WD so
            # the stacks never need a separate (k,n,n) scaling pass.
            # Dense legs live in ping-pong slot buffers (F53: no fresh (k,n,n)
            # results, no concatenates): transport k slots into the spare buffer,
            # write the newborn into slot k, swap.
            pre_done = False
            Kl = inv_l = Wgm = mo = None
            var_p = WDq = None
            ml = n
            if A_st is not None and not skip_src and fis >= 1 and newborn is not None:
                # (a) W C y C_pre con la dimension interna reducida a las conservadas de la capa anterior
                if mp < n:
                    Wg = fnp.take(W, dperm_prev, axis=1, out=NN("wg"))
                    Wgm = Wg[:, :mp]
                if mp < n and not cent:
                    Cc = fnp.take(fnp.take(C, dKp, axis=0), dKp, axis=1, out=NN("cc")[:mp, :mp])
                    wcc = pool.get("wcc", (1, 1, n, n))[..., :mp]
                    smm.smin = DEAD_SMIN
                    smm.mm(Wgm[None, None], Cc[None, None], wcc, smm.level(n, mp, mp, s_lev))
                    smm.smin = STRASSEN_MIN
                    WCm = wcc[0, 0]
                    if trim:
                        var = fnp.maximum(fnp.sum(fnp.multiply(WCm, Wgm, out=T1[:, :mp]), axis=1), 1e-10)
                    elif comp_p:
                        # (p) diagonal por filas; el resto de C_pre solo en las filas conservadas (abajo)
                        var_p = fnp.maximum(fnp.sum(fnp.multiply(WCm, Wgm, out=T1[:, :mp]), axis=1), 1e-10)
                        Xpre, Ypre, mipre = WCm, Wgm, mp
                    else:
                        C_pre = self._sym_product(WCm, Wgm.T, n, NN("cpre"), min(CPRE_LEV, s_lev), mi=mp)
                else:
                    wcc = pool.get("wcc", (1, 1, n, n))
                    _kc = 2 * A_st.shape[0] + 1
                    smm.mm(W[None, None], legs4["AP0"][_kc:_kc + 1], wcc, smm.level(n, n, n, s_lev))
                    if trim:
                        var = fnp.maximum(fnp.sum(fnp.multiply(wcc[0, 0], W, out=T1), axis=1), 1e-10)
                    elif comp_p:
                        var_p = fnp.maximum(fnp.sum(fnp.multiply(wcc[0, 0], W, out=T1), axis=1), 1e-10)
                        Xpre, Ypre, mipre = wcc[0, 0], W, n
                    else:
                        C_pre = self._sym_product(wcc[0, 0], w32, n, NN("cpre"), min(CPRE_LEV, s_lev))
                pre_done = True
                if fis >= 2:
                    # (c) alfa de ESTA capa antes de transportar: solo depende de mu y de diag(C_pre)
                    var_e = var if trim else (fnp.maximum(fnp.diag(C_pre), 1e-10) if var_p is None else var_p)
                    alive_e, nal, ml, perm_e = _vivas(mu / fnp.sqrt(var_e), DEAD_TAU_L.get(li, dead_tau), n, f32, dround)
                    if ml < n:
                        Kl = perm_e[:ml]
                        inv_l = fnp.minimum(fnp.argsort(perm_e, kind="stable"), ml)
                        if VB_HOT > 0:
                            _hot = fnp.greater(fnp.take(mu / fnp.sqrt(var_e), Kl), VB_HOT)
                            Kl = fnp.take(Kl, fnp.argsort(_hot.astype(f32), kind="stable"))
                            mo = min(ml, -(-(ml - int(fnp.sum(_hot.astype(f32)))) // DEAD_MULT) * DEAD_MULT)
                            inv_l = fnp.minimum(fnp.argsort(fnp.concatenate([Kl, perm_e[ml:]]), kind="stable"), ml)
                    if var_p is not None and Kl is None:
                        C_pre = self._sym_product(Xpre, Ypre.T, n, NN("cpre"), min(CPRE_LEV, s_lev),
                                                  mi=(mipre if mipre < n else None))
                    elif var_p is not None:
                        # (p) C_pre = X Y^T solo en filas/columnas conservadas: los tres bloques del producto simetrico
                        # como una familia (como _sym_product), con fondos de tamano completo y vistas
                        h2 = ml // 2
                        Xk = fnp.take(Xpre, Kl, axis=0, out=NN("cpx")[:ml, :mipre])
                        Yk2 = fnp.take(Ypre, Kl, axis=0, out=NN("cpy")[:ml, :mipre])
                        X3 = pool.get("sym_x", (3, 1, n // 2, n))[:, :, :h2, :mipre]
                        Y3 = pool.get("sym_y", (3, 1, n, n // 2))[:, :, :mipre, :h2]
                        O3 = pool.get("sym_o", (3, 1, n // 2, n // 2))[:, :, :h2, :h2]
                        fnp.copyto(X3[0, 0], Xk[:h2]); fnp.copyto(X3[1, 0], Xk[:h2]); fnp.copyto(X3[2, 0], Xk[h2:])
                        YkT = Yk2.T
                        fnp.copyto(Y3[0, 0], YkT[:, :h2]); fnp.copyto(Y3[1, 0], YkT[:, h2:]); fnp.copyto(Y3[2, 0], YkT[:, h2:])
                        smm.smin = DEAD_SMIN
                        smm.mm(X3, Y3, O3, smm.level(h2, mipre, h2, min(CPRE_LEV, s_lev)))
                        smm.smin = STRASSEN_MIN
                        cpc = pool.get("cpc", (n + 1, n + 1))
                        fnp.copyto(cpc[:h2, :h2], O3[0, 0]); fnp.copyto(cpc[:h2, h2:ml], O3[1, 0])
                        fnp.copyto(cpc[h2:ml, h2:ml], O3[2, 0]); fnp.copyto(cpc[h2:ml, :h2], O3[1, 0].T)
                        fnp.copyto(cpc[ml, :ml + 1], 0.0)
                        fnp.copyto(cpc[:ml + 1, ml], 0.0)
                        C_pre = fnp.take(fnp.take(cpc[:ml + 1, :ml + 1], inv_l, axis=0), inv_l, axis=1, out=NN("cpre"))
            if A_st is not None and not skip_src:
                k = A_st.shape[0]
                WD = fnp.multiply(W, (w1_prev)[None, :], out=NN("wd"))
                WDb = WD[None]
                # V21: one source per layer crosses the age gate; join it to the shared
                # basis (post-wick legs, like the lean chain), skip the join at the last
                # layer (its legs are used once more, densely, for D3 only).
                ka_t = ka
                if confine and not last:
                    ka_t = max(ka, min(k, li - age_old))
                if ka_t > ka:
                    j = ka  # the joiner (exactly one per layer)
                    w1c = (w1_prev)[:, None]
                    jc = fis >= 3 and pre_done and r_old <= mp < n
                    mcj = mcol[j] if (jc and fis >= 5) else n
                    jw.append(mcj)
                    if jc:
                        APj = fnp.take(legs["AP0"][j] if mcj == n else legs["AP0"][j][..., :mcj], dKp, axis=1)   # (2, mp, mcj)
                        Aj = APj[0]
                        Pj = APj[1]
                    else:
                        Aj = A_st[j]   # V29: dense legs are already w1-scaled (wick stage)
                        Pj = P_st[j]
                    dAj = (dA_list[j])[:, None]
                    dPj = (dP_list[j])[:, None]
                    if mcj < n:
                        dAj = dAj[:mcj]
                        dPj = dPj[:mcj]
                    Om = pool.get("om", (n, r_old))         # fixed sketch (n, r)
                    fnp.copyto(Om, w32[:, :r_old])
                    Qp = fnp.multiply(w1c, Qc, out=pool.get("qp", (n, r_old))) if ka > 0 else None
                    Yq = pool.get("yq", (n, r_old))
                    Yq1 = pool.get("yq1", (n, r_old))
                    Yq2 = pool.get("yq2", (n, r_old))
                    if jc:
                        Om = fnp.take(Om, dKp, axis=0)
                        if Qp is not None:
                            Qp = fnp.take(Qp, dKp, axis=0)
                        Yqf = Yq
                        invp = fnp.minimum(fnp.argsort(dperm_prev, kind="stable"), mp)
                        Yq = Yq[:mp]
                        Yq2 = Yq2[:mp]
                        if mcj < n:
                            Yq1 = Yq1[:mcj]
                    APs = (APj[:, None] if jc else legs4["AP0"][2 * j:2 * j + 2]) if _g(smm, "rf") or _g(smm, "proj") else None
                    bug = BUG_L > 0 and ka > 0 and APs is not None
                    if bug:
                        # e20 (X_BUG=l): fusion BUG. Base = Rayleigh-Ritz del Gram ponderado (viejas + entrante) en
                        # span[Qp | G_j Om_l]; sustituye al buscador de rango y a la proyeccion de la entrante.
                        nrb, rl = Om.shape[0], r_old + BUG_L
                        Sfull = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                        smm.smin = P2B_MIN
                        y1t = pool.get("bg_y", (2, 1, BUG_L, n))[..., :mcj]
                        smm.mm(Om[:, :BUG_L].T[None, None], APs, y1t, smm.level(BUG_L, nrb, mcj, P2B_LEV))
                        fnp.multiply(y1t[0, 0], dAj.T, out=y1t[0, 0])
                        fnp.multiply(y1t[1, 0], dPj.T, out=y1t[1, 0])
                        Yl = pool.get("bg_yl", (1, n, BUG_L))[:, :nrb]
                        smm.hub(APs, y1t, Yl, smm.level(nrb, mcj, BUG_L, P2B_LEV))
                        Bq, _ = fnp.linalg.qr(fnp.concatenate([Qp, Yl[0]], axis=1))       # (nrb, r + l)
                        abt = pool.get("bg_ab", (2, 1, rl, n))[..., :mcj]                    # B^T [Aj | Pj]
                        smm.mm(Bq.T[None, None], APs, abt, smm.level(rl, nrb, mcj, P2B_LEV))
                        ys = pool.get("bg_ys", (2, 1, rl, n))[..., :mcj]
                        fnp.multiply(abt[0, 0], dAj.T, out=ys[0, 0])
                        fnp.multiply(abt[1, 0], dPj.T, out=ys[1, 0])
                        Kj = pool.get("bg_k", (1, rl, rl))
                        smm.hub(abt, ys, Kj, smm.level(rl, mcj, rl, P2B_LEV))
                        smm.smin = STRASSEN_MIN
                        BQ = Bq.T @ Qp                                                       # (r + l, r)
                        Kk = Kj[0] + BQ @ (Sfull @ BQ.T)
                        Uk = _eye_cols(rl, r_old, f32)
                        for _pass in range(BUG_PASS):
                            Uk, _ = fnp.linalg.qr(Kk @ Uk)
                        Qn = Bq @ Uk
                        Tq_b = Uk.T @ BQ
                        Sj_b = Uk.T @ (Kj[0] @ Uk)
                    fija = FIJA and ka > 0 and APs is not None
                    if fija:
                        if FIJA_PROJ == "ne":
                            Mt = fnp.linalg.solve(fnp.einsum("ia,ib->ab", Qp, Qp), Qp.T)          # (r, filas) = Qp^+
                        else:
                            Q0, R0 = fnp.linalg.qr(Qp)
                            Mt = ((_tri_solve(R0, Q0.T, E28_TRI) if E28_TRI > 0 else fnp.linalg.solve(R0, Q0.T))
                                  if FIJA_PROJ == "qrm" else Q0.T)
                        fj_dir = E28_FJ and FIJA_PROJ == "qrm"
                        if fj_dir:
                            # e28 fj: la familia escribe [A|P] proyectados directamente en el hueco de la fuente (sin ct, concatenate ni copias)
                            _fp, _fp4 = pool.get_pair(("fap", fa_side), (L - 1, 2, r_old, n))
                            _s0 = 2 * (fa_off + ka - kb)
                            ct = _fp4[_s0:_s0 + 2][..., :mcj]
                        else:
                            ct = pool.get("fj_c", (2, 1, r_old, n))[..., :mcj]
                        smm.smin = P2B_MIN
                        smm.mm(Mt[None, None], APs, ct, smm.level(r_old, Mt.shape[1], mcj, P2B_LEV))
                        smm.smin = STRASSEN_MIN
                        if fj_dir:
                            Fj = None
                        elif FIJA_PROJ == "qr":
                            Fj = fnp.linalg.solve(R0, fnp.concatenate([ct[0, 0], ct[1, 0]], axis=1))   # (r, 2 mcj): Qp Fj = Q0 Q0^T [Aj|Pj]
                        else:
                            Fj = fnp.concatenate([ct[0, 0], ct[1, 0]], axis=1)
                        Qn = Qp
                    app = APP_L >= 0 and ka > 0 and APs is not None
                    if app:
                        nrb = Om.shape[0]
                        Q0, R0 = fnp.linalg.qr(Qp[:, :rc_app])
                        ra = min(APP_L, r_old - rc_app)
                        if ra > 0:
                            smm.smin = P2B_MIN
                            y1t = pool.get("ap_y", (2, 1, APP_L, n))[..., :mcj]
                            smm.mm(Om[:, :APP_L].T[None, None], APs, y1t, smm.level(APP_L, nrb, mcj, P2B_LEV))
                            fnp.multiply(y1t[0, 0], dAj.T, out=y1t[0, 0])
                            fnp.multiply(y1t[1, 0], dPj.T, out=y1t[1, 0])
                            Yl = pool.get("ap_yl", (1, n, APP_L))[:, :nrb]
                            smm.hub(APs, y1t, Yl, smm.level(nrb, mcj, APP_L, P2B_LEV))
                            smm.smin = STRASSEN_MIN
                            Ya = Yl[0]
                            for _pass in range(2):
                                Ya = Ya - Q0 @ (Q0.T @ Ya)
                            Q1, _ = fnp.linalg.qr(Ya)
                        else:
                            Q1 = Q0[:, :0]
                        Qn = fnp.concatenate([Q0, Q1[:, :ra], fnp.zeros((nrb, r_old - rc_app - ra), dtype=f32)], axis=1)
                        Tq_a = fnp.concatenate([fnp.concatenate([R0, fnp.zeros((rc_app, r_old - rc_app), dtype=f32)], axis=1),
                                                fnp.zeros((r_old - rc_app, r_old), dtype=f32)], axis=0)
                        rc_app += ra
                    for _pass in range(0 if (bug or app or fija) else QPASS):
                        # Yq = G Om with G = Aj dA Aj^T + Pj dP Pj^T + Qp Sg Qp^T
                        if _g(smm, "rf"):
                            # P2: [Aj | Pj]^T Om como familia (izquierdo Om^T compartido) y la vuelta como hub (suma A + P)
                            y1t = pool.get("p2y1", (2, 1, r_old, n))[..., :APs.shape[3]]
                            smm.smin = P2B_MIN
                            smm.mm(Om.T[None, None], APs, y1t, smm.level(r_old, APs.shape[2], APs.shape[3], P2B_LEV))
                            fnp.multiply(y1t[0, 0], dAj.T, out=y1t[0, 0])
                            fnp.multiply(y1t[1, 0], dPj.T, out=y1t[1, 0])
                            smm.hub(APs, y1t, Yq[None], smm.level(APs.shape[2], APs.shape[3], r_old, P2B_LEV))
                            smm.smin = STRASSEN_MIN
                        else:
                            fnp.matmul(Aj.T, Om, out=Yq1)
                            fnp.multiply(dAj, Yq1, out=Yq1)
                            fnp.matmul(Aj, Yq1, out=Yq)
                            fnp.matmul(Pj.T, Om, out=Yq1)
                            fnp.multiply(dPj, Yq1, out=Yq1)
                            fnp.matmul(Pj, Yq1, out=Yq2)
                            fnp.add(Yq, Yq2, out=Yq)
                        if ka > 0:
                            # V24: full core of the confined legs = tier-1 core + lifted tier-2 core
                            Sfull = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                            if _g(smm, "rfq"):
                                t_ = smm.mm2(Qp.T, Om, pool.get("p2t", (r_old, r_old)))
                                smm.mm2(Qp, Sfull @ t_, Yq2)
                            else:
                                fnp.matmul(Qp, Sfull @ (Qp.T @ Om), out=Yq2)
                            fnp.add(Yq, Yq2, out=Yq)
                        if jc:
                            fnp.copyto(Yqf[mp], 0.0)
                            Qnn, _ = fnp.linalg.qr(fnp.take(Yqf[:mp + 1], invp, axis=0))   # filas naturales
                            Qn = fnp.take(Qnn, dKp, axis=0)
                        else:
                            Qn, _ = fnp.linalg.qr(Yq)           # (n, r) orthonormal
                        Om = Qn
                    if APP_L >= 0 and ka == 0:
                        Qn = fnp.concatenate([Qn[:, :APP_R0], fnp.zeros((Qn.shape[0], r_old - APP_R0), dtype=f32)], axis=1)
                        rc_app = APP_R0
                    # V27: factors live in ping-pong slot buffers (rotation reads one side,
                    # writes the other; the joiner takes slot m1 = number of tier-1 members)
                    if not fija:
                        fa_side ^= 1
                    fap_new, fap4_new = pool.get_pair(("fap", fa_side), (L - 1, 2, r_old, n))
                    m1 = ka - kb
                    if fija:
                        # sin rotacion: los factores viejos se quedan en su sitio; la entrante va detras (hueco fa_off + m1)
                        FAj = fap_new[fa_off + m1, 0][:, :mcj]
                        FPj = fap_new[fa_off + m1, 1][:, :mcj]
                        if Fj is not None:
                            fnp.copyto(FAj, Fj[:, :mcj])
                            fnp.copyto(FPj, Fj[:, mcj:])
                        if mcj < n:
                            fnp.copyto(fap_new[fa_off + m1][..., mcj:], 0.0)
                    elif bug:
                        smm.smin = P2B_MIN
                        smm.mm(Uk.T[None, None], abt, fap4_new[2 * m1:2 * m1 + 2][..., :mcj],
                               smm.level(r_old, rl, mcj, P2B_LEV))
                        smm.smin = STRASSEN_MIN
                        FAj = fap_new[m1, 0][:, :mcj]
                        FPj = fap_new[m1, 1][:, :mcj]
                        if mcj < n:
                            fnp.copyto(fap_new[m1][..., mcj:], 0.0)
                    elif _g(smm, "proj"):
                        smm.smin = P2B_MIN
                        smm.mm(Qn.T[None, None], APs, fap4_new[2 * m1:2 * m1 + 2][..., :mcj],
                               smm.level(r_old, APs.shape[2], mcj, P2B_LEV))
                        smm.smin = STRASSEN_MIN
                        FAj = fap_new[m1, 0][:, :mcj]
                        FPj = fap_new[m1, 1][:, :mcj]
                        if mcj < n:
                            fnp.copyto(fap_new[m1][..., mcj:], 0.0)
                    elif mcj < n:
                        FAj = fnp.matmul(Qn.T, Aj, out=fap_new[m1, 0][:, :mcj])
                        FPj = fnp.matmul(Qn.T, Pj, out=fap_new[m1, 1][:, :mcj])
                        fnp.copyto(fap_new[m1][..., mcj:], 0.0)
                    else:
                        FAj = fnp.matmul(Qn.T, Aj, out=fap_new[m1, 0])
                        FPj = fnp.matmul(Qn.T, Pj, out=fap_new[m1, 1])
                    if ka > 0 and not fija:
                        if bug:
                            Tq = Tq_b
                        elif app:
                            Tq = Tq_a
                        elif _g(smm, "rotsm"):
                            Tq = smm.mm2(Qn.T, Qp, pool.get("p2tq", (r_old, r_old)))
                        else:
                            Tq = Qn.T @ Qp                      # (r, r) rotation of the old factors
                        if ka > kb and _g(smm, "rot"):
                            mcr = max(mcol[kb:ka]) if (_g(smm, "rotc") and fis >= 4) else n
                            smm.smin = P2B_MIN
                            smm.mm(Tq[None, None], fap4[2 * fa_off:2 * (fa_off + m1)][..., :mcr], fap4_new[:2 * m1][..., :mcr],
                                   smm.level(r_old, r_old, mcr, P2B_LEV))
                            smm.smin = STRASSEN_MIN
                            if mcr < n:
                                fnp.copyto(fap4_new[:2 * m1][..., mcr:], 0.0)
                            if _g(smm, "rotsm"):
                                Sg = smm.mm2(Tq, smm.mm2(Sg, Tq.T, pool.get("p2sa", (r_old, r_old))),
                                             pool.get(("p2sg", fa_side), (r_old, r_old)))
                            else:
                                Sg = Tq @ (Sg @ Tq.T)
                        elif ka > kb:
                            fnp.matmul(Tq[None, None], FAP, out=fap_new[:m1])
                            Sg = Tq @ (Sg @ Tq.T)
                        else:
                            Sg = None
                        if kb > 0:
                            U = Tq @ U                      # V24: the sub-basis rides along
                    if fija:
                        FAP = fap_new[fa_off:fa_off + m1 + 1]
                    else:
                        FAP = fap_new[:m1 + 1]
                        fa_off = 0
                    fap4 = fap4_new
                    FAo = FAP[:, 0]
                    FPo = FAP[:, 1]
                    if fija:
                        Sj = None
                    elif bug:
                        Sj = Sj_b
                    elif _g(smm, "sj"):
                        mcs = FAj.shape[1]
                        ys = pool.get("p2sjy", (2, 1, r_old, n))[..., :mcs]
                        fnp.multiply(FAj, dAj.T, out=ys[0, 0])
                        fnp.multiply(FPj, dPj.T, out=ys[1, 0])
                        sjb = pool.get(("p2sj", fa_side), (1, r_old, r_old))
                        smm.smin = P2B_MIN
                        smm.hub(fap4_new[2 * m1:2 * m1 + 2][..., :mcs], ys, sjb, smm.level(r_old, mcs, r_old, P2B_LEV))
                        smm.smin = STRASSEN_MIN
                        Sj = sjb[0]
                    else:
                        Sj = (FAj @ (dAj * FAj.T)) + (FPj @ (dPj * FPj.T))
                    if Sj is not None:
                        Sg = Sj if Sg is None else Sg + Sj
                    r_ord = FIJA_RC if FIJA_RC > 0 else (r2 if nest else 0)
                    if FIJA and r_ord > 0 and ka == 0 and kb == 0:
                        # orden de la base fija: las r_ord direcciones mayores del Gram de la primera entrante primero
                        # (con nivel 2, su subespacio = las r2 primeras columnas de la base)
                        Ur = fnp.asarray(Sg[:, :r_ord])
                        for _pass in range(FIJA_PASS):
                            Ur, _ = fnp.linalg.qr(Sg @ Ur)
                        Vr, _ = fnp.linalg.qr(fnp.concatenate([Ur, _eye_cols(r_old, r_old - r_ord, f32)], axis=1))
                        fnp.copyto(FAP[0, 0], Vr.T @ FAP[0, 0])
                        fnp.copyto(FAP[0, 1], Vr.T @ FAP[0, 1])
                        Qn = Qn @ Vr
                        if not (E29_SGDEAD and RPROJ == 0 and BUG_L == 0 and APP_L < 0):
                            Sg = Vr.T @ (Sg @ Vr)   # e29 sgdead: con la base fija Sg no se vuelve a leer (solo lo leen X_RPROJ, X_BUG, X_APP y los caminos sin base fija)
                    # V24: nested move of the oldest tier-1 member into tier 2, in the
                    # orthonormal coordinates Qn of this rebuild (lean: youngest gate first,
                    # then the older gate on the already-confined legs).
                    kb_t = kb
                    if nest:
                        kb_t = max(kb, min(ka_t - 1, li - age_old2))
                    if kb_t > kb and fija:
                        # nivel 2 con base fija: la saliente se proyecta (minimos cuadrados en la metrica de la base,
                        # G = R0^T R0) sobre las r2 primeras columnas de la base fija; U = [I; 0] fijo, sin rotaciones
                        assert kb_t == kb + 1 and FIJA_PROJ != "ne", (li, kb, kb_t)
                        fap2_cur, fap24_cur = pool.get_pair(("fap2", f2_side), (L - 1, 2, r2, n))
                        if E28_NEST:
                            wj_ = jw[kb]
                            Fcat = fnp.concatenate([FAo[0][:, :wj_], FPo[0][:, :wj_]], axis=1)
                            if E28_NEST2 and E29_NESTG:
                                F2 = fnp.add(Fcat[:r2], _tri_solve(R0[:r2, :r2], R0[:r2, r2:], E28_TRI2) @ Fcat[r2:])
                            elif E28_NEST2:
                                F2 = fnp.add(Fcat[:r2], _tri_solve(R0[:r2, :r2], R0[:r2, r2:] @ Fcat[r2:], E28_TRI2))
                            else:
                                F2 = _tri_solve(R0[:r2, :r2], R0[:r2, :] @ Fcat, E28_TRI2)
                            fnp.copyto(fap2_cur[kb, 0][:, :wj_], F2[:, :wj_])
                            fnp.copyto(fap2_cur[kb, 1][:, :wj_], F2[:, wj_:])
                            if wj_ < n:
                                fnp.copyto(fap2_cur[kb, 0][:, wj_:], 0.0)
                                fnp.copyto(fap2_cur[kb, 1][:, wj_:], 0.0)
                        else:
                            Gm = R0.T @ R0
                            F2 = fnp.linalg.solve(Gm[:r2, :r2], Gm[:r2, :] @ fnp.concatenate([FAo[0], FPo[0]], axis=1))
                            fnp.copyto(fap2_cur[kb, 0], F2[:, :n])
                            fnp.copyto(fap2_cur[kb, 1], F2[:, n:])
                        FAP2 = fap2_cur[:kb + 1]
                        fap24 = fap24_cur
                        FA2 = FAP2[:, 0]
                        FP2 = FAP2[:, 1]
                        U = _eye_cols(r_old, r2, f32)
                        S2 = None
                        FAP = FAP[1:]
                        fa_off += 1
                        FAo = FAP[:, 0]
                        FPo = FAP[:, 1]
                        kb = kb_t
                    elif kb_t > kb:
                        assert kb_t == kb + 1, (li, kb, kb_t)
                        jb = kb                             # stack slot of the mover
                        FA1 = FAo[0]
                        FP1 = FPo[0]
                        dAb = (dA_list[jb])[:, None]
                        dPb = (dP_list[jb])[:, None]
                        mcb = mcol[jb] if (_g(smm, "rotc") and fis >= 4) else n
                        if _g(smm, "nests"):
                            ys = pool.get("p2sjy", (2, 1, r_old, n))[..., :mcb]
                            fnp.multiply(FA1[:, :mcb], dAb[:mcb].T, out=ys[0, 0])
                            fnp.multiply(FP1[:, :mcb], dPb[:mcb].T, out=ys[1, 0])
                            ssb = pool.get("p2ss", (1, r_old, r_old))
                            smm.smin = P2B_MIN
                            smm.hub(fap4_new[0:2][..., :mcb], ys, ssb, smm.level(r_old, mcb, r_old, P2B_LEV))
                            smm.smin = STRASSEN_MIN
                            S_s = ssb[0]
                        else:
                            S_s = (FA1 @ (dAb * FA1.T)) + (FP1 @ (dPb * FP1.T))   # (r1, r1)
                        G2 = S_s if kb == 0 else S_s + U @ (S2 @ U.T)
                        Om2 = fnp.copy(w32[:r_old, :r2])                        # sketch (r1, r2)
                        for _pass in range(int(self.QPASS2)):
                            Un, _ = fnp.linalg.qr(smm.mm2(G2, Om2, pool.get("p2g2", (r_old, r2))) if _g(smm, "nestsm") else G2 @ Om2)
                            Om2 = Un
                        f2_side ^= 1
                        fap2_new, fap24_new = pool.get_pair(("fap2", f2_side), (L - 1, 2, r2, n))
                        if _g(smm, "nestu"):
                            smm.smin = P2B_MIN
                            smm.mm(Un.T[None, None], fap4_new[0:2][..., :mcb], fap24_new[2 * kb:2 * kb + 2][..., :mcb],
                                   smm.level(r2, r_old, mcb, P2B_LEV))
                            if mcb < n:
                                fnp.copyto(fap24_new[2 * kb:2 * kb + 2][..., mcb:], 0.0)
                            smm.smin = STRASSEN_MIN
                        else:
                            fnp.matmul(Un.T, FA1, out=fap2_new[kb, 0])
                            fnp.matmul(Un.T, FP1, out=fap2_new[kb, 1])
                        if kb > 0 and _g(smm, "nestt"):
                                smm.smin = P2B_MIN
                                T2 = Un.T @ U
                                mc2r = max(mcol[:kb]) if (_g(smm, "rotc") and fis >= 4) else n
                                smm.mm(T2[None, None], fap24[:2 * kb][..., :mc2r], fap24_new[:2 * kb][..., :mc2r],
                                       smm.level(r2, r2, mc2r, P2B_LEV))
                                if mc2r < n:
                                    fnp.copyto(fap24_new[:2 * kb][..., mc2r:], 0.0)
                                smm.smin = STRASSEN_MIN
                        elif kb > 0:
                            T2 = Un.T @ U                                       # (r2, r2)
                            fnp.matmul(T2[None, None], FAP2, out=fap2_new[:kb])
                        FAP2 = fap2_new[:kb + 1]
                        fap24 = fap24_new
                        FA2 = FAP2[:, 0]
                        FP2 = FAP2[:, 1]
                        if _g(smm, "nestsm"):
                            S2 = smm.mm2(Un.T, smm.mm2(G2, Un, pool.get("p2g2", (r_old, r2))), pool.get(("p2s2", f2_side), (r2, r2)))
                        else:
                            S2 = Un.T @ (G2 @ Un)
                        U = Un
                        FAP = FAP[1:]
                        fa_off = 1
                        FAo = FAP[:, 0]
                        FPo = FAP[:, 1]
                        Sg = Sg - S_s
                        kb = kb_t
                    qc_side ^= 1
                    if comp_q and Kl is not None:
                        # (q) la base nueva solo en las filas conservadas de esta capa
                        mij = mp if jc else n
                        Xqj = fnp.take(Wgm if jc else W, Kl, axis=0, out=NN("wqj")[:ml, :mij])
                        qcp = pool.get(("qcp", qc_side), (n + 1, r_old))
                        Qcc = smm.mm2(Xqj, Qn, qcp[:ml]) if _g(smm, "qc") else fnp.matmul(Xqj, Qn, out=qcp[:ml])
                        fnp.copyto(qcp[ml], 0.0)
                        Qc = fnp.take(qcp[:ml + 1], inv_l, axis=0, out=pool.get(("qc", qc_side), (n, r_old)))
                    else:
                        Qc = (smm.mm2(Wgm if jc else W, Qn, pool.get(("qc", qc_side), (n, r_old))) if _g(smm, "qc")
                              else fnp.matmul(Wgm if jc else W, Qn, out=pool.get(("qc", qc_side), (n, r_old))))  # wick already inside Qn
                    ka = ka_t
                elif ka > 0 and comp_q and Kl is not None:
                    # (q) transporte de la base solo en filas conservadas y con la dimension interna reducida
                    qc_side ^= 1
                    if mp < n:
                        WDq = fnp.multiply(fnp.take(Wgm, Kl, axis=0, out=NN("wq")[:ml, :mp]),
                                           (fnp.take(w1_prev, dKp))[None, :], out=NN("wdc")[:ml, :mp])
                        Qin = fnp.take(Qc, dKp, axis=0)
                    else:
                        WDq = fnp.multiply(fnp.take(W, Kl, axis=0, out=NN("wq")[:ml, :n]),
                                           (w1_prev)[None, :], out=NN("wdc")[:ml, :n])
                        Qin = Qc
                    qcp = pool.get(("qcp", qc_side), (n + 1, r_old))
                    Qcc = smm.mm2(WDq, Qin, qcp[:ml]) if _g(smm, "qc") else fnp.matmul(WDq, Qin, out=qcp[:ml])
                    fnp.copyto(qcp[ml], 0.0)
                    Qc = fnp.take(qcp[:ml + 1], inv_l, axis=0, out=pool.get(("qc", qc_side), (n, r_old)))
                elif ka > 0:
                    qc_side ^= 1
                    Qc = (smm.mm2(WD, Qc, pool.get(("qc", qc_side), (n, r_old))) if _g(smm, "qc")
                          else fnp.matmul(WD, Qc, out=pool.get(("qc", qc_side), (n, r_old))))
                if not (comp_q and Kl is not None and ka > 0):
                    Qcc = fnp.take(Qc, Kl, axis=0) if (Kl is not None and ka > 0) else None   # (ml, r): filas conservadas
                if ka > kb and Kl is not None:
                    m1 = ka - kb
                    mc1 = max(mcol[kb:ka]) if fis >= 4 else n
                    _mo = ml if mo is None else mo
                    _g2 = apc_buf[2 * kb:2 * ka] if apc_buf is not None else gy2[:2 * m1]
                    _sbon(smm, 1); smm.mm(Qcc[:_mo][None, None], fap4[2 * fa_off:2 * (fa_off + m1)][..., :mc1],
                           _g2[:, :, :_mo, :mc1], smm.level(_mo, r_old, mc1, s_sb)); _sboff(smm)
                    if _mo < ml:
                        fnp.copyto(_g2[:, :, _mo:ml, :mc1], 0.0)
                    if apc_buf is None:
                        fnp.copyto(gy2[:2 * m1, :, ml, :mc1], 0.0)
                        fnp.take(gy2[:2 * m1, :, :ml + 1, :mc1], inv_l, axis=2, out=legs4["AP1"][2 * kb:2 * ka][..., :mc1])
                        if mc1 < n:
                            fnp.copyto(legs4["AP1"][2 * kb:2 * ka][..., mc1:], 0.0)
                elif ka > kb:
                    # formed dense legs of the tier-1 sources (pre-wick of this layer):
                    # V28 ONE Strassen family over the interleaved (A, P) factor slab
                    m1 = ka - kb
                    _sbon(smm, 2); smm.mm(Qc[None, None], fap4[2 * fa_off:2 * (fa_off + m1)],
                           legs4["AP1"][2 * kb:2 * ka], smm.level(n, r_old, n, s_sb)); _sboff(smm)
                if kb > 0 and Kl is not None:
                    _mo = ml if mo is None else mo
                    if E28_UEYE:
                        QUc = pool.get("p2qu", (n, r2))[:_mo]
                        fnp.copyto(QUc, Qcc[:_mo, :r2])
                    else:
                        QUc = smm.mm2(Qcc[:_mo], U, pool.get("p2qu", (n, r2))[:_mo]) if _g(smm, "nestsm") else fnp.matmul(Qcc[:_mo], U)
                    mc2 = max(mcol[:kb]) if fis >= 4 else n
                    _g3 = apc_buf[:2 * kb] if apc_buf is not None else gy2[:2 * kb]
                    _sbon(smm, 3); smm.mm(QUc[None, None], fap24[:2 * kb][..., :mc2], _g3[:, :, :_mo, :mc2],
                           smm.level(_mo, r2, mc2, s_sb)); _sboff(smm)
                    if _mo < ml:
                        fnp.copyto(_g3[:, :, _mo:ml, :mc2], 0.0)
                    if apc_buf is None:
                        fnp.copyto(gy2[:2 * kb, :, ml, :mc2], 0.0)
                        fnp.take(gy2[:2 * kb, :, :ml + 1, :mc2], inv_l, axis=2, out=legs4["AP1"][:2 * kb][..., :mc2])
                        if mc2 < n:
                            fnp.copyto(legs4["AP1"][:2 * kb][..., mc2:], 0.0)
                elif kb > 0:
                    # V24: tier-2 legs through the sub-basis
                    if E28_UEYE:
                        QU = pool.get("qu", (n, r2))
                        fnp.copyto(QU, Qc[:, :r2])
                    else:
                        QU = (smm.mm2(Qc, U, pool.get("qu", (n, r2))) if _g(smm, "nestsm") else fnp.matmul(Qc, U, out=pool.get("qu", (n, r2))))
                    _sbon(smm, 4); smm.mm(QU[None, None], fap24[:2 * kb], legs4["AP1"][:2 * kb],
                           smm.level(n, r2, n, s_sb)); _sboff(smm)
                # V29: the dense young legs were pre-scaled by w1_prev at the previous
                # wick, so the family's left operand is the raw W; the newborn's A leg
                # (slot k A position, born w1-scaled) and the covariance C (slot k P
                # position) ride along: out [ka:k] = transported legs, [k,0] = W a_b,
                # [k,1] = W C (consumed by the C_pre block below, then overwritten by W).
                # Level 0 falls back to the dense batched matmul inside mm().
                extra = 2 if newborn is not None else 0
                if pre_done:
                    # familia sin C (ya hecha arriba): patas jovenes + pata A de la recien nacida
                    nb = 2 * (k - ka) + 1
                    mcf = max(mcol[ka:k + 1]) if fis >= 4 else n
                    s_in = legs4["AP0"][2 * ka:2 * k + 1]
                    s_out = legs4["AP1"][2 * ka:2 * k + 1]
                    s_outv = s_out
                    if mcf < n:
                        s_in = s_in[..., :mcf]
                        s_outv = s_out[..., :mcf]
                    if mp < n:
                        Yg = fnp.take(s_in, dKp, axis=2, out=gy[:nb, :, :mp, :mcf])
                        Xw = Wgm
                    else:
                        Yg = s_in
                        Xw = W
                    smm.smin = DEAD_SMIN
                    if Kl is not None:
                        Xr = fnp.take(Xw, Kl, axis=0, out=NN("wr")[:ml, :mp])
                        smm.mm(Xr[None, None], Yg, gy2[:nb, :, :ml, :mcf], smm.level(ml, mp, mcf, s_lev))
                        fnp.copyto(gy2[:nb, :, ml, :mcf], 0.0)
                        fnp.take(gy2[:nb, :, :ml + 1, :mcf], inv_l, axis=2, out=s_outv)
                    else:
                        smm.mm(Xw[None, None], Yg, s_outv, smm.level(n, mp, mcf, s_lev))
                    smm.smin = STRASSEN_MIN
                    if mcf < n:
                        fnp.copyto(s_out[..., mcf:], 0.0)
                elif 2 * k + extra > 2 * ka:
                    smm.mm(W[None, None], legs4["AP0"][2 * ka:2 * k + extra],
                           legs4["AP1"][2 * ka:2 * k + extra], smm.level(n, n, n, s_lev))
                z_side ^= 1
                if fis >= 5 and pre_done and (Kl is not None or mp < n):
                    # W diag(w1) solo en filas conservadas de esta capa y columnas conservadas de la anterior
                    if mp < n:
                        Xz = Xr if Kl is not None else Wgm
                        w1g = fnp.take(w1_prev, dKp)
                        Zin = fnp.take(Z_st, dKp, axis=1)
                    else:
                        Xz = Xr
                        w1g = w1_prev
                        Zin = Z_st
                    if WDq is not None and Kl is not None:
                        WDc = WDq
                    else:
                        WDc = fnp.multiply(Xz, (w1g)[None, :], out=NN("wdc")[:Xz.shape[0], :Xz.shape[1]])
                    zb_ = pool.get(("z", z_side), (L - 1, n, r + 2))
                    if Kl is not None:
                        zc = pool.get("zc", (L - 1, n + 1, r + 2))
                        fnp.matmul(WDc[None], Zin, out=zc[:k, :ml])
                        fnp.copyto(zc[:k, ml], 0.0)
                        Z_st = fnp.take(zc[:k, :ml + 1], inv_l, axis=1, out=zb_[:k])
                    else:
                        Z_st = fnp.matmul(WDc[None], Zin, out=zb_[:k])
                    if Zf_st is not None and k > ZK1:
                        zfb_ = pool.get(("zf", z_side), (L - 1, n, 2 * rfb))
                        Zfin = fnp.take(Zf_st[ZK1:k], dKp, axis=1) if mp < n else Zf_st[ZK1:k]
                        if Kl is not None:
                            zfc = pool.get("zfc", (L - 1, n + 1, 2 * rfb))
                            fnp.matmul(WDc[None], Zfin, out=zfc[ZK1:k, :ml])
                            fnp.copyto(zfc[ZK1:k, ml], 0.0)
                            Zf_st = fnp.take(zfc[ZK1:k, :ml + 1], inv_l, axis=1, out=zfb_[ZK1:k])
                        else:
                            Zf_st = fnp.matmul(WDc[None], Zfin, out=zfb_[ZK1:k])
                else:
                    Z_st = fnp.matmul(WDb, Z_st, out=pool.get(("z", z_side), (L - 1, n, r + 2))[:k])
                    if Zf_st is not None and k > ZK1:
                        Zf_st = fnp.matmul(WDb, Zf_st[ZK1:k], out=pool.get(("zf", z_side), (L - 1, n, 2 * rfb))[ZK1:k])
                legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]
                legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]
            else:
                k = 0
            if newborn is not None and not skip_src:
                a_b, Rr, Lr, s_b, e_b, Ff = newborn
                if A_st is None:
                    # V29: layer 1 has no stack yet: the newborn + C family alone (slot 0
                    # written at birth on the AP0 side; output to AP1, then swap)
                    smm.mm(W[None, None], legs4["AP0"][0:2], legs4["AP1"][0:2],
                           smm.level(n, n, n, s_lev))
                    legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]
                    legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]
                # V29: C_pre from the transported covariance W C in the newborn's P slot
                WC = legs["AP0"][k, 1]
                if pre_done:
                    pass
                elif trim:
                    # var = diag(W C W^T) = rowsum((W C) * W)
                    var = fnp.maximum(fnp.sum(fnp.multiply(WC, W, out=T1), axis=1), 1e-10)
                else:
                    C_pre = self._sym_product(WC, w32, n, NN("cpre"), min(CPRE_LEV, s_lev))
                if tau_b > 0.0 and alive_b_prev is not None:
                    # (b) columnas de la fuente con umbral propio: W sin las columnas cortadas (permutada con fis >= 4)
                    if fis >= 4 and perm_b_prev is not None:
                        fnp.take(fnp.multiply(W, (alive_b_prev)[None, :], out=NN("wb")), perm_b_prev, axis=1,
                                 out=legs["AP0"][k, 1])
                    else:
                        fnp.multiply(W, (alive_b_prev)[None, :], out=legs["AP0"][k, 1])
                elif fis >= 4 and mp < n:
                    if not pre_done:
                        Wg = fnp.take(W, dperm_prev, axis=1, out=NN("wg"))
                    fnp.copyto(legs["AP0"][k, 1], Wg)
                    fnp.copyto(legs["AP0"][k, 1][:, mp:], 0.0)
                elif DEAD_PMASK and alive_prev is not None:
                    fnp.multiply(W, (alive_prev)[None, :], out=legs["AP0"][k, 1])
                else:
                    fnp.copyto(legs["AP0"][k, 1], W)
                A_st = legs["AP0"][:k + 1, 0]
                P_st = legs["AP0"][:k + 1, 1]
                # V27: the newborn's thin columns go into slot k of the current Z side
                # (the transport above wrote slots [:k] of that side); L was written into
                # its slot at birth.
                zb = pool.get(("z", z_side), (L - 1, n, r + 2))
                fnp.matmul(W, Rr, out=zb[k])
                Z_st = zb[:k + 1]
                L_st = pool.get("l", (L - 1, n, r + 2))[:k + 1]
                if Ff is not None:
                    zfb = pool.get(("zf", z_side), (L - 1, n, 2 * rfb))
                    if not (AUD_ZS and k == 0):   # la fuente de la capa 0 no tiene patas finas: W @ 0
                        fnp.matmul(W, Ff, out=zfb[k])
                    Zf_st = zfb[:k + 1]
                s_list.append(s_b)
                e_list.append(e_b)
                newborn = None

            # ---- WK slices ----
            if trim:
                C_off = None
            else:
                var = fnp.maximum(fnp.diag(C_pre), 1e-10) if var_p is None else var_p
                C_off = _zero_diag(C_pre)
            mode = 0 if A_st is None else 1
            if mode == 1 and skip_src:
                D3, D21 = fnp.zeros(n, dtype=f32), None
                if regen:
                    WW = fnp.multiply(W, W, out=NN("ww"))
                    dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev
                    g4row = dG * METRIC_C
                    wk4m = wk431 = None
                elif riders:
                    g4row = ((W * W) @ K4_vec) * 0.5 * float(st["wk4_c4"] * metric2)
                    wk4m = None
                else:
                    g4row = ones_n * (K4_sigma * float(st["wk4_c4"] * metric2))
                    wk4m = None
            elif mode == 1:
                if bufs is None:
                    bufs = {nm: pool.get(nm, (L - 1, n, n))
                            for nm in ("ap", "pp", "t", "mp", "xt", "yt", "u")}
                    bufs["lap"], bufs["lap4"] = pool.get_pair("lap", (L - 1, 2, n, n))   # V26: [LA | LP]
                    bufs["hub"] = pool.get("hub", (n, n))             # V26: hub result
                    bufs["ppl"] = pool.get("ppl", (L - 1, n, r + 2))  # V27: PP L stack
                    bufs["d21"] = NN("d21")
                    bufs["t1"] = T1
                    if rfb > 0:
                        bufs["gyr"] = pool.get("gyr", (L - 1, n, rfb))
                        bufs["gxr"] = pool.get("gxr", (L - 1, n, rfb))
                        bufs["d21fb"] = NN("d21fb")
                if RPROJ > 0 and ka > 0:
                    self._ora_S = Sg if kb == 0 else (U @ (S2 @ U.T) if Sg is None else Sg + U @ (S2 @ U.T))
                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                                        c1_list, c2_list, y_list, n, bufs,
                                        r, rfb, Zf_st, R1T_st, R2T_st,
                                        need_d21=not trim,
                                        ka=ka, Qc=(Qc[:, :FIJA_RC] if (FIJA and FIJA_RC > 0 and Qc is not None) else Qc), kb=kb, U2=U,
                                        apb=legs["AP0"], apb4=legs4["AP0"],
                                        sb1=((fap4[2 * fa_off:2 * (fa_off + ka - kb)][:, :, :FIJA_RC] if (FIJA and FIJA_RC > 0)
                                              else fap4[2 * fa_off:2 * (fa_off + ka - kb)]) if ka > kb else None),
                                        sb2=(fap24[:2 * kb] if kb > 0 else None), s_sb=s_sb,
                                        dz=((Kl, ml, inv_l, gy, gy2, (mcol if fis >= 4 else None), comp_f, mo, apc_buf) if Kl is not None else None))
                if regen:
                    # F68: exact transported diagonal of the regenerated core
                    # G = diag(g) + lam C_off:  dG = (W*W) g + lam (diag(C_pre)
                    # - (W*W) var).  NOT an einsum: same-object repeated operands
                    # crash flopscope (F45).
                    WW = fnp.multiply(W, W, out=NN("ww"))
                    if BETA != 0.0:
                        # V25: adaptive lambda. dG is affine in lam: t_g + lam * t_v.
                        t_g = WW @ g_prev
                        t_v = var - WW @ var_prev
                        dG0 = t_g + t_v * lam_prev                # table value first
                        rr = fnp.mean(dG0) / fnp.mean(var)
                        ref = float(REF_R[min(li - 1, len(REF_R) - 1)])
                        # clamp: an odd MLP must never turn the rule into NaN/inf (zero fallback)
                        lam_prev = lam_prev * fnp.power(fnp.clip(rr / ref, 0.5, 2.0), BETA)
                        dG = t_g + t_v * lam_prev                 # var == diag(C_pre)
                    else:
                        dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev  # var == diag(C_pre)
                    g4row = (dG * (METRIC_C * C1_A) + (D3 * D3) * (C1_B / var)) if K4C1 else dG * METRIC_C
                    g22c = (dG * (METRIC_C / 6.0))[:, None]
                    wk4m = _zero_diag(fnp.add(g22c, g22c.T, out=NN("wk4m")))
                    wk431 = None if trim else fnp.multiply(C_off, (0.5 * METRIC_C * lam_prev), out=NN("wk431"))
                    if _os.environ.get("V17_DEBUG", "0") == "1":
                        DEBUG.append(dict(layer=li, dG=dG, g_prev=g_prev, var_prev=var_prev,
                                          lam=lam_prev, var=var, D3=D3, D21=D21))
                elif riders:
                    # vec+ww: per-neuron K4 diagonal content transported by the
                    # true (W*W) row action; one avg-metric cup replaced
                    # (/metric_c = *0.5). NOT an einsum: same-object repeated
                    # operands crash flopscope (F45).
                    gv = ((W * W) @ K4_vec) * 0.5
                    g4row = gv * float(st["wk4_c4"] * metric2)
                    g22 = gv * float(0.5 * st["wk4_c22"] * metric2)
                    g22c = (g22)[:, None]
                    wk4m = _zero_diag(fnp.add(g22c, g22c.T, out=NN("wk4m")))
                else:
                    g4v = K4_sigma * float(st["wk4_c4"] * metric2)
                    g22v = K4_sigma * float(st["wk4_c22"] * metric2)
                    g4row = ones_n * g4v
                    wk4m = _zero_diag(ones2 * g22v)
            else:
                D3 = D21 = g4row = wk4m = None

            # ---- TT (2,2) (c2, 22.10.2026): exact coherent transport of the previous layer's post-ReLU (2,2) slice and
            # kappa4 diagonal, blended into the regenerated wk4m (k4slice: (4)+(2,2) exact transport = 71% of K4 on 128) ----
            if TT22_LAM > 0.0 and wk4m is not None and tt22_prev is not None:
                K22p, K4vp = tt22_prev
                WW2 = W * W
                T22 = fnp.matmul(fnp.matmul(WW2, K22p + fnp.diag(K4vp)), WW2.T)
                wk4m = wk4m + (_zero_diag(T22) - wk4m) * TT22_LAM
                if g4row is not None:   # exact coherent (4): sum W^4 K4v + 3 sum_{i!=j} W^2 W^2 K22 = 3 diag(T22) - 2 W^4 K4v
                    g4ex = fnp.diagonal(T22) * 3.0 - ((WW2 * WW2) @ K4vp) * 2.0
                    g4row = g4row + (g4ex - g4row) * TT22_LAM
            # ---- wick matrix ----
            sigma = fnp.sqrt(var)
            alpha = mu / sigma
            phi = flops.stats.norm.pdf(alpha).astype(f32)
            Phi = flops.stats.norm.cdf(alpha).astype(f32)
            if lt_on:
                fd = {"mu_pre": mu, "var": var, "alpha": alpha, "phi": phi, "Phi": Phi}
                if mode == 1:
                    fd["D3"] = D3
                    fd["g4row"] = g4row
                    fd["lam"] = ones_n * lam_prev
                    if D21 is not None:
                        sD = fnp.abs(D21, out=NN("lt_s"))
                        fd["d21_abs"] = fnp.sum(sD, axis=1)
                        fd["d21T_abs"] = fnp.sum(sD, axis=0)
                        fnp.multiply(D21, D21, out=sD)
                        fd["d21_sq"] = fnp.sum(sD, axis=1)
                        fd["d21_colsum"] = fnp.sum(D21, axis=0)
                        fd["d21_rowsum"] = fnp.sum(D21, axis=1)
                if C_off is not None:
                    sC = fnp.multiply(C_off, C_off, out=NN("lt_s"))
                    fd["coff_sq"] = fnp.sum(sC, axis=1)
                    fd["coff_sum"] = fnp.sum(C_off, axis=1)
                    if lt_ext:
                        fd["cn2"] = sC @ (1.0 / var)
                        fd["coff_abs"] = fnp.sum(fnp.abs(C_off, out=sC), axis=1)
            if PWR:
                APOW = fnp.power((alpha)[:, None], EXA)
                inv = 1.0 / sigma
                SB = fnp.concatenate([fnp.power((inv)[:, None], EXI), fnp.power((sigma)[:, None], EXS)], axis=1)
            else:
                a2 = alpha * alpha
                a3 = a2 * alpha
                a4 = a3 * alpha
                a5 = a4 * alpha
                APOW = fnp.stack([ones_n, alpha, a2, a3, a4, a5], axis=1)
                inv = 1.0 / sigma
                i2 = inv * inv
                i3 = i2 * inv
                i4 = i3 * inv
                i5 = i4 * inv
                i6 = i5 * inv
                s2 = sigma * sigma
                s3 = s2 * sigma
                s4 = s3 * sigma
                SB = fnp.stack([i6, i5, i4, i3, i2, inv, ones_n, sigma, s2, s3, s4], axis=1)
            phic = (phi)[:, None]
            Phic = (Phi)[:, None]
            W_all = ((APOW @ C1) * phic + (APOW @ C2) * Phic) * (SB @ SELC)
            lead_dead = None
            W_u = C_u = None
            if dead_tau > 0.0 and riders:
                if fis >= 1 or dround != "arriba":
                    alive, nal_, mp_next, perm_next = _vivas(alpha, DEAD_TAU_L.get(li, dead_tau), n, f32, dround)
                    stat.append((li, nal_, mp_next, ml))
                else:
                    alive = fnp.greater(alpha, -DEAD_TAU_L.get(li, dead_tau)).astype(f32)
                alive_now = alive
                if tau_b > 0.0:
                    alpha_b = (alpha - 1.0e6 * fnp.greater(alpha, on_b).astype(f32)) if on_b > 0.0 else alpha
                    if fis >= 4:
                        alive_b, _nb, mb_next, perm_b_next = _vivas(alpha_b, tau_b, n, f32, dround)
                    else:
                        alive_b = fnp.greater(alpha_b, -tau_b).astype(f32)
                m2 = alive[:, None] * alive[None, :]
                if DEAD_MODE == "k3":
                    if D3 is not None:
                        D3 = D3 * alive
                    if D21 is not None:
                        D21 = D21 * m2
                else:
                    lead_dead = W_all[:, 0] * (1.0 - alive)
                    if cent and not trim:
                        # (1) C entera: los pesos de Wick sin mascara entran solo en los terminos no lineales
                        W_u = W_all
                        C_u = C_off
                        if D3 is not None:
                            D3 = D3 * alive
                    W_all = W_all * alive[:, None]
                    if not trim:
                        C_off = C_off * m2
                        if D21 is not None:
                            D21 = D21 * m2
            WT = W_all.T if W_u is None else W_u.T

            # ---- nonlin terms (fused) ----
            pmode = mode if (mode == 0 or (regen and not NO_WK431)) else 2
            prog = self._progs[pmode]
            fpm = FP[pmode]
            b2 = {"ones2": ones2, "c_off": C_off if C_u is None else C_u}
            b1 = {"ones1": ones_n}
            if mode == 1:
                if not trim:
                    d3col = fnp.multiply((D3)[:, None], ones_row, out=NN("d3col"))
                    g4col = fnp.multiply((g4row)[:, None], ones_row, out=NN("g4col"))
                    b2.update({"d21": D21, "d21T": D21.T, "wk4m": wk4m,
                               "d3col": d3col, "d3row": d3col.T,
                               "g4col": g4col, "g4row": g4col.T})
                    if regen and not NO_WK431:
                        b2["wk431"] = wk431
                b1.update({"d3": D3, "g4": g4row})
            ab_cache = {}

            def ab(pair, tbl):
                if pair in ab_cache:
                    return ab_cache[pair]
                a, b = pair
                if a in ("ones2", "ones1"):
                    val = tbl[b]
                elif b in ("ones2", "ones1"):
                    val = tbl[a]
                else:
                    val = tbl[a] * tbl[b]
                ab_cache[pair] = val
                return val

            if not trim:
                ab2 = prog["AB2"]
                for t_, (a_, b_) in enumerate(ab2):
                    if a_ in ("ones2", "ones1"):
                        fnp.copyto(abbuf[t_], b2[b_])
                    elif b_ in ("ones2", "ones1"):
                        fnp.copyto(abbuf[t_], b2[a_])
                    else:
                        fnp.multiply(b2[a_], b2[b_], out=abbuf[t_])
                ABstack = abbuf[:len(ab2)]
                WL2 = fpm["SL2"] @ WT
                WR2 = fpm["SR2"] @ WT
                if AUD_IND2:
                    PK2 = pool.get("pk2aud", (len(D2_IPS), n, n))
                    if RELOJ and pmode != 0:
                        # misma aritmetica que el einsum de tres operandos, en el sitio sobre la pila (ya no se usa)
                        fnp.multiply(ABstack, WL2[:, :, None], out=ABstack)
                        fnp.multiply(ABstack, WR2[:, None, :], out=ABstack)
                    for _G, (_a, _b) in enumerate(self._ind2b[pmode]):
                        if RELOJ and pmode != 0:
                            fnp.sum(ABstack[_a:_b], axis=0, out=PK2[_G])
                        else:
                            fnp.einsum("tij,ti,tj->ij", ABstack[_a:_b], WL2[_a:_b], WR2[_a:_b], out=PK2[_G])
                else:
                    PK2 = fnp.einsum("tij,ti,tj,tg->gij", ABstack, WL2, WR2, fpm["IND2"])
            B1stack = fnp.stack([ab(p, b1) for p in prog["B1"]], axis=0)
            WL1 = fpm["SL1"] @ WT
            PK1 = fnp.einsum("ti,ti,tg->gi", B1stack, WL1, fpm["IND1"])

            if not trim:
                pk11 = fnp.add(PK2[0], PK2[0].T, out=NN("pk11"))
                pk11 = _zero_diag(fnp.multiply(pk11, 0.5, out=pk11))
                pk21 = _zero_diag(PK2[1])
                pk22 = fnp.add(PK2[2], PK2[2].T, out=NN("pk22"))
                pk22 = _zero_diag(fnp.multiply(pk22, 0.5, out=pk22))
            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]

            # ---- online mean correction: delta = feats @ beta[l] (13 dots/neuron);
            # mu still holds the pre-nonlin mean here, pk1v == K(1,) is the pred.
            if riders and not NO_CORR:
                if mode == 1:
                    D3f = D3
                    D21n = fnp.abs(D21, out=T1) @ ones_n if D21 is not None else fnp.zeros(n, dtype=f32)
                    k4f = ones_n * K4_sigma
                else:
                    D3f = fnp.zeros(n, dtype=f32)
                    D21n = fnp.zeros(n, dtype=f32)
                    k4f = fnp.zeros(n, dtype=f32)
                if _FZ is None or TAB_DUMP:
                    feats = fnp.stack([ones_n, mu, var, sigma, alpha, fnp.abs(alpha),
                                       phi, Phi, pk1v, D3f, D21n, k4f, sigma * phi],
                                      axis=1)
                if _FZ is not None:
                    g4f = g4row if (mode == 1 and g4row is not None) else zeros_n
                    d21s = crs = cr2 = zeros_n
                    if _FZ["pares"]:
                        if D21 is not None:
                            d21s = D21 @ ones_n
                        if C_off is not None:
                            crs = C_off @ ones_n
                            cr2 = fnp.einsum("ij,ij->i", C_off, C_off)
                    sp = sigma * phi
                    fb = {"1": ones_n, "mu": mu, "var": var, "sigma": sigma, "alpha": alpha, "phi": phi,
                          "Phi": Phi, "pred": pk1v, "D3": D3f, "D21n": D21n, "K4": k4f, "g4": g4f,
                          "pk2": PK1[1], "pk3": PK1[2], "pk4": PK1[3], "sigma*phi": sp}
                    if "|alpha|" in _FZ["base"]:
                        fb["|alpha|"] = fnp.abs(alpha)
                    fm = {"": ones_n, "s3": D3f, "s4": g4f, "dn": D21n, "v": var, "ds": d21s, "cr": crs,
                          "c2": cr2, "k4": k4f}
                    if "s33" in _FZ["mods"]:
                        fm["s33"] = D3f * D3f
                    Mod = fnp.stack([fm[m] for m in _FZ["mods"]], axis=1) * fnp.power(var[:, None], fz_E)
                    G = Mod @ fz_BH[li]
                    V = fnp.power(alpha[:, None], fz_J)
                    delta = (fnp.stack([fb[nm] for nm in _FZ["base"]], axis=1) @ fz_BB[li]
                             + fnp.einsum("ij,ij,i->i", V, G, sp))
                elif CORR_FEATS is None or TAB_OLD:
                    delta = feats @ beta_rows[li]
                else:
                    g4f = g4row if (mode == 1 and g4row is not None) else zeros_n
                    d21s = crs = cr2 = zeros_n
                    if any(nm.endswith(("*ds", "*cr", "*c2")) for nm in CORR_FEATS):
                        # sumas por fila de pares (n^2 cada una); no existen en la ultima capa (trim)
                        if D21 is not None:
                            d21s = D21 @ ones_n
                        if C_off is not None:
                            crs = C_off @ ones_n
                            cr2 = fnp.multiply(C_off, C_off, out=T1) @ ones_n
                    if CORR_F64:   # diagnostico: suma de la correccion en float64
                        delta = (_tab_feats(CORR_FEATS, ones_n, mu, var, sigma, alpha, phi, Phi, pk1v,
                                            D3f, D21n, k4f, g4f, PK1, d21s, crs, cr2).astype(fnp.float64)
                                 @ fnp.asarray(CORR_BETA[li], dtype=fnp.float64)).astype(f32)
                    else:
                        delta = _tab_feats(CORR_FEATS, ones_n, mu, var, sigma, alpha, phi, Phi, pk1v,
                                           D3f, D21n, k4f, g4f, PK1, d21s, crs, cr2) @ beta_rows[li]
                if TAB_DUMP:
                    g4d = g4row if (mode == 1 and g4row is not None) else zeros_n
                    if D21 is not None:
                        d21s = D21 @ ones_n
                    else:
                        d21s = zeros_n
                    if C_off is not None:
                        crs = C_off @ ones_n
                        cr2 = fnp.multiply(C_off, C_off) @ ones_n
                    else:
                        crs = cr2 = zeros_n
                    _tab_rec(li, mu=mu, var=var, pk1=pk1v, pk2=PK1[1], pk3=PK1[2], pk4=PK1[3], d3=D3f, d21n=D21n,
                             d21s=d21s, g4=g4d, k4s=k4f, crs=crs, cr2=cr2,
                             lam=float(lam_prev) if not hasattr(lam_prev, "item") else float(lam_prev.item()),
                             base=(pk1v if lead_dead is None else pk1v + lead_dead),
                             delta_old=feats @ fnp.asarray(CORR_BETA_OLD[li], dtype=f32),
                             delta=delta)
            else:
                delta = None

            if lt_ext and delta is not None:
                fd["delta"] = delta
            if riders and li>0:
                if not last:
                    pr=C_off/(sigma[:,None]*sigma[None,:]);pt=D21/(var[:,None]*sigma[None,:]);pr2=pr*pr;pt2=pt*pt
                    prpt=pr*pt;prptt=pr*pt.T
                    pf=fnp.stack([fnp.sum(pr2,axis=1),fnp.sum(pr2*pr,axis=1),fnp.sum(pr2*pr2,axis=1),fnp.sum(pt2,axis=1),fnp.sum(pt2,axis=0),fnp.sum(prpt,axis=1),fnp.sum(prptt,axis=1),pr2@(g4row/(var*var))],axis=1)
                    tau=fnp.sum(D21,axis=0)+D3
                    ex=fnp.stack([fnp.sum(pt,axis=1),fnp.sum(pt,axis=0),pr@alpha,pt@alpha,pt.T@alpha,prpt@alpha,prptt@alpha,tau/(sigma*fnp.mean(var)),ones_n*(fnp.mean(tau*mu)/(fnp.mean(mu*mu)*fnp.mean(var)+1e-30))],axis=1)
                else:
                    pf=fnp.zeros((n,8),dtype=f32);ex=fnp.zeros((n,9),dtype=f32)
                ga3=D3/(sigma*var);ga4=g4row/(var*var)
                features=fnp.concatenate([fnp.stack([alpha,ga3,ga4,fnp.log(var/fnp.mean(var))],axis=1),pf,fnp.stack([ones_n*fnp.mean(ga4),ones_n*fnp.mean(ga3*ga3),ones_n*(li/15.)],axis=1)],axis=1)
                features=fnp.concatenate([features,ex],axis=1)
                features=fnp.concatenate([features,fnp.broadcast_to(fnp.mean(features,axis=0),features.shape)],axis=1)
                features=fnp.concatenate([features,features[:,:24]],axis=1)
                block=0 if li<=5 else (1 if li<=10 else (2 if not last else 3))
                normalized=fnp.clip((features-self._nn_mean)/self._nn_scale,-8.,8.)
                hidden=fnp.tanh(normalized@self._nn_hw.T+self._nn_hb)
                learned=.001*sigma*phi*(hidden@self._nn_ow[block]+self._nn_ob[block])
                res_e=Phi*(W@res_e)+learned
                if lt_dump:
                    LT_RES.append((li, fnp.copy(res_e)))

            if last:
                _r = pk1v if delta is None else pk1v + delta
                if lt_on:
                    fd["pred"] = pk1v
                    if lt is not None and lt.get("v2", False):
                        lt_u, lt_prev = self._lt2_layer(lt, li, fd, W, NN("ww"), lt_prev, n, True)
                    else:
                        lt_u, lt_prev = self._lt_layer(lt, li, fd, W, NN("ww"), lt_prev, lt_u, n, None, lt_dump)
                    if lt is not None:
                        _r = _r + lt_u
                if riders:_r=fnp.maximum(_r+res_e,0)
                rows.append(_r if lead_dead is None else _r + lead_dead)
                break

            # ---- wick old blocks + dslice scalings ----
            if dead_tau > 0.0 and riders and DEAD_MODE == "k3":
                W_all = W_all * alive[:, None]
                C_off = C_off * alive[None, :]
            w1 = W_all[:, self._i11]
            w2 = W_all[:, self._i21]
            w1col = (w1)[:, None]
            w1_prev = w1  # thin stacks / basis pick up this wick via WD at the next linear
            if mode == 1 and ka < A_st.shape[0]:
                # V29: dense young legs pre-scaled by w1 (rows) in place, so the next
                # transport family multiplies by the raw W (W (w1 A) == (W w1) A up to
                # rounding); the joiner reads them scaled too.
                kk = A_st.shape[0]
                fnp.multiply(legs4["AP0"][2 * ka:2 * kk], w1col[None, None],
                             out=legs4["AP0"][2 * ka:2 * kk])
            if mode == 1:
                D3_w = D3 * w1 ** 3
                D21_w = fnp.multiply(w1col * w1col, D21, out=NN("d21w"))
                fnp.multiply(D21_w, (w1)[None, :], out=D21_w)
            else:
                D3_w = None
                D21_w = None

            # ---- new source (V1.6, struct-free Y1 = a_b * D(w2)) ----
            if legs is None:
                # V26: A and P legs interleaved per slot: [:, 0] = A, [:, 1] = P
                # V28: with one-time (2 (L-1), 1, n, n) views for the Strassen batches
                pairs = {nm: pool.get_pair(nm, (L - 1, 2, n, n)) for nm in ("AP0", "AP1")}
                legs = {nm: pairs[nm][0] for nm in pairs}
                legs4 = {nm: pairs[nm][1] for nm in pairs}
            # V29: the newborn's A leg is born into its stack slot (slot = number of
            # sources present at this layer); the next layer's family transports it
            k_b = 0 if A_st is None else A_st.shape[0]
            cperm = (perm_b_next if tau_b > 0.0 else perm_next) if fis >= 4 else None
            if cperm is not None:
                a_b = fnp.multiply(w1col, C_off, out=NN("abnat"))
                if tau_b > 0.0:
                    fnp.multiply(a_b, (alive_b)[None, :], out=a_b)
                fnp.take(a_b, cperm, axis=1, out=legs["AP0"][k_b, 0])
            else:
                a_b = fnp.multiply(w1col, C_off, out=legs["AP0"][k_b, 0])
                if tau_b > 0.0:
                    fnp.multiply(a_b, (alive_b)[None, :], out=a_b)
            if rfb > 0 and mode == 1:
                # V18 (F69): D21 feedback thin legs. D21 ~ Qf Bf (rank rfb range finder,
                # one power iteration, sketch = a slice of the layer weight).
                w3 = W_all[:, self._i31]
                Omf = pool.get("omf", (n, rfb))  # V20: contiguous sketch
                fnp.copyto(Omf, w32[:, :rfb])
                Yf = D21 @ Omf
                Qf, _ = fnp.linalg.qr(Yf)
                Yf = D21 @ (D21.T @ Qf)
                Qf, _ = fnp.linalg.qr(Yf)
                Bf = Qf.T @ D21                       # D21 ~= Qf @ Bf
                F1_b = (w2)[:, None] * Qf            # Xt = F1 R1, R1 = 1.5 Bf
                R1T_b = Bf.T * 1.5                              # (n, rfb) = R1^T
                F2_b = w1col * Bf.T                             # Yt = F2 R2, R2 = 0.5 Qf^T d(w3)
                R2T_b = Qf * (w3 * 0.5)[:, None]     # (n, rfb) = R2^T
                Xt_b = fnp.matmul(F1_b, R1T_b.T, out=NN("xtb"))
                Yt_b = fnp.matmul(F2_b, R2T_b.T, out=NN("ytb"))
                X1_b = fnp.multiply(a_b, 3.0, out=NN("x1b"))
                fnp.add(X1_b, Xt_b, out=X1_b)
                Y1_b = fnp.multiply(a_b, (w2)[None, :], out=NN("y1b"))
                fnp.add(Y1_b, Yt_b, out=Y1_b)
                xd = fnp.diag(Xt_b)
                yd = fnp.diag(Yt_b)
                # birth dslices of B1 with P = I (general form; diag(a_b) = 0)
                # (same operation order as the V26 expression: bit-identical)
                D21_new = fnp.multiply((xd)[:, None], Y1_b.T, out=NN("d21new"))
                fnp.multiply(X1_b, Y1_b, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply((yd)[:, None], X1_b.T, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(D21_new, 1.0 / 3.0, out=D21_new)
                D3_new = xd * yd
            else:
                # birth dslices of B1 with P = I: diag(a_b) = 0, so only the a*Y1 term
                # survives: D21_new = a_b^2 * D(w2)-row; D3_new = 0.
                D21_new = fnp.multiply(a_b, a_b, out=NN("d21new"))
                fnp.multiply(D21_new, (w2)[None, :], out=D21_new)
                D3_new = None
                if rfb > 0:
                    F1_b = fnp.zeros((n, rfb), dtype=f32)
                    F2_b = F1_b
                    R1T_b = F1_b
                    R2T_b = F1_b
            if regen and mode == 1 and not NO_FEED:
                # K4->K3 feed (F68): X3 = diag(w1^2 dG) + lam a_b d(w1), Y3 = y 1^T
                # with y = (m/4) w2; M_t1 = u v^T, u = (m/4) w2*dG, v = w1^2.
                # Birth (P = I) dslices: D21 += (1/3)[dgw y^T + 2 y (.) X3_off]
                # + (1/3) v u^T;  D3 += dgw*y + u*v.
                w1sq = w1 * w1
                y_b = w2 * (0.25 * METRIC_C)
                dgw = w1sq * dG
                u_b = y_b * dG
                c1_b = w1 * lam_prev
                # (V26 expression, same operation order, out= into the pooled buffers)
                fnp.multiply((dgw)[:, None], (y_b)[None, :], out=T1)
                fnp.multiply(T1, 1.0 / 3.0, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(a_b, (c1_b)[None, :], out=T1)
                fnp.multiply((y_b * (2.0 / 3.0))[:, None], T1, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply((w1sq)[:, None], (u_b * (1.0 / 3.0))[None, :], out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                D3_new = ((dgw * y_b + u_b * w1sq) if D3_new is None
                          else D3_new + dgw * y_b + u_b * w1sq)
            else:
                y_b = fnp.zeros(n, dtype=f32)
                u_b = y_b
                c1_b = y_b
                dgw = y_b
                w1sq = y_b
            rep21 = D21_new if D21_w is None else fnp.add(D21_w, D21_new, out=NN("rep21"))
            rep21 = _zero_diag(rep21)

            # ---- pK -> K (per-entry; d=1 outputs are vector ops, d=2 few entries) ----
            pk = {(1,): pk1v, (2,): pk2v, (3,): pk3v, (4,): pk4v,
                  (1, 1): pk11, (2, 1): pk21, (2, 2): pk22}

            def pk_slice(vec):
                nz = tuple(sorted((x for x in vec if x > 0), reverse=True))
                base = pk[nz]
                if len(vec) == 1:
                    return base
                if len(nz) == 2:
                    return base if vec == tuple(sorted(vec, reverse=True)) else base.T
                return ((base)[:, None] if vec[0] > 0
                        else (base)[None, :])

            K = {}
            for out_part, entries in PK2K_TABLE.items():
                if len(out_part) == 2:
                    continue    # V27: matrix parts assembled below with out= (same order)
                if KVEC and len(out_part) == 1:
                    continue
                acc2 = None
                for vpart, coef in entries:
                    prod = None
                    for blk in vpart:
                        fct = pk_slice(blk)
                        prod = fct if prod is None else prod * fct
                    contrib = prod * coef
                    acc2 = contrib if acc2 is None else acc2 + contrib
                K[out_part] = acc2
            if KVEC:
                _s = fnp.multiply(pk1v, pk1v)
                _mono = fnp.stack([pk1v, pk2v, pk3v, pk4v, _s, fnp.multiply(pk1v, pk2v), fnp.multiply(pk1v, pk3v),
                                   fnp.multiply(pk2v, pk2v), fnp.multiply(_s, pk1v), fnp.multiply(_s, pk2v),
                                   fnp.multiply(_s, _s)], axis=1)
                _kv = _mono @ KC
                K[(1,)], K[(2,)], K[(3,)], K[(4,)] = _kv[:, 0], _kv[:, 1], _kv[:, 2], _kv[:, 3]
            # V27 (F80): the three matrix parts of PK2K_TABLE, unrolled in the generic
            # loop's operation order (bit-identical), written into pooled buffers:
            #   (1,1) = pk11 * 1.0 = pk11
            #   (2,1) = (pk1 (.) pk11) * (-2) + pk21
            #   (2,2) = ((pk1^T (.) pk1) (.) pk11) * 4 + (pk1^T (.) pk21) * (-2)
            #           + (pk1 (.) pk21^T) * (-2) + (pk11 * pk11) * (-2) + pk22
            pk1c = (pk1v)[:, None]
            pk1r = (pk1v)[None, :]
            K11 = pk11
            if FOLD2:
                m2v = fnp.multiply(pk1v, -2.0)
                m2c = (m2v)[:, None]
                m2r = (m2v)[None, :]
                K21 = fnp.multiply(m2c, pk11, out=NN("k21"))
                fnp.add(K21, pk21, out=K21)
                K22 = fnp.multiply(m2r, m2c, out=NN("k22"))
                fnp.multiply(K22, pk11, out=K22)
                fnp.multiply(m2r, pk21, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.multiply(m2c, pk21.T, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.multiply(pk11, pk11, out=T1)
                fnp.multiply(T1, -2.0, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.add(K22, pk22, out=K22)
                # las diagonales de pk11, pk21 y pk22 ya son cero: las de K21 y K22 tambien
            else:
                K21 = fnp.multiply(pk1c, pk11, out=NN("k21"))
                fnp.multiply(K21, -2.0, out=K21)
                fnp.add(K21, pk21, out=K21)
                K22 = fnp.multiply(pk1r, pk1c, out=NN("k22"))
                fnp.multiply(K22, pk11, out=K22)
                fnp.multiply(K22, 4.0, out=K22)
                fnp.multiply(pk1r, pk21, out=T1)
                fnp.multiply(T1, -2.0, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.multiply(pk1c, pk21.T, out=T1)
                fnp.multiply(T1, -2.0, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.multiply(pk11, pk11, out=T1)
                fnp.multiply(T1, -2.0, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.add(K22, pk22, out=K22)
                K21 = _zero_diag(K21)
                K22 = _zero_diag(K22)
            K1v = K[(1,)]
            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]
            tt22_prev = (fnp.copy(K22), fnp.copy(K4v)) if TT22_LAM > 0.0 else None
            if W_u is not None:
                # (1) solo C (K11 y su diagonal K2) conserva a las apagadas; lo demas, como la mascara
                fnp.multiply(K21, m2, out=K21)
                fnp.multiply(K22, m2, out=K22)
                K1v = K1v * alive
                K3v = K3v * alive
                K4v = K4v * alive

            if lt_on:
                fd["pred"] = K1v
                fd["K2v"] = K2v
                fd["K3v"] = K3v
                fd["K4v"] = K4v
                fd["k21row"] = fnp.sum(K21, axis=1)
                fd["k21col"] = fnp.sum(K21, axis=0)
                if lt_ext:
                    sX = NN("lt_s")
                    fd["k22abs"] = fnp.sum(fnp.abs(K22, out=sX), axis=1)
                    fd["k22sq"] = fnp.sum(fnp.multiply(K22, K22, out=sX), axis=1)
                    fd["k21sq"] = fnp.sum(fnp.multiply(K21, K21, out=sX), axis=1)
                    fd["k11sq"] = fnp.sum(fnp.multiply(K11, K11, out=sX), axis=1)
                    fd["k11sum"] = fnp.sum(K11, axis=1)
            mu_pre_saved=mu
            # ---- assemble ----
            mu = K1v if delta is None else K1v + delta
            if lead_dead is not None:
                mu = mu + lead_dead
            if TAB_DUMP:
                _tab_rec(li, base_asm=(K1v if lead_dead is None else K1v + lead_dead))
            if TAB_FORCE and _TAB_TRUTH is not None:
                mu = _TAB_TRUTH[li]   # 8.a-quater: media forzada a la verdadera (solo diagnostico)
            # V27: diagflat + add -> pooled buffer, diagonal written in place (bit-identical:
            # K11 has a zero diagonal)
            if FOLD2:
                C = NN("c")
                fnp.copyto(C, K11)   # K11 = pk11 ya es simetrico: (K11 + K11^T) * 0.5 = K11 (exacto)
            else:
                C = fnp.add(K11, K11.T, out=NN("c"))
                fnp.multiply(C, 0.5, out=C)
            fnp.fill_diagonal(C, K2v)
            if riders:
                cov_delta,dv=_fused_cov_delta(self,li,mu_pre_saved,var,alpha,phi,Phi,C_off,D3 if mode==1 else zeros_n,g4row if mode==1 else zeros_n,D21 if mode==1 else None,wk4m if mode==1 else None,wk431 if mode==1 else None,C)
                K2v=K2v+dv
                C=fnp.add(C,cov_delta,out=C)
            if not E29_NOSYMC:   # e29 nosymc: la etiqueta (7 n^2 facturados por capa) no la lee ninguna operacion posterior
                C = flops.as_symmetric(C, symmetry=(0, 1))
            # V29: C rides the next layer's transport family in the P position of the
            # newborn's slot (the family writes W C there; the newborn's P leg W is
            # copied over it once C_pre is formed)
            fnp.copyto(legs["AP0"][k_b, 1], C)
            S3c = K3v - D3_w if D3_w is not None else K3v
            if D3_new is not None:
                S3c = S3c - D3_new
            S21 = fnp.subtract(K21, rep21, out=NN("s21"))
            if not FOLD2:
                S21 = _zero_diag(S21)
            if dead_tau > 0.0 and riders and DEAD_MODE == "k3":
                S3c = S3c * alive
                fnp.multiply(S21, m2, out=S21)
            # S21 = S_sep + Rres: S_sep = exact leading (2,1) Wick term (the (c_off,
            # ones2) term of pk21 minus 2*pk1*[(c_off, ones2) term of pk11]); M_b^T
            # part 3*S_sep^T = 3*a_b*diag(e) rides on the A leg for free. Rres is
            # compressed to rank r (randomized range finder, one power iteration).
            e_b = (W_all[:, self._i12] + m2v * w1) if FOLD2 else (W_all[:, self._i12] - 2.0 * pk1v * w1)
            if lt_on:
                fd["e_b"] = e_b
                fd["w1"] = w1
                fd["w2"] = w2
                fd["s3c"] = S3c
            S_sep = fnp.multiply((e_b)[:, None], C_off, out=NN("ssep"))
            S_sep = fnp.multiply(S_sep, (w1)[None, :], out=S_sep)   # C_off tiene la diagonal a cero: S_sep tambien
            if not FOLD2:
                S_sep = _zero_diag(S_sep)
            Rres = fnp.subtract(S21, S_sep, out=S21)
            if tau_b > 0.0:
                # (b) como el oraculo: las columnas cortadas salen del residuo antes del buscador de rango
                fnp.multiply(Rres, (alive_b)[:, None], out=Rres)
            Om = pool.get("om_r", (n, r))  # V20: contiguous sketch
            fnp.copyto(Om, w32[:, :r])
            Y = fnp.matmul(Rres, Om, out=pool.get("y", (n, r)))
            Q, _ = fnp.linalg.qr(Y)
            Y2 = fnp.matmul(Rres.T, Q, out=pool.get("y2", (n, r)))
            Y = fnp.matmul(Rres, Y2, out=Y)
            Q, _ = fnp.linalg.qr(Y)
            Bm = fnp.matmul(Q.T, Rres, out=pool.get("bm", (r, n)))   # Rres ~= Q @ Bm
            # M_b = diag(S3c) + 3 S21^T = diag(S3c) + 3 a_b diag(e_b) + 3 Rr Lr^T
            # with Rr = Bm^T (n,r) [transported as Z = P Rr] and Lr = Q (static).
            # extra thin columns: M_t1 = u v^T -> Z column P u, L column v; and the Y3
            # row vector y is a LEFT factor of the hub tensor, so it is transported
            # like a leg (y(l) = P(l) y): Z column P y with a ZERO L column (keeps it
            # out of the M leg). Zeros off-suite / at layer 0 so every source keeps
            # r+2 columns.
            # V27: Rr into a persistent (n, r+2) buffer (read by next layer's W @ Rr);
            # Lr written straight into its stack slot k (static, never transported).
            Rr_full = pool.get("rr", (n, r + 2))
            fnp.multiply(Bm.T, 3.0, out=Rr_full[:, :r])
            fnp.copyto(Rr_full[:, r], u_b)
            fnp.copyto(Rr_full[:, r + 1], y_b)
            # slot of the source born here = number of sources present at this layer
            # (the newborn added above included): it joins the stacks at the next layer
            k_b = 0 if A_st is None else A_st.shape[0]
            lb = pool.get("l", (L - 1, n, r + 2))
            if cperm is not None:
                fnp.take(Q, cperm, axis=0, out=lb[k_b, :, :r])
                fnp.copyto(lb[k_b, :, r], fnp.take(w1sq, cperm))
            else:
                fnp.copyto(lb[k_b, :, :r], Q)
                fnp.copyto(lb[k_b, :, r], w1sq)
            fnp.copyto(lb[k_b, :, r + 1], zeros_n)
            Lr_full = None
            # V18: thin feedback legs [F1 | F2] travel in their own stack Zf (transported
            # like Z, never entering the M-leg einsums); right factors R1T/R2T are static.
            Ff_b = fnp.concatenate([F1_b, F2_b], axis=1) if rfb > 0 else None
            if cperm is not None:
                def cg(v):
                    return fnp.take(v, cperm)
            else:
                def cg(v):
                    return v
            newborn = (a_b, Rr_full, Lr_full, cg(S3c), cg(e_b), Ff_b)
            mcol.append((mb_next if tau_b > 0.0 else mp_next) if cperm is not None else n)
            if rfb > 0:
                r1b = pool.get("r1t", (L - 1, n, rfb))
                r2b = pool.get("r2t", (L - 1, n, rfb))
                if cperm is not None:
                    fnp.take(R1T_b, cperm, axis=0, out=r1b[k_b])
                    fnp.take(R2T_b, cperm, axis=0, out=r2b[k_b])
                else:
                    fnp.copyto(r1b[k_b], R1T_b)
                    fnp.copyto(r2b[k_b], R2T_b)
                R1T_st = r1b[:k_b + 1]
                R2T_st = r2b[:k_b + 1]
            w2b_list.append(cg(w2))
            # V21: hub-column Gram weights of this source's legs (X1 = 3A, Y1 ~ A d(w2),
            # M ~ P d(s) + 3 A d(e)): A-type 9 + w2^2 + 9 e^2, P-type 1 + s^2
            if FOLD2:
                _q = fnp.stack([w2, e_b, S3c], axis=1)
                _dd = fnp.multiply(_q, _q, out=_q) @ CDD + C0DD
                dA_list.append(cg(_dd[:, 0]))
                dP_list.append(cg(_dd[:, 1]))
            else:
                dA_list.append(cg(9.0 + w2 * w2 + 9.0 * e_b * e_b))
                dP_list.append(cg(1.0 + S3c * S3c))
            c1_list.append(cg(c1_b))
            c2_list.append(cg(dgw))
            y_list.append(y_b)
            if regen:
                # post-ReLU kappa4 diagonal core (r=1 matrix-core harmonic projection)
                k22row = K22 @ ones_n
                g_prev = ((K4v + k22row) * float(st["cA"])
                          + (fnp.sum(K4v) + fnp.sum(k22row)) * float(st["cI"]))
                var_prev = K2v
                lam_prev = float(LAM[min(li, len(LAM) - 1)])
                if lt_on:
                    fd["g_post"] = g_prev
                    fd["k22row"] = k22row
            K4_sigma = (fnp.sum(K4v) * float(st["k4_c4"])
                        + fnp.sum(K22) * float(st["k4_c22"])) * float(st["P2"])
            # per-neuron K4 diagonal content (mean equals K4_sigma by construction;
            # K22 is symmetric so K22 @ ones == its column sums)
            if riders:
                K4_vec = (K4v * float(st["k4_c4"])
                          + (K22 @ ones_n) * float(st["k4_c22"])) * float(n * st["P2"])
            if lt_on:
                if lt is not None and lt.get("v2", False):
                    lt_u, lt_prev = self._lt2_layer(lt, li, fd, W, NN("ww") if lt_prev is not None else None,
                                                    lt_prev, n, False)
                else:
                    lt_u, lt_prev = self._lt_layer(lt, li, fd, W, NN("ww") if lt_prev is not None else None,
                                                   lt_prev, lt_u, n, None, lt_dump)
            rows.append(mu)
            alive_prev = alive_now
            alive_b_prev = alive_b
            perm_b_prev = perm_b_next
            if fis >= 1:
                dperm_prev = perm_next
                mp = mp_next
                dKp = perm_next[:mp] if perm_next is not None else None

        if fis >= 1:
            DEAD_STAT[getattr(self, "_ncall", 0)] = stat
        _out = fnp.stack(rows, axis=0)
        return _out

    def _lt2_layer(self, lt, li, fd, W, WW, st, n, last):
        if "fast" in lt:
            return self._lt2_layer_fast(lt, li, fd, W, WW, st, n, last)
        return self._lt2_layer_ref(lt, li, fd, W, WW, st, n, last)

    def _lt2_layer_fast(self, lt, li, fd, W, WW, st, n, last):
        """c2 21.10.2026: the same LT2 layer in fewer fnp calls (on the grader every call and every
        slice is a round trip). The state is carried as the two right-hand sides T1 = [prev8|zo|h]
        (for W) and T2 = [prev8|zo|h?|u^2] (for WW), so the next layer needs 3 products instead of
        up to 9; x is one column_stack in a permuted column order; the sign-log runs once on the
        whole x under a column mask; standardisation and b1 are folded into W1 at setup (a ones
        column carries b1)."""
        f32 = fnp.float32
        fa = lt["fast"]
        zero = [None]

        def col(k):
            v = fd.get(k)
            if v is None:
                if zero[0] is None:
                    zero[0] = fnp.zeros(n, dtype=f32)
                v = zero[0]
            return v

        own = [col(k) for k in fa["names"]]
        ones = self.__dict__.get("_lt2_ones")
        if ones is None or ones.shape[0] != n:
            ones = fnp.ones(n, dtype=f32)
            self._lt2_ones = ones
        if st is None:
            zb = self.__dict__.get("_lt2_zb")
            if zb is None or zb.shape[0] != n:
                zb = fnp.zeros((n, fa["nrest"]), dtype=f32)
                self._lt2_zb = zb
            xr = fnp.column_stack(own + [zb, ones])
            PWu = None
        else:
            M = W @ st["T1"]
            MM = WW @ st["T2"]
            Wu = W @ st["u"]
            PWu = fd["Phi"] * Wu
            xr = fnp.column_stack(own + [M, MM] + ([Wu, PWu] if lt["uin"] else []) + [ones])
        xs = fnp.where(fa["mask"], fnp.copysign(fnp.log1p(fnp.abs(xr) * 1e3), xr), xr)
        if fa["fold"]:
            x = xs
        else:
            x = (xs - fa["mL" if st is not None else "mL0"][li]) * fa["iL" if st is not None else "iL0"][li]
        cdf = flops.stats.norm.cdf
        h = x @ fa["W1L"][li]
        h = fnp.maximum(h, 0.0) if lt["relu"] else h * cdf(h).astype(f32)
        for Wm, bm in lt["mids"]:
            h = h @ Wm + bm
            h = fnp.maximum(h, 0.0) if lt["relu"] else h * cdf(h).astype(f32)
        g = h @ lt["woL"][li] + lt["boL"][li]
        u = g if PWu is None else PWu + g
        if last:
            return u, None
        p8 = [col(k) for k in LT_XF]
        mid = []
        if lt["xall"]:
            mid.append(x[:, :lt["Fo"]])
        hs = fnp.tanh(h @ lt["WhL"][li] + lt["bhL"][li]) if lt["D"] else None
        T1 = fnp.column_stack(p8 + mid + ([hs] if hs is not None else []))
        if lt["uin"] or (lt["D"] and lt["embw"]):
            T2 = fnp.column_stack(p8 + mid + ([hs] if hs is not None and not lt["embw"] else [])
                                  + ([u * u] if lt["uin"] else []))
        else:
            T2 = T1
        return u, {"u": u, "T1": T1, "T2": T2}

    def _lt2_layer_ref(self, lt, li, fd, W, WW, st, n, last):
        """LT2: inputs [tf(own) | tf(W prev8, WW prev8) | W zo, WW zo (xall) | W h, WW h (emb) |
        W u, Phi W u, WW u^2 (uin)] -> MLP -> local error g (and the next emb state h); carried
        correction u = Phi * (W u_prev) + g. st: previous layer's state in this layer's unit order."""
        f32 = fnp.float32
        zero = [None]

        def col(k):
            v = fd.get(k)
            if v is None:
                if zero[0] is None:
                    zero[0] = fnp.zeros(n, dtype=f32)
                v = zero[0]
            return v

        Fo = lt["Fo"]; D = lt["D"]
        names = LT_FE + (LT_FE2 if Fo > len(LT_FE) else [])
        X0 = fnp.stack([col(k) for k in names], axis=1)
        Z0 = (fnp.sign(X0) * fnp.log1p(fnp.abs(X0) * 1e3) - lt["mzoL"][li]) * lt["izoL"][li]
        if st is None:
            Xv = fnp.zeros((n, 2 * len(LT_XF)), dtype=f32)
        else:
            Xv = fnp.concatenate([W @ st["prev8"], WW @ st["prev8"]], axis=1)
        parts = [Z0, (fnp.sign(Xv) * fnp.log1p(fnp.abs(Xv) * 1e3) - lt["mzvL"][li]) * lt["izvL"][li]]
        if lt["xall"]:
            parts.append(fnp.zeros((n, 2 * Fo), dtype=f32) if st is None
                         else (fnp.concatenate([W @ st["zo"], WW @ st["zo"]], axis=1) - lt["m2L"][li]) * lt["i2L"][li])
        if D:
            if st is None:
                parts.append(fnp.zeros((n, D if lt["embw"] else 2 * D), dtype=f32))
            elif lt["embw"]:
                parts.append(W @ st["h"])
            else:
                parts.append(fnp.concatenate([W @ st["h"], WW @ st["h"]], axis=1))
        Wu = None if st is None else W @ st["u"]
        if lt["uin"]:
            parts.append(fnp.zeros((n, 3), dtype=f32) if st is None
                         else fnp.stack([Wu, fd["Phi"] * Wu, WW @ (st["u"] * st["u"])], axis=1))
        x = fnp.concatenate(parts, axis=1)
        cdf = flops.stats.norm.cdf
        h = x @ lt["W1L"][li] + lt["b1L"][li]
        h = fnp.maximum(h, 0.0) if lt["relu"] else h * cdf(h).astype(f32)
        for Wm, bm in lt["mids"]:
            h = h @ Wm + bm
            h = fnp.maximum(h, 0.0) if lt["relu"] else h * cdf(h).astype(f32)
        g = h @ lt["woL"][li] + lt["boL"][li]
        u = g if st is None else fd["Phi"] * Wu + g
        if last:
            return u, None
        nst = {"u": u, "prev8": fnp.stack([col(k) for k in LT_XF], axis=1)}
        if lt["xall"]:
            nst["zo"] = Z0
        if D:
            nst["h"] = fnp.tanh(h @ lt["WhL"][li] + lt["bhL"][li])
        return u, nst

    def _lt_layer(self, lt, li, fd, W, WW, prev, u, n, inv_perm, dump):
        """LT: this layer's corrector inputs X (n, LT_F) = [own columns | W prev | (W*W) prev],
        local error g = MLP(X), carried correction u = Phi * (W u_prev) + g. Returns (u, the
        columns the next layer carries in). Both live in this layer's unit order."""
        f32 = fnp.float32
        zeros = None
        cols = []
        for k in LT_FE:
            v = fd.get(k)
            if v is None:
                if zeros is None:
                    zeros = fnp.zeros(n, dtype=f32)
                v = zeros
            cols.append(v)
        X0 = fnp.stack(cols, axis=1)
        if prev is None:
            X = fnp.concatenate([X0, fnp.zeros((n, 2 * len(LT_XF)), dtype=f32)], axis=1)
        else:
            X = fnp.concatenate([X0, W @ prev, WW @ prev], axis=1)
        nxt = []
        for k in LT_XF:
            v = fd.get(k)
            if v is None:
                if zeros is None:
                    zeros = fnp.zeros(n, dtype=f32)
                v = zeros
            nxt.append(v)
        nxt = fnp.stack(nxt, axis=1)
        if dump:
            ext = []
            for k in LT_FE2:
                v = fd.get(k)
                if v is None:
                    if zeros is None:
                        zeros = fnp.zeros(n, dtype=f32)
                    v = zeros
                ext.append(v)
            X = fnp.concatenate([X, fnp.stack(ext, axis=1)], axis=1)
            LT_DUMP.append((li, fnp.take(X, inv_perm, axis=0) if inv_perm is not None else fnp.copy(X),
                            fnp.take(fd["Phi"], inv_perm) if inv_perm is not None else fnp.copy(fd["Phi"])))
        if lt is None:
            return None, nxt
        cdf = flops.stats.norm.cdf
        Z = fnp.sign(X) * fnp.log1p(fnp.abs(X) * 1e3)
        h = Z @ lt["W1L"][li] + lt["b1L"][li]
        h = h * cdf(h).astype(f32)
        h = h @ lt["W2"] + lt["b2"]
        h = h * cdf(h).astype(f32)
        h = h @ lt["W3"] + lt["b3"]
        h = h * cdf(h).astype(f32)
        g = h @ lt["w4L"][li] + lt["b4L"][li]
        u = g if u is None else fd["Phi"] * (W @ u) + g
        return u, nxt

    # ------------------------------------------------------------------
    def _sym_product(self, X, Y, n, out, lev, mi=None):
        """V29: out = X Y for a SYMMETRIC result (X = W C, Y = w32 = W^T): the three block
        products (11, 12, 22) of the 2x2 partition as ONE Strassen family (0.75 of the
        dense count before the recursion); the 21 block is the copy of 12^T. Odd n (smoke
        shapes) falls back to the plain product."""
        h = n // 2
        if n % 2 or n < 4:
            return fnp.matmul(X, Y, out=out)
        pool = self._pool
        X3 = pool.get("sym_x", (3, 1, h, n))
        Y3 = pool.get("sym_y", (3, 1, n, h))
        O3 = pool.get("sym_o", (3, 1, h, h))
        if mi is not None and mi < n:
            X3 = X3[..., :mi]
            Y3 = Y3[:, :, :mi, :]
        else:
            mi = n
        fnp.copyto(X3[0, 0], X[:h]); fnp.copyto(X3[1, 0], X[:h]); fnp.copyto(X3[2, 0], X[h:])
        fnp.copyto(Y3[0, 0], Y[:, :h]); fnp.copyto(Y3[1, 0], Y[:, h:]); fnp.copyto(Y3[2, 0], Y[:, h:])
        self._smm.mm(X3, Y3, O3, self._smm.level(h, mi, h, lev))
        fnp.copyto(out[:h, :h], O3[0, 0]); fnp.copyto(out[:h, h:], O3[1, 0])
        fnp.copyto(out[h:, h:], O3[2, 0]); fnp.copyto(out[h:, :h], O3[1, 0].T)
        return out

    def _hub2(self, bufs, apb4, k0, k, n):
        """V26: sum over slots k0..k-1 of LA_k A_k^T + LP_k P_k^T as ONE Strassen family
        (LA/LP live interleaved in bufs["lap"], A/P in the legs, so the pair axis folds
        into the contraction batch; V28: on the one-time 4-D views)."""
        out = bufs["hub"]
        self._smm.hub(bufs["lap4"][2 * k0:2 * k], apb4[2 * k0:2 * k], out[None],
                      self._smm.level(n, n, n, min(STRASSEN_HUB, self._s_hub)))
        return out

    # ------------------------------------------------------------------
    def _dslices(self, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                 c1_list, c2_list, y_list, n, bufs, rres, rfb, Zf_st, R1T_st, R2T_st,
                 need_d21=True, ka=0, FAo=None, FPo=None, Qc=None,
                 kb=0, FA2=None, FP2=None, U2=None, apb=None, apb4=None,
                 sb1=None, sb2=None, s_sb=0, dz=None):
        """(3,) and (2,1) dslices with the M leg expressed through A, P and the thin
        residual leg: M = P*diag(s) + 3*A*diag(e) + Z L^T (the factor 3 of the
        residual is folded into Z at birth). Two dense contractions (right factors
        A and P) + O(n^2 r) thin terms.
          D21 = 2H + T2 + T3/3 + 2T4/3,  H=(A*P*w2)A^T, T2=(A*A*w2)P^T,
                T3=(P*P)M^T, T4=(M*P)P^T
          D3  = 3 sum_j w2_j A_ij^2 P_ij + sum_j M_ij P_ij^2
        Every (k,n,n) intermediate is written with out= into pooled buffers
        (F53: fresh result buffers dominate residual time).
        need_d21=False (V19, final layer): D3 only -- no left factors, no dense
        contractions, no thin right-factor terms; returns D21 = None.
        """
        k = len(w2b_list)
        W2B = fnp.stack(w2b_list, axis=0)[:, None, :]
        Sb = fnp.stack(s_list, axis=0)[:, None, :]
        Eb = fnp.stack(e_list, axis=0)[:, None, :]
        cz = dz is not None and len(dz) > 6 and bool(dz[6])
        mo_ = dz[7] if (cz and len(dz) > 7) else None   # e24hot: filas no calientes del nivel viejo (None = todas)
        k1 = 1 if (AUD_ZS and rfb > 0) else 0
        bufs0 = bufs
        if cz:
            # (f) compacta: toda la rebanada en las filas conservadas de esta capa (ml_) y las columnas [:mca]
            ml_, inv_ = dz[1], dz[2]
            mca = max(dz[5]) if dz[5] else n
            if len(dz) > 8 and dz[8] is not None:
                # e28 oldc: las viejas (slots [0:2ka]) ya estan compactas en dz[8]; solo se recogen las jovenes y se ponen a cero
                # las columnas entre el ancho de cada nivel y mca
                APc = dz[8][:2 * k, :, :ml_, :mca]
                fnp.take(apb4[2 * ka:2 * k][..., :mca], dz[0], axis=2, out=APc[2 * ka:2 * k])
                if ka > kb:
                    mc1_ = max(dz[5][kb:ka])
                    if mc1_ < mca:
                        fnp.copyto(APc[2 * kb:2 * ka][..., mc1_:], 0.0)
                if kb > 0:
                    mc2_ = max(dz[5][:kb])
                    if mc2_ < mca:
                        fnp.copyto(APc[:2 * kb][..., mc2_:], 0.0)
            else:
                APc = fnp.take(apb4[:2 * k][..., :mca], dz[0], axis=2, out=dz[4][:2 * k, :, :ml_, :mca])
            A_st = APc[0::2, 0]
            P_st = APc[1::2, 0]
            Z_st = fnp.take(Z_st, dz[0], axis=1)
            L_st = L_st[:, :mca]
            if Zf_st is not None and k > k1:
                Zf_st = fnp.take(Zf_st[k1:k], dz[0], axis=1)
            if R1T_st is not None:
                R1T_st = R1T_st[k1:k][:, :mca]
                R2T_st = R2T_st[k1:k][:, :mca]
            W2B = W2B[..., :mca]
            Sb = Sb[..., :mca]
            Eb = Eb[..., :mca]
            bufs = dict(bufs0)
            for nm in ("ap", "pp", "t", "mp", "xt", "yt", "u"):
                bufs[nm] = bufs0[nm][:, :ml_, :mca]
            bufs["lap"] = bufs0["lap"][:, :, :ml_, :mca]
            bufs["lap4"] = bufs0["lap4"][:, :, :ml_, :mca]
            bufs["ppl"] = bufs0["ppl"][:, :ml_]
            bufs["t1"] = bufs0["t1"][:ml_, :ml_]
            if rfb > 0:
                bufs["gyr"] = bufs0["gyr"][:, :ml_]
                bufs["gxr"] = bufs0["gxr"][:, :ml_]
                bufs["d21fb"] = bufs0["d21fb"][:ml_, :ml_]
        AP = fnp.multiply(A_st, P_st, out=bufs["ap"][:k])
        PP = fnp.multiply(P_st, P_st, out=bufs["pp"][:k])
        T = bufs["t"][:k]
        # M*P = PP*s + 3 AP*e + (Z L^T)*P
        MP = fnp.matmul(Z_st, fnp.swapaxes(L_st, -1, -2), out=bufs["mp"][:k])   # V27: BLAS path
        fnp.multiply(MP, P_st, out=MP)
        fnp.multiply(PP, Sb, out=T)
        fnp.add(MP, T, out=MP)
        fnp.multiply(AP, Eb * 3.0, out=T)
        fnp.add(MP, T, out=MP)
        if need_d21:
            # leftA = 2 AP*w2 + PP*e   (right factor A)
            LA = fnp.multiply(AP, W2B * 2.0, out=bufs["lap"][:k, 0])
            fnp.multiply(PP, Eb, out=T)
            fnp.add(LA, T, out=LA)
            # leftP = A*A*w2 + PP*(s/3) + (2/3) M*P   (right factor P)
            LP = fnp.multiply(A_st, A_st, out=bufs["lap"][:k, 1])
            fnp.multiply(LP, W2B, out=LP)
            fnp.multiply(PP, Sb * (1.0 / 3.0), out=T)
            fnp.add(LP, T, out=LP)
            fnp.multiply(MP, 2.0 / 3.0, out=T)
            fnp.add(LP, T, out=LP)
        fnp.multiply(P_st, W2B, out=T)
        D3 = (fnp.einsum("kij,kij,kij->i", A_st, A_st, T) * 3.0
              + fnp.einsum("kij,kij->i", MP, P_st))
        if rfb > 0 and k > k1:
            # V18 (F69): B1 pair with X1 = 3A + Xt, Y1 = A*w2 + Yt; Xt = F1 R1, Yt = F2 R2
            # (F1/F2 = transported thin Z columns, R1T/R2T static (k, n, rfb)).
            # AUD_ZS: las rebanadas [k1:k] (la 0, de la capa 0, tiene Xt = Yt = 0 y no entra)
            if not cz:
                Zf_st = Zf_st[k1:k]
                R1T_st = R1T_st[k1:k]
                R2T_st = R2T_st[k1:k]
            F1 = Zf_st[:, :, :rfb]
            F2 = Zf_st[:, :, rfb:]
            Xt = fnp.einsum("kiq,kjq->kij", F1, R1T_st, out=bufs["xt"][k1:k])
            Yt = fnp.einsum("kiq,kjq->kij", F2, R2T_st, out=bufs["yt"][k1:k])
            if mo_ is not None and mo_ < ml_ and ka > k1:
                fnp.copyto(Xt[:ka - k1, mo_:], 0.0)
                fnp.copyto(Yt[:ka - k1, mo_:], 0.0)
            U = bufs["u"][k1:k]
            T_ = T[k1:]
            A_ = A_st[k1:]
            P_ = P_st[k1:]
            W2_ = W2B[k1:]
            AP_ = AP[k1:]
            fnp.multiply(A_, Yt, out=T_)
            fnp.multiply(Xt, A_, out=U)
            fnp.multiply(U, W2_ * (1.0 / 3.0), out=U)
            fnp.add(T_, U, out=T_)
            fnp.multiply(Xt, Yt, out=U)
            fnp.multiply(U, 1.0 / 3.0, out=U)
            fnp.add(T_, U, out=T_)
            D3 = D3 + fnp.einsum("kij,kij->i", P_, T_) * 3.0
            if need_d21:
                LP_ = LP[k1:]
                LA_ = LA[k1:]
                fnp.add(LP_, T_, out=LP_)
                fnp.multiply(Xt, P_, out=T_)
                fnp.multiply(T_, W2_ * (1.0 / 3.0), out=T_)
                fnp.add(LA_, T_, out=LA_)
                fnp.multiply(P_, Yt, out=U)
                fnp.add(LA_, U, out=LA_)
                fnp.multiply(Xt, P_, out=T_)
                fnp.multiply(T_, 1.0 / 3.0, out=T_)
                fnp.add(T_, AP_, out=T_)                 # GY
                GYR = fnp.matmul(T_, R2T_st, out=bufs["gyr"][k1:k])
                if cz:
                    Xd = fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=bufs0["xt"][k1:k, :ml_, :ml_])
                    D21_fb = fnp.sum(Xd, axis=0, out=bufs["d21fb"])
                else:
                    fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=Xt)
                    D21_fb = fnp.sum(Xt, axis=0, out=bufs["d21fb"])
                fnp.multiply(AP_, W2_, out=T_)
                fnp.add(T_, U, out=T_)
                fnp.multiply(T_, 1.0 / 3.0, out=T_)
                GXR = fnp.matmul(T_, R1T_st, out=bufs["gxr"][k1:k])
                if cz:
                    Xd = fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=bufs0["xt"][k1:k, :ml_, :ml_])
                    fnp.sum(Xd, axis=0, out=bufs["t1"])
                else:
                    fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=Xt)
                    fnp.sum(Xt, axis=0, out=bufs["t1"])
                fnp.add(D21_fb, bufs["t1"], out=D21_fb)
            else:
                D21_fb = None
        else:
            D21_fb = None
        # F68 feed pair (X3 = A*c1 + P*c2 column scalings, Y3 = y 1^T):
        #   D21 += (1/3) R y^T + (2/3) (y (.) X3) P^T,  D3 += y * R,
        #   R = rowsum(X3*P) = (A*P) c1 + (P*P) c2; y is the TRANSPORTED Y3 row vector
        #   (last Z column).   MP/T are free scratch now.
        C1 = fnp.stack(c1_list, axis=0)                 # (k, n) static column scalings
        C2 = fnp.stack(c2_list, axis=0)
        if cz:
            C1 = C1[:, :mca]
            C2 = C2[:, :mca]
        Yk = Z_st[:, :, rres + 1]                        # (k, n) transported y = P(l) y
        R = fnp.einsum("kij,kj->ki", AP, C1) + fnp.einsum("kij,kj->ki", PP, C2)
        D3 = D3 + fnp.einsum("ki,ki->i", R, Yk)
        if cz:
            # (f) D3 de las filas conservadas; las demas a cero (la mascara no las lee)
            d3p = self._pool.get("d3p", (n + 1,))
            fnp.copyto(d3p[:ml_], D3)
            fnp.copyto(d3p[ml_:ml_ + 1], 0.0)
            D3 = fnp.take(d3p[:ml_ + 1], inv_, axis=0)
        if not need_d21:
            return D3, None
        fnp.multiply(P_st, C2[:, None, :], out=MP)
        fnp.multiply(A_st, C1[:, None, :], out=T)
        fnp.add(T, MP, out=T)
        fnp.multiply(T, (Yk * (2.0 / 3.0))[:, :, None], out=T)
        fnp.add(LP, T, out=LP)
        if dz is not None:
            # (c) muertas/impl: solo las filas/columnas conservadas de esta capa (las demas se anulan en la mascara)
            Kl, ml, inv, g1, g2, mcol = dz[:6]
            mc2 = max(mcol[:kb]) if (mcol and kb > 0) else n
            mc1 = max(mcol[kb:ka]) if (mcol and ka > kb) else n
            mch = max(mcol[ka:k]) if (mcol and ka < k) else n
            smm = self._smm
            if cz:
                lapc = bufs["lap4"][:2 * k]
            else:
                lapc = fnp.take(bufs["lap4"][:2 * k], Kl, axis=2, out=g1[:2 * k, :, :ml, :])
            hubc = self._pool.get("hubc", (n, n))
            hv = hubc[:ml, :ml]
            mo2 = ml if (mo_ is None or not cz) else mo_
            inner = None
            if ka > kb:
                r1 = sb1.shape[2]
                innerc = self._pool.get("inner", (1, n, r1))[:, :mo2]
                _sbon(smm, 5); smm.hub(lapc[2 * kb:2 * ka][:, :, :mo2, :mc1], sb1[..., :mc1], innerc, smm.level(mo2, mc1, r1, s_sb)); _sboff(smm)
                inner = innerc[0]
            if kb > 0:
                r2_ = sb2.shape[2]
                inner2c = self._pool.get("inner2", (1, n, r2_))[:, :mo2]
                _sbon(smm, 6); smm.hub(lapc[:2 * kb][:, :, :mo2, :mc2], sb2[..., :mc2], inner2c, smm.level(mo2, mc2, r2_, s_sb)); _sboff(smm)
                if E28_UEYE:
                    lift = self._pool.get("lift", (n, U2.shape[0]))[:mo2]   # columnas r2_.. = 0 desde la creacion del fondo (solo este sitio lo escribe)
                    fnp.copyto(lift[:, :r2_], inner2c[0])
                else:
                    lift = (smm.mm2(inner2c[0], U2.T, self._pool.get("lift", (n, U2.shape[0]))[:mo2]) if _g(smm, "lift")
                            else fnp.matmul(inner2c[0], U2.T, out=self._pool.get("lift", (n, U2.shape[0]))[:mo2]))
                inner = lift if inner is None else fnp.add(inner, lift, out=inner)
            if ka < k:
                if cz:
                    apc = APc[2 * ka:2 * k][..., :mch]
                else:
                    apc = fnp.take(apb4[2 * ka:2 * k][..., :mch], Kl, axis=2, out=g2[:2 * (k - ka), :, :ml, :mch])
                smm.smin = DEAD_SMIN
                smm.hub(lapc[2 * ka:2 * k][..., :mch], apc, hv[None],
                        smm.level(ml, mch, ml, min(STRASSEN_HUB, self._s_hub)))
                smm.smin = STRASSEN_MIN
                if inner is not None and RPROJ > 0:
                    fnp.add(hv, _rproj_right(inner, Qc, self._ora_S, RPROJ, Kl), out=hv)
                elif inner is not None:
                    Qcc = fnp.take(Qc, Kl, axis=0)
                    hvo = hv[:mo2]
                    fnp.add(hvo, (smm.mm2(inner, Qcc.T, bufs["t1"][:mo2, :ml]) if _g(smm, "inner")
                                  else fnp.matmul(inner, Qcc.T, out=bufs["t1"][:mo2, :ml])), out=hvo)
            elif RPROJ > 0:
                fnp.copyto(hv, _rproj_right(inner, Qc, self._ora_S, RPROJ, Kl))
            else:
                Qcc = fnp.take(Qc, Kl, axis=0)
                smm.mm2(inner, Qcc.T, hv[:mo2]) if _g(smm, "inner") else fnp.matmul(inner, Qcc.T, out=hv[:mo2])
                if mo2 < ml:
                    fnp.copyto(hv[mo2:], 0.0)
            if cz:
                # (f) todos los sumandos de D21 en filas/columnas conservadas, y una sola devolucion al orden natural
                fnp.matmul(R.T, Yk, out=bufs["t1"])
                fnp.multiply(bufs["t1"], 1.0 / 3.0, out=bufs["t1"])
                fnp.add(hv, bufs["t1"], out=hv)
                PPL = fnp.matmul(PP, L_st, out=bufs["ppl"][:k])
                t1 = fnp.einsum("kiq,kcq->ic", PPL, Z_st)
                fnp.multiply(t1, 1.0 / 3.0, out=t1)
                fnp.add(hv, t1, out=hv)
                if D21_fb is not None:
                    fnp.add(hv, D21_fb, out=hv)
                fnp.copyto(hubc[ml, :ml + 1], 0.0)
                fnp.copyto(hubc[:ml + 1, ml], 0.0)
                D21 = fnp.take(fnp.take(hubc[:ml + 1, :ml + 1], inv, axis=0), inv, axis=1, out=bufs0["d21"])
                return D3, _zero_diag(D21)
            fnp.copyto(hubc[ml, :ml + 1], 0.0)
            fnp.copyto(hubc[:ml + 1, ml], 0.0)
            D21 = fnp.take(fnp.take(hubc[:ml + 1, :ml + 1], inv, axis=0), inv, axis=1, out=bufs["hub"])
            fnp.matmul(R.T, Yk, out=bufs["t1"])
            fnp.multiply(bufs["t1"], 1.0 / 3.0, out=bufs["t1"])
            fnp.add(D21, bufs["t1"], out=D21)
        elif ka > 0:
            # V21: old sources through the shared basis: [sum LA FAo^T + LP FPo^T] Qc^T
            # V24: tier-2 sources through U: [sum LA FA2^T + LP FP2^T] U2^T lifted first (U2 = sub-basis; U is the scratch buffer)
            # V28: each tier's contraction sum_k LA_k FA_k^T + LP_k FP_k^T is ONE hub
            # family over the interleaved slabs (rectangular right operand (2m, 1, r, n));
            # level 0 = the dense batched-matmul leaf + k-sum (same billing as the einsum)
            inner = None
            smm = self._smm
            if ka > kb:
                r1 = sb1.shape[2]
                inner = self._pool.get("inner", (1, n, r1))
                _sbon(smm, 7); smm.hub(bufs["lap4"][2 * kb:2 * ka], sb1, inner, smm.level(n, n, r1, s_sb)); _sboff(smm)
                inner = inner[0]
            if kb > 0:
                r2_ = sb2.shape[2]
                inner2 = self._pool.get("inner2", (1, n, r2_))
                _sbon(smm, 8); smm.hub(bufs["lap4"][:2 * kb], sb2, inner2, smm.level(n, n, r2_, s_sb)); _sboff(smm)
                if E28_UEYE:
                    lift = self._pool.get("lift", (n, U2.shape[0]))
                    fnp.copyto(lift[:, :r2_], inner2[0])
                else:
                    lift = (smm.mm2(inner2[0], U2.T, self._pool.get("lift", (n, U2.shape[0]))) if _g(smm, "lift")
                            else fnp.matmul(inner2[0], U2.T, out=self._pool.get("lift", (n, U2.shape[0]))))
                inner = lift if inner is None else fnp.add(inner, lift, out=inner)
            # V27: D21 accumulates in one pooled buffer (a + b == b + a exactly, so the
            # hub-first order is bit-identical to V26's D21 + hub)
            if ka < k and STRASSEN_HUB > 0:
                D21 = self._hub2(bufs, apb4, ka, k, n)
                fnp.add(D21, (_rproj_right(inner, Qc, self._ora_S, RPROJ) if RPROJ > 0 else
                              smm.mm2(inner, Qc.T, bufs["t1"]) if _g(smm, "inner") else fnp.matmul(inner, Qc.T, out=bufs["t1"])), out=D21)
            elif RPROJ > 0:
                D21 = _rproj_right(inner, Qc, self._ora_S, RPROJ)
                if ka < k:
                    D21 = (D21 + fnp.einsum("kij,kcj->ic", LA[ka:], A_st[ka:])
                           + fnp.einsum("kij,kcj->ic", LP[ka:], P_st[ka:]))
            else:
                D21 = smm.mm2(inner, Qc.T, bufs["d21"]) if _g(smm, "inner") else fnp.matmul(inner, Qc.T, out=bufs["d21"])
                if ka < k:
                    D21 = (D21 + fnp.einsum("kij,kcj->ic", LA[ka:], A_st[ka:])
                           + fnp.einsum("kij,kcj->ic", LP[ka:], P_st[ka:]))
            fnp.matmul(R.T, Yk, out=bufs["t1"])
            fnp.multiply(bufs["t1"], 1.0 / 3.0, out=bufs["t1"])
            fnp.add(D21, bufs["t1"], out=D21)
        else:
            if STRASSEN_HUB > 0:
                D21 = self._hub2(bufs, apb4, 0, k, n)
                fnp.matmul(R.T, Yk, out=bufs["t1"])
                fnp.multiply(bufs["t1"], 1.0 / 3.0, out=bufs["t1"])
                fnp.add(D21, bufs["t1"], out=D21)
            else:
                D21 = (fnp.einsum("kij,kcj->ic", LA, A_st)
                       + fnp.einsum("kij,kcj->ic", LP, P_st)
                       + fnp.einsum("ki,kc->ic", R, Yk) * (1.0 / 3.0))
        PPL = fnp.matmul(PP, L_st, out=bufs["ppl"][:k])   # V27: BLAS path (1.2 s -> 8 ms)
        t1 = fnp.einsum("kiq,kcq->ic", PPL, Z_st)
        fnp.multiply(t1, 1.0 / 3.0, out=t1)
        D21 = fnp.add(D21, t1, out=bufs["d21"])   # in place, or hub/dense result -> d21
        if D21_fb is not None:
            fnp.add(D21, D21_fb, out=D21)
        return D3, _zero_diag(D21)


if __name__ == "__main__":
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=1024, depth=16, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, seed=0)

COVARIANCE_COEFFICIENTS=[[1.0382541045893163e-05, 0.0007710218576036287, -0.012165568069665397, -0.011388110544040587, 1.4473075452210729, 3.8433742058955414], [-2.796561221904119e-05, 0.0012435417488319625, -0.006877637131227627, 0.006280200357623293, 1.1124452966162313, 1.308774408292828], [-0.00010564134969361276, -0.0002536298504826164, -0.009325009127789574, 0.02533833771921164, -0.0024262825753221325, 7.865466525054918]]
EXTRA_COEFFICIENTS=[[-0.005156968844468532, -0.16767011563561823, 0.06038217271452533, 0.6591303884741143, 6.166508700328775, 0.11266456714655175, 3.8468961780552458, -4.180300125589212], [-0.029200778384418716, -0.16847787037371462, 0.17385834428334637, 1.9063195997451166, 5.706385238550148, 0.3013650320033185, 3.4634844319983005, 10.318251075433114], [0.0036498330250020307, -0.3120718214444178, 0.10502265412967432, 1.2613295695556428, 0.42702999725120117, 0.25867015626427836, 7.44454299403288, 8.969591487527225]]
MIXED_COEFFICIENTS=[[-0.004974211698730539, 0.16223504027019156, 2.369041328436166, -0.03597434673853842, 0.005530736287689819, -0.1762562960884645, 2.3516017068615054, -0.002961413297481591], [-0.011777187126862943, 0.28653697945917683, 1.1904024432659965, -0.031353618217330646, 0.76262784906978, -0.15297389440251605, 3.5534597905422274, -0.0012491059063203915], [-0.014827599056538267, 0.06618333943695882, -0.6707987230419777, -0.09037993602047752, 1.235201736363402, -0.3844521487678545, 5.838495233218684, -0.003122737756504137]]

_MIXED_GL12=[(0.009219682876640378, 0.023587668193255706), (0.0479413718147626, 0.05346966299765953), (0.11504866290284765, 0.08003916427167321), (0.20634102285669126, 0.10158371336153287), (0.31608425050090994, 0.1167462682691773), (0.43738329574426554, 0.12457352290670134), (0.5626167042557344, 0.12457352290670134), (0.6839157494990901, 0.1167462682691773), (0.7936589771433087, 0.10158371336153287), (0.8849513370971523, 0.08003916427167321), (0.9520586281852375, 0.05346966299765953), (0.9907803171233596, 0.023587668193255706)]
"""Source fragment: permitted-API mixed covariance fields, weighted before addition."""
def _mixed_delta(alpha,var,coff,d21,k22,k31,post,beta):
    sd=fnp.sqrt(var);mu=alpha*sd;ph=flops.stats.norm.pdf(alpha).astype(fnp.float32);g=flops.stats.norm.cdf(alpha).astype(fnp.float32)
    f=[mu*g+sd*ph,g,ph/sd,-alpha*ph/var,(alpha*alpha-1)*ph/(sd*var),-(alpha**3-3*alpha)*ph/(var*var)]
    out=fnp.zeros_like(coff)
    if d21 is not None:
        power=fnp.ones_like(coff)
        for rr in range(3):
            term=d21*f[rr+2][:,None]*f[rr+1][None,:]
            out=out+power*(term+term.T)*(beta[rr]/(2.*[1.,1.,2.][rr]))
            if rr<2:power=power*coff
        out=out+k22*f[2][:,None]*f[2][None,:]*(beta[3]/4.)
        out=out+k22*coff*f[3][:,None]*f[3][None,:]*(beta[4]/4.)
        for rr in range(2):
            term=k31*f[rr+3][:,None]*f[rr+1][None,:]
            out=out+(term+term.T)*(beta[5+rr]/6.)*(coff if rr else 1.)
    rho=fnp.clip(coff/(sd[:,None]*sd[None,:]),-.999999,.999999)
    power=rho
    gauss=power*g[:,None]*g[None,:]
    q0=ph;q1=-alpha*ph;factorial=1.
    for order in range(2,9):
        factorial*=order;power=power*rho
        if order==2:h=ph
        else:
            h=q1;q0,q1=q1,-alpha*q1-(order-2)*q0
        gauss=gauss+power*h[:,None]*h[None,:]/factorial
    gauss=gauss*sd[:,None]*sd[None,:]
    out=out+beta[7]*(post-gauss);fnp.fill_diagonal(out,0.)
    return out

QUADRATIC_COEFFICIENTS=[[8.550483365484562, -20.901728589848897, -5.4815832597549, -0.11273768559633637, -1.544679124152432, 0.5625453450297174, 10.726369496008756, -2.176067157081971, 0.7520496644348557, -15.846853208253512], [-2.1152557993180623, 36.08690054690872, -2.0442070152831024, -4.259166093984986, -1.2215573506781139, 0.9964420108260761, -6.710636175349492, -0.9808671077292589, -4.46750558090562, 12.14733309017343], [-0.7896541603489463, 11.32175854871564, -1.4574788179555167, -4.752937956917181, -1.5357830696670418, 0.11637636991028551, -3.786948156810273, -20.339001523234174, -3.056885707498587, -3.9712272842180245]]
def _quadratic_delta(alpha,var,coff,k3,d21,beta):
    sd=fnp.sqrt(var);ph=flops.stats.norm.pdf(alpha).astype(fnp.float32);g=flops.stats.norm.cdf(alpha).astype(fnp.float32);f=[None,g,ph/sd]
    for rr in range(2,7):f.append(-alpha*f[rr]/sd-(rr-2)*f[rr-1]/var)
    a=k3[:,None];b=k3[None,:];c=d21;d=d21.T
    def outer(p,q):return f[p][:,None]*f[q][None,:]
    out=coff*(a*a*outer(7,1)+b*b*outer(1,7))*(beta[0]/72.)+coff*a*b*outer(4,4)*(beta[1]/36.)
    for rr in range(2):
        off=2+4*rr
        part=(c*c*outer(rr+4,rr+2)+d*d*outer(rr+2,rr+4))*(beta[off]/8.)
        part=part+c*d*outer(rr+3,rr+3)*(beta[off+1]/4.)
        part=part+(a*c*outer(rr+5,rr+1)+b*d*outer(rr+1,rr+5))*(beta[off+2]/12.)
        part=part+(b*c*outer(rr+2,rr+4)+a*d*outer(rr+4,rr+2))*(beta[off+3]/12.)
        out=out+part*(coff if rr else 1.)
    fnp.fill_diagonal(out,0.)
    return out

# T172: exact regrouping of our frozen32-field covariance defect dictionary.
_FUSED_COV_SPECS=[{'C1': [[0, 0, 0.002961413297481591], [8, 0, -0.006082784034832698], [0, 8, -0.006082784034832698], [9, 0, 0.010063695452420888], [0, 9, 0.010063695452420888], [12, 0, 0.004694356964439656], [0, 12, 0.004694356964439656], [15, 0, 0.1187567134095078], [0, 15, 0.1187567134095078], [9, 9, -0.5806035719402471]], 'C2': [[1, 1, -0.004213348623279498], [10, 1, 0.05492753237284286], [1, 10, 0.05492753237284286], [13, 1, 0.08014367037615096], [1, 13, 0.08014367037615096]], 'C3': [[2, 2, 0.24171149308642573], [11, 2, 0.17129190834246596], [2, 11, 0.17129190834246596], [14, 2, -0.029029861983258416], [2, 14, -0.029029861983258416]], 'C4': [[3, 3, 0.16026398413304263]], 'C5': [[4, 4, 2.4678444145679924e-05]], 'C6': [[5, 5, 4.113074024279988e-06]], 'C7': [[6, 6, 5.875820034685697e-07]], 'C8': [[7, 7, 7.344775043357121e-08]], 'D0': [[1, 0, -0.0024871058493652694], [10, 0, -0.128723260346036], [1, 9, 0.04687877875247645]], 'D1': [[2, 1, 0.08111752013509578], [11, 1, 0.0626708053695713], [2, 10, -1.3205711006877927]], 'D2': [[3, 2, 0.5922603321090415]], 'K22_0': [[1, 1, -0.008993586684634606]], 'K31_0': [[2, 0, -0.029376049348077415]], 'DD_0': [[3, 1, -0.6851979074693625]], 'DX_0': [[2, 2, -0.028184421399084093]], 'K22_1': [[2, 2, 0.0013826840719224547]], 'K31_1': [[3, 1, 0.3919336178102509]], 'DD_1': [[4, 2, 1.3407961870010945]], 'DX_1': [[3, 3, -0.5440167892704928]]}, {'C1': [[0, 0, 0.0012491059063203915], [8, 0, -0.0034388185656138133], [0, 8, -0.0034388185656138133], [9, 0, 0.028976390713891062], [0, 9, 0.028976390713891062], [12, 0, 0.012556876333471604], [0, 12, 0.012556876333471604], [15, 0, -0.02937855276830642], [0, 15, -0.02937855276830642], [9, 9, 1.0024139040807978]], 'C2': [[1, 1, 0.003764653131971842], [10, 1, 0.15885996664542637], [1, 10, 0.15885996664542637], [13, 1, 0.07215592566663126], [1, 13, 0.07215592566663126]], 'C3': [[2, 2, 0.18561573375375862], [11, 2, 0.15851070107083742], [2, 11, 0.15851070107083742], [14, 2, 0.0716545213571744], [2, 14, 0.0716545213571744]], 'C4': [[3, 3, 0.05458431309163119]], 'C5': [[4, 4, 1.0409215886003262e-05]], 'C6': [[5, 5, 1.734869314333877e-06]], 'C7': [[6, 6, 2.4783847347626814e-07]], 'C8': [[7, 7, 3.097980918453352e-08]], 'D0': [[1, 0, -0.005888593563431472], [10, 0, -0.10179644588984282], [1, 9, 0.08303683423550634]], 'D1': [[2, 1, 0.14326848972958842], [11, 1, -0.372292131742135], [2, 10, 1.0122777575144524]], 'D2': [[3, 2, 0.29760061081649913]], 'K22_0': [[1, 1, -0.007838404554332662]], 'K31_0': [[2, 0, -0.02549564906708601]], 'DD_0': [[3, 1, -0.2555258769103878]], 'DX_0': [[2, 2, -1.0647915234962464]], 'K22_1': [[2, 2, 0.190656962267445]], 'K31_1': [[3, 1, 0.5922432984237046]], 'DD_1': [[4, 2, -0.8388295219186865]], 'DX_1': [[3, 3, -0.24521677693231472]]}, {'C1': [[0, 0, 0.003122737756504137], [8, 0, -0.004662504563894787], [0, 8, -0.004662504563894787], [9, 0, 0.017503775688279054], [0, 9, 0.017503775688279054], [12, 0, 0.010777923177678264], [0, 12, 0.010777923177678264], [15, 0, -0.010967418893735365], [0, 15, -0.010967418893735365], [9, 9, 0.3144932930198789]], 'C2': [[1, 1, 0.014230537737857889], [10, 1, 0.10511079746297024], [1, 10, 0.10511079746297024], [13, 1, 0.15509464570901835], [1, 13, 0.15509464570901835]], 'C3': [[2, 2, 0.0001160758635303341], [11, 2, 0.011861944368088921], [2, 11, 0.011861944368088921], [14, 2, 0.06228882977449462], [2, 14, 0.06228882977449462]], 'C4': [[3, 3, 0.32785788595047594]], 'C5': [[4, 4, 2.6022814637534475e-05]], 'C6': [[5, 5, 4.337135772922413e-06]], 'C7': [[6, 6, 6.195908247032018e-07]], 'C8': [[7, 7, 7.744885308790023e-08]], 'D0': [[1, 0, -0.007413799528269134], [10, 0, -0.12798192247225348], [1, 9, 0.009698030825857126]], 'D1': [[2, 1, 0.03309166971847941], [11, 1, -0.2547404756248823], [2, 10, -0.3309356070181687]], 'D2': [[3, 2, -0.16769968076049444]], 'K22_0': [[1, 1, -0.02259498400511938]], 'K31_0': [[2, 0, -0.06407535812797575]], 'DD_0': [[3, 1, -0.1821848522444396]], 'DX_0': [[2, 2, -1.1882344892292953]], 'K22_1': [[2, 2, 0.3088004340908505]], 'K31_1': [[3, 1, 0.9730825388697807]], 'DD_1': [[4, 2, -0.4733685196012841]], 'DX_1': [[3, 3, -5.0847503808085435]]}]
_fused_parent_setup=Estimator.setup
def _fused_cov_setup(self,ctx):
    _fused_parent_setup(self,ctx)
    self._cov_specs=[]
    for groups in _FUSED_COV_SPECS:
        current={}
        for name,items in groups.items():
            if len(items)==1:current[name]=(items[0],None,None)
            else:
                current[name]=(fnp.asarray([x[0] for x in items],dtype=fnp.int64),fnp.asarray([x[1] for x in items],dtype=fnp.int64),fnp.asarray([x[2] for x in items],dtype=fnp.float32))
        self._cov_specs.append(current)
Estimator.setup=_fused_cov_setup

def _cov_kernel(F,spec):
    left,right,coef=spec
    if right is None:
        i,j,c=left
        return (F[:,i]*c)[:,None]*F[:,j][None,:]
    a=fnp.take(F,left,axis=1)
    b=fnp.take(F,right,axis=1)*coef[None,:]
    return a@b.T

def _fused_cov_delta(self,li,mu,var,alpha,phi,Phi,coff,d3,g4,d21,k22,k31,post):
    cb=COVARIANCE_COEFFICIENTS[min(li//5,2)];ex=EXTRA_COEFFICIENTS[min(li//5,2)];mx=MIXED_COEFFICIENTS[min(li//5,2)]
    sd=fnp.sqrt(var);gm=mu*Phi+sd*phi
    ff=[None,Phi,phi/sd]
    for r in range(2,8):ff.append(-alpha*ff[r]/sd-(r-2)*ff[r-1]/var)
    gg=g4/(var*var)
    dv=((mu*mu+var)*Phi+mu*sd*phi-gm*gm)*(cb[0]+cb[1]*gg)
    dv=dv+ex[0]*d3*(ff[2]-gm*ff[3])/3.+ex[1]*g4*(ff[3]-gm*ff[4])/12.
    F=fnp.stack(ff[1:]+[gg*ff[1],d3*ff[4],d3*ff[5],d3*ff[6],g4*ff[5],g4*ff[6],g4*ff[7],d3*d3*ff[7]],axis=1)
    specs=self._cov_specs[min(li//5,2)]
    out=mx[7]*post
    c2=coff*coff
    power=coff
    for r in range(1,9):
        out=fnp.add(out,power*_cov_kernel(F,specs['C'+str(r)]),out=out)
        if r<8:power=power*coff
    if d21 is not None:
        for r in range(3):
            term=d21*_cov_kernel(F,specs['D'+str(r)])
            if r:term=term*(coff if r==1 else c2)
            out=fnp.add(out,term+term.T,out=out)
        dd=d21*d21;dx=d21*d21.T
        for r in range(2):
            for name,obj,sym in [('K22_',k22,False),('K31_',k31,True),('DD_',dd,True),('DX_',dx,False)]:
                term=obj*_cov_kernel(F,specs[name+str(r)])
                if r:term=term*coff
                if sym:term=term+term.T
                out=fnp.add(out,term,out=out)
    fnp.fill_diagonal(out,dv)
    return out,dv

_nn_parent_setup=Estimator.setup
def _nn_setup(self,ctx):
    _nn_parent_setup(self,ctx)
    from pathlib import Path
    import json
    z=json.loads((Path(ctx.submission_dir)/"nn_asset.json").read_text())
    for key in ("mean","scale","hw","hb","ow","ob"):setattr(self,"_nn_"+key,fnp.asarray(z[key],dtype=fnp.float32))
Estimator.setup=_nn_setup
