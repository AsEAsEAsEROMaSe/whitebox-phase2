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
# f0 21.10.2026: ALG058 nn_asset carried residual and its local learned term (after the res_e update);
# names are cut to the weights' Fo, so older weights (Fo 35) see exactly the old columns
LT_FE3 = ["res_e", "learned"]
LT_FILE = "lt_weights.npz"
LT2_FILE = "lt2_weights.npz"
# LT2 weights embedded by lab/lt2_embed.py (name -> (shape, array typecode, base64 little-endian)): the
# grader's setup cannot read a shipped data file, so the weights travel inside this file.
import array as _array
import base64 as _b64
_LT2_BLOB = {
    "W1": ((118, 128), "f", "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"),
    "b1": ((16, 128), "f", "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"),
    "usc": ((16, 3), "f", "AFBhRgBQYUYAAAAAbMoER2zKBEe5TUZN1mL7RtZi+0bAwolOolDdRqJQ3Ub12nZOqofGRqqHxkZSVD9OFxizRhcYs0Yj9hlO5YKlRuWCpUZklfpN0zmbRtM5m0b+A9ZNQyGTRkMhk0YmPrxNpEKMRqRCjEZvHqlNdMaGRnTGhkbpsZlN3RqDRt0ag0a36I1NsgF/RrIBf0YBSYZNQ+54RkPueEZgBH5Nsx50RrMedEZ/DnJNP+xuRj/sbkaJymhN"),
    "Wo": ((128, 33), "f", "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"),
    "bo": ((33,), "f", "0f7YPBY+Gb5fetS8kzMWvUlOFrxxGr888AjgvTbYxjxmEKM9+tiQPIiSQrwuoGa9FhsVPYJrcr3ImFK8lAPlO4hmlz3JeRO8Y5UbPiD99TyLkQS925x/vMNMpjsw2ka8eblgPK1KozyYZQq8B8t2O1JOBDwKPN68OjwZvTdjM7yoopI9"),
    "es": ((16,), "f", "/m6ROKXD9jdsWQI4ew8UOJ4NJTgc9zY4/fpFOEwZUzj3tl44bZ9pOHohczjt73k4pn+AOJGigzinOoY4HSaJOA=="),
    "mz": ((16, 51), "f", "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"),
    "iz": ((16, 51), "f", "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"),
    "m2": ((16, 70), "f", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="),
    "i2": ((16, 70), "f", "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPw=="),
    "flags": ((7,), "q", "IwAAAAAAAAAAAAAAAAAAACAAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
    "Wm0": ((128, 128), "f", "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"),
    "bm0": ((128,), "f", "Ha5+uyeJzL1R3p++++BePCikgr3wZbi9MZClvXFsCr58VxS+0FZPPfZ3xL1woPu9VOtKPStpLL4vlIa+HuP1vdIqND2hYSS+m9yCvMSunrz3Mr086r6fvXziNbwP1RW+raw0vTktir36wDe+I5Q8vYtV373MJy29/uaQvdkYTTvNy0M8suIgvpOzzLxUaVQ9nNXePfrJfLtD0+S9qzyUvSuvPL4yIHS+DOaUvQaqgr5KapY953VlPbQ9Ab7oY5K9M6UnvuWsxryqnfm9NxtevSeuEb41klq9pR+2PBxy672xACw8BspqvSWIg7382+q+g5xRvuNy8L2Q4ZS94cX4vZ/jm73J8Gi8OTw2vmuPOTyZb5S+XY4JvlMbyrzeEYK9084dvWQHxjy/ZC6+rPS+vcP/0b19EOW9TftmvqvcN76UPSO6bo1Evh8iVb3RF+i9X+TqvoMGyzvmeRa+3sDjPLPWn72x50u+MSdqvXdlQ7xlOyK+dfjaPBcLTr6IVAm+CN5vviZKOr1tnI+9kmucvQK8PL115UW+Zo6ePE3QEr7kMUW93yFCvmz0qb3Hkz++sggVvYsoQb5iL8a9oOyqvW6gAr4Mb5289k0evahVtb1wdvu8O3OPvTJ8e76ztwq9ExchvtrJYr1V0p29BNBXvDymVr6JSAm+8i+Qvap2kr0="),
    "Wm1": ((128, 128), "f", "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"),
    "bm1": ((128,), "f", "UhhLPTqH/L20ikG+r4y1vCjYQLziMry9xSK7vXtb3rzDnOi8eX1hvlI0k73L1z49V4rwOyU6pr1cSRG7fBy5vAiO9LseLzq952levQMvj73Pd4y+bt8TvhhCZL668r68s93HvRGw0L3nwNS9rAGLuwZvab53LxC+4gsmvRM1l70osPi9olETPXBb170kszS+6mmBvAGrlb0mphK+RDQWOsAHEb7tkUa9TDAPuwjKPDztE5u8ia6BPV15772+GH68g/oPve/nHL6Qj1y+/rMovZOAO70gSUi+5Ezpu+HOuL6Ljpu94YkkvefVmrvgwbq9SQSDu+r8071bj0a+6MaYva+MSr2dcvq9gpvMvVZAODxfLE+99QrBvWol/L0w+eC7Updeva2yLTz5IJG9EW7DuU3sKL5Td06+0sFgveaoKDzI4Nm9bwUYvVsrPT2JwmO9nhuDvCqDrL07dwC9SkMGPGzzh7xm0o487VcVvp0Hmr0OFEi+kO6GvRRder3grI29KXYMvX9QOr41Jhm+69tEvX0Il73Fvf+83LbPvTTfGz2hyCe8w+DwPOmRazzzBaC9yU8BPU/rpL2nLT29Yd3MvfrQ0r2mfQy+g4cTPWLaEL5CzqK9gSIvvfYIMjzoyeW9MLJnO3Hr4LxrOV6+16qJuy2jkztimoa8dIEVPR/iv70="),
}


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
                if fd is not None:
                    fd["res_e"] = res_e; fd["learned"] = learned
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
        names = (LT_FE + LT_FE2 + LT_FE3)[:Fo] if Fo > len(LT_FE) else LT_FE
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
            for k in LT_FE2 + LT_FE3:
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
