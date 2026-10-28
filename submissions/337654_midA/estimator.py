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
LT2_FAST = True   # c2 21.10.2026: _lt2_layer_fast (same values within fp32, fewer fnp calls)
# LT2 weights embedded by lab/lt2_embed.py (name -> (shape, array typecode, base64 little-endian)): the
# grader's setup cannot read a shipped data file, so the weights travel inside this file.
import array as _array
import base64 as _b64
_LT2_BLOB = {
    "W1": ((78, 128), "f", "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"),
    "b1": ((16, 128), "f", "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"),
    "usc": ((16, 3), "f", "wCI9RsAiPUYAAAAAmN31Rpjd9UZUvAtN2wgAR9sIAEfmIWxOM//oRjP/6Ea3EYBOQ5rRRkOa0UaKD1RONyK9RjcivUY+nStOJ1ixRidYsUaKuwtOzBWmRswVpkYJtvVNdKucRnSrnEaOgNdNJPyURiT8lEbSwr9NSm6ORkpujkYDaa1N+SiJRvkoiUYUfZ5NimKERopihEbB+ZJNwSCARsEggEaI64hNYet5RmHreUaLQYBNPudpRj7naUa8+3NN"),
    "Wo": ((128, 17), "f", "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"),
    "bo": ((17,), "f", "SqU1vCb1LL1GsVC94QXPvHJTMb1gyyO9cPOZPLGiwju+Zx47iHUGPo39qLwZNzE+fONAvO1oSz3JIMG9YSBnvL0Vgr0="),
    "es": ((16,), "f", "UUCtOKxGBThL7v83GKMMOH5VHDjOQC04PsU4OOBLRThBJ1E4FfFbOAQQZjhF5244SYV3OI6+fzg/HYM4kxeMOA=="),
    "mz": ((16, 43), "f", "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"),
    "iz": ((16, 43), "f", "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"),
    "m2": ((16, 54), "f", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    "i2": ((16, 54), "f", "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"),
    "flags": ((7,), "q", "GwAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
    "Wm0": ((128, 128), "f", "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"),
    "bm0": ((128,), "f", "YDf0vSjavLwXFNA8ooWovU+P0b1exp699GtovNbBsTxlb+s8MUzIuxblyDweYk08h8mWvQKTsr0/Rsk9ta2AN7m1Br3Ov7W97h+NvYAq8rydyZG9BWnEvfqhDD2L/DE9n4vWPCrZoD1VALI8flD7vYMwz7xTmJe864vbvdowHr1WKtK98Y+/veiUr72TKCc93rtNPavRAb5xFiS9u+IDvi4FW73LTMG8q3McPZVblDtkdjO9RXYPvEu8Dz39m6I9mraGPRfWPL2TyRG9UFXAPBszpr2Aoai8wRLyO3h1sDycXDU93GWNPMVdCb3hakG9PwppvAm+Lb2PEwg8qLAIvpolub0oIBO9s157vbZFKDzV7mq8UMxKvdqyA72BRdW9/SJWvX1pFL1Jhxo9dbw+PbTqTD1pDZO9qGmDvabp0L0bFx69jl4FvaZlFTwlSeo8M6mhvJAVor1ptpq9TkfbvYduDL2B6PG9PeL3vdfApL0g2QQ9OakDvpNgV72MbQi9yQeXPB+3KT1V9s8650QyvQK99rynirm9K1/FvbZMH7vtQ1W7w+B0vVftzr2T+Ja91jqlOFCkyTzp1NK95RunvOQS9r3hutG9fWlOvX3zML18pZU9rUVCvTdBEr54oII9XKlgPMH0nb3hEYC93clzPbBMajwfMRk8t7Oxu12jjLw="),
    "Wm1": ((128, 128), "f", "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"),
    "bm1": ((128,), "f", "dJHBvNFG27zD2oM9a0VhPdCQtzw/v4u9skcIvJ9SYLu3UL88mRnKPJbQib3wbi69D7i7vOWY/zwpE9E6oTkwvYignjnlm2y9Q1hcvYsTGj24hpY9Wkj9PM7jpb07SDG94/HlPIwHLD2t1ms7XjkUvXR12rsvZRU9jj/CPNzvaj24HdK8eb1LvU9MDT2F22s9+3cXPd7KBT2h16k9Ie9vvUfivr3ZXoy9lKK6u2dO5rsL53c86z9MPWNO0DviyPW8EP1kvbG7ZTz/0S29TkWNPMv3Yr1hxDW9ylY/vRVrFr2abx48Xn+pvVCGkr2ktsG8KRzKvfQQbry/h1O9VpEAvXFgFzxmmsa9cY0cPb2Z1zuyh5a9gTt/vPwJI73x99m8ybonvWgmrLw9+Bs9L4jLuszL2zqXc8S8VF2AvahSprzsZF29XR3rvI6mDrwmYTQ8wT3vPIG2azySjYI8fc6Avf3U0Tv2txg9HATbvSchJ7wfLzk9BcwRvTdTQbzBmrm8+P9qPEXDl72t1hM9KJmXvbG+IL3ARVk8oqa1u5wF7DwP99s8oAhfPY/knbwtx1O8Z2i/vIQLxzw9umq9jXDavNnbi72l7eE8c6aivTP4PLtQZY49QzE8OyLzlTn1L4e8/EUWvTCnPLu0Wq69HHgLPbGbor21uOw8L+2LPDlKDT0="),
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
    lt["fast"] = {"names": ((LT_FE + LT_FE2 + LT_FE3)[:Fo] if Fo > len(LT_FE) else LT_FE), "nrest": fin - Fo,
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
        names = ((LT_FE + LT_FE2 + LT_FE3)[:Fo] if Fo > len(LT_FE) else LT_FE)
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
