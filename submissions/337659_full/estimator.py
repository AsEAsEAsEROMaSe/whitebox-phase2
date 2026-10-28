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
    "W1": ((118, 128), "f", "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"),
    "b1": ((16, 128), "f", "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"),
    "usc": ((16, 3), "f", "/iU9Rv4lPUYAAAAAM/j1RjP49UYfwQtNUhcAR1IXAEcCVWxO5sfoRubH6EaoLoBOMKvRRjCr0UbtqlNOaVy9RmlcvUb3uCtO6q6xRuqusUaVEQxO5ZCmRuWQpkavpvZNOM2cRjjNnEZ1wNhN+gqVRvoKlUaBFcBNe2iORntojkaQi61NInWJRiJ1iUYmcJ5NR6yERkeshEYonZNNA/N/RgPzf0Y6hIlNgTx5RoE8eUYG5n9NKyZpRismaUbBpnJN"),
    "Wo": ((128, 33), "f", "i+JpPZFB2zwWvJO9eaYjPaZqpDxpBR2+LFCxOx3NtD2s/rc8c3cdvdaJhr1kyny9cll3PRD8Aj1zHr49RQOfPVBtdT3cgFW9ZnrbPckx5DwwwUo9IK5AvfqXMr0Jez6+/PCKuwuVCj2mkR6+gbIvvSsXibx6two+1oSlvSDaBT1WPxg9o9SzvDhJ4Lzi1OM8TM3tPG82ej2Lnh09EhuKPXcjrD1i9Yu9tDKIPQYvsD3REFk8o2MVu2fFUDxiRcm9XrEGvAg/c71ytqw9Bbg5vMwWyruzDs49cjUuPBWj2D2A5QE+Y3J8PZ38tr0ZBr493kuMvShn4DzGcwa9JqWgvVCWETy1z3C70ghRvMQhkz32Qwg97b6yvDe3VD14W+W9nUUNvvAhPL1ZJ0I8tCaoPVDMFr55CbE8qytHPVBt0D1w5fc8hWRMvfYsob0P7pM8FpdavUBi6DtDMaY9Wxk0PdTda7025Pa9j4LSPS/1nj3qa1A9Fe27vcJFGL4dg2w9qmSivA7X8r0bCkK9r/8XPcedurkhiig9eRmnPbEnzruWqnO9mKt/vdNSbD1wK7c9X0hjPbHxorzynSq9K+HfPZlqEL5kMgM81s8UvX5eGrx/ZXE8TXZcPAEsO7wAiIk9hFKhvaxo3b1nGsi97bDGvfQ3gz2QprO88DEnvXv+Sbz6mB4+9sAcvW+OCr5Ja9a8M63LPUMftb2JHUK+SRbEu/E6xD06oHK9Cl9MPvY8Wz6HNhG9TU/ZvVt34T0Y7Qi99d15O7u0HzwWxuG8PaG4PAVSOz3eJBG+3S4qvfG8rjyo5wy+fBgivKd0MLwqWEu+69gmPM8rnb1ADzi+hZSPvBtPIj0vAnK9AyjIva5+mD3erR29/oypPTM5PbwN9vM8/6uuPb4J1D12IEA9QP4PPJWd1D3zru88lWN9PLHOrT3Z8+g8FINUvf8+Nr1Z6Gs9XJ0KPlraPzs1lVA9IRRQPERvgby1FGk9Cbe2POxaWL3FE229lDZ7PeCDBD33MJA9j2OBvc5hv721sRc9j9lmvc3dgz011j8+MMp1Pe/vdz3ELQQ9ytMLvpBcOz0be449eTrVu+GLvLqXhTE9523BvTqAjj0RBuy9++P/PGCOVj2KD609QRiXPoCOpb1SHq68TGSqPeMkn70eB2W8cygMvFQyqL3URZK9Cq4dvU9l1j3V/B29MT/QvO1u0z0EcM09twx6PUsAIL06dvm9oQBUPZ5Fp7oHXes6ELVDvQOsTr2fDTE9wQ2hvc+ZID0F54K8KS4/PYRxpr2YUyg9qdIAPJeHsj3obFs9VghePe+pozybeTq9v6RQPbfNnr1MJXa9Fi/cPUiCJT26lCk9NeqBPfnc5rwVbXq9cYu8PdeffT2ik/681gdPvbGmgT0KAIG82qSzvZoWgjtO2aS8v6pbPaaRhT3jGQe+I1TsPEvQzryOKuu9qcYxPYRMA75YE6k9jWgRvITMxD0o+yw8IHm0PVSOObsNWj+9phOVvRb7fD3O/0S9cvWFvXsQyD1Ji5e9vM3LPT0XMj0ldhe98WZ+vRJiwL1LGqo82lirvRR2Ur2i1ye8sM/GPOcZsj05Dii9dpQ7vvgkfj5pN529sKLYPS9tP70PmJ+8rxWFPWq7P7wLS2M9XLbPvd8HgD52gci8EFsrPix0Hb4iy/O8uuEpPKudl71SkhQ9Z3WJPoYKLz4xxBU9toGovGT7bz072K09XaMIPeACDL7521A+tswTPQTKvT0aMNm81bcQvaT0cL7OipU9JJr0OrdCxrxJJaw8IMZ/u9lkgr32HNG89m4JPVpcKj7TSPw9oD6ivUjXNr1o1Eo8JZQxvmgtbD3WoKA9D2QPPtMQvjzkmqo9F639vQl90z0Ah2Q9cosYvRC0RD0yez49wdWYPX0+sz0vkRO+HfALPZcypT3XGOE97IGbvb58e713gtk85ce/PekiybwPypG9pv72vE3p7TtbX3q9yJNeuzLxED2NKNy8emdAPWa+l73xh/e7zm8yvcs1nj3v82E9R8vZPEinG72wMaW9+ys/O56V2DyiZ449Y3MBvv4wZr1WoAi+raJ9Pfztob23Ew28VMp5PLiDDD0QBY48n7Q/vT1scTzS4Ae9PaDbPPWFobzxw9q9xbWoPcsk/bym7uG8XaiyPSLywb2AqkW9vjwbvf6Ddr2rqNI9sbb6uwrIIT5mqBA8qXsRvUpapD23NVc9thPZvZy5dLwIcNq8MGgXPaLES718PIS+UGOOvBP0fL0uF5y9HgYsva8JOLzH1M275mOWvZbUpT3T8nM9ZyvLPYYjw7zbVdA9MKBQvYsFtDtAPgc9f7WWupIPuD3S9Nk9AXfEvBBRHb4y0qk9pv2sPNnrJTsSXnm9Y2u+uztsID7IkxK8IdX+vNIuQ7xKpgE+InCGPZ+Krr2hqc093atuvgNsgz2VoPS9P34/Pb0q1r0HPzc9kpjJPBEmm71a1HM9nFmaPc71Iz1QDmI92E1hu0RKLD2Iat49+2rVvOuOBD3xD1e9aypnvRGku7wGjyq9MhNUvbxebbxajaM9M4i0vc0NiDvrFWy8Ms6luaZBPj626Ia7R3B6vBMdlD1mViw+4e7ivU8NNL7dC08+rmNDPXNgEj0eaq49BXZQPHXdED2bO+Y9HiDuvaO9+TzmDnu9U8A9PQDsDL4AWBK9SSRTPXJcND2Wg4a9GhQVvE1mkL0srBQ+tleqPUp2pL20+ou8Dw+aPZdCOL0Ljwk+r9GSvDHGkTxyjy49Hyu9PZQcrT17zHg8cX2NPaHWPj3GkMk89URIPJw1rD3cpgI96WRLOJ9Pcr5ZEeQ8NAbOOzKieL08Tsq95L0uvXvogr0z2hS+twtNvB3LlL307Fk9cSMDO0+Vnr3huQ08FFokvhpmkLs25+m9xK+zPKh3CjyyxiM9SrExvvIOCT6D7zi8wZ6APJ7DizzOFHC+wV7yPemcir1EUxe+c8vAvbwgPj5lfGc89iQavYG4Pb0tK/48EO16vFTOpD2aku09sE+iPLBk6L2No0U961rxPEfHLDsuwh09E64NPaf+oD2bCxI+ngzPPKQhgjyi5Ee9vuuuvHx0pz2jPjE9mS8suk8VrL1DQr+7dd4WPRRxj7wZFJk99JiyvbJyD722Lq091H7oPVBDFLyGu2K7NNgEPYY3DD3gYnE7hfmpve+tMzzVbV+852eHvZHkTDwMRd+8QsmqPTyFOj0XYIK9HJQpvX30DLtnNS89sffovfvrfb0JJfe82u6GPKKGo71FJV+9OOdyPXcRqD0jQkS9VBefu7QRrTwErhe9XWlfPRbDKr0+yw093u2pPRNA7TjBOm89wHpzvKD8xT0/ZGe8eREGPZ70xb3hhEI8zUhJveT+W73jl7y9kilRvf6l1L2Hhak9Ry0uPTVa+7x7jaw9iuyCPbWAiDu+yBo9hMEcvSBXgT0p+ky9XI0mPa2nWT10FP69bsLAPE7B+j1XpoK9rV2evEAL0bw18aS7/MpAvTelZT27rFY9n4fhvDa8oj22omY9ZcJ8PTYKkr2xZ5I8t0IePcIrULxEEJ29kvFUvGoWSD20r3w97W0vvP0x671ciYe8wilJPmHLkz0Jv8S9l34bvbdbmz1EgHy9lBsdPJ2KWL00Dtk9HL2LvX2+QDwtZg49TR2hvfBoID6ziGm8SbcpPf32cb2MwnM9TPLLvQgdKTx0AN48wWRcPYwtIz3y3/a8M1G3O1cyDb6r4C6+IvqnPQaaNb4mFNo965H0PN9jJz2DnDs+La0PvZkoKb4plQW+QRCfPT4F0by8O7y8h9Z7PSo0pj1Tcpg+vrIjPmN9Iz0jnwc+4U1/vZ1aP77f6By+sTTfPaqHHz6sH5099RbjO71LubzlR5K9r+yqvVyUQ70cxNG9QEmnPYVvtb2fpwW9we4LvXgO7buJZBM9THq7PHWTwbuRG+S6IjPOvdJdDD3yjmI9or1bPfaCCL2Fg1C9V3ChPQLoq7yHD6e9YDqZvUg8hDzxr688Me2tPcAdgb0dlcE8wjUDvSpO3DwuiX27OKAJPE0URD2btQa9n30pPc2Ui72vIqc9oKMAPROAXz1RegI+b1owO7fyrb1xVTg9rmCDvN4zNDz1WkA9KlCXvJhqWL3WEiw+e8qdvXkTnD3S6F88EQhBPeuUPT1fh368/DS9PSTRfryD3Rk+IlmAvI3Lpb3x6lS9SeGFvDgNiD2B8Jw9gyEvvPusqT64e8Q9WPDNvZbfzj2f+F495brqvctWuzzn1fo8LgiavXZdkL1vX9E8wIkQvsFnZj3QN8g7LUVIPg7FIbx+ezg+RSrOvchIfDv3zW69SVyruzvr0r0/LLg97mkxvf40BD4CvJE+YN6DvdvFwTxDNXy7NiyUvKmy9L0MW6g9kf1KPSZqBz4u1y099vZ7PW21B7xLoJ09vibnuxmRlj1KTho+KtLmvVfWvzx3wmo8ZaeZvWEMtj2A2/m8tvfCPbWHm73EhME9c+KXuwtGWL0Nj9a9bSmRO95xdr306hs9QyO6vLgUFz27lUI+QLNXvcSNzT0Pe4M9Y7n3O6ZZST1mkgM+tU27vE8zAD4MXee8Iw2IO1/72j03S5E9wUKiPW7R9j3Afs28WuhtPbMMzL3Ip+g9t5/KO3wffb1NscA8Uy2vPEJUD7zdR/U8mCECPV7sqj2Ewci8rr2EPdqRsb3U1Ya98UaOPL5swb1pSus7NufFvRT1rLxWLjE+4WqNvdgMkr2FtqC9HkslvCp+jz1sDqC88XEZPaaEuD23WJ49OPUDvaikZD0xBgy8EPKAvBN9Fjxd/J89h2gcvVuGK73OViI95K3HvYlJ4r2tEa07Ym4VPVZLSr2DFYq9zr48OJ3FaT3CnyA8eBXNPTgJgj244u09VqYgvYqTr735mZS97sG2PQwfLL1KnVG88e0fOrk+Kz5zE7K9AM6wPNuW9zyIWSM+lvxGvYgvV73mNQI+KmCKu6+bAT3t+TU+xMhVvXD2Jb74eUg9jD+pvM84az2uXMg9vJ+AvYVHAL3R2Jw90Ez3vR03iTzcab893AD4PPIDQb0tsoQ+s051vknI3T2ARwQ+dgFbvCpqVb4es4g9L5uVPCp7vzuZnxm93k63PQrcur1o3ty8W2m5OzC5eDq1LnG9y3GRvXnSKj1ssCc8bj8zPFtp7D1R5YG9keYFvRKAGL2iAm09MaQkPbAO+DyRxaW870l/PIwI1j0at9q7zkvAu+4K+j0u/+g7Ue6SPS3H570JnP29g8syvcT6dz0HB4s93EAZvrbJkT1kvhy9piCyOmuVNz2yPre9PsSmvZzkITzpYoq9vsnpvDNb1rxb0jM9nMsKPtcjHzyoJAA83U9sPRl4WDtM21e97hxOvTczXj1+JZA9w28APipqNbzfHhE+etsPPvTNo7yqCCO9RLwfvvpWLT0xagQ9cZPHPBGTmb0sPFw97oqwPdziHLzE1pM82bNCPYAZ1j2jhhg8o/cpPUrhKj2VoKW9OWUlvfKhTT1iM5q9P0u8vXrDnz2V+pY91s2WPeJa4b2j7Lo9YjXDve6kiL2kVxc9GeCuvMXdjb1HS4w9m21NvRWd8rzFT/g8qDhkPdiI7TyIpbW8z1e4PB13WD1JA0g9pyzbPFlfu7w/WVO8F91jPZFymD0/rxm+8OytPe6u6ryJ+A695s+7PFs/cLyx+cG9lziyvAdAoj01Y2e9vZYIO3Lg1T0R/7C8tw4uPW/MzbzzpYQ9e5hBvQnKFr2Yo+i9vhhhvQ1kED0UvEe9ITwwPafBZLv1szu9yTDcvC0hXz2Hfde8vnEHvYmADT2VHKs9xVF7PQhVQb3v8DC95K+3vCDv4zzUPKi9hLC7PaytqLw/vvQ9FPFhPReKCL6f2Iw9HZ/TvR+Liz1kuzs9iwI4PSd7ijwTIRc9rzWzPDDUdzxpFp08UNIBPd8o773OfjQ9KOjsvdDQCTxUdZk9u5TmPDfUxzwgxw28J7McvXN6kD2tleS9Ey3nPa5ItT3pwtg9MqwhPqR3Tz5dA4c8B3aLvVhpOzzfpjM+GAGkO43mk7yN6mu9xR71vVVRsjt3lzW+C3knvrsOCz6KrT2+v19AvLUxQ70mM7a8CbKBvf+U8b02btq9PErEvXeU7D099Wi9gflPvkusOjxFZBG+TN3bPen/nbwb7309pImzPRwsk70qhY89z1uMPSiHQTt54uE9BaOivfJJHT4XS6i9XVbMvU5Mr71bhdi9ypQaPj7C2zxgbVe9vY6BOiIjsb3STrM7xgQGvUT7gL2KT0G8BeeZvefS1T0bTJ29n4scPtmHcD3uYpw9DTz3vRqchD1KVoq4QvYbPQD70T3NKUK8Fi+fvX5/b7zz1M69K+iGvARu7L1TpDC8IEj5vXofmj3R+wq9/HbGvUxmjb2mcNI8FXK+PYnV4TxP6YU8Mr89PF8qRD3j7hk+/cloumPE6LzZD6E9Uz4fvfnA/Dyzy7W9B3DOPf7Kgj2eOpu9dhZkvTKhmL3fTeM8TVh+veDJmTzrbjo9mHgGPQPDxT3yfG28NfQBPVX2rzwLTOk9oYbJvDTvlj15NTW9JDi/PVyLLL01ZRM+ScBpPcF2kLu0PrW9ld94PS2Uwj1otPg82uj4PQGPNb6vlXc92WwQOoxtPT0Xp/Y9GxVivhXvyrx6niA9wr4HPm7nEb1hE7k99kgovOx18DyV4cI7kyj2u/h2OL19sce9yYJJvRmQlb2eRXg9SlHVPDWusL0UlwE9DtUzvVdO5jxieoS9KIXGvX+9j7wKo2M9LfM4vDcVIL2sUta8UQf1Paz2I71ktdS8zy0APpR3xLzSrWg87IYLvfyWir1f9Se9be3/PGccMD09f9082Z4KPYW3DT2IvzC7abkYPFg1Ez5678q9n2w8vWqFpT2z4YQ8ehtcvGr1yL2YWZa9+uuZvVu+Ij4Iz6e8LQ2MPQtcjr1lRIw9J928vNvmhjxhPb69J37RPXlTmj3ggSu+g8yPvIrkyD3E4yi9wW+iPGQpLr5FbKU9nxbTvM+Usj33hca8O0ZXvZRBTz5hTbo9TKewvSW0Gj5YUbc9oouwvaiRnD00xSi9rEAZPeFH3D0jq4c79GVFvXq2wT2ZLRA9UOK8PXx8Mz0DlS68L6UDPY6TjD1QudU9BIQnPUHjEr7ppfE8zswmvXLj17zNiLc92fiZvV9uWT7sg04+2oNOvbMoVz2SCKw9zeRjve5uRjymMHK8NLcfvWVGbj0nu748gjY4vIxvbzw/9OE7WQoMPVFKYzzy4Dg8EZJ2PfC+Ur3V9349qFOiPSFTTbsXHeu9JLWMPWKraz0wFGA9O4/IvZONz723oqi9dMl5va8G5jwC21s9spn+vSCyAL1rexE9/cQUvZHOB70kD3m9G6QkvSlTsTu2iC69WehWPYhwab0A/8K8cXJ3PDGLPL1An+k9SpvUPc/ZMb1IefA9GJjmPMLmBb0E4Ek7PFH4vb3DeL3I8iq8fonwPaZtab06/oI9s8BlvRKN3T0fEoC9k9N/PaFUpb3Tveo9y5COvbUCHz3g6aY99PECPTu4Zz1PC5E9FOnvOvNKmj2wZo+8pjlEva8y3j0o0Aw8u+NKvcJMEj2BBoe9ZDl/PYvaqb30QmW9LepZPeeSwDwycro84S3WvYRITz0E2W08cjKVvUlf0zx58Js9xO42PUKFLr0pFBM9FSR4vXV5QD1zspM9szXtvSTasL3zWy69zSjaPYCjrb1zbA+9L0MPPLY3yDx6O/C9tZG4PJDzNL7HoWi+2tDrOov7Oz0HvK+9KrdvPSvSTb5DMU09/xuBPp9JE7zTm0s8nDXTvRtxbT3e4gy+pcSPPUx/xLsFIpS9adDbvJGGzb13eTW+IKoEvSmkqjpfVrS+1r3+PbU0VL5niu48SYLePMW+Ur38g1e969kpPMVkmLvXZ1g9EcyVO6ILUTtDd/o8TZOIvWOj7z3xT5M9C7srPek1kr1CHr+8gT20PE67GD0aed08j9J+vY4n1z0diQM+G9S5PAldHb16QEi93rGOvdClTT1fWr889+yTPX+YhjtxkPg5NlIpPMZuVr1qY6M9KSr6PLDjSb1fP6W9bcbWuz5yubwznxA8Z582PYcgE7xvDS89XZTVvTA0Ab0RTyg+gNcPvSxj8z3RH4Y9TiLxvSOIFb2Xkgc+iXQzPiZeijqWPUo9wvVfvcSuoj0yGga9V+udPWeJYztkbhM+RvcfvTXoKDzd3gA+ycouvtWffz2jHw66tH7zPZu9JLxh5fs9DoG6vMVikzwIhk66kgowPQlqsT02WAI9GtievNyxiD394pW8iW9iu7Yklb3PgK89J1ASPaq/ljytDYY9H2P8PR8LMLxD6lQ8ewz7uyney7zO6Ds9o9xuvWbBnD1Dagy9wErUPTxz+720MbC7Jn2mPEZgsbv1EDG8hO48PYGtEb4rdqO8ANruPKU9zzwg4qs9+wvXvQKQIL0PeJE92GfWPVo97Lzo4io99rbJPMX13rxfboO9dU+6vcKzgr16W6e9FuMDPjuycryw0Be9jkaGvLrAjDsW3+O8gQQvvacvbzy9FIo9b+Lavb+K+zuA0Kw9CxhoPQqRk7zIcxq9upwePaYLjj3FYiu8rCGrPZiHsz3HZtK8yUWqvLr3b70mL6I97UyLvNYW/bz2rE48a9zTvf64f70nq+U8lqOgvUoVHj01Z088pep/u5DztDx+OHA9iMKHvZRVpT1//Z+9tV0yPRiOlzvUCHm9C46YvTcDrT3LPbS8EF0sPdauy7yfR/G7QAtUvMbOWb1bcgI+dX37PEZ0szwmKoM8Q3DRvDDRPrxI3y0824NoPKzOn70mNSk9vOV6vbi8Gb3xf1u8d7SZvah0oj3Dw0G9qB9nPl1s1r2ulPE95zzPvWAnkD3iS/I8mEB7PbhyqzzKxDo9KayaO7ikoj19nEA9j8d+uyVXgT1f6yo6zEzZPGp8kTxKs+89VbG1vO+sjzwrNKe9zyMfPFqVmryaSyA9CkzBO76xH7yvJdO92djGO8xmIz24Hig9Ezu7Pf/8X72ERTk9lDWKPZbjcbxf91W99FfoPKKiXbxvn8q8A6mFPDunQzsXxio99QYAvb3BBDwiPby93Qs9vVmTYj0VdM07lp7EPdfcVr16m/Y89j/rPL9HBD5Ch3+89O7DPWVZd7yWxy49JodPPbrK4T3GVZQ9BdILPSyuNT4T67K9xG+sPaFVw7pFwQk9ziZNPr2BJT7X+KW9sRMMPRFqoTsiJY48PiKIvb87Bb6UZv+9nxzDvWWFXr0Nic69PNj3OtjtIT54Dc894+UJviRLCr4mcL89hz4XO4TDNr76cra8AANEPauKSb5Mx5w9106BunNYzb15SRY9D4PBOytmAD452Zg9PqkbPTvuzb1kMri95wXvPECgoL3WRwq9/w/EvJyZiztYNZq7paoHPcs/oL1UWMw9r95AvKEenz0K3HY91hfzPcPhLz7JLxi+9FYFvl2kaz0j+jM9VGp6PTQvPj0ZEso9Wm/5vUe+Cz4Njxy8j08PPRjBiT13las7EUAVO/3V7T3JOg6+mpmjvI/mdj3Yu/a8SVFevdMPjD3vQ3M9wompPGJh/Ds47Pc9qfYTvTneQz3jX9y9c/SgvbH8MT2Trvu9OCb+PFv9qDvm+7Q9G+1gvcPHFz1vjPw8sap4vR6SWzxBW1I9TVSVPSRGsr0pJ7y8VyKrPaFXATyeN4e9V4ebPaofi72qfIi9Kc4NvWleVj3lqYq9T6l6ve9cib0dPhG+dsKhvczlQb33JZ89XHcKPlcwvLyrGSy8qX2UPfbekb1RBxM9/ChnvUV6xLqyfqO95ZSAPWO0mzv6QB49zSpdvZjztz1Rw9a7eRZ/vaYESbxRMGs9yGdwPTi9mz1j1K29mgiQvSxMFbxH/py9d/D1PUb8uLwSOzc9PbGGO8+pQTse4Zi9vS00PEUoTD2D2Qa93EOoPcP/IL79Bm26jlwbPTEmwzxlfMW9YbioO9MXsTyneLC7wSpTPRSpiDwsM5i9bbR+PG5y8Lwi2Z89zmFKvbsvib2TOmo9h0LZPWaUoL1Hycc8gF2zvT9TDT0rteK8+JKKvXy5jr3ylzW9iauoPbFcFztKOlQ8qUtJvXMdo7tmuaO9Isu9OW5Zxj3PmOg9R4pZO/iNlz3nPC0+QA2UPbK1KT2Imx6+HAYUvIsR8T04cBw+ntCXvD6iIz5EaCK9eDACPj06sb0xgY296cGwvZ9ZwDpjJow9qrhQvYKHmb1mJi89dWr0u6w2nb3EqEI9FfEyPSr3Nrywvh69F6h+vaTXtz0WrrS9mvgEPXE/prr+eNI8drYDPtnSTL1e6m698dGavaQW370qgfC9WcSYvW8fsb0mE368lBEpPaBQ0D2JmL497EEOvmjUpD16YtA899UuPVNaFD1js7k9XmagvRR5gD5fNya9yu1kPNuXyr30KYe91KWlPun4BL5FhA++Nt+TPYgjkDvF5089x86TPdfa6b3XVCE92PPRvY6WiTypBFs+AAaevZ+qPT4OdQO9ZnOjPcVUBb4vM409x/GZvATLXDyOUom9eVYdPdH6iz19LJu9VNKgPapzIb1rNMW9LJ2ZPYq6sT0Z+VY9ATMvPV4KAr5Anje9ZAzFvbU9vT1gVDs9eD/fvP8HRrydSyA99xLQvBDoPz2s3928rjZrvT5SCD3udIq9ufijPQh7MT2sAAY+PuXhPXNxwDtFcRs8yq61u81VGD7phrC9meBVPQZFpLnrJ8m93/eIPDLCnrxuncE9FcTwPMCjEr4P5Lo7OPU/PfoQ6z3KuSE+uTNQPviezz3W8Ks94Sq4vSuLgT20ZxO9b695vTC/17z1g5894xQxvR+ZrT3puJC9M9PiPP2rCr6XU8i8ZgoXvVDYur3FQxe9YGvrvSaNxLwahaU8ZPTYPbBfRr05Zw+9zgcPPVXBvL2wa0S9rlm2PAkr4jw9UUE9GTE3vR78jr0ml4w9TBYNveWJJz0eovC9680iPdLDFj3X3Ow996HGvZbMZL3haTo8jwiuPb9cm72dUOU8TNO8vWYzILuV9hC+KlCOveH4tL0EnhC9qKaJu6YSWL2+2nc9Hk4PPg/fxL3E6Gy9ChKSOBfg5z3ZGjg9GymPPC1fjz3U1iO7wX0lPZm8jryX+A89SGOivWCxCT4I/GA6qn3MvC3tNTwbF+e8Ekz2vV79mT363R49oo/2vBKrmjyOSZ697K9pO35Uu7tqd6k8lI+LPPrzZr2bx1K7CJHqvFh+gL36Z9U9hPXHPKH0d71FiZg9Pt8Nvhh9NbyY3rO7BRUHPoyO5jzud3K90j64vUyx1r2d2zA+iMsoPaOUhz1oXhK7KtzlPYta6zyEcBE+3Og8u24jgboQ2PQ8Yif2vJ7zpzyXD/49HC2dO2ckyb3byN082FArPoJglDsaDwM+yLM8vIehCb7ciOU8QcfHPfvvgL3RgnY9WTKtPbQtyL2a8Wc8vXOKuyaa3b1X9PE714wJulHkBr3XGuE8/gy0vBsdtr2HseI8zcvpO0nrQj3OWoe9vX6Qu1DW8r3v3KA9LXpXPT2iKz12Kxo9cJ2lvbkV4z1jjDM9E+adPRWokjzaDgI9bDCZvVP/rz0ArQI+r1Q6PXJrLj08drG9YeIFvmKHGr7CcCA8adIEPm7iDz0Gjps6k5GWPUrQiT3cZSc+2YM7PD0hvj16+xY9pNjCPaW/VzyFjWo94+i8vUMFgT07i5A9UzouPcADgT2neCS9LFs7PfsNCL4RZR89vRABvPVlKD2K6gO9uWylvJSVHb7FIkG8MSdVPY/c/b3CNWu9jjINPOUwL73ZUuw9w+M3vTzRub2UC/E9Ekr7u9bnlT3Sgfm8akgMvpZnuzwZEAM+Zzz7Pa8x0z1WpVM9EJcKvKKVez057aY9G8/jPfbg7b2XAcc8T7jRPQ1ghb1UO3q+bx5yvX9MrD22U7m9GOKEPdQL4DwMa/+8AszzPIi8wT2sOOa9v+0APfHduLxcEsu9WPW+POLTGj5s8VG9LeUPvvzH+jxyl6g8J+H9PeMP9TrJCYI9i7XUvODUJLw6z9+88aHJPCbzGD4rSbO95aQKPYPfij1x9Fa+rEYlvQDUsr0K+q49E9ylvbA/uj0vavo9p5KSvVXtbzwtWxy9oTjZvBB2Oj3ddrA9eRvJvSH5XzwNVUm8nZPVPDRIdr2TQFI9+evZvcxUlTxakPk9eyeevIieQD09ML48UlJsPVcQej1NfOU9WWLBPAG9sr1ylSY9kg6/vL4JtbyzwjI9fl3avBaPtDr7tti9KnpcPbV1o7o82+y8AtBiPEdJkz1kAPg9QVQTvjVskb3oSiG9YL8xvTRD2r2PAVe9TUyHvAudtz1qH3W9lZy3vZvIR71voJg9zafQPcnRgL22rw2+mGngPQSWb73K+gk+DGbYvQYHhz1+CuM8xZVTvRy+hT1ldFA9b1mfPTDRLj06QQw9msKOPKAM5ryJ5Vy91NH2vXkQVz3C2sq8QhJ+vLoFiT0ni4w9qoK5vQapyDzkVpI9lSkWPiz7Qr7qk+s9etdVPa8BqrzS+Sm7rn8evRQbsTxxQNo5pZUWvda6/j3WwZa9byHmvIvLED1zgsI9gzmHPtxeB7o7d8I9E0RavaYgQ70P7hS+Up+2PV+i/LtexMC9DzFTPe6zOz3l9UQ9k1KTvWhI6zwv+zu9KxmiPP4/irtLVZU8bzXiPDGChT0OIn86O4dDPbt4hjw4c7s9uXF3PsVnjj1qCYa96B2hPfGItj2uLve7yU2lPbapqb0Gyo+9TIvqPEWvETq4zOO9FPzlPNM1Bz7ka0S91nzGPf1TMD2Ycxi9uObwPIzJ1L3FygU9IiYdPZRLV73TmYC99Qn0vA8qgb1ubwA9PeCmvbG7fjxBfBA9BstTvTF1ULzKj9I8oKguvs/shLtKNAM9w2FVvB9lJD3bcDQ+dZbgvYlqwTwowZC8UZVXuw8n0j1g58i9C8cEPhJtQL3/joU93XChvZN6Hrz+Sg0+geRpvTYSJj3bgEC9a3FrPWWW57xWSsa8TmNKPRoVyL10MN+9bBkYPnEVnD2VDws9XcpyvZBWbr3AXlk8yw9oPonEHD1p3sG9nxnUPGSvnz1zgci95cIRPV1Yb72KSyY+uKYgPnkeMr0UUKg8h8nHPbe58jnUcI69cUB3vUpRj713aGQ9ezuvuyeRAD5ZcNM8cxtavaCGDL2zIIa8t6EZPGxV9D2ROrE8dAsDPigJkj0eBo091rFgPdUOjL0ZQgG9JksAPasKvT29o2i98T+sO7LXbjxQQZO9JLuhvSFI37s++Jq9+ZiWvGGonjzUyt29v60Hvc4SZr0rrc09qrgxPWQHeLqYjja9Bxp0va0FBDzlYr4923Qzvd2gBr2Sr7y7RiUivRcXGL7OYiU9CMTuPFaDUT1PtYu9VhTtvSDyQL0H1u083gkWPtJsgr0hpJg7lGskPBfPlj3Ufs08qQa3upesNr289eQ9SxatPUbutT31KAY+zaTNPELoI73KgCO+v7Z1PQ5dor109Oe82GGOPNYHWj2x9rQ7acizvQNZiL1rVxs80a2DvamQ3TxunFI9YrPXvBT7rL12aKG9VNB2O/A2mj2z+aO9ZMAEPcbgUj2sRju9PQe4vdeWtDzrIHi92pqVPW4D3L1hqE48FE2APUPgBj7vuUI9uLdhPUcqiL1QwXQ9OyqQvbDUG73mS6W6ENGbvP96gz28WA29FfD/PQ9/HL3TklI6BbXnvWyHgzxG1y+9OuWfPXlD9b2J7og814LmvTk6jb3Ldk+96x3sPBmJnb2AIQM+Z3jXvNY+Pz2DL+87EfcePQp9I708Dr49BLBZOxHepTzZnMi9wQOivalmKTmQd9a9TQJXPdr8xr2ZRz69TwcJPW3tjj2ADuM8LWqcPZDIvLxgItu7J3mrPSFyWT25EG48Vb4yvK7O7D3WYGm9CZKSvWg+iTwoOFY7kdH+PYB9MT0kL0C9hXLBPava1DzDwqU9m+AGvuZB+7ytSxc9ujVhPXBjO71RZDa+cCQbPutDhj03JRG9SZ8evTS/zLyvTPm75kYXvWffPLrwboi8tUDVvGykobxpyCG9ca6Ku0Wsbzzzia69OLahvans9zz70189vd1cPXQqUj3KGmy8rjIgPgtsULw6IwK++9utPFJOCr2omsY9cqt8PTJ7UL3cJ4C9xWzQPBXnOb0Q++g82C2YvXILDL5liji9jjtBvYM/2Dra2I69HL9LPREhjT26Ea08Rm0nvA4GL7193vq9E5GzPTS6Dj3x5NI9Vr6BvXWXGL1SDMW9rpjiPSYCHL0/9s88H/J2PcxV6z3t0TC9Z/mWvYR4Ab7CvSm9fKqgvIYx2L2fDi2+f/4qvlQoFD1dClO+66EIPrPK3bowhy8++xocPJRIP73IqJM9nSaivU8g8ruMVTS9bSA+vJU1Zb3/SiC9haSvvbjtzrwEJkA9wGwSvXZ+wj13l0A7Ye7nvH6hur3Y/mo97MaMPKLsJD7NWHw9pmnCPc0ceLzFC/O7YGAzvVrVLj3mVZ29qLBtveATaD3yq4G8w0v0PdQcCz0SaFk8/Vj4vCkC5bzw+ci9LCeAu91ejjzUrRc+yUQQvnT/dj32Txc9fWDevFz2Tr0KqAq9+TwwvI05hD1IlL69nmKMPb8oKD15sxY+BEPoPQNSxztpE669SUgLPsPR7b3TT9M9Ph2EPWCOg7x5d4C9XPluPBkvFD1qbiQ8DvI7PIVlBL5b/oQ9ZNLCPc+eLj0kRdw8jexdPW7LXj0lqxi9ZwVNvfYI5DiLLpO9GwmEPH9Zqr04GgY87rbGvURAyLxsmXw8AhfKvaa4mD0+AVI9eZfiPEQdmjyjhKY8kVsJvYsB37zWf0q9XTqGPKfE5j3ocqM8UXrAPZgIKT03naE9e+3EPfN8Eb5fJra9gMJUPNF2vL1e9jm9+o+0PfGl2Dt4we28sIgmvcLAcz25D/u8L9/DPZ5oe7xaSIO9mh+AvTxZqz2dSAq++8G6vWcMfL2fzUI9j10qPbiOcb2Uh0W9Wa9aPe/1RD2nKag94nJ7vOqrV73BJj+9/qurvZ2qz7t2Ixa+1mOXPbQOpLzBaMs6Y8HWPL1h9LvUmC48Vj+OPX2YVr09gLu939r5PSA/KL2Ledc93u5EPj8NjD0Y0x48YH4Nvu+vvT28rxU+KzbcPIuaiD0O8q69h7lsOlVEAD1mgn69LRWEvULFpz0GvUq+J85FvWfweL2J0e+97H3xPKlq1b0WQ7a9gfVMvT9ciboCIZG9tfZWvoTtvj3/H2c90kl5OxweGL2GNo08SBoEPpoObL1xXZa8qjmRvSFj3TwuJ6A9PcBbPdAGpj2PS2K9J592PYe3Or5a4p29KyfEOnW1oj1W70M90tGjvN/XnTyNPuC8MTs0vo2cbL0jOqs9hgq/PchDJT0J9nM8zyb0PGpS9z0PbbW9Ew7Duxocdb0q4rw9os8CvQvbSb6YhkI7q825PBxMm70Xps+9IxIzPXNTnb0TKZY9RnKHPQRlaL0Iqa89i3WGOg/ldb2n7dg93wgYvh/Mnr0HytC98cAyPDdx1D04h9o9G4NCvfaOBD73e0Q9KGwVPuk7b7zS7JO9Gn8DvQizCr7YdhC+NXcWPomeeT29OXs9cawnvbD2Br6Cz7i9TEmDvUAu6zxtmDS9IcRSPYVGaL2H3Gc8RLCrveO3WDx/fEM8bsAavsqz4rz4tZ09Xh7WPKgZDb7lgsu9hbnPPMKCfT1vZ7y80NJHvdeiLz4z6Sg9cmPyPbIFKL3VXVQ9dzwYvIrE8L0CIiy+iNKFvTKwgD1IaVE9cV/UPfXLIz4g7CS9FqVzvY3tB73y+9M9+578PGpgPj2zjeQ9iNzaveKtNLx7BH89hredvalvt7zwATw9wyu6vH7ZnrzULf87PrI2vXCKmj3DN969XTmxPX9bh70Luz09Jx3zvJFVzrwf4789fQi0PWLMp7xBX+Q8k6RQvYC+ZDyFi9q7OH6fPOHGmD6jiss9xVKgPc+3Gj71uxA+p3GcvVKVWT0KGac9UTflvK+Xor3OXXU9ZrXoPMLc5z1yvJK92oXpvRnAZj2TbIo+7eakva+C0r0z/eI8kaqkPZ4QQb3A+tu8quaevaUSibxL7Qs+DsXTu0rLabtPA2Y+JvWUPeszDb4dJ1k9+b0zPbIgBr4juog9OHKRPaO3V71UBdS9MGuovOUzej2dwGW9UqU8PbV/gL0NZku95vTdvKP+KL1y+MS9uP06vuPJOT6OfcE7ysEbPHg5V730j6U8RAcHviEEGLvPA0e9FservHCR5j392K+9TNviPSp0Ej3nt2W9FK4mvW4OqD1Qk109ffc1vmE8WD4vUJ88bumYPRAfrT3Z0ue8GnqWvXffcL1ju1E8fn+pPQIvrL3/mSQ+zN21PVL1yz2WGCA+j8AFvddO2b2W4M29kZBEPZq2Dz6s1Y69vvs+PSUovr0KSHc8ErpMPrYFX7v5kVY+2DoZvnFvqb1+6Qe+SX04PjFwK7oChwQ8qsq2Pbytij3PiH08FAoevdaVvz15ERS9aVujPQ008j1ObNI91O9UvZD8/jzEIKK9CCeAvDjSn7ywSAm+w6XXPWlb4jzNnyU99SahvWo227rS3B6+vRj7PLsmOjzIkRe+/k/jvBQVGr27m2Q9aaROvXnIFb5B8vI5//8OvkpB0r1fFrS8/RDtu+n3gj0qGty8UVWMPHvIIL6hw6K9eTnXvZLl97yHCha+l39Eu5TJzb1L9Fg9M9O5u2z11z0re4m9onB5vQ284Tzsiss8xiGfPct8Vz2jCzy9ruxzPQBi6L3kEYE8abZsvD72u73SrUG+BMO9PeqKuLxqaHW9ML9ZPZ0KZjyIpyC90xKovSX/cD30Rjg8u42cPD74Gr4q4wS9be9Jvg1ucL0Ky689ZXXJPfAUpL0Nss+8vCTJPRrewb0jMis+W8ZavazUgLzTKwk9FweePKRxS7wehIg9Lb/PvfxoFb4O+LW7ivQFPvAYQj0b1Jc8MsYRvW1PRzvT4pe9sPEMPqv+wbwXNTs8c9oiPV3MAz4BacI7khU+veE4sD0XEKC9NRU1PQlbC7xjroW9uxzQvMX+jb06iou9KKttvXBFTD6cqhw9qBSFPJwpsr0QZv28SwuivZ6Porxbz7098F1cPV6Iqj2asri9QCPPvD9IVLwGcgU948tMvZcPLb0QcBk9x7TvvCEABrwua468fPyFvajjZ7xAOKe9lxsMvbjeAb6Urca9ehuMPLEPSD0wo2G9lcSEPbGi0L11KO09gvbvPQFIDr6xOjs+R2dtvaEhrL2v0zS+xVwovfOGBr3CTak8D0hxPOFD5bzZkHA9dWOTPJQTcb3eiM+9VjcMvlUnpb0A8vk81Ls2PteoA73VCZo9y5ufPRX7Mbyb4Be8CGi4PeJsL740G3k9L0opvRzpkT3G0rU8p1nuPGvHjT1kNGI99E8RPSVr2LzAhv69aKtBPGMYGD6bNp08uPYdveAcUT1nhhm8WfYgvnwbBb6HarK9vRezvVfZbT35LQS+0aqrPSFPvj0lr6c7WmVEviTEgLzeqR8+5xQVPXYa9b3Z+6Y9uz23u4B6PT7EIsK9bZeaPIp2aj0sat+8nGDdveK/2D0O2im9OtwTvqK23j1iFho9eSCKvSrREL6ihMi9zK+dve2K9zyX8uQ9bqZoPcrDyT0P4jO8mle4vXhJgr3gsHG946WVOx2IdD3XwDI9um6FvK4HNj2O5F49BN0CPeHko7tDC649YCQePDrhiDxSpJY8j9eyvWgQ1T2e8Hg8HW3SPeFk/b2AOkC8VERwPlCxkT2Tr4e8MlqIvrG0Ez4vf7I9tt4+PRac/DyT2aE9fqNEvk9QoTvNODm+aWk1vdeaMD1iXd29GnaDPJ05y72vTsk9zAysvT8NyD0a/pG9KiEnPcoopbtMzQ87wqiUO8HkNbsfVqg9S66MPWoSjbyikSs80ZZcPBB6+TyToGO927o7vTLUN73iaYG8X9wuvbkwuz3Log09ymljPM6va7wJm5+9cZawO8Tr2rtuBuy8KQGVvS2lxj2E5VE9AaHtPWRcNT2RXAm80isjPT6ZaD3tQkU+y2R4vZUNqL2Y2+O7ZKgTPJZvpD0MeBy8BdsoPBcip7sL2PM8sK6SPQkSpb0fyQE+RJQOPSJlWz0isKw9eW81PLEn8r2BTiW+Ial6PTfTTLwWbME9IIq8PLE6nz33mrC8dQfkPbELCb4FloC9Qo0FvHjoT70FP/A9NUXUPTN5cj3XvKu8wVT6PGFc4z3amAE9xkMtPcqmrT2TR3g9Jt0HPe63k71YJ808aWrGveqeij3SLIc7mPaCvQ+tzT3fBYo7Dal6OsfV2jxGftE9lDd3PUe8DL27vZW9fcWAvXRe9DvMyvS9Eg1OPHi3RD33paQ9XJWSvGMQmr2efBs9SM3ovQ/1A72h5ce7DB8pPU2+m7ls/6c9JvLJvWp5dLx3eie+3LdgPA+q7j3Bk/A8lV4XvSu7Ej5moBi9x4ApvatjmD04y6y8QIT5PNTn4j0VE6W9gi46Pc4zuz2f/mw88MQ5vTiHqb3apK29pb8/vRo5UTycgpK+UTSCPfBuCb5jZZW86dE3vhyYHb2wQJA9tgeRPNTpaD3MCRM+suqfPQwjkr0FzRW8fo6XvZbglL0kaCE+SeixPTwPtD1fe2C9qcb8vWB/Br5ou8Q9Z3VHvdF/Cz51O6M9Ys4kPuPVFb65yjk+elspvu0K6r2RgVi95h9/PBKa7j3YpYk9SaamPe9OED1kWBy+koB7PaFZBr4bnbY9Itayve5duj2bF6y920G9PdfzFT7SDV29ttMLPHxaUD0mzxG+OuAhvS+dl70ZxCS8WL84vWv26z3u5lS7Qw09Pp0rRr5kM5C+Zg6Zvc6WED4y9/a95B1VPiGeRD71LOo9qZQ6Ps1rc724LpE9OUGRvS5MAb5UVqu908JoPdUZKz6uvFK99dRFPe+wZD6Z2Le9XxWyPZdNgr0Cn/q9mXTdvSWeVT0QYCs97ThrPYnKw73Tlpw9U7WRPb1dvDwykcA9d+O9O2h03z1rwqq8ffsYPadNljns/zY9yblgPbVZlL0DAxe++NedPXFBt7xto4S9qlrfvbtfLr4zmRw+ZSh5vQi10jxdVbQ8YnazvJlrkT3FD7U8QE+EPQUDAz0fKmc9TbIQvhOKKL2BDtQ9azE4PTLgRb2CRLk7Ai/WvDaG/T2V0Ao+ZFnsPVbZjT0ewvs9DgvIvHWojD1oIQG9sdlLPXLrhDyDlbe8B2lWu1qiXr3ToeQ8GQDvOqgD5DziiVw94nJFPaw/6L3dZjG8n30rPfQJHj20nRK+lqMbPZGfGjtl45K95e1IvWCXxz1S7OI8OPJjuknY8bzHBcO9PssEPdDmXT3TcjQ9NHafvcUPOr0Us9m95v/fvQeV0LvLrhm9mWq1vMiwJD19nzu+dgasvPqChLzIySu+jJyRvccEob1tujg8wgs3PBySWbyGdQy8IL03PKO6sT00wgk95QlEvABJer2qRKA9SI2MvRx7SD0fb6M9sU6XOxUbIb2FEw694MB7PQ+sXjw8a2m9Qg6wvCy2pD2SKlm9MeZmvdvxTL1I+h49y3u3PZ3WO722oaI8pI/kPLLBvzxP5Ik8JQbSPV5xeT2Hj5K7oOevvWgutrtwHHw9omghPQCCMr4Xs2M9FBCbPE9hFD7Vyga+4nKJu8Sfmb0a+rK9x6CyvRQPeL2gS6q9XQ4Xvluo67z94eM9BADkPIt2ub2uwSm80QasvQ20GjzxUuw9yoV8PS3qCz4hbVK8+lmjvCmaZzzG2TG+yMooO7tPQL1lABW9YWnTPQWSJD4GhiU957a2vGz5nTuvqwo+BEImPTnDsL04F5y9IhOFvWtFxL2yq4k9rcLIO4MPnLwv2FG9uEWVvYAQvDyos3890liHPSiQzTxjL4K9U6QuPvZo7r1JitU9Kci/vAANWj3stKM9G02nvUnZXz2GQHk9v3kSPue5gr2OHwG9JfqXvAuNL70bhba95NguvZbm+Dv6lH29les0vZulJj11m7u7BgeVvaN1Q72RV4074i4aPgpsCr4ave48yNnaPb/rRD2BfRU9hDszvggTcjwQG3i936N7PejigLy3leu8y1tRPDEO+L29b/+9my6JPKMYwTy5XE88Fl+vvaDdVr3z2m0+Lp1AOnIpib1iT2W9XT5nvU+9trvSvdU9+EOqPMrHI700ETk7TqIzPNYp+LzOBLu9yskjPe2nlzyt5kc8mGKyvcs3WDzeDU89NlOOPfv5hT0VsYE8w8ZFvXsvgL1YBhU9v2DgO3CgIT2f6Hq8rSOaPXOHxjwVu3g6HU4aPUMZhz3kfS2+g0t7PSoKBz7UPgw+P1+kvSpnHL2QDb08jMVVvLuyr70TPTe9ZvczPW3g/r2zcp699TtQvH9hvT1kDoq97nVavOKjgr2/yL48iWIOPTTkBT6dOTi99NquPertiL2ks9U9spM7vsEhq73ZV9Y9y6vAvFZe+js1JTi7GEEpPT07db2W5Pm9mC+1u2ezlD2TVwe9JHgPPVUxhD3kGaQ9RP2OvcNHvD0U7p29M+TCva/c873hI7e86J7EvX7PIL2mtv29gzvAOcdZkr1powU+kODLu/5v6D3I3RU9fK8HPl1F1L1wvuw9FSa0PLfXLL28cES9HONrPAuloT0j2iG9etaCuZ2fP7wk66I9U7fuPSzZzbzYOTq8utxNvEPv/T1IWm27wI+6veIoar0e2Bq+6cmXvbMITjzFlIO8gKDuvPqQVbpx93q90qmkPY5N97xgoYQ93GgyPh5lX73N3Va+fqeBvSu0IT5vYA294gj2PT+eQb4fFjS+LfiuPe2c1r1sXSe+kp4oPc0RDb60JB++C4rRvUATuL0Bv4q9BxY+vFLqsryc6Vg8FleSvdBZgb1J58q89+PhPeF0Nb0C7QG+l5iIPR4T0D3Dd4W85iAsPI982L3q+Kw9BwFsPoUVPTt4Mu+9gkDCvSu1jrzasru8ixjePQfBDL5VrhC9LE4FOuv4kDrt0fe8xkTrPG3lSD6LUxs+6wK7vAzD8zxWDWy8rnA7PIBmkL5541U9C4ZNvZS6zL39msM93+0FPtejoTzaXt69GdvbPchHET7WOTu9XnEOvhb30L3XMcm8GM97vcRqvD39fpS+w4cAPeJoJD1C1L48K9wQvvPnmT19d+I9MO6IPVkEob20RY+9zAwQPo5n3z1CWw69fTMbvR/4rj0dPyE92oeYuzVHVj51ZpQ9URVpPAk5HbzHIZ89IOCIvVqUpT0cj7k9ek48PHcC2b1GhJc7x51qPk8Un734MIc9SdVduyIAzD1C/cs9kJdPvISD0j2dL0C9/CUgPUBX1r0rd2y9SNGjvSIsnrxXSek9oBlFvNIKjz3mFIY+nRsRPnb/uL1HLaU84XCNPJ+K5T0Vy3O9mrQ7vO/iDj2LETI8KvFPPWEMHb1TkZ09wewWPdTpaz2s3UE9y3WjvaT82bnLIb29wPXXPVbWhr1Elrc8cJmpvWvOdj1dr3q9UQ27PRMF1bwm2qo9rmhzvViEo7sP3749uLYXvTREVL1kyKM9sUUtvN2MrD1PYkI9ix0ovdW5Sj2Rraa9GMgavTvusr12TdQ840Y1vVP8FrzOUaS8Mq6lPc/10btX84O86AXjPWcVTr77CSo7VOIEvprMJT7tHP29Oijjuw/qmr3ki4C9hAnFvRp1mzy1pN69WOIYuyDXS70P6va9Zjp5ulg5Ej0RQBY+OuKWvQDQrLyreak88Xd3PezVzr1IYDi9NbihvUUdoj2QDrC91EarvNQ5CLyBVi+8VeV+vYgg1DtVIJI8AZK2vU7leLu/jMa9ga/5vf9lXz3QoQW8EjwePCCpBz3HHvS80u6NvXTh2r0BL+y98dEZvlJ8BL1RsKK9aBeDPQjAOLzeYlk+pXG4PEv9eL3eTUw9Kj0Vu0Y1Dbwt1SQ9FG0gPc6f5T0RVPi8j1IOvqivpD1zezc9qMZKvRd6Ib7gNwi94pspPhnrAD6/vIS9Ae+QvjgdXD17nxA+5UcyvHvV17w66MU7kjSGveOQc7woc06+qFrCPfd/ET0O0jq8qwkdvq7zT77R31Q+berZPSIZbjzt0F+9kdKBvLnlGL1QXNm9RaI4PXfEmjw26Gw7ThWqvdX7Bz4rg5U9/F8GPWYL2z3NfL09uZ2xvdXO271/9pq9onlXPP7SUzw/saU6sgyPPVNT3T2ptzy9XsaYvWAw0jzZu0C8JJc7vekVvr10cI09yHrWve4CqjyTIBo9BIlnPQx4Jz0BmpU87vuxO/zAVD0MRES9NTgOPuALPL1fkQc9H2tDPby2OT0bQg4+YSygvSitIz7qYLs9e5k2PRwfAL1f7zU9CW/rPbQHOj3U3QK+oXq7vYty8Ly/rUO8ARE8PUA/WL08zXK9y+BDvfbiwL0MbcU9zkHxvdDRNrwUeJw92567PKr93L0wdaK7"),
    "bo": ((33,), "f", "Li5oPSfLkDyuCva68KVYPXvrtr08U/c8uusJPXTF2LpmlLg9mlxUvcipFT3Wusy8U+gZvTGCyb2JSe68pMvEPAJDDz5nGPi7ceYBPnWxOjtxL3K9FC7qvbF1Kr0tg9W9xnlQvL3s8DwRc/k7k6mhPAyj7Dy67oi98ZmIvSPPwb34GiY9"),
    "es": ((16,), "f", "WD2tOEI4BThl0f83gcQMON9IHDiPCy04BWs4OBK6RDg4+lA4L9tbOGcZZjjmYm44t/t2OH8GgDg+eYM4louMOA=="),
    "mz": ((16, 51), "f", "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"),
    "iz": ((16, 51), "f", "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"),
    "m2": ((16, 70), "f", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="),
    "i2": ((16, 70), "f", "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPw=="),
    "flags": ((7,), "q", "IwAAAAAAAAAAAAAAAAAAACAAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
    "Wm0": ((128, 128), "f", "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"),
    "bm0": ((128,), "f", "u9EMPMhry7xKBcK9yGp7vLpsibvmL368tt/KvPeiz70dGZ69BKFrPUACO73IRwa9uqxPPLgz1L2Glxe9qVtIPck9zbx7trm8/hMVPJh7ND3QVdM8n1BRvXSfrrwQkL69ohO9u07hGD1LkJG9p8SRPPKVDT3cise8Xvb/PNISKr2Udy28mL+pvR0ciz2Y6T09wScCPXW8Jr1ighm9XLtYO+/oAb7o5vq92UJwvaoahb3EVag9r3UOPfWmBb12xNK9xTnJvWyPKrwKTc+9RA+rvcJ62jyhu2y9ZOGuvVKrSb1/oKI9WLvVPAFxOjysN3k93RkvvVBUiruxaQK9ab6dvNEBYb21tdU8Zh3+OnuJxj0GE4O9QTajvaTzub1wVoy82q7wuhF6Nz278P69sl2lve76Vj3Hyh+9ba6UvX61z72G65E91YKjvN80yDwuz4u9Nw9bvUdDSDwAHbu9SCkTPY1bqr3HdN69032BvK6bLj07Vey6/7crvQG36Tw85qM78MQJvQb4yzzAaHS95PS+vbcE1LwtO4+9T8H/PPaCPr0VCp28VeGPva/577x3jB08lsYfPW3LrL0KWSS9VdmMvf5Iuruppje9CTYYPMpgcrwU5E89h4QPPeWZvL2xo907mztVvaer1bx7woa8d85wPU68Ub3xHI691acDvHyhMb0="),
    "Wm1": ((128, 128), "f", "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"),
    "bm1": ((128,), "f", "NJ6CPREK6TzjFQe9elK3PPVdZTvCY2a8XjYhvZYdl70+1nA9AISgvWLryzy3mXQ9YXOpPZq7Vz0/Q3m8GZNlve6+nTxIqBC9gcO6vR6L/7uQKi293GwTvfPO6Lzt6AW7zM6gvTsvd70X4EM9RmMyPL9HHb0TfGK9e+bHvJRPvr1kshy8hnRcPUnFFj1xldA7oT+yuqyRYL3lTjW9W2GLOzDOkL26gd67r/FvPRU7obs2xUU9h4xCPW2GRL1VY7M8k00jvSmjLb3L9aq9q+xUPZy98rsqPF29HKE1PNlBib2l34e9FOknvd7Qij1c0N68dmIJPekos7yMSiI9yxDnPOQxR72BZpM78tjgvIv4Bj1U8R07yJebvIRdWb00TvI8+i/xvOIhCz1MvHs8nSgXPdoTUb27O1y9TCIDPYHEJD0qTDI9sWBluxblhD3Kfj+9LW+evSE2ib3qKOY8PFhjve6OdDws5R89vw48vSUatjwM8n88OT1lPCy7qrtoNUi8AknIPNGQgb2tvzm7l7Q8PVxb27xEZUo9EnS5vJ+YLj2T/i28yvciPRxr+bvQPCC99yToPClXhr1yYbY8MhFOvT+WybsFGZW9ljgGuwmrfr3hnTU85HlYvTyFNjpx7/28kb3jONZSmrzr97C9yV+0PVSRrzwNIgy9/Ln6PE+hKb0="),
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
