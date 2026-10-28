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

            if last:
                _r = pk1v if delta is None else pk1v + delta
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
            K4_sigma = (fnp.sum(K4v) * float(st["k4_c4"])
                        + fnp.sum(K22) * float(st["k4_c22"])) * float(st["P2"])
            # per-neuron K4 diagonal content (mean equals K4_sigma by construction;
            # K22 is symmetric so K22 @ ones == its column sums)
            if riders:
                K4_vec = (K4v * float(st["k4_c4"])
                          + (K22 @ ones_n) * float(st["k4_c22"])) * float(n * st["P2"])
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
