# Paper changes from the 2026-06-11 audit + clean run

Derived from `AUDIT_2026-06-11.md` and the Middlebury-free regeneration
(`scripts/transfer_analysis_v5/CLEAN_RUN_2026-06-11.md`). Every from→to below is
verified against the current `.tex` and the regenerated artifacts. Nothing here is
applied yet. Priority: **P1 = wrong/contradictory, fix before sharing; P2 = should
fix; P3 = polish; ADD = new content; GATE = waiting on a running experiment.**

Reassurance first: regenerating all 15 generated tables from the clean CSVs leaves
**13/15 bit-identical** to the committed paper; the 2 that differ move one
cross-architecture-retraining reference cell by ~0.01 (bootstrap reshuffle, not
Middlebury). So the *tables* are fine — almost everything below is **prose numbers**
and the **supplement**, which lags one reframe behind the main text.

---

## A. Number corrections in main.tex (P1/P2 — mechanical, safe)

| # | loc | current | change to | why |
|---|---|---|---|---|
| A1 | **main.tex:849** | `($\dtb$ $+0.64$, $\dbt$ $+0.31$, sym $+0.43$)` | `($\dtb$ $+0.64$, $\dbt$ $+0.21$, sym $+0.37$)` | **P1.** Contradicts the paper's own `tab_dirtest`/`intervention_oos.csv` (verified +0.64/+0.21/+0.37). DRAFT_NOTES already flagged this re-sync. |
| A2 | **main.tex:166 and :448** | `within-context $\rho \approx -0.35$` / `(mean Spearman $\approx{-}0.35$)` | `$\approx -0.37$` | **P2.** −0.35 only reproduces *with* Middlebury; the clean value is −0.374 (`anticorr_by_benchmark.csv`). |
| A3 | **main.tex:679** | `the policy is right ${\approx}80\%$ of the time` | `${\approx}78\%$` | **P2.** Policy >10-gap accuracy is 0.771–0.782 (`pairwise_gap_rule.csv`); ~80% overstates. "Bracketing cross-arch retraining" stays true. |
| A4 | **main.tex:441–442** | `leave-one-benchmark-out never weakens it below $90\%$ of its value` | `…below $\sim\!90\%$` or `by more than ~10%` | **P3.** Actual LOBO low is 89.8% (0.807/0.899) — strictly just under 90%. |
| A5 | **main.tex:620–648** (weight-transfer para: `+0.47/+0.56/+0.48 vs +0.50`, `+0.29–0.50 vs +0.61`, `$\rho$ $+0.51$–$0.65$`) | re-sync to the regenerated `loao_weight_transfer.csv` | scratch fit `+0.51–0.65`→`+0.51–0.64`; **pretrained held-out rho rose** (catspp fit 0.25→0.31, rule 0.44→0.51; glunet rule 0.67→0.72) | **P2.** These are the numbers fed by the *contaminated* loao CSV (now clean). Re-grep the paragraph against the new CSV. |

## B. Supplement — it is one reframe behind main (P1/P2)

| # | loc | issue | change |
|---|---|---|---|
| B1 | **supp_main.tex:178–185** | "pretrained (fine-tuned) arm … contains only three sources, whose dtb and dbt orderings coincide; cannot discriminate" — **directly contradicts main** (n=9, precision>recall). | **P1.** Rewrite to the current n=9 state and the TT-arm adjudication (D2): KITTI precision cells are quantitatively void (spread ≤4%, rank self-corr 0.15–0.28 under 5% noise), TT-flyingthings precision +0.733 is a genuine nominal miss the mirror grid will test. Drop "fine-tuned" → "frozen" (it is pt1/fz1). |
| B2 | **supp_main.tex:120–123** | "Both pretrained GLU-Net variants exceed their oracles (fractions $>$1.3): the two distances rank training sets better than retraining does there." | **P2.** O3 fractions are 1.26/1.14 (not >1.3); and "better than retraining" is the claim main.tex retracts to **parity**. Reword to "≈ parity (fractions ≈1.1–1.4)". |
| B3 | **supp_main.tex:241–242** | "rule $\rho$ $0.49/0.50/0.52$ instead of $0.50/0.51/0.51$" — neither triple matches any current artifact. | **P2.** Re-sync the Middlebury-sensitivity numbers: current rule 0.48/0.50/0.49, policy 0.52/0.52/0.50 (`tab_predictors`, `summary.csv`). |
| B4 | **supp_main.tex:239** | "9 benchmarks, 1{,}458 models" | **P2.** 1,458 = cells, not models. Use "99 models / 891 evaluations" (matches main abstract). |
| B5 | **supp_main.tex:135** | "main-paper Fig.~5" (gap figure) | **P3.** Gap figure is **Fig. 4** (order: splats1, law2, dino3, gap4, abs5). Fix cross-ref. |
| B6 | **supp_main.tex:86** | DINO control caption "No direction carries signal and no flip exists." | **P3.** The control's own RAFT row has flip CI [+0.02,+0.16] (excludes 0). Soften to "no *aggregate* flip (p=0.43); one per-variant CI marginally excludes 0." |

## C. Dangling main→supp promises + table/generator bugs (P2)

| # | loc | issue | fix |
|---|---|---|---|
| C1 | **main.tex:957** ("… replicates on basic KuBasic assets and, compressed, with a pretrained backbone (supp.)") | the KuBasic + pretrained-2×2 replication **does not exist in supp**. | **P2.** Data exists in `decomp_2x2.csv` — add the supp table/para, **or** drop the "(supp.)" pointer. (Pretrained 2×2 also needs the 4th cell — see GATE-1.) |
| C2 | **main.tex:547** ("95% bootstrap CIs in supp.") | the Table-3 predictor bootstrap-CI table **is not in supp**. | **P2.** Add it (CIs are computable from the bootstrap stage) or drop the pointer. |
| C3 | **make_paper_tables.py ~l.460 → tab_supp_gap.tex** | "rule accuracy" columns are a mean over 4 families (incl. appearance). | **P2.** Filter `family=="motion_rule"` before the pivot. True motion_rule >10 = **0.766/0.772/0.791** (JOINT 0.791 *exceeds* cross-arch retraining 0.777) — this **strengthens** the paper and removes the false "rule below retraining in every bin" reading. Same bug in `blocks.py` p8a. |
| C4 | **tab_utility.tex / caption ~main.tex:698** | policy bolded as best in the both-held pairwise column where matched-direction (0.79) > policy (0.78); caption "beats both of its ingredients" false in JOINT. | **P3.** Fix the bold; soften caption to "best on regret everywhere; ties/loses by ≤0.01 on pairwise accuracy in JOINT." |
| C5 | **tab_law caption main.tex:406** | "the average is the best single number in 4 of 9 rows---all from-scratch or near-neutral" | **P3.** The 4th such row is **pretrained** CATs++ frozen (flip CI excludes 0). Reword "all from-scratch or near-neutral". |
| C6 | **main.tex:440** | "$p{=}0.0079$ with RAFT in the scratch group" | **P2.** Never recomputed post-Middlebury (regenerated REPORT.md has only p=0.0286). Recompute the RAFT-as-scratch C(9,5) permutation on 9 benchmarks, or drop the 0.0079. |
| C7 | repo hygiene | `tab_dirtest.tex`, `tab_perregime_ablation.tex` generated but `\input` nowhere. | **P3.** Delete from the repo or leave (don't affect the compiled PDF). |

## D. New content worth ADDING (high value — preempts the obvious attacks)

| # | where | content | evidence |
|---|---|---|---|
| D1 | **§4 robustness para or supp** | **Regime-vs-level deconfound.** At the context grain the flip follows *regime*, not transfer level: OLS coef regime **+0.81** (p=9.5e-6 clustered) vs level n.s.; partial ρ gap↔regime\|level **+0.61** vs gap↔level\|regime +0.12; within pretrained variants gap↔level is *negative* (−0.375). A level-threshold rule loses LOVO (0.449 vs 0.507). | `regime_vs_level_deconfound.py`. **Kills the "it's a dynamic-range artifact" reviewer attack** (your own REGIME doc records variant-grain ρ(flip,level)=−0.80). |
| D2 | **§7 / tab_oos caption** | **TT-arm adjudication** (replaces the bare dagger): KITTI precision cells are *quantitatively* uninterpretable (d_tb spread 2–4% of median, rank self-corr 0.15–0.28 under 5% noise), so they can't speak to the law; **but disclose TT-flyingthings precision +0.733 (p=0.031, 3.1× spread) as a genuine nominal miss** the recall-varying mirror grid will test. | `tt_arm_adjudication.py`. Honesty + preempts "your own grid contradicts you". |
| D3 | **§4 (anti-corr sentence) + supp table** | Disclose **heterogeneity**: mean −0.37 but per-benchmark −0.77 (tss) … **+0.41 (kitti2012)** — the tight-vs-broad trade-off is benchmark-dependent and *positive* on one flagship. | `anticorr_by_benchmark.csv`. A reviewer will recompute the mean and see it. |
| D4 | **§3 footnote** | 4 of 11 sources are also benchmarks; dropping the 36 train==benchmark diagonal cells moves the rule 0.507→0.498, no sign changes. | `diagonal_sensitivity.csv`. Closes an in-domain-leakage objection. |
| D5 | **abstract/§7** | Report the headline FF +0.67 with its **exact n=9 p=0.059** beside the pre-registration. | `tt_arm_adjudication` exact null. A reviewer will compute it; better to own it. |
| D6 | **§7** | The "5-seed-averaged" OOS distances claim is **verified** (paper CSV == 5-seed mean bit-exactly; +0.67 stable +0.58–0.67 across seeds). Keep the claim; **retire the "single-seed collapse to +0.317" line** in the REGIME finding doc (does not reproduce). | `seed_audit_oos_per_seed.csv`. |

## E. GATED on running experiments (track, don't write yet)

- **GATE-1 (decomp + the 9–11 / 6–10 prose).** `tab_decomp`, and main.tex:810 "$9$–$11$~\pck" and main.tex:980 "$6$–$10$~\pck wrong structure", match no artifact (twin gaps 3.7–11.8; main effects 6.2–9.7; wrong-structure cells 1.3–7.9). Regenerate `decomp_*` once the **4th pretrained 2×2 cell** (`kitti_badmotion_ft_gso_matte_pt1_fz1`, training now) lands, then re-sync both numbers. This also satisfies C1's pretrained 2×2.
- **GATE-2 (tab_supp_eps).** Source `eps_rule_table.csv` has **no producer** and is Middlebury-contaminated (quarantined). Either rewrite the ε-radius directed-coverage producer (eps@1/4/16px per variant) and regenerate, or **cut tab_supp_eps** from the supp. Until then it cannot be built clean.
- **GATE-3 (architecture generality).** The GLU-Net intervention arm (training now, ~5/11) → the dose-response architecture-generality table (paper Table 6). FlowFormer 2×2 (RC) → the 3 pre-registered predictions; exclude Middlebury at harvest and compare sources at matched epochs.
- **GATE-4 (benchsim/F5).** `benchsim_rule` was fit with Middlebury + an extra variant; F5 (`fig:abs`) builds on it. Re-run `context_scale_calibration.py` (rule/sym/policy) on the clean table before trusting F5.

## Safe-to-apply-now subset (if you want me to do it)
A1, A2, A3, A4, B4, B5, C3 (the tab_supp_gap generator fix), C5 — all mechanical, verified,
and each strictly improves correctness. A5/B1/B2/B3/C6 need a sentence of judgment.
D1–D4 are the high-value additions. GATE items wait on the runs already in flight.

---

## APPLIED 2026-06-11 ~13:18 (conservative prose pass; tables kept per request)

Backup: `ACCV_2026/_backup_pre_conservative_2026-06-11/` (main.tex, supp_main.tex, tables/).
Both documents recompile clean (main 23pp, supp 9pp). **No tables touched** — the
directional tables (tab_law, tab_dirtest, tab_oos, tab_predictors, …) are left in
place for your review; the new analyses (jackknife, target-invariance, level-vs-
selection, partial-correlation) were **not** injected.

main.tex:
- A1 ✓ l.849 TT-arm estimates `+0.31/+0.43` → `+0.21/+0.37` (now matches tab_dirtest/CSV).
- A2 ✓ anti-correlation `−0.35` → `−0.37` (Middlebury-free value) at intro l.166 + law-sec l.448.
- A3 ✓ gap accuracy `≈80%` → `≈78%` (l.679).
- A4 ✓ LOBO `below 90%` → `below ~90%` (l.442).
- Contribution 4 reframed: "Absolute prediction" → "**Ranking zero-shot, calibration
  only with anchors**" — states absolute PCK is set by a source-level appearance/realism
  effect the distances don't see; anchors give MAE≈9–10 (few-shot), absent anchors ranking only.

supp_main.tex:
- B1 ✓ TT-arm subsection rewritten: "only three sources, orderings coincide" (contradicted
  main) → n=9, distances near-constant so estimator-noise-limited, point estimates nominally
  favor the *opposite* (off-target) direction (+0.64/+0.21/+0.37), reported undiscounted, not
  treated as a test of the pretrained half. "(fine-tuned)" → "(frozen-backbone)".
- B2 ✓ ">1.3 … better than retraining" → "≈1.1–1.4 … parity, not an information gain".
- B4 ✓ "1,458 models" → "99 models, 891 evaluations".
- B5 ✓ "main-paper Fig.~5" → "Fig.~4" (verified gap is the 4th figure; recovery=7th checks out).
- B6 ✓ DINO caption "no flip exists" → "no usable signal; aggregate flip null (+0.018, p=0.43)".

NOT applied (deliberately): the abstract/contribution-1 flip framing (left for your table
review); B3 sensitivity triple (softened to "≤0.03 shift" rather than assert a recomputed
triple); C-series generator/table bugs; D-series new content; GATE items.

---

## NEW TABLES INJECTED 2026-06-11 ~14:00 (supplement)

Generator: `scripts/transfer_analysis_v5/make_new_tables.py` (reads the clean CSVs,
writes `ACCV_2026/tables/tab_new_*.tex` — nothing hand-transcribed). New supplement
section **"Additional Robustness and Disentangling Analyses"** (\S\ref{supp:newrobust},
before Additional Figures) collects all eight with honest framing prose. supp now 13pp
(was 9), compiles clean, no new layout regressions (worst overfull 24.6pt is pre-existing).

| table | label | what it shows |
|---|---|---|
| tab_new_jackknife | Tab.~9 | per-regime source/family jackknife: pretrained recall robust [+0.57,+0.65], scratch precision soft + spair-leaning |
| tab_new_partial | Tab.~10 | appearance vs motion partial corr: motion survives controlling appearance (+0.54); appearance wrong-signed |
| tab_new_fixedmotion | Tab.~11 | fixed-motion appearance = level not selection (same winner every benchmark, cross-bench ρ≈+0.8–1.0) |
| tab_new_regimelevel | Tab.~12 | flip is regime (+0.81, p=9.5e-6) not level (n.s.); within-pretrained gap–level negative |
| tab_new_ttarm | Tab.~13 | which OOS cells are interpretable (FF-fly real; TT-KITTI void; TT-fly the flagged miss) |
| tab_new_seedaudit | Tab.~14 | 5-seed OOS distances; FF off-target stable +0.58–0.67 |
| tab_new_anticorr | Tab.~15 | anti-corr heterogeneity: mean −0.37, +0.41 on KITTI-2012 |
| tab_new_diagonal | Tab.~16 | in-domain diagonal-cell sensitivity: rule 0.507→0.498, no sign change |

All in the SUPPLEMENT; promote any to main as you see fit after review. Note: tab_new_fixedmotion
reads the FF kitti_recovered cells of intervention_breakdown.csv (complete + verified); the
pending decomp/2x2 regen (GATE-1) does not change those FF cells.
