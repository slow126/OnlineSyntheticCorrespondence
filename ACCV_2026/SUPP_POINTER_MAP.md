# Supplement Pointer Alignment Map

Every place the main paper points the reader to the supplement, and where that
pointer should land.

Main compiles `sections/07_interventions_v2.tex` (not the v1 file), so all
intervention pointers below are read from v2.

**Status (2026-07-02):** the xr-hyper wiring and the two broken `\ref{sec:law}`
pointers (#1, #2) are FIXED and verified in the rendered PDFs. Rows #3–#10 remain
prose-only, and the two content gaps are still open. Cross-refs use the `S-` prefix
(e.g. `\ref{S-supp:sampling}`).

## Blockers (structural)

1. ✅ **DONE — Main can now resolve `supp:` labels.** Added `\usepackage{xr-hyper}`
   + `\externaldocument[S-]{supp_main}` after hyperref in `main.tex:18-24`. Supplement
   already references main (`supp_main.tex:13-14`); the link is now bidirectional.
   Build order matters — see "Rebuild" below.
2. ✅ **DONE — the two mis-targeted pointers are fixed.** `03_setup.tex:90` →
   `\ref{S-supp:sampling}` (renders "supp., Sect. 5"); `04_coverage_law.tex:85` →
   `\ref{S-supp:law}` (renders "supp., Sect. 2"). Both previously resolved to main §4.
   Verified in `main.pdf`, no undefined-reference warnings.
3. ⬜ Everything else is still prose-only (`(supp.)`, "in the supplement") with no
   number. Not broken, but vague — candidates for a concrete `S-` `\ref`.

## Rebuild (correct order, from `ACCV_2026/`)

Cross-refs go both ways, so interleave the passes (`-g` defeats latexmk's cache):

```
latexmk -g -pdf -interaction=nonstopmode main.tex
latexmk -g -pdf -interaction=nonstopmode supp_main.tex
latexmk -g -pdf -interaction=nonstopmode main.tex
latexmk -g -pdf -interaction=nonstopmode supp_main.tex
```

## Supplement section index (targets)

| Label | Supp section |
|---|---|
| `supp:designcut` | From-Scratch Training: An Out-of-Distribution Probe |
| `supp:law` | Per-Configuration Evidence and the Pooled Flip |
| `supp:robust` | Robustness Controls |
| `supp:predict` | Predictor Analyses (contains `supp:recovery`) |
| `supp:sampling` | Sampling Stability of the Estimators |
| `supp:mid` | The Middlebury Exclusion |
| `supp:newrobust` | Additional Robustness and Disentangling Analyses |
| `supp:figs` | Additional Figures |
| `supp:precision` | Off-Target Mass at Fixed Coverage (the Precision Ladder) |

## Pointer-by-pointer map

| # | Main location | Pointer as written | Currently resolves to | Should point to (section) | Specific anchor in supp | Status / action |
|---|---|---|---|---|---|---|
| 1 | `03_setup.tex:90` | "…$k$-NN density estimate that is sampling-sensitive and fails to converge … (supp., Sect.~\ref{S-supp:sampling})" | **supp §5** ✓ (renders "Sect. 5") | `supp:sampling` — Sampling Stability of the Estimators | prose: "kNN-KL estimators plateau near ρ=0.5 … never stabilize" (`supp_main.tex:490-496`) | ✅ **DONE.** Retargeted to `\ref{S-supp:sampling}`, verified in PDF. |
| 2 | `04_coverage_law.tex:85` | "two alternative distance estimators (epsilon radius, and k-means clustering, supp., Sect.~\ref{S-supp:law})" | **supp §2** ✓ (renders "Sect. 2") | `supp:law` — Per-Configuration Evidence… | ε-radius `tab:supp_eps` + k-means `tab:supp_kmeans` | ✅ **DONE** (ref fixed + content gap closed). k-means table added as `tab:supp_kmeans` (§2, pretrained-only, count-weighted codebook AUC; coverage +0.32..+0.45, off-target ~0, symmetric hedges). |
| 3 | `04_coverage_law.tex:93` | "Per-configuration tables, the two estimators, and the sampling and pooling controls are in the supplement." | — (prose) | spans `supp:law` + `supp:sampling` | per-config: `tab:supp_asym`, `tab:law_full`; estimators: `tab:supp_eps`; sampling: `supp:sampling`; pooled flip: `supp:law` | Textual only. Consider one concrete ref (`supp:law`). Same k-means caveat as #2 ("the two estimators"). |
| 4 | `05_selecting_data.tex:51` | "Learned combiners … do not improve over our zero-shot coverage (Table~\ref{tab:predictors}, supp.)." | `tab:predictors` = main table; "supp." = prose | `supp:predict` (+ `supp:law`) | feature-set combiner `tab:supp_featuresets` (`supp_main.tex:404-412`); ε-combiner overfit note (`supp_main.tex:281-286`) | Textual "supp." Add ref to `supp:predict` / `tab:supp_featuresets`. |
| 5 | `05_selecting_data.tex:52` | "The full source×target landscape is in the supplement." | — (prose) | `supp:predict` — The source-ranking landscape | `fig:consensus` (`supp_main.tex:378-391`) | Textual only. Point to `fig:consensus`. |
| 6 | `06_absolute_prediction.tex:79` | "…run-to-run training noise that no dataset descriptor can predict (supp.)." | — (prose) | `supp:predict` — Reproducibility oracles | `tab:supp_oracles` (`supp_main.tex:349-355`) | Textual only. Point to `supp:predict` / `tab:supp_oracles`. |
| 7 | `07_interventions_v2.tex:78` (fig:ladder caption) | "Per-model and per-variant breakdowns are in the supplement." | — (prose) | `supp:newrobust` (closest) | `fig:kubric_appearance` (`supp_main.tex:672-683`) — appearance-variant gallery | **CONTENT GAP.** No per-model / per-magnitude ladder breakdown table exists in supp. `fig:kubric_appearance` covers appearance variants only, not the per-model motion ladder. Either add the breakdown or soften the caption. |
| 8 | `07_interventions_v2.tex:94` | "…the full parameter set is in the supplement." (recovered generator θ) | — (prose) | `supp:predict` → `supp:recovery` | `tab:recovery_full` (`supp_main.tex:472-478`) | Textual only. Point to `tab:recovery_full` / `supp:recovery`. |
| 9 | `07_interventions_v2.tex:135` (tab:interv caption) | "Full thresholds and strides are in the supplement." | — (prose) | `supp:designcut` — Full thresholds and strides | `tab:interv_kitti_full` + `tab:interv_davis_full` (`supp_main.tex:205-227`) | Textual only. Point to both tables. |
| 10 | `07_interventions_v2.tex:4` (comment) | "% the appearance gallery lives in the supplement." | — (LaTeX comment, not rendered) | `supp:newrobust` | `fig:kubric_appearance` (`supp_main.tex:672-683`) | Not a live pointer. No body-text pointer to the appearance gallery exists in v2 — add one if we want the gallery cited. |

## Content gaps surfaced (not just alignment)

- ✅ **k-means clustering estimator** (promised in `04_coverage_law.tex:85`, and implied
  by "the two estimators" in `:93`) — **RESOLVED**. Added `tab:supp_kmeans` in §2
  (pretrained-only; count-weighted codebook AUC; coverage +0.32..+0.45, off-target ~0,
  symmetric hedges). Data: `analysis/coverage_v2_flow_only_raw_kmeans_curve_summary.csv`
  joined to `transfer_table_nomid.csv`; generator `scripts/transfer_analysis_v5/make_tab_supp_kmeans.py`.
  Note: pretrained-only because the published ε-table's scratch rows come from an orphan
  CSV (no producer) whose scratch peak-PCK basis we could not reproduce.
- ✅ **KL-kNN sampling-stability table** (§5 was prose-only) — **RESOLVED**. Added
  `tab:sampling` from `analysis_v3/density_invariance_pair_sharded/stability_flow_train_eval__eval_eval.csv`
  (blocks.py R2A method): mean-NN → 1.00 by 8M, kNN-KL stuck at ρ≈0.47–0.49 (dT→B) /
  0.04–0.21 (dB→T). Generator `scripts/transfer_analysis_v5/make_tab_sampling_stability.py`.
- ⬜ **Per-model / per-variant ladder breakdown** (promised in `fig:ladder` caption,
  `07_interventions_v2.tex:78`) still has **no counterpart**. `fig:kubric_appearance`
  shows appearance variants at fixed motion, not a per-model breakdown across magnitude
  rungs. Decide: add the breakdown figure/table, or reword the caption.

## Remaining work

1. ✅ ~~Add `xr-hyper` + `\externaldocument` to `main.tex`.~~ Done (`S-` prefix).
2. ✅ ~~Fix the two wrong `\ref{sec:law}`.~~ Done (#1 → `S-supp:sampling`, #2 → `S-supp:law`).
3. ⬜ Optionally upgrade the prose "supplement" pointers (#3–#10) to concrete `S-` `\ref`s
   per the table above.
4. ⬜ Resolve the two content gaps (k-means estimator, per-model ladder breakdown) by
   adding content or rewording main.
