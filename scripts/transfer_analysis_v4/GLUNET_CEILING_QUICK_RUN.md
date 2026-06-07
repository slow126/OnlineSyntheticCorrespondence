# Camera-Ready Ceiling Strategy and GLU-Net Quick Run

**Date:** 2026-06-06

## Recommendation

Do not claim a theoretical ceiling or that the predictor reaches 98% of all
possible data-only performance. The present experiments cannot identify that
quantity without assumptions about architecture interactions and training
noise.

Use the following reporting hierarchy:

1. Primary result: variant-aware, within-context Spearman correlation of `g`.
2. Primary comparison: the paired motion-minus-appearance gap under
   generator-family resampling.
3. Empirical context: leave-one-variant-out rank consensus.
4. Sensitivity: aggregate variants within architecture families, then hold out
   one architecture family so CATs++ toggles do not dominate the reference.
5. Appendix only: pairwise agreement and classical reliability approximations.

Call the consensus result an **empirical oracle reference** or
**cross-variant consensus benchmark**, not a ceiling. It uses outcomes from
other model variants, so it is intentionally stronger than a static descriptor,
but it is not an upper bound.

## Bugs Fixed

1. The old ceiling script evaluated `L + g` while claiming to evaluate motion
   `g`. The revised analysis evaluates `g`.
2. Mean pairwise variant agreement is now labeled target reproducibility, not a
   hard ceiling.
3. GLU-Net run names were classified as CATs++ because `steps100` was matched
   before `glunet`.
4. GLU-Net rows with fewer than three early AUC checkpoints were discarded even
   though peak PCK was valid. They are now retained as peak-only rows with AUC
   set to missing.
5. Singleton contexts were ignored by the ranking metric but still entered
   ridge preprocessing. Contexts with fewer than three sources are now removed
   before fitting.
6. GLU-Net partial runs had unequal progress. The quick run uses a common
   10,000-step peak horizon rather than each source's global peak.

The default raw-flow coverage CSV currently contains concatenated 17-column and
37-column schemas. A clean pure-source file already exists at
`analysis/coverage_v2_flow_only_raw_joint_full_new.csv`; `build_table.py` now
accepts `--flow-raw-csv` so a rebuild can select it explicitly.

## Corrected Current Result

Using CATs++ and RAFT, after removing invalid singleton contexts:

| quantity | rho |
|---|---:|
| motion LOTO | +0.506 |
| appearance LOTO | -0.237 |
| held-variant consensus | +0.665 |
| architecture-balanced held-variant consensus | +0.645 |
| architecture-aggregated motion | +0.595 |
| held-architecture consensus | +0.680 |

Motion is 76% of the held-variant empirical reference, or 78% under the
architecture-balanced held-variant sensitivity. These are descriptive ratios.

## Preliminary GLU-Net Result

All 11 canonical GLU-Net validation CSVs were found under:

`/home/spencer/rc_glunet_val_csvs/snapshots`

No additional download is needed for this quick run. The runs were configured
for 5,000 epochs at 100 steps per epoch, but the snapshots range from 100 to 680
epochs. The longest horizon shared by every source is therefore 10,000 steps.

At that common provisional horizon:

| quantity | rho |
|---|---:|
| motion LOTO | +0.451 |
| appearance LOTO | -0.249 |
| held-variant consensus | +0.666 |
| architecture-balanced held-variant consensus | +0.643 |
| architecture-aggregated motion | +0.498 |
| held-architecture consensus | +0.668 |

Motion reaches 68% of the held-variant empirical reference, 70% under the
architecture-balanced variant sensitivity, and 74% for the architecture-level
aggregated estimand.

Interpretation:

- GLU-Net does not invalidate the main claim. Motion remains positive and
  appearance remains negative.
- The estimate becomes more conservative when the third architecture is added.
- The GLU-Net result is not camera-ready because 10,000 steps is only 2% of the
  planned 500,000-step budget and different source jobs are still at very
  different stages.
- The full 64-run GLU-Net grid is not required for the consensus analysis. One
  standardized GLU-Net variant across the 11 sources is sufficient. The larger
  grid is useful only if pretraining/freeze effects become a separate claim.

## Camera-Ready GLU-Net Procedure

1. Choose one training-step horizon before looking at source rankings.
2. Let every canonical GLU-Net source reach that horizon.
3. Export `validation_results.csv` for all 11 sources. Config and summary files
   are useful for provenance but are not required by the importer.
4. Use peak PCK through the same horizon to match the current paper target.
   Also report fixed-horizon final PCK as a sensitivity if space permits.
5. Refit the variant-aware predictor with GLU-Net as another row context.
6. Report held-variant and held-architecture consensus as empirical references.
7. Preserve the motion-minus-appearance paired comparison as the main claim.

## Generated Outputs

- `scripts/transfer_analysis_v4/results_ceiling_fixed_current/EMPIRICAL_REFERENCES.md`
- `scripts/transfer_analysis_v4/results_glunet_snapshot_10k/EMPIRICAL_REFERENCES.md`
- `scripts/transfer_analysis_v3/transfer_table_glunet_snapshot_10k_rebuilt.csv`
- `analysis/leakage_free_flow_kmeans_manifold/auc_results_glunet_snapshot_10k.csv`

