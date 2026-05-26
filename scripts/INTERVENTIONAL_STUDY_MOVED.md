This study moved to /home/spencer/Projects/interventional-study on 2026-05-26.

The `scripts/interventional_study/` directory was relocated to a standalone repo.
See `/home/spencer/Projects/interventional-study/MIGRATION.md` for the briefing.

The new repo still depends on this one for:
  - sys.path imports from scripts/transfer_analysis_v3, scripts/transfer_analysis_v4
  - scripts/coverage module
  - data files: scripts/transfer_analysis_v3/transfer_table.csv,
                analysis_v3/pairwise_self_distances.csv

The dependency is via the `OSC_REPO` env var (defaulting to this repo's path).
