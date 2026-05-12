#!/usr/bin/env python3
"""
Generate supplementary method tables directly from repository configs and run metadata.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "supplementary_feature_and_training_details.md"


def load_yaml(rel_path: str) -> dict:
    path = ROOT / rel_path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def load_json(rel_path: str) -> dict:
    with (ROOT / rel_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_column_values(rel_path: str, column: str) -> list[str]:
    values: set[str] = set()
    with (ROOT / rel_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or column not in reader.fieldnames:
            return []
        for row in reader:
            value = str(row.get(column, "")).strip()
            if value:
                values.add(value)
    return sorted(values)


def fmt_list(values: list) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def parse_default_model_type() -> str:
    text = load_text("src/training/correspondence_lightning.py")
    match = re.search(r"model_type = self\.model_config\.get\('type', '([^']+)'\)\.lower\(\)", text)
    if not match:
        raise RuntimeError("Could not resolve default model type.")
    return match.group(1)


def parse_dino_model_name() -> str:
    text = load_text("src/mmd/encoders.py")
    match = re.search(r'model_name: str = "([^"]+)"', text)
    if not match:
        raise RuntimeError("Could not resolve DINO model name.")
    return match.group(1)


def parse_dino_spatial_layer_note() -> str:
    text = load_text("models/DinoV3/DinoV3.py")
    uses_last = "outputs.last_hidden_state" in text
    drops_prefix = "last_layer_output = last_layer_output[:, 5:, :]" in text
    if uses_last and drops_prefix:
        return "final transformer layer (`last_hidden_state`), keeping spatial tokens only after dropping the first 5 prefix tokens"
    if uses_last:
        return "final transformer layer (`last_hidden_state`)"
    return "spatial feature extraction implemented in `models/DinoV3/DinoV3.py`"


def parse_dino_prefix_drop_count() -> int:
    text = load_text("models/DinoV3/DinoV3.py")
    match = re.search(r"last_layer_output\s*=\s*last_layer_output\[:,\s*(\d+):,\s*:\]", text)
    if not match:
        raise RuntimeError("Could not resolve DINO prefix-token drop count.")
    return int(match.group(1))


def parse_dino_normalization_stats() -> tuple[list[float], list[float]]:
    text = load_text("models/DinoV3/DinoV3.py")
    match = re.search(
        r"Normalize\(\s*mean=\(([^)]*)\),\s*std=\(([^)]*)\)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError("Could not resolve DINO normalization stats.")
    mean = [float(v.strip()) for v in match.group(1).split(",") if v.strip()]
    std = [float(v.strip()) for v in match.group(2).split(",") if v.strip()]
    return mean, std


def parse_multistep_milestones(raw_value: str | list[int]) -> list[int]:
    if isinstance(raw_value, list):
        return [int(v) for v in raw_value]
    parsed = ast.literal_eval(str(raw_value))
    return [int(v) for v in parsed]


def split_model_family_encoder(label: str) -> tuple[str, str | None]:
    if "_" not in label:
        return label, None
    family, suffix = label.rsplit("_", 1)
    return family, suffix


def parse_encoder_config_suffix(suffix: str | None) -> tuple[str, str, str]:
    if suffix is None or len(suffix) != 2 or any(ch not in {"T", "F"} for ch in suffix):
        return "n/a", "n/a", "No explicit encoder variant suffix"
    pretrained = "true" if suffix[0] == "T" else "false"
    freeze = "true" if suffix[1] == "T" else "false"
    meaning = f"`pretrained={pretrained}`, `freeze={freeze}`"
    return pretrained, freeze, meaning


def family_display_name(family: str) -> str:
    mapping = {
        "catspp": "CATs++",
        "raft": "RAFT",
        "flowformer": "FlowFormer",
    }
    return mapping.get(family, family)


def collect_observed_model_family_encoder_labels() -> list[str]:
    rel_paths = [
        "analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/asym_and_mmd/prediction_lobo_rank_detail.csv",
        "analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/asym_and_mmd/prediction_jointood_rank_detail.csv",
        "analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/mmd_only/prediction_lobo_rank_detail.csv",
        "analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/mmd_only/prediction_jointood_rank_detail.csv",
    ]
    labels: set[str] = set()
    for rel_path in rel_paths:
        labels.update(load_csv_column_values(rel_path, "model_family_encoder"))
    order = {"catspp_FF": 0, "catspp_FT": 1, "catspp_TF": 2, "catspp_TT": 3, "raft": 4, "raft_TF": 5, "flowformer": 6}
    return sorted(labels, key=lambda value: (order.get(value, 100), value))


@dataclass
class MethodRecipe:
    label: str
    config_path: str
    model_summary: str
    batch_size: int
    epochs: int
    steps_per_epoch: int
    total_steps: int
    lr: float
    lr_backbone: float
    weight_decay: float
    scheduler: str
    dataset_downsample_flow: str


def build_method_recipe(rel_path: str, label: str, default_model_type: str) -> MethodRecipe:
    cfg = load_yaml(rel_path)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    model_type = model_cfg.get("type", default_model_type)
    if model_type == "raft":
        model_summary = (
            f"RAFT (`small={model_cfg.get('small')}`, `iters={model_cfg.get('iters')}`, "
            f"`alternate_corr={model_cfg.get('alternate_corr')}`, `mixed_precision={model_cfg.get('mixed_precision')}`, "
            f"`dropout={model_cfg.get('dropout')}`)"
        )
    else:
        model_summary = (
            f"CATs-style default (`type={model_type}`), backbone=`{model_cfg.get('backbone')}`, "
            f"`freeze={model_cfg.get('freeze')}`, `pretrained_backbone={model_cfg.get('pretrained_backbone')}`"
        )
    milestones = parse_multistep_milestones(train_cfg["step"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    epochs = int(train_cfg["epochs"])
    return MethodRecipe(
        label=label,
        config_path=rel_path,
        model_summary=model_summary,
        batch_size=int(train_cfg["batch_size"]),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        total_steps=epochs * steps_per_epoch,
        lr=float(train_cfg["lr"]),
        lr_backbone=float(train_cfg["lr_backbone"]),
        weight_decay=float(train_cfg["weight_decay"]),
        scheduler=f"MultiStepLR(milestones={milestones}, gamma={float(train_cfg['step_gamma'])})",
        dataset_downsample_flow=str(dataset_cfg.get("downsample_flow")),
    )


def main() -> None:
    dino_cov = load_yaml("src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml")
    hof_fp = load_yaml("src/configs/hof_configs/hof_fingerprint_full.yaml")
    hof_cov = load_yaml("src/configs/coverage_configs/coverage_hof_full_occ_diag_k20.yaml")
    flow_cov = load_yaml("src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml")
    mmd_cfg = load_yaml("src/configs/mmd_configs/mmd_config.yaml")
    dino_mmd = load_yaml("src/configs/mmd_configs/mmd_dino_v2.yaml")
    flow_mmd = load_yaml("src/configs/mmd_configs/mmd_flow_v2.yaml")
    run_meta = load_json(
        "analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/asym_and_mmd/run_metadata.json"
    )
    run_script = load_text("scripts/run_comprehensive_sweep_latest.sh")
    default_model_type = parse_default_model_type()
    dino_model_name = parse_dino_model_name()
    dino_layer_note = parse_dino_spatial_layer_note()
    dino_prefix_drop = parse_dino_prefix_drop_count()
    dino_mean, dino_std = parse_dino_normalization_stats()
    observed_model_groups = collect_observed_model_family_encoder_labels()

    flow_eps_match = re.search(r'FLOW_EPS_VALUES="([^"]+)"', run_script)
    if not flow_eps_match:
        raise RuntimeError("Could not find FLOW_EPS_VALUES in run_comprehensive_sweep_latest.sh.")
    flow_eps_values = flow_eps_match.group(1).split(",")

    unique_ridge = set()
    for meta_path in sorted(
        (
            ROOT
            / "analysis_comprehensive_runs"
            / "ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3"
        ).rglob("run_metadata.json")
    ):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "ridge_alpha" in meta:
            unique_ridge.add(float(meta["ridge_alpha"]))
    if not unique_ridge:
        raise RuntimeError("Could not resolve ridge_alpha from run metadata.")

    recipes = [
        build_method_recipe("src/configs/CorrespondenceConfigs/synthetic_rc.yaml", "CATs++ synthetic RC", default_model_type),
        build_method_recipe("src/configs/CorrespondenceConfigs/synthetic_rc_raft.yaml", "RAFT synthetic RC", default_model_type),
    ]

    dino_total_rff = len(mmd_cfg["dino_features"]["sigmas"]) * int(mmd_cfg["dino_features"]["features_per_sigma"])
    flow_total_rff = len(mmd_cfg["flow_vectors"]["sigmas"]) * int(mmd_cfg["flow_vectors"]["features_per_sigma"])
    dino_resize_size = 512
    dino_tokens_per_image = int(dino_cov["sampling"]["vectors_per_image"])
    dino_grid_side = int(math.isqrt(dino_tokens_per_image))
    if dino_grid_side * dino_grid_side != dino_tokens_per_image:
        raise RuntimeError("DINO vectors_per_image is not a square number.")
    dino_patch_size = dino_resize_size // dino_grid_side

    lines: list[str] = []
    lines.append("# Supplementary: Feature Spaces, Transfer Estimator, and Training Recipes")
    lines.append("")
    lines.append(
        "This file is generated by `scripts/generate_supplementary_method_details.py` from repository configs, trainer code, and run metadata."
    )
    lines.append("")
    lines.append("## 1. Model families and encoder variations")
    lines.append("")
    lines.append(
        "The analysis code constructs `encoder_config` from the checkpoint metadata flags `pretrained` and `freeze`, then forms `model_family_encoder = model_family + '_' + encoder_config` whenever the suffix is known."
    )
    lines.append("")
    lines.append("| Observed analysis label | Family | `pretrained` | `freeze` | Interpretation | Source |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for label in observed_model_groups:
        family, suffix = split_model_family_encoder(label)
        pretrained, freeze, meaning = parse_encoder_config_suffix(suffix)
        lines.append(
            f"| `{label}` | {family_display_name(family)} | `{pretrained}` | `{freeze}` | {meaning} | "
            "`scripts/build_heldout_model_cv.py`, `scripts/build_leakage_free_eval.py`, "
            "`analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/asym_and_mmd/prediction_lobo_rank_detail.csv` |"
        )
    lines.append("")
    lines.append(
        f"The current paper analysis artifacts expose the model groups {', '.join(f'`{label}`' for label in observed_model_groups)}."
    )
    lines.append("")
    lines.append(
        f"Plotting code converts the CATs++ suffixes to human-readable labels such as `CatsPP FF`, `CatsPP FT`, `CatsPP TF`, and `CatsPP TT`, while RAFT is shown as `RAFT` (`scripts/plot_smoothness_metrics.py`)."
    )
    lines.append("")
    lines.append(
        "In the current analysis CSVs, RAFT rows still carry the metadata suffix `TF`; the grouping code would collapse them to plain `raft` only if `pretrained`/`freeze` were missing or marked unknown."
    )
    lines.append("")
    lines.append(
        "Repository configs also include `flowformer` recipes, but no `flowformer_*` group appears in the current paper analysis root, so the main supplement tables should focus on CATs++ and RAFT."
    )
    lines.append("")
    lines.append("## 2. Feature spaces and extraction details")
    lines.append("")
    lines.append("### 2.1 Appearance features (`f_app`)")
    lines.append("")
    lines.append("| Field | Value | Source |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| coverage config alias | `{md_escape(str(dino_cov.get('encoder')))}` | `src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml` |"
    )
    lines.append(
        f"| instantiated feature extractor | `{md_escape(dino_model_name)}` | `src/mmd/encoders.py`, `models/DinoV3/DinoV3.py` |"
    )
    lines.append(
        f"| spatial embedding layer | {md_escape(dino_layer_note)} | `models/DinoV3/DinoV3.py` |"
    )
    lines.append(
        f"| resize size | `{dino_resize_size}` | `src/mmd/encoders.py`, `models/DinoV3/DinoV3.py` |"
    )
    lines.append(
        f"| image normalization | mean `{fmt_list(dino_mean)}`, std `{fmt_list(dino_std)}` | `models/DinoV3/DinoV3.py` |"
    )
    lines.append(
        f"| prefix-token drop | first `{dino_prefix_drop}` tokens removed before keeping spatial tokens | `models/DinoV3/DinoV3.py` |"
    )
    lines.append(
        f"| patch grid / patch size | `{dino_grid_side} x {dino_grid_side}` tokens, implying `{dino_patch_size} x {dino_patch_size}` pixel patches at `{dino_resize_size} x {dino_resize_size}` input | derived from `vectors_per_image` in `src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml` and resize size in `models/DinoV3/DinoV3.py` |"
    )
    lines.append(
        f"| patch sampling | `{dino_cov['sampling']['vectors_per_image']}` spatial tokens per image (`{dino_grid_side} x {dino_grid_side}` at `{dino_resize_size} x {dino_resize_size}`) | `src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml` |"
    )
    lines.append(
        f"| PCA reduction | output dim `{dino_cov['pca']['output_dim']}`, fit on `{dino_cov['pca']['fit_on']}`, max train vectors `{dino_cov['pca']['max_train_vectors']}`, whiten=`{dino_cov['pca']['whiten']}`, L2 normalize=`{dino_cov['pca']['l2_normalize']}` | `src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml` |"
    )
    lines.append(
        f"| vector budget for coverage cache | max vectors `{dino_cov['sampling']['max_vectors']}` | `src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml` |"
    )
    lines.append("")
    lines.append(
        "Note: the config label still says `dinov2_vitg14`, but the current implementation instantiates the DINOv3 HuggingFace checkpoint above. The supplement should document the implemented model, not just the alias."
    )
    lines.append("")
    lines.append("The implemented appearance pipeline is:")
    lines.append("")
    lines.append("1. Resize each RGB image to `512 x 512` and apply ImageNet normalization with the mean/std above.")
    lines.append(
        f"2. Run the DINOv3 backbone and take `outputs.last_hidden_state`, then discard the first `{dino_prefix_drop}` prefix tokens so only spatial patch embeddings remain."
    )
    lines.append(
        f"3. Treat the remaining `{dino_tokens_per_image}` spatial tokens as a `{dino_grid_side} x {dino_grid_side}` patch grid, with one embedding `z_p` per patch location `p`."
    )
    lines.append(
        f"4. Fit PCA on training-split patch vectors only, using at most `{dino_cov['pca']['max_train_vectors']}` vectors, and project each raw token `z_p in R^D` to `{dino_cov['pca']['output_dim']}` dimensions."
    )
    lines.append("5. L2-normalize the PCA output before downstream coverage or MMD calculations.")
    lines.append("")
    lines.append(r"\[")
    lines.append(
        rf"\tilde{{z}}_p = W (z_p - \mu), \qquad W \in \mathbb{{R}}^{{{dino_cov['pca']['output_dim']} \times D}},"
    )
    lines.append(r"\]")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"f_{\mathrm{app}}(p) = \frac{\tilde{z}_p}{\lVert \tilde{z}_p \rVert_2}.")
    lines.append(r"\]")
    lines.append("")
    lines.append("### 2.2 HOF motion fingerprints (`f_HOF`)")
    lines.append("")
    lines.append("| Field | Value | Source |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| grid size | `{fmt_list(hof_fp['hof']['grid_hw'])}` | `src/configs/hof_configs/hof_fingerprint_full.yaml` |"
    )
    lines.append(
        f"| orientation bins (`B_theta`) | `{hof_fp['hof']['angle_bins']}` | `src/configs/hof_configs/hof_fingerprint_full.yaml` |"
    )
    lines.append(
        f"| magnitude bin edges | `{fmt_list(hof_fp['hof']['mag_edges'])}` | `src/configs/hof_configs/hof_fingerprint_full.yaml` |"
    )
    lines.append(
        f"| magnitude bins (`B_r`) | `{len(hof_fp['hof']['mag_edges']) - 1}` | derived from `mag_edges` in `src/configs/hof_configs/hof_fingerprint_full.yaml` |"
    )
    lines.append(
        f"| magnitude clip | `{hof_fp['hof']['mag_clip']}` | `src/configs/hof_configs/hof_fingerprint_full.yaml` |"
    )
    lines.append(
        f"| occupancy tau | `{hof_fp['hof']['occupancy_tau']}` | `src/configs/hof_configs/hof_fingerprint_full.yaml` |"
    )
    lines.append(
        f"| normalization flags | `normalize_hist={hof_fp['hof']['normalize_hist']}`, `use_sqrt_mag={hof_fp['hof']['use_sqrt_mag']}`, `zero_is_invalid={hof_fp['hof']['zero_is_invalid']}` | `src/configs/hof_configs/hof_fingerprint_full.yaml` |"
    )
    lines.append("")
    lines.append("### 2.3 Directed coverage hyperparameters")
    lines.append("")
    lines.append("| Component | Values used | Source |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| DINO k-NN coverage | `self_radius_k={dino_cov['coverage']['self_radius_k']}`, `radius_quantile={dino_cov['coverage']['radius_quantile']}`, `neighbor_agg={dino_cov['coverage']['neighbor_agg']}`, `k_values={fmt_list(dino_cov['coverage']['k_values'])}` | `src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml` |"
    )
    lines.append(
        f"| HOF k-NN coverage sweep | `self_radius_k={hof_cov['coverage']['self_radius_k']}`, `radius_quantile={hof_cov['coverage']['radius_quantile']}`, `neighbor_agg={hof_cov['coverage']['neighbor_agg']}`, `k_values={fmt_list(hof_cov['coverage']['k_values'])}`, `filter_duplicates={hof_cov['coverage']['filter_duplicates']}` | `src/configs/coverage_configs/coverage_hof_full_occ_diag_k20.yaml` |"
    )
    lines.append(
        f"| BFV epsilon ladder | `{fmt_list(flow_cov['epsilon_curves']['values_px'])}` px | `src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml` |"
    )
    lines.append(
        f"| main registered appearance predictor names | `dino_rnorm_k5`, `dino_kl_k5` | `scripts/run_comprehensive_sweep_latest.sh` |"
    )
    lines.append(
        f"| main registered HOF predictor names | `hof_motion_k1`, `hof_kl_k5` | `scripts/run_comprehensive_sweep_latest.sh` |"
    )
    lines.append("")
    lines.append(
        f"For the main paper runs, the shell sweep registers DINO appearance predictors at `k=5`, while the single directed HOF motion baseline is registered as `hof_motion_k1`. The flow epsilon ladder used by the density pipeline is `{', '.join(flow_eps_values)}` px."
    )
    lines.append("")
    lines.append("## 3. Transfer estimator and leakage-free evaluation recipe")
    lines.append("")
    lines.append("| Field | Value | Source |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| estimator family | `{run_meta['linear_model']}` | run metadata from `analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/asym_and_mmd/run_metadata.json` |"
    )
    lines.append(
        f"| resolved ridge alpha | `{sorted(unique_ridge)}` across discovered run metadata; final paper root uses `{run_meta['ridge_alpha']}` | run metadata under `analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3` |"
    )
    lines.append(
        f"| predictor standardization | `standardize={run_meta['standardize']}`; intercept is kept unpenalized in `fit_linear_model` | `scripts/build_leakage_free_eval.py` and run metadata |"
    )
    lines.append(
        f"| target residualization | `cv_residualize_target_by_context={run_meta['cv_residualize_target_by_context']}`, context cols=`{run_meta['cv_residual_context_cols']}`, transform=`{run_meta['cv_residual_target_transform']}`, eval space=`{run_meta['cv_residual_eval_space']}` | run metadata |"
    )
    lines.append(
        f"| fit weighting | `fit_sample_weighting={run_meta['fit_sample_weighting']}`, `fit_balance_real_synth={run_meta['fit_balance_real_synth']}` | run metadata |"
    )
    lines.append(
        f"| aggregation | `overall_aggregation={run_meta['overall_aggregation']}` | run metadata |"
    )
    lines.append(
        f"| pairwise context grouping | `pairwise_group_cols={run_meta['pairwise_group_cols']}` | run metadata and `scripts/build_leakage_free_eval.py` |"
    )
    lines.append("")
    lines.append("The ridge solver used in the leakage-free evaluator is")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"\hat{\beta} = \arg\min_{\beta_0,\beta}\ \lVert y - \beta_0 \mathbf{1} - X\beta \rVert_2^2 + \lambda \lVert \beta \rVert_2^2,")
    lines.append(r"\]")
    lines.append("")
    lines.append(
        "where predictors are optionally standardized fold-by-fold, and the intercept is excluded from the penalty matrix (`penalty[0,0]=0`)."
    )
    lines.append("")
    lines.append("Leakage prevention is implemented directly in the CV splits:")
    lines.append("")
    lines.append("- Held-out training-dataset or benchmark protocols use `run_group_cv`, where each fold trains on rows outside the held-out group and tests only on the held-out group.")
    lines.append("- Held-out `(train_dataset, benchmark)` joint protocols use `run_joint_ood_group_cv`, where training excludes any row sharing the held-out training dataset or the held-out benchmark, and testing uses only the held-out pair.")
    lines.append("- The current evaluator does not perform an internal ridge-alpha grid search. `ridge_alpha` is supplied as a fixed CLI argument, and the final paper sweep scripts default it to `10`.")
    lines.append("")
    lines.append("## 4. Correspondence model training recipes")
    lines.append("")
    lines.append(
        "The Lightning trainer in `src/training/correspondence_lightning.py` configures `AdamW` with separate parameter groups for backbone and non-backbone weights, then applies `MultiStepLR` when `scheduler: step`."
    )
    lines.append("")
    lines.append("| Recipe | Model summary | Batch size | Epochs | Steps/epoch | Total steps | LR | Backbone LR | Weight decay | Scheduler | `downsample_flow` | Config |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for recipe in recipes:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(recipe.label),
                    md_escape(recipe.model_summary),
                    str(recipe.batch_size),
                    str(recipe.epochs),
                    str(recipe.steps_per_epoch),
                    str(recipe.total_steps),
                    f"`{recipe.lr}`",
                    f"`{recipe.lr_backbone}`",
                    f"`{recipe.weight_decay}`",
                    md_escape(recipe.scheduler),
                    f"`{recipe.dataset_downsample_flow}`",
                    f"`{recipe.config_path}`",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Common recipe details shared by both synthetic RC configs:")
    lines.append("")
    lines.append("- `augmentation=true`, `seed=2021`, `n_threads=0`, `weight_decay=0.05`, `scheduler=step`, `step=[70, 80, 90]`, `step_gamma=0.5`.")
    lines.append("- Both configs train on `512 x 512` synthetic pairs with `batch_size=8` for `50` epochs and `1000` steps per epoch, for `50,000` optimizer updates.")
    lines.append("- The default non-RAFT model type is `cats`, so `synthetic_rc.yaml` is the CATs-style recipe unless `model.type` is overridden.")
    lines.append("")
    lines.append("## 5. Symmetric MMD baseline details")
    lines.append("")
    lines.append("| Baseline | Preprocessing before RFF | RBF sigmas | Features/sigma | Total RFF features | Seed | Backend | Unbiased | Runtime input dim | Source |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| Flow MMD | normalize `[x,y,dx,dy]` to `[-1,1]`, then project to motion-only flow space `[dx,dy]` (`mmd_space=flow`) | `{fmt_list(mmd_cfg['flow_vectors']['sigmas'])}` | `{mmd_cfg['flow_vectors']['features_per_sigma']}` | `{flow_total_rff}` | `{mmd_cfg['flow_vectors']['seed']}` | `{mmd_cfg['flow_vectors']['backend']}` | `{mmd_cfg['flow_vectors']['unbiased']}` | `2` (preset starts at `4`, then `calculate_mmd_v2.py` overwrites `input_dim` with the actual post-transform dim) | `src/configs/mmd_configs/mmd_flow_v2.yaml`, `src/configs/mmd_configs/mmd_config.yaml`, `scripts/calculate_mmd_v2.py` |"
    )
    lines.append(
        f"| DINO MMD | extract DINO patch tokens, apply PCA to `{dino_mmd['pca']['output_dim']}` dims, then L2-normalize | `{fmt_list(mmd_cfg['dino_features']['sigmas'])}` | `{mmd_cfg['dino_features']['features_per_sigma']}` | `{dino_total_rff}` | `{mmd_cfg['dino_features']['seed']}` | `{mmd_cfg['dino_features']['backend']}` | `{mmd_cfg['dino_features']['unbiased']}` | `{dino_mmd['pca']['output_dim']}` (preset starts at `4096`, then `calculate_mmd_v2.py` overwrites `input_dim` with the actual post-PCA dim) | `src/configs/mmd_configs/mmd_dino_v2.yaml`, `src/configs/mmd_configs/mmd_config.yaml`, `scripts/calculate_mmd_v2.py` |"
    )
    lines.append("")
    lines.append(
        "The RFF maps are shared across datasets and are seeded once (`seed=42`). For each bandwidth `sigma`, the code samples Gaussian frequencies and random phases, then concatenates `features_per_sigma` cosine features per scale."
    )
    lines.append("")
    lines.append("## 6. Primary source files")
    lines.append("")
    for rel_path in [
        "src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml",
        "src/configs/hof_configs/hof_fingerprint_full.yaml",
        "src/configs/coverage_configs/coverage_hof_full_occ_diag_k20.yaml",
        "src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml",
        "src/mmd/encoders.py",
        "models/DinoV3/DinoV3.py",
        "scripts/plot_smoothness_metrics.py",
        "scripts/build_heldout_model_cv.py",
        "src/configs/CorrespondenceConfigs/synthetic_rc.yaml",
        "src/configs/CorrespondenceConfigs/synthetic_rc_raft.yaml",
        "src/configs/CorrespondenceConfigs/synthetic_rc_flowformer.yaml",
        "src/training/correspondence_lightning.py",
        "scripts/build_leakage_free_eval.py",
        "src/configs/mmd_configs/mmd_config.yaml",
        "src/configs/mmd_configs/mmd_dino_v2.yaml",
        "src/configs/mmd_configs/mmd_flow_v2.yaml",
        "scripts/run_comprehensive_sweep_latest.sh",
        "scripts/run_ridge_weighted_ablation_triplet.sh",
        "analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/asym_and_mmd/run_metadata.json",
    ]:
        lines.append(f"- `{rel_path}`")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
