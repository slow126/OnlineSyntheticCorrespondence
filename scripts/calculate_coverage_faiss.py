#!/usr/bin/env python3
"""
Calculate coverage metrics using FAISS approximate nearest neighbors.

This produces directed coverage scores that can be used as drop-in predictors
for the existing leakage-free analysis (recall/precision/outside columns).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coreset.validation import extract_flow_vectors_from_batch
from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.data.synth.datasets.MixedCorrespondenceDataset import MixedCorrespondenceDataset
from src.mmd.encoders import BaseFeatureEncoder, ResNet101Encoder, DinoV3Encoder

try:
    import faiss  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "faiss is required for calculate_coverage_faiss.py. "
        "Install faiss-cpu or faiss-gpu in your environment."
    ) from exc


def create_dataset_from_config(
    dataset_name: str,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    entry_overrides: Optional[dict] = None,
) -> CorrespondenceDataset:
    dataset_config = common_params.copy()
    if dataset_name in dataset_overrides:
        dataset_config.update(dataset_overrides[dataset_name])
    if entry_overrides:
        dataset_config.update(entry_overrides)

    if "size" in dataset_config and isinstance(dataset_config["size"], list):
        dataset_config["size"] = tuple(dataset_config["size"])
    dataset_config["split"] = split
    if "max_kps" in dataset_config and dataset_config["max_kps"] is None:
        dataset_config["max_kps"] = None

    if dataset_name == "tss":
        dataset_config["thres"] = dataset_config.get("thres", "img")
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
    elif dataset_name == "middlebury":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
    elif dataset_name == "pointodyssey":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", True)
        dataset_config["thres"] = dataset_config.get("thres", "img")
    elif dataset_name in ["kitti2012", "kitti2015"]:
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
        dataset_config["thres"] = dataset_config.get("thres", "img")
        if dataset_config.get("kitti_val_use_full_training", False) and split == "val":
            dataset_config["split"] = "training"
    elif dataset_name == "flyingthings":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", True)

    print(f"Creating dataset: {dataset_name} (split: {split})")
    return CorrespondenceDataset(dataset_name, **dataset_config)


def create_mixed_dataset_from_config(
    datasets_list: list,
    percentages: list,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    epoch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> MixedCorrespondenceDataset:
    if len(datasets_list) != len(percentages):
        raise ValueError("Number of datasets must match number of percentages.")

    created_datasets = []
    for dataset_name in datasets_list:
        ds_config = common_params.copy()
        if dataset_name in dataset_overrides:
            ds_config.update(dataset_overrides[dataset_name])
        if "size" in ds_config and isinstance(ds_config["size"], list):
            ds_config["size"] = tuple(ds_config["size"])
        ds_config["split"] = split
        if "max_kps" in ds_config and ds_config["max_kps"] is None:
            ds_config["max_kps"] = None

        if dataset_name == "tss":
            ds_config["thres"] = ds_config.get("thres", "img")
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", False)
        elif dataset_name == "middlebury":
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", False)
        elif dataset_name == "pointodyssey":
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", True)
            ds_config["thres"] = ds_config.get("thres", "img")
        elif dataset_name in ["kitti2012", "kitti2015"]:
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", False)
            ds_config["thres"] = ds_config.get("thres", "img")
            if ds_config.get("kitti_val_use_full_training", False) and split == "val":
                ds_config["split"] = "training"
        elif dataset_name == "flyingthings":
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", True)

        print(f"Creating sub-dataset: {dataset_name} (split: {split})")
        created_datasets.append(CorrespondenceDataset(dataset_name, **ds_config))

    print(f"Creating mixed dataset with {len(created_datasets)} datasets")
    return MixedCorrespondenceDataset(
        datasets=created_datasets,
        percentages=percentages,
        epoch_size=epoch_size,
        seed=seed,
    )


def create_encoder(encoder_name: str, device: torch.device) -> BaseFeatureEncoder:
    if encoder_name == "resnet101":
        return ResNet101Encoder(device=device)
    if encoder_name == "dino":
        return DinoV3Encoder(device=device)
    raise ValueError(f"Unknown encoder: {encoder_name}. Supported: 'resnet101', 'dino'")


def extract_features_from_batch(batch: dict, encoder: BaseFeatureEncoder) -> np.ndarray:
    if "src_img" in batch:
        img = batch["src_img"]
    elif "source" in batch:
        img = batch["source"]
    elif "image0" in batch:
        img = batch["image0"]
    else:
        raise ValueError(f"Could not find source image in batch. Keys: {batch.keys()}")

    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    feats = encoder.extract_features(img)
    feats = feats.float().cpu().numpy().astype(np.float32, copy=False)
    return feats


def _subsample_dense(
    vectors: np.ndarray,
    threshold: Optional[int],
    fraction: float,
    min_keep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if threshold is None or vectors.shape[0] <= threshold:
        return vectors
    keep = max(min_keep, int(vectors.shape[0] * fraction))
    keep = min(keep, vectors.shape[0])
    if keep == vectors.shape[0]:
        return vectors
    idx = rng.choice(vectors.shape[0], size=keep, replace=False)
    return vectors[idx]


class VectorCollector:
    def __init__(self, max_vectors: Optional[int], rng: np.random.Generator):
        self.max_vectors = max_vectors
        self.rng = rng
        self.buffers = []
        self.total = 0

    def add(self, vectors: Optional[np.ndarray]) -> None:
        if vectors is None or vectors.size == 0:
            return
        self.buffers.append(vectors)
        self.total += vectors.shape[0]
        if self.max_vectors and self.total > self.max_vectors * 2:
            data = np.concatenate(self.buffers, axis=0)
            idx = self.rng.choice(data.shape[0], size=self.max_vectors, replace=False)
            self.buffers = [data[idx]]
            self.total = self.max_vectors

    def finalize(self) -> np.ndarray:
        if not self.buffers:
            return np.empty((0, 0), dtype=np.float32)
        data = np.concatenate(self.buffers, axis=0)
        if self.max_vectors and data.shape[0] > self.max_vectors:
            idx = self.rng.choice(data.shape[0], size=self.max_vectors, replace=False)
            data = data[idx]
        return data.astype(np.float32, copy=False)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _build_index(
    vectors: np.ndarray,
    index_factory: str,
    metric: str,
    use_gpu: bool,
    nprobe: Optional[int],
) -> "faiss.Index":
    dim = vectors.shape[1]
    if metric == "cosine":
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        metric_type = faiss.METRIC_L2

    if index_factory.lower() == "flat":
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.index_factory(dim, index_factory, metric_type)

    if index.is_trained is False:
        index.train(vectors)
    index.add(vectors)

    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = nprobe

    if use_gpu and faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)

    return index


def _nn_distances(
    index: "faiss.Index",
    vectors: np.ndarray,
    metric: str,
    k: int,
    agg: str = "first",
) -> np.ndarray:
    if vectors.size == 0:
        return np.array([], dtype=np.float32)
    dists, _ = index.search(vectors, k)
    if metric == "cosine":
        dists = 1.0 - dists
    else:
        dists = np.sqrt(np.maximum(dists, 0.0))
    if k <= 1 or agg in ("first", "min"):
        out = dists[:, 0]
    elif agg in ("kth", "last", "max"):
        out = dists[:, -1]
    elif agg == "mean":
        out = dists.mean(axis=1)
    elif agg == "median":
        out = np.median(dists, axis=1)
    else:
        raise ValueError(f"Unsupported neighbor aggregation: {agg}")
    return out.astype(np.float32, copy=False)


def _self_radius(
    index: "faiss.Index",
    vectors: np.ndarray,
    metric: str,
    quantile: float,
    k: int = 1,
    agg: str = "first",
) -> float:
    if vectors.shape[0] < 2:
        return float("nan")
    k = max(int(k), 1)
    search_k = min(vectors.shape[0], k + 1)
    if search_k <= 1:
        return float("nan")
    dists, _ = index.search(vectors, search_k)
    if metric == "cosine":
        dists = 1.0 - dists
    else:
        dists = np.sqrt(np.maximum(dists, 0.0))
    neigh_dists = dists[:, 1:]
    if neigh_dists.size == 0:
        return float("nan")
    if agg in ("first", "min"):
        sample = neigh_dists[:, 0]
    elif agg in ("kth", "last", "max"):
        sample = neigh_dists[:, -1]
    elif agg == "mean":
        sample = neigh_dists.mean(axis=1)
    elif agg == "median":
        sample = np.median(neigh_dists, axis=1)
    else:
        raise ValueError(f"Unsupported neighbor aggregation: {agg}")
    return float(np.quantile(sample, quantile))


def _sample_dataset_vectors(
    ds_config: dict,
    common_params: dict,
    dataset_overrides: dict,
    representation: str,
    encoder: Optional[BaseFeatureEncoder],
    batch_size: int,
    num_workers: int,
    sampling_cfg: dict,
    rng: np.random.Generator,
) -> Dict[str, object]:
    is_mixed = ds_config.get("mixed", False) or "datasets" in ds_config
    split = ds_config["split"]
    num_batches = ds_config.get("num_batches")
    entry_overrides = ds_config.get("overrides", None)

    if is_mixed:
        datasets_list = ds_config.get("datasets", [])
        percentages = ds_config.get("percentages", [])
        label = ds_config.get("name")
        if not label:
            if len(percentages) == 2 and len(datasets_list) == 2:
                pct1 = int(percentages[0] * 100)
                pct2 = int(percentages[1] * 100)
                label = f"{datasets_list[0]}_{datasets_list[1]}_{pct1}_{pct2}"
            else:
                label = "+".join(datasets_list)
        dataset = create_mixed_dataset_from_config(
            datasets_list,
            percentages,
            split,
            common_params,
            dataset_overrides,
            epoch_size=ds_config.get("epoch_size"),
            seed=ds_config.get("seed"),
        )
        has_synthetic = "synthetic" in datasets_list
        workers = 0 if has_synthetic else num_workers
    else:
        label = ds_config["name"]
        dataset_name = ds_config.get("dataset_name", label)
        dataset = create_dataset_from_config(
            dataset_name, split, common_params, dataset_overrides, entry_overrides
        )
        workers = 0 if dataset_name == "synthetic" else num_workers

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=False,
    )

    dense_threshold = sampling_cfg.get("dense_threshold")
    dense_fraction = sampling_cfg.get("dense_fraction", 0.05)
    dense_min_keep = sampling_cfg.get("dense_min_keep", 2000)
    max_vectors = sampling_cfg.get("max_vectors")

    collector = VectorCollector(max_vectors, rng)
    total_batches = None if num_batches is None else int(num_batches)
    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        if representation == "flow":
            vectors = extract_flow_vectors_from_batch(batch)
        else:
            if encoder is None:
                raise ValueError("Feature encoder is required for non-flow representations.")
            vectors = extract_features_from_batch(batch, encoder)

        if vectors is None or vectors.size == 0:
            continue

        vectors = _subsample_dense(
            vectors,
            dense_threshold,
            dense_fraction,
            dense_min_keep,
            rng,
        )
        collector.add(vectors)
        if (batch_idx + 1) % 5 == 0:
            print(
                f"    [{label}] batches={batch_idx + 1}"
                f"{'' if total_batches is None else f'/{total_batches}'}"
                f" vectors={collector.total}"
            )

    vectors = collector.finalize()
    print(f"    [{label}] done: vectors={vectors.shape[0]}")
    return {
        "label": label,
        "split": split,
        "is_eval": bool(ds_config.get("is_eval", False)),
        "representation": representation,
        "vectors": vectors,
    }


def _apply_pca(
    vectors_by_name: Dict[str, Dict[str, object]],
    pca_cfg: dict,
    metric: str,
) -> None:
    if not pca_cfg.get("enabled", False):
        return

    fit_on = pca_cfg.get("fit_on", "train")
    max_train = pca_cfg.get("max_train_vectors", 200000)
    output_dim = int(pca_cfg.get("output_dim", 256))

    fit_vectors = []
    for info in vectors_by_name.values():
        is_eval = info["is_eval"]
        if fit_on == "train" and is_eval:
            continue
        if fit_on == "eval" and not is_eval:
            continue
        fit_vectors.append(info["vectors"])

    if not fit_vectors:
        return

    train_vectors = np.concatenate(fit_vectors, axis=0)
    if max_train and train_vectors.shape[0] > max_train:
        rng = np.random.default_rng(0)
        idx = rng.choice(train_vectors.shape[0], size=max_train, replace=False)
        train_vectors = train_vectors[idx]

    dim = train_vectors.shape[1]
    if output_dim >= dim:
        return

    pca = faiss.PCAMatrix(dim, output_dim, eigen_power=-0.5 if pca_cfg.get("whiten") else 0.0)
    pca.train(train_vectors.astype(np.float32, copy=False))

    for info in vectors_by_name.values():
        vecs = info["vectors"]
        if vecs.size == 0:
            continue
        info["vectors"] = pca.apply_py(vecs.astype(np.float32, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS-based coverage metrics.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to FAISS coverage config YAML.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    representation = config.get("representation", "flow")
    encoder_name = config.get("encoder", "resnet101")
    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 4))

    sampling_cfg = config.get("sampling", {})
    rng = np.random.default_rng(int(sampling_cfg.get("seed", 42)))

    faiss_cfg = config.get("faiss", {})
    index_factory = faiss_cfg.get("index_factory", "HNSW32")
    metric = faiss_cfg.get("metric", "l2")
    use_gpu = bool(faiss_cfg.get("use_gpu", False))
    nprobe = faiss_cfg.get("nprobe")

    coverage_cfg = config.get("coverage", {})
    radius_quantile = float(coverage_cfg.get("radius_quantile", 0.95))
    k = int(coverage_cfg.get("k", 1))
    neighbor_agg = str(coverage_cfg.get("neighbor_agg", "first"))
    self_radius_k = int(coverage_cfg.get("self_radius_k", 1))

    output_cfg = config.get("output", {})
    output_file = output_cfg.get("results_file", "coverage_faiss_results.csv")

    datasets_config = config.get("datasets", [])
    if not datasets_config:
        raise ValueError("No datasets specified in config.")

    common_params = config.get("dataset_params", {})
    dataset_overrides = config.get("dataset_overrides", {})

    encoder = None
    if representation != "flow":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = create_encoder(encoder_name, device)

    vectors_by_name: Dict[str, Dict[str, object]] = {}
    for ds_config in datasets_config:
        ds_repr = ds_config.get("representation", representation)
        info = _sample_dataset_vectors(
            ds_config,
            common_params,
            dataset_overrides,
            ds_repr,
            encoder,
            batch_size,
            num_workers,
            sampling_cfg,
            rng,
        )
        label = info["label"]
        split = info["split"]
        key = f"{label}_{split}"
        vectors_by_name[key] = info
        print(f"  Collected {info['vectors'].shape[0]} vectors for {key}")

    _apply_pca(vectors_by_name, config.get("pca", {}), metric)

    if metric == "cosine":
        for info in vectors_by_name.values():
            vecs = info["vectors"]
            if vecs.size == 0:
                continue
            info["vectors"] = _l2_normalize(vecs)

    indices = {}
    radii = {}
    for key, info in vectors_by_name.items():
        vecs = info["vectors"]
        if vecs.size == 0:
            indices[key] = None
            radii[key] = float("nan")
            continue
        index = _build_index(
            vecs.astype(np.float32, copy=False),
            index_factory,
            metric,
            use_gpu,
            nprobe,
        )
        indices[key] = index
        radii[key] = _self_radius(
            index,
            vecs,
            metric,
            radius_quantile,
            k=self_radius_k,
            agg=neighbor_agg,
        )

    train_keys = [k for k, v in vectors_by_name.items() if not v["is_eval"]]
    eval_keys = [k for k, v in vectors_by_name.items() if v["is_eval"]]

    results = []
    for train_key in train_keys:
        train_info = vectors_by_name[train_key]
        train_vecs = train_info["vectors"]
        train_idx = indices[train_key]
        if train_idx is None or train_vecs.size == 0:
            continue
        train_label = train_info["label"]
        train_split = train_info["split"]

        for eval_key in eval_keys:
            eval_info = vectors_by_name[eval_key]
            eval_vecs = eval_info["vectors"]
            eval_idx = indices[eval_key]
            if eval_idx is None or eval_vecs.size == 0:
                continue

            dist_eval_to_train = _nn_distances(train_idx, eval_vecs, metric, k, agg=neighbor_agg)
            dist_train_to_eval = _nn_distances(eval_idx, train_vecs, metric, k, agg=neighbor_agg)

            radius_train = radii.get(train_key, float("nan"))
            radius_eval = radii.get(eval_key, float("nan"))

            recall = (
                float(np.mean(dist_eval_to_train <= radius_train))
                if np.isfinite(radius_train) and dist_eval_to_train.size
                else float("nan")
            )
            precision = (
                float(np.mean(dist_train_to_eval <= radius_eval))
                if np.isfinite(radius_eval) and dist_train_to_eval.size
                else float("nan")
            )

            result = {
                "dataset1": train_label,
                "split1": train_split,
                "dataset2": eval_info["label"],
                "split2": eval_info["split"],
                "representation": train_info["representation"],
                "k": k,
                "neighbor_agg": neighbor_agg,
                "self_radius_k": self_radius_k,
                "radius_quantile": radius_quantile,
                "radius_train": radius_train,
                "radius_eval": radius_eval,
                "recall": recall,
                "precision": precision,
                "outside": 1.0 - precision if np.isfinite(precision) else float("nan"),
                "train_to_eval_coverage": recall,
                "eval_to_train_coverage": precision,
                "mean_nn_eval_to_train": float(np.mean(dist_eval_to_train)) if dist_eval_to_train.size else float("nan"),
                "median_nn_eval_to_train": float(np.median(dist_eval_to_train)) if dist_eval_to_train.size else float("nan"),
                "p90_nn_eval_to_train": float(np.quantile(dist_eval_to_train, 0.9)) if dist_eval_to_train.size else float("nan"),
                "mean_nn_train_to_eval": float(np.mean(dist_train_to_eval)) if dist_train_to_eval.size else float("nan"),
                "median_nn_train_to_eval": float(np.median(dist_train_to_eval)) if dist_train_to_eval.size else float("nan"),
                "p90_nn_train_to_eval": float(np.quantile(dist_train_to_eval, 0.9)) if dist_train_to_eval.size else float("nan"),
                "n_train_vectors": int(train_vecs.shape[0]),
                "n_eval_vectors": int(eval_vecs.shape[0]),
            }
            results.append(result)
            print(
                f"[{train_label} -> {eval_info['label']}] "
                f"train->eval={recall:.3f} eval->train={precision:.3f}"
            )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        import csv

        fieldnames = list(results[0].keys())
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} results to: {output_path}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
