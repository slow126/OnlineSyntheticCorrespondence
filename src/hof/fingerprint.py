"""
HOF (Histogram of Optical Flow) fingerprint extraction.

Spec summary (dense + sparse friendly):
- 32x32 spatial grid (default)
- normalize flow by image size: u=dx/W, v=dy/H
- angle bins: 8 over [0, 2pi), soft binning
- magnitude bins: fixed edges in normalized units (L2 by default), soft binning
- per-cell histogram L1-normalized (if any samples)
- per-cell occupancy channel based on count / tau
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np


@dataclass
class HOFFingerprintConfig:
    grid_hw: Tuple[int, int] = (32, 32)
    angle_bins: int = 8
    mag_edges: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.08, 0.25)
    mag_clip: Optional[float] = None  # defaults to mag_edges[-1]
    occupancy_tau: float = 5.0
    normalize_hist: bool = True
    use_sqrt_mag: bool = True
    zero_is_invalid: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fingerprint_dim(cfg: HOFFingerprintConfig) -> int:
    gh, gw = cfg.grid_hw
    mag_bins = len(cfg.mag_edges) - 1
    return gh * gw * (1 + cfg.angle_bins * mag_bins)


def _validate_config(cfg: HOFFingerprintConfig) -> None:
    if cfg.angle_bins <= 0:
        raise ValueError("angle_bins must be > 0")
    if len(cfg.mag_edges) < 2:
        raise ValueError("mag_edges must have at least 2 values")
    edges = np.asarray(cfg.mag_edges, dtype=float)
    if not np.all(np.isfinite(edges)):
        raise ValueError("mag_edges must be finite")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("mag_edges must be strictly increasing")
    if cfg.grid_hw[0] <= 0 or cfg.grid_hw[1] <= 0:
        raise ValueError("grid_hw must be positive")


def _normalize_flow(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized (u, v) in fraction-of-image units."""
    h, w = flow.shape[:2]
    u = flow[..., 0] / float(w)
    v = flow[..., 1] / float(h)
    return u, v


def _soft_bin_angles(theta: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Soft binning for angles in [0, 2pi)."""
    two_pi = 2.0 * np.pi
    theta = np.mod(theta, two_pi)
    ang_idx = theta / two_pi * bins
    ang0 = np.floor(ang_idx).astype(np.int64)
    ang0 = np.mod(ang0, bins)
    frac = ang_idx - np.floor(ang_idx)
    ang1 = (ang0 + 1) % bins
    w0 = 1.0 - frac
    w1 = frac
    return ang0, ang1, w0, w1


def _soft_bin_magnitude(mag: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft binning for magnitude using fixed edges.
    For values inside a bin [e_i, e_{i+1}), weight linearly toward the right edge.
    The last bin has no right neighbor (all weight stays in the last bin).
    """
    mag_bins = len(edges) - 1
    mag_idx = np.searchsorted(edges, mag, side="right") - 1
    mag_idx = np.clip(mag_idx, 0, mag_bins - 1)

    mag0 = mag_idx.astype(np.int64)
    mag1 = np.minimum(mag0 + 1, mag_bins - 1)

    denom = edges[mag0 + 1] - edges[mag0]
    denom = np.where(denom == 0, 1.0, denom)
    w1 = (mag - edges[mag0]) / denom
    w1 = np.clip(w1, 0.0, 1.0)

    # No right neighbor for last bin
    last_mask = mag0 == (mag_bins - 1)
    w1 = np.where(last_mask, 0.0, w1)
    w0 = 1.0 - w1

    return mag0, mag1, w0, w1


def compute_hof_fingerprint(
    flow: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    cfg: Optional[HOFFingerprintConfig] = None,
    return_components: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute HOF fingerprint for a single flow field.

    Args:
        flow: [H, W, 2] array in pixel units.
        valid_mask: Optional [H, W] mask (True=valid). Combined with finite check.
        cfg: HOFFingerprintConfig
        return_components: If True, return occupancy + histogram in meta.

    Returns:
        fingerprint: 1D float32 vector
        meta: dict with small diagnostic info (counts, shapes)
    """
    if cfg is None:
        cfg = HOFFingerprintConfig()
    _validate_config(cfg)

    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"flow must be [H, W, 2], got {flow.shape}")

    h, w = flow.shape[:2]
    gh, gw = cfg.grid_hw
    edges = np.asarray(cfg.mag_edges, dtype=np.float64)
    mag_clip = float(cfg.mag_clip) if cfg.mag_clip is not None else float(edges[-1])

    finite_mask = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
    if valid_mask is None:
        mask = finite_mask
    else:
        mask = finite_mask & valid_mask.astype(bool)

    if cfg.zero_is_invalid:
        zero_mask = (flow[..., 0] == 0) & (flow[..., 1] == 0)
        mask = mask & (~zero_mask)

    if not np.any(mask):
        mag_bins = len(edges) - 1
        occ = np.zeros((gh, gw), dtype=np.float32)
        hist = np.zeros((gh, gw, cfg.angle_bins, mag_bins), dtype=np.float32)
        cell = np.concatenate([occ[..., None], hist.reshape(gh, gw, -1)], axis=2)
        fingerprint = cell.reshape(-1).astype(np.float32)
        meta = {
            "height": int(h),
            "width": int(w),
            "valid_count": 0,
            "grid_hw": (int(gh), int(gw)),
        }
        if return_components:
            meta["occupancy"] = occ
            meta["histogram"] = hist
        return fingerprint, meta

    ys, xs = np.nonzero(mask)

    u, v = _normalize_flow(flow)
    u = u[ys, xs]
    v = v[ys, xs]

    if cfg.use_sqrt_mag:
        mag = np.hypot(u, v)
    else:
        mag = u * u + v * v

    mag = np.clip(mag, edges[0], mag_clip)
    theta = np.arctan2(v, u)

    # Spatial bins
    ci = (ys * gh / float(h)).astype(np.int64)
    cj = (xs * gw / float(w)).astype(np.int64)
    ci = np.clip(ci, 0, gh - 1)
    cj = np.clip(cj, 0, gw - 1)

    # Soft bins
    ang0, ang1, w_ang0, w_ang1 = _soft_bin_angles(theta, cfg.angle_bins)
    mag0, mag1, w_mag0, w_mag1 = _soft_bin_magnitude(mag, edges)

    mag_bins = len(edges) - 1

    # Occupancy counts
    counts = np.zeros((gh, gw), dtype=np.float32)
    np.add.at(counts, (ci, cj), 1.0)

    # Histogram
    hist = np.zeros((gh, gw, cfg.angle_bins, mag_bins), dtype=np.float32)

    w00 = (w_ang0 * w_mag0).astype(np.float32)
    w01 = (w_ang0 * w_mag1).astype(np.float32)
    w10 = (w_ang1 * w_mag0).astype(np.float32)
    w11 = (w_ang1 * w_mag1).astype(np.float32)

    np.add.at(hist, (ci, cj, ang0, mag0), w00)
    np.add.at(hist, (ci, cj, ang0, mag1), w01)
    np.add.at(hist, (ci, cj, ang1, mag0), w10)
    np.add.at(hist, (ci, cj, ang1, mag1), w11)

    if cfg.normalize_hist:
        sums = hist.sum(axis=(2, 3), keepdims=True)
        hist = np.divide(hist, sums, out=np.zeros_like(hist), where=sums > 0)

    occ = np.minimum(1.0, counts / float(cfg.occupancy_tau))

    cell = np.concatenate([occ[..., None], hist.reshape(gh, gw, -1)], axis=2)
    fingerprint = cell.reshape(-1).astype(np.float32)

    meta = {
        "height": int(h),
        "width": int(w),
        "valid_count": int(mask.sum()),
        "grid_hw": (int(gh), int(gw)),
    }
    if return_components:
        meta["occupancy"] = occ
        meta["histogram"] = hist

    return fingerprint, meta
