"""
Coverage analysis modules for multi-space (xy/flow/joint) coverage metrics.

This package implements a clean 5-step pipeline:
0. Sampling: Load cached vectors
1. Calibration: Compute global alpha for flow scaling
2. Spaces: Transform vectors into different spaces (xy, flow, joint)
3. Radius: Compute self-radius for each dataset/space
4. Metrics: Compute cross-dataset directed distances and coverage

Modules:
--------
- cache: Unified caching utilities (vectors, radii, alpha) + preprocessing (PCA, L2 norm)
- faiss_ops: Core FAISS operations (build_index, knn_distances, self_radius, directed_distances)
- spaces: Space transformations (xy, flow, joint extraction from flow vectors)
- calibration: Alpha calibration for flow scaling (Step 1)
- metrics: Coverage metrics computation (Steps 4-5)
- kl: kNN KL divergence utilities
"""

__version__ = "2.0.0"
__all__ = [
    "cache",
    "faiss_ops",
    "spaces",
    "calibration",
    "metrics",
    "kl",
]
