"""
Space definitions and transformations for coverage analysis.

For flow vectors [x, y, dx, dy]:
- XY space: [x, y] - position only
- Flow space: [dx, dy] - motion only  
- Joint space: [x, y, α*dx, α*dy] - position + scaled motion

All coordinates normalized to [-1, 1] range for comparable distances.
"""

from typing import Tuple
import numpy as np


def normalize_flow_vectors(
    vectors: np.ndarray,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Normalize flow vectors from pixel space to [-1, 1] range.
    
    Maps pixel coordinates to [-1, 1] centered at image center:
    - x ∈ [0, W] → [-1, 1] via 2*x/W - 1
    - y ∈ [0, H] → [-1, 1] via 2*y/H - 1
    - dx, dy scaled consistently: 2*dx/W, 2*dy/H
    
    Args:
        vectors: (N, 4) array [x, y, dx, dy] in pixel space
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        (N, 4) normalized vectors [x_norm, y_norm, dx_norm, dy_norm]
    """
    if vectors.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {vectors.shape}")
    
    result = vectors.copy().astype(np.float32)
    
    # Positions: center at image center
    result[:, 0] = 2.0 * result[:, 0] / img_width - 1.0   # x: [0,W] → [-1,1]
    result[:, 1] = 2.0 * result[:, 1] / img_height - 1.0  # y: [0,H] → [-1,1]
    
    # Flow: scale consistently with positions
    result[:, 2] = 2.0 * result[:, 2] / img_width   # dx scaled
    result[:, 3] = 2.0 * result[:, 3] / img_height  # dy scaled
    
    return result


def to_xy_space(vectors: np.ndarray) -> np.ndarray:
    """
    Extract XY (position) space from flow vectors.
    
    Args:
        vectors: (N, 4) flow vectors [x, y, dx, dy]
        
    Returns:
        (N, 2) position vectors [x, y]
    """
    if vectors.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {vectors.shape}")
    
    return vectors[:, [0, 1]].copy()


def to_flow_space(vectors: np.ndarray) -> np.ndarray:
    """
    Extract flow (motion) space from flow vectors.
    
    Args:
        vectors: (N, 4) flow vectors [x, y, dx, dy]
        
    Returns:
        (N, 2) flow vectors [dx, dy]
    """
    if vectors.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {vectors.shape}")
    
    return vectors[:, [2, 3]].copy()


def to_joint_space(vectors: np.ndarray, alpha: float) -> np.ndarray:
    """
    Create joint space with α-scaled flow component.
    
    Args:
        vectors: (N, 4) flow vectors [x, y, dx, dy]
        alpha: Scaling factor for flow components
        
    Returns:
        (N, 4) joint vectors [x, y, α*dx, α*dy]
    """
    if vectors.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {vectors.shape}")
    
    result = vectors.copy()
    result[:, 2:4] *= alpha
    
    return result


def get_space_names(representation: str) -> list[str]:
    """
    Get list of space names for a representation.
    
    Args:
        representation: "flow", "dino", or "resnet"
        
    Returns:
        List of space names
    """
    if representation == "flow":
        return ["xy", "flow", "joint"]
    elif representation in ["dino", "resnet"]:
        return ["features"]
    else:
        raise ValueError(f"Unknown representation: {representation}")


def get_space_dim(space_name: str, base_dim: int = 4) -> int:
    """
    Get dimensionality for a space.
    
    Args:
        space_name: "xy", "flow", "joint", or "features"
        base_dim: Base dimensionality (4 for flow, feature_dim for dino/resnet)
        
    Returns:
        Dimensionality of the space
    """
    if space_name == "xy":
        return 2
    elif space_name == "flow":
        return 2
    elif space_name == "joint":
        return 4
    elif space_name == "features":
        return base_dim
    else:
        raise ValueError(f"Unknown space: {space_name}")


def transform_to_space(
    vectors: np.ndarray,
    space_name: str,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Transform vectors to specified space.
    
    Args:
        vectors: Input vectors (flow: (N,4), features: (N,D))
        space_name: "xy", "flow", "joint", or "features"
        alpha: Scaling factor for joint space (ignored for other spaces)
        
    Returns:
        Transformed vectors
    """
    if space_name == "xy":
        return to_xy_space(vectors)
    elif space_name == "flow":
        return to_flow_space(vectors)
    elif space_name == "joint":
        return to_joint_space(vectors, alpha)
    elif space_name == "features":
        # Features are already in the right space
        return vectors.copy()
    else:
        raise ValueError(f"Unknown space: {space_name}")
