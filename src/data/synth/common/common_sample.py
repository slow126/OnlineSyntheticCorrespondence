from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch


@dataclass
class CommonSample:
    src_img: Optional[torch.Tensor] = None
    trg_img: Optional[torch.Tensor] = None
    flow_full: Optional[torch.Tensor] = None   # pixel space
    flow_feat: Optional[torch.Tensor] = None   # feature space (e.g., 32x32)
    src_kps: Optional[torch.Tensor] = None     # [2, N]
    trg_kps: Optional[torch.Tensor] = None     # [2, N]
    n_pts: Optional[int] = None
    pckthres: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_img": self.src_img,
            "trg_img": self.trg_img,
            "flow": self.flow_full,  # full resolution alias
            "flow_full": self.flow_full,
            "flow_downsampled": self.flow_feat,
            "flow_feat": self.flow_feat,
            "src_kps": self.src_kps,
            "trg_kps": self.trg_kps,
            "n_pts": self.n_pts,
            "pckthres": self.pckthres,
            "meta": self.meta,
        }
