from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class PreprocMotionDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str = "train",
        index_file: str = "index.jsonl",
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.split = split
        self.index_path = self.root / dataset_name / split / index_file

        if self.index_path.exists():
            self.entries = self._load_index(self.index_path)
        else:
            data_dir = self.root / dataset_name / split
            self.entries = [
                {"file": path.name, "dataset": dataset_name, "sample_id": path.stem}
                for path in sorted(data_dir.glob("*.pt"))
            ]

    def _load_index(self, path: Path) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        entry = self.entries[idx]
        file_name = entry["file"]
        path = self.root / self.dataset_name / self.split / file_name
        payload = torch.load(path, map_location="cpu")

        dx = payload["dx"].float()
        dy = payload["dy"].float()
        mask = payload["mask"].float()

        X = torch.stack([dx, dy, mask], dim=0)
        V_target = X[:2]
        M0 = X[2:3]

        metadata = {
            "dataset_name": payload.get("dataset", self.dataset_name),
            "sample_id": payload.get("sample_id", Path(file_name).stem),
            "n_valid": int(payload.get("n_valid", mask.sum().item())),
            "orig_h": int(payload.get("orig_h", dx.shape[0])),
            "orig_w": int(payload.get("orig_w", dx.shape[1])),
        }

        return X, V_target, M0, metadata
