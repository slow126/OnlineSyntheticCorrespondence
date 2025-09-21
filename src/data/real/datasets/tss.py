from pathlib import Path
from typing import Union

from PIL import Image
import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import normalize

from src.io import read_flo_file


class TSSDataset(object):
    def __init__(
        self,
        root,
        size: Union[tuple, int] = 256,
        normalize: Union[bool, str, tuple, list] = 'imagenet',
    ):
        self.root = Path(root)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        self.labels = {}

        self.pairs = []
        idx = 0
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir(): continue
            self.labels[sub.name] = idx
            idx += 1
            self.pairs.extend(sorted(sub.iterdir()))

        self.flipped = []
        for p in self.pairs:
            self.flipped.append(int(p.joinpath('flip_gt.txt').open().read()))

        if normalize == 'imagenet':
            from src import imagenet_stats
            normalize = imagenet_stats
        elif normalize == True:
            normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize = normalize

    def __len__(self):
        return len(self.pairs)

    def _read_image(self, path, name):
        img = Image.open(path.joinpath(name)).convert('RGB')
        if self.size is not None:
            img = img.resize(self.size, Image.Resampling.BILINEAR)
        img = torch.from_numpy(np.array(img, dtype=np.float32)).div_(255).moveaxis(-1, 0)
        if self.normalize:
            img = normalize(img, *self.normalize)
        return img

    def _read_flow(self, path, name):
        flow = read_flo_file(path.joinpath(name))
        h, w = flow.shape[:2]

        flow = torch.from_numpy(flow).moveaxis(-1, 0)
        if self.size is not None:
            flow = interpolate(flow.unsqueeze(0), self.size, mode='nearest-exact').squeeze(0)
            flow[flow > 1e9] = torch.inf
            flow[0] *= (self.size[1] / w)
            flow[1] *= (self.size[0] / h)
        else:
            flow[flow > 1e9] = torch.inf

        return flow
    
    def __getitem__(self, idx):
        pair_dir = self.pairs[idx]

        label = self.labels[pair_dir.parent.name]

        img1 = self._read_image(pair_dir, 'image1.png')
        img2 = self._read_image(pair_dir, 'image2.png')
        flow1 = self._read_flow(pair_dir, 'flow1.flo')
        flow2 = self._read_flow(pair_dir, 'flow2.flo')
        flipped = self.flipped[idx]

        return {
            'src_img': img1,
            'trg_img': img2,
            'src_flow': flow1,
            'trg_flow': flow2,
            'flipped': flipped,
            'label': label,
        }