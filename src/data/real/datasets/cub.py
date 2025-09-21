from pathlib import Path
from typing import Union

from PIL import Image
import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import normalize


def _parse_ints(vals):
    for i in range(len(vals)):
        try:
            vals[i] = int(vals[i])
        except:
            pass
    return vals


def _read_file(fname):
    with open(fname) as f:
        txt = f.read().strip()
        res = dict(_parse_ints(x.split(' ', 1)) for x in txt.split('\n'))
    return res

# TODO: NEED TO DECIDE HOW TO MAKE PAIRS
class CUBDataset(object):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        size: Union[int, tuple] = 256,
        normalize: Union[bool, str, tuple, list] = 'imagenet',
    ):
        self.root = Path(root)
        self.train = train
        self.size = size
        self.normalize = normalize

        images = _read_file(self.root / 'images.txt')
        split = _read_file(self.root / 'train_test_split.txt')
        classes = _read_file(self.root / 'classes.txt')
        labels = _read_file(self.root / 'image_class_labels.txt')
        boxes = _read_file(self.root / 'bounding_boxes.txt')
        parts = _read_file(self.root / 'parts/parts.txt')

        image_keys = sorted(images.keys())

        imgs, targets = [], []
        cls = set()
        for id in image_keys:
            if id in labels and split[id] == self.train:
                imgs.append(images[id])
                targets.append(labels[id])
                cls.add(labels[id])
        self.imgs = imgs
        self.targets = targets

        self.class_to_idx = {}
        idx_shift = {}
        i = 0
        for k, v in sorted(classes.items()):
            if k in cls:
                self.class_to_idx[v] = i
                idx_shift[k] = i
                i += 1
        self.classes = [x[0] for x in sorted(self.class_to_idx.items(),
                                             key=lambda a:a[1])]
        self.targets = [idx_shift[i] for i in self.targets]
            
        self.bboxes = []
        for id in image_keys:
            if id in labels and split[id] == self.train:
                self.bboxes.append([float(x) for x in boxes[id].split(' ')])

        part_locs = {i: np.zeros((len(parts), 3)) for i in image_keys}
        with root.joinpath('parts/parts_locs').open('r') as f:
            for row in f.read().strip().split('\n'):
                vals = row.strip().split(' ')
                img_id = int(vals[0])
                part_id = int(vals[1])
                x, y = float(vals[2]), float(vals[3])
                vis = float(vals[4])
                part_locs[img_id][part_id - 1, :] = (x, y, vis)
        self.part_locs = [part_locs[i] for i in image_keys]

    def __len__(self):
        return len(self.imgs)

    def filepath(self, index):
        '''Returns Path to image in ``self.imgs[index]``'''
        return self.root / 'images' / self.imgs[index]

    def __getitem__(self, index):
        path = self.filepath(index)
        target = self.targets[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target