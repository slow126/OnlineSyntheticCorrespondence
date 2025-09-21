import json
import mmap
import os
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import numpy as np

from src.data.image_ops import img_from_byte_array


__all__ = [
    'FlatImageFolder',
    'Hdf5ImageDataset',
]


class FlatImageFolder(object):
    '''Collects all images nested under a root folder into a dataset.
    '''
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        EXT = ('.jpg', '.jpeg', '.png')
        self.root = Path(root)

        self.imgs = []
        for root, dirs, files in os.walk(self.root):
            if len(files) > 0:
                root = Path(root)
                for f in files:
                    f = root / f
                    if f.suffix.lower() in EXT:
                        self.imgs.append(f)

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


class Hdf5ImageDataset(object):
    '''Dataset using images stored in HDF5 files located in root directory.
    '''
    def __init__(self, root, segs=None, transform=None, limit=None, offset=0):
        self.root = Path(root)

        all_segs = sorted(self.root.glob('*.hdf5'))

        self.data_files = all_segs[:segs]
        self.meta_files = [x.with_suffix('.json') for x in self.data_files]

        limit = limit or 1e9

        self.samples = []
        n = -1
        for i, meta in enumerate(self.meta_files):
            j = json.load(meta.open('r'))['data']
            for k in j:
                n += 1
                if n < offset: continue
                if len(self.samples) >= limit: break
                sample = [i, k, *j[k]]
                self.samples.append(sample)
            if len(self.samples) >= limit: break

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def fetch_from_file(self, idx):
        sample = self.samples[idx]
        path = self.data_files[sample[0]]
        offset = sample[2]
        shape = sample[3]
        dtype = sample[4]
        with path.open('rb') as fp:
            fileno = fp.fileno()
            mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
            data = np.frombuffer(mapping, dtype=dtype, count=np.prod(shape), offset=offset)
            data = data.reshape(shape)
        return data

    def __getitem__(self, idx):
        data = self.fetch_from_file(idx)
        img = img_from_byte_array(data).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img