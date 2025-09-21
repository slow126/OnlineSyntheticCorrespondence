import os
from pathlib import Path
import random
import time
from typing import Optional, Union

import torch
import torchvision

from .base import BaseDatamodule, make2
from .real.datasets import img_datasets
from .synth.datasets import hdf5_dataset
from .synth.datasets import transforms as synth_transforms
from src.flow import corr2flow


class JointDataset(object):
    def __init__(
        self,
        synth_root: str,
        imgs_root: str,
        imgs_type: str = 'files',
        synth_args: Optional[dict] = None,
        imgs_args: Optional[dict] = None,
        sampling: str = 'random',
    ):
        self.synth_root = Path(synth_root)
        self.imgs_root = Path(imgs_root)
        self.sampling = sampling

        synth_args = synth_args or {}
        imgs_args = imgs_args or {}

        if imgs_type == 'files':
            self.real_image_dataset = img_datasets.FlatImageFolder(imgs_root, **imgs_args)
        elif imgs_type == 'hdf5':
            self.real_image_dataset = img_datasets.Hdf5ImageDataset(imgs_root, **imgs_args)
        else:
            raise RuntimeError(f'Unsupported dataset type: "{imgs_type}"')

        self.synth_dataset = hdf5_dataset.Hdf5CorrespondenceDataset(synth_root, **synth_args)

    def __len__(self):
        return len(self.synth_dataset)

    def __getitem__(self, idx):
        src_img, trg_img, pts = self.synth_dataset[idx]
        if self.sampling == 'random':
            ridx = random.randint(0, len(self.real_image_dataset) - 1)
        else:
            ridx = idx % len(self.real_image_dataset)
        real_img = self.real_image_dataset[ridx]

        return real_img, src_img, trg_img, pts


class JointCorrespondenceDatamodule(BaseDatamodule):
    def __init__(
        self,
        synth_root: str,
        imgs_root: str,
        imgs_type: str = 'files',
        img_res: Union[int, tuple, list] = 256,
        flow_res: Union[int, tuple, list] = 64,
        synth_args: Optional[dict] = None,
        imgs_args: Optional[dict] = None,
        num_val_synth: int = 10,
        batch_size: int = 64,
        num_workers: int = 8,
        copy_synth_local: Optional[str] = None,
        copy_real_local: Optional[str] = None,
    ):
        self.synth_root = Path(synth_root)
        self.imgs_root = Path(imgs_root)
        self.imgs_type = imgs_type
        self.img_res = make2(img_res)
        self.flow_res = make2(flow_res)
        self.synth_args = synth_args or {}
        self.imgs_args = imgs_args or {}
        self.num_val_synth = num_val_synth
        self.copy_real_local = copy_real_local
        self.copy_synth_local = copy_synth_local
        super().__init__('', batch_size, num_workers, None)

    def prepare_data(self):
        if self.copy_synth_local is not None:
            import subprocess
            datadir = os.environ['DATADIR']
            dst_root = Path(self.copy_synth_local)
            src_root = self.synth_root
            dst = dst_root.joinpath(src_root.relative_to(datadir))
            if not dst.is_dir():
                dst.mkdir(parents=True)
            src = str(src_root) + '/'
            dst = str(dst) + '/'
            t = time.time()
            print(f'Copying data from {src} to {dst}...')
            cmd = ' '.join(['rsync', '-av', src+'*.{hdf5,json}', dst])
            subprocess.run(cmd, shell=True)
            t = time.time() - t
            print(f'Finished copying in {t:.5f} seconds')
            self.original_synth_root = self.synth_root
            self.synth_root = Path(dst)
            print(f'Setting root for dataloading to {str(self.synth_root)}')

        if self.copy_real_local is not None:
            import subprocess
            datadir = os.environ['DATADIR']
            dst_root = Path(self.copy_real_local)
            src_root = self.imgs_root
            dst = dst_root.joinpath(src_root.relative_to(datadir))
            if not dst.is_dir():
                dst.mkdir(parents=True)
            src = str(src_root) + '/'
            dst = str(dst) + '/'
            t = time.time()
            print(f'Copying data from {src} to {dst}...')
            extra = []
            if self.imgs_type == 'hdf5':
                src = src + './*/*.{hdf5,json}'
                extra.append('--relative')
            cmd = ' '.join(['rsync', '-av', src, dst] + extra)
            subprocess.run(cmd, shell=True)
            t = time.time() - t
            print(f'Finished copying in {t:.5f} seconds')
            self.original_imgs_root = self.imgs_root
            self.imgs_root = Path(dst)
            print(f'Setting root for dataloading to {str(self.imgs_root)}')

    def get_transforms(self, data='real'):
        if data == 'synth':
            r = [int(8 / 7 * s) for s in self.img_res]
            train = synth_transforms.Compose([
                synth_transforms.ToTensor(),
                synth_transforms.Resize(r),
                synth_transforms.RandomCrop(self.img_res),
                synth_transforms.ColorJitter(0.2, 0.2, 0.2, 0.5),
                synth_transforms.ScalePoints(self.flow_res),
                synth_transforms.Normalize(0.5, 0.5),
            ])
            val = synth_transforms.Compose([
                synth_transforms.ToTensor(),
                synth_transforms.Resize(self.img_res),
                synth_transforms.ScalePoints(self.flow_res),
                synth_transforms.Normalize(0.5, 0.5),
            ])
        elif data == 'real':
            train = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(self.img_res, (0.2, 1)),
                torchvision.transforms.ColorJitter(*(0.2,) * 4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ])
            val = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.img_res),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ])
        else:
            raise RuntimeError(f'Unknown dataset type: {data}')

        return train, val

    def setup(self, stage):
        tform_train_real, tform_val_real = self.get_transforms('real')
        tform_train_synth, tform_val_synth = self.get_transforms('synth')
        offset = self.num_val_synth or 0

        self.imgs_args['transform'] = tform_val_real
        self.synth_args['transform'] = tform_val_synth
        self.synth_args['obj_limit'] = offset
        self.val_data = JointDataset(
            self.synth_root, self.imgs_root/'val', self.imgs_type,
            self.synth_args, self.imgs_args, sampling='fixed',
        )

        if stage == 'fit':
            self.imgs_args['transform'] = tform_train_real
            self.synth_args['transform'] = tform_train_synth
            self.synth_args['offset'] = offset
            self.synth_args['obj_limit'] = None
            self.train_data = JointDataset(
                self.synth_root, self.imgs_root/'train', self.imgs_type,
                self.synth_args, self.imgs_args, sampling='random',
            )

    def collate(self, batch):
        imgs = [x[:3] for x in batch]
        pts = [x[3] for x in batch]
        imgs = torch.utils.data.default_collate(imgs)
        flow = corr2flow(pts, self.flow_res)
        return {'real_img': imgs[0], 'src_img': imgs[1], 'trg_img': imgs[2], 'flow': flow}