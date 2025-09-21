import os
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from ..base import BaseDatamodule, make2
from .datasets.tss import TSSDataset
from .datasets.optical_flow.kitti import KITTI_noc, KITTI_only_occ, KITTI_occ
from .datasets.optical_flow.sintel import mpi_sintel
from .datasets.pose.awa import AWADataset
from .datasets.pose.cub import CUBDataset
from .datasets.geometric.hpatches import HPatchesDataset
from .datasets.load_dataset import load_dataset
from src.warp.warped_pairs import SyntheticPairWarper
from src import flow



# Transforms, move this to seperate file?
class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, get_float=True):
        self.get_float = get_float

    def __call__(self, array):

        if not isinstance(array, np.ndarray):
            array = np.array(array)
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        if self.get_float:
            # carefull, this is not normalized to [0, 1]
            return tensor.float()
        else:
            return tensor


def get_transforms(normalize):
    if normalize == 'imagenet':
        from src import imagenet_stats
        normalize = imagenet_stats
    elif normalize == True:
        normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    input_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
    ])
    target_t = transforms.Compose([
        ArrayToTensor(),
    ])
    return input_t, target_t


class SemanticFlowDatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        benchmark: str,
        thres: str = 'auto',
        data_kw: Optional[dict] = None,
        batch_size: int = 64,
        num_workers: int = 6,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        self.benchmark = benchmark
        self.thres = thres
        self.data_kw = data_kw if data_kw is not None else {}
        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local)

    def prepare_data(self):
        if self.copy_data_local is not None:
            from src.data.real.datasets.copy_local import get_file_list
            droot, file_list = get_file_list(self.benchmark)
            self.copy_from = self.root + '/' + droot
            self.file_list = file_list
        super().prepare_data()

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_data = load_dataset(
                self.benchmark,
                self.root,
                val = False,
                thres=self.thres,
                augmentation=True,
                **self.data_kw
            )
        self.val_data = load_dataset(
            self.benchmark,
            self.root,
            val = True,
            thres=self.thres,
            augmentation=False,
            **self.data_kw,
        )

class AWADatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        thres: str = 'bbox',
        split: str = 'random',
        size: int = 256,
        normalize: Union[bool, str] = True,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        self.thres = thres
        self.size = size
        self.normalize = normalize
        self.split = split
        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local)

    def setup(self, stage):
        if stage == 'fit':
            raise RuntimeError('AWA dataset is meant for evaluation, not training.')

        self.val_data = AWADataset(self.root, self.thres, self.split, size=self.size, normalize=self.normalize)


class CUBDatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        thres: str = 'bbox',
        split: str = 'random',
        size: int = 256,
        normalize: Union[bool, str] = True,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        self.thres = thres
        self.size = size
        self.normalize = normalize
        self.split = split
        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local)

    def setup(self, stage):
        if stage == 'fit':
            raise RuntimeError('CUB dataset is meant for evaluation, not training.')

        self.val_data = CUBDataset(self.root, self.thres, self.split, size=self.size, normalize=self.normalize)

class TSSDatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        size: Union[int, tuple] = 256,
        normalize: Union[bool, str, tuple, list] = 'imagenet',
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        self.size = size
        self.normalize = normalize
        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local)

    def setup(self, stage):
        if stage == 'fit':
            raise RuntimeError('TSS dataset is meant for evaluation, not training.')

        self.val_data = TSSDataset(self.root, self.size, self.normalize)


class HPatchesDatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        normalize: Union[bool, str, tuple, list] = 'imagenet',
        batch_size: int = 64,
        num_workers: int = 6,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        self.normalize = normalize
        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local)

    def setup(self, stage):
        if stage == 'fit':
            raise RuntimeError('Hpatches dataset is meant for evaluation, not training.')
        
        # target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
        # input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel firs
        input_transform, target_transform = get_transforms(self.normalize)

        self.val_data = HPatchesDataset(
            self.root,
            os.path.join(self.root, 'hpatches_all.csv'),
            image_transform = input_transform,
            flow_transform = target_transform,
            co_transform = None,
            use_original_size=False,
        )


class SintelDatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        dtype: str = 'clean',
        normalize: Union[bool, str, tuple, list] = 'imagenet',
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        self.normalize = normalize
        self.dtype = dtype
        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local)

    def setup(self,stage):
        if stage == 'fit':
            raise RuntimeError('Sintel dataset is meant for evaluation, not training.')

        # target_transform = transforms.Compose([ArrayToTensor()])
        # input_transform = transforms.Compose([ArrayToTensor(get_float=False)])
        input_transform, target_transform = get_transforms(self.normalize)
        
        # self.val_data = MPISintelTestData(self.root,input_transform,target_transform)
        _, self.val_data = mpi_sintel(
            self.root,
            source_image_transform=input_transform,
            target_image_transform=input_transform,
            flow_transform=target_transform,
            load_occlusion_mask=True,
            dstype=self.dtype,
        )


class KITTIDatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        occ: bool = False,
        onlyocc: bool = False,
        normalize: Union[bool, str, tuple, list] = 'imagenet',
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        self.occ = occ
        self.onlyocc = onlyocc
        self.normalize = normalize
        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local)

    def setup(self,stage):
        if stage == 'fit':
            raise RuntimeError('KITTI dataset is meant for evaluation, not training.')

        # target_transform = transforms.Compose([ArrayToTensor()])
        # input_transform = transforms.Compose([ArrayToTensor(get_float=False)])
        input_transform, target_transform = get_transforms(self.normalize)
        
        if self.onlyocc:
            _, self.val_data = KITTI_only_occ(
                self.root,
                source_image_transform=input_transform,
                target_image_transform=input_transform,
                flow_transform=target_transform,
            )
        elif self.occ:
            _, self.val_data = KITTI_occ(
                self.root,
                source_image_transform=input_transform,
                target_image_transform=input_transform,
                flow_transform=target_transform,
            )
        else:
            _, self.val_data = KITTI_noc(
                self.root,
                source_image_transform=input_transform,
                target_image_transform=input_transform,
                flow_transform=target_transform,
            )


class WarpSupervisionDatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        dataset_config: dict,
        batch_img_keys: Optional[Union[list, tuple]] = None,
        warp_params: Optional[Union[str, dict]] = 'basic',
        warp_scale_factor: float = 1.0,
        warp_scale_interval: Union[int, float] = 2,
        flow_map: Literal['absolute', 'relative'] = 'relative',
        size: Union[int, Tuple[int, int]] = (256, 256),
        crop_scale: float = 1.0,
        normalize: Union[str, list, tuple] = 'imagenet',
        batch_size: int = 64,
        num_workers: int = 6,
        copy_data_local: Optional[str] = None,
    ):
        self.dataset_config = dataset_config
        self.batch_img_keys = batch_img_keys
        self.warp_params = warp_params
        self.warp_scale_factor = warp_scale_factor
        self.warp_scale_interval = warp_scale_interval
        self.flow_map = flow_map
        self.size = make2(size)
        self.crop_scale = crop_scale
        self.normalize = normalize
        super().__init__(root, batch_size, num_workers, copy_data_local)

    def post_init(self):
        if isinstance(self.normalize, str) and self.normalize == 'center':
            self.normalize_vals = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        elif isinstance(self.normalize, str) and self.normalize == 'imagenet':
            from src import imagenet_stats
            self.normalize_vals = imagenet_stats
        else:
            self.normalize_vals = self.normalize

        kw = self.warp_params or {}
        if isinstance(kw, str):
            if kw == 'basic':
                kw = {
                    'affine': {},
                    'homography': {},
                    'tps': {},
                    'elastic': {},
                    'transform_src': 0.5,
                }
            elif kw == 'advanced':
                pass

        self.warper = SyntheticPairWarper(**kw)

        self.photo_aug = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.02),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(5, (0.2, 2))], 0.2),
        ])

    def prepare_data(self):
        if self.copy_data_local is not None:
            from src.data.real.datasets.copy_local import get_file_list
            droot, file_list = get_file_list(self.benchmark)
            self.copy_from = self.root + '/' + droot
            self.file_list = file_list
        super().prepare_data()

    def setup(self, stage='fit'):
        name = self.dataset_config.pop('name')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size, antialias=True),
            torchvision.transforms.ToTensor(),
            # self.warper,
            # _wrap(self.photo_aug, [1]),
            # _wrap(torchvision.transforms.Normalize(*self.normalize_vals), [0, 1]),
        ])
        if stage == 'fit':
            self.train_data = load_dataset(name, self.root, transform=transform, **self.dataset_config)
        self.val_data = load_dataset(name, self.root, val=True, transform=transform, **self.dataset_config)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int):
        if self.batch_img_keys is None:
            batch = {'src': batch}
            keys = ['src']
        else:
            keys = self.batch_img_keys

        for key in keys:
            imgs = batch[key]
            src, trg, field = self.warper(imgs)
            trg = self.photo_aug(trg)
            if self.flow_map == 'relative':
                field = flow.convert_mapping_to_flow(field, is_normalized=False)
            batch[key] = torchvision.transforms.functional.normalize(src, *self.normalize_vals)
            batch[key + '_warp'] = torchvision.transforms.functional.normalize(trg, *self.normalize_vals)
            batch[key + '_flow'] = field
        # batch = {
        #     'src': batch[0],
        #     'src_warp': batch[1],
        #     'src_flow': batch[2],
        # }

        return batch
    
    def adjust_warp_strength(self, batch_idx, epoch, num_batches):
        interval = self.warp_scale_interval
        idx = epoch * num_batches + batch_idx
        update = int(num_batches * interval)
        
        if idx > 0 and idx % update == 0:
            self.warper.homography['sigma'] = min(0.4, self.warper.homography['sigma'] * self.warp_scale_factor)
            self.warper.tps['sigma'] = min(0.2, self.warper.tps['sigma'] * self.warp_scale_factor)
            jitter = self.photo_aug.transforms[0]
            for k in ('brightness', 'contrast', 'saturation'):
                v = getattr(jitter, k)
                v = tuple([(x - 1) * self.warp_scale_factor + 1 for x in v])
                setattr(jitter, k, v)
            # jitter.hue = tuple([x * self.warp_scale_factor for x in jitter.hue])

    def predict_dataloader(self):
        return self.val_dataloader()


def _wrap(tform, keys):
    def wrapped_transform(inputs):
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        for key in keys:
            inputs[key] = tform(inputs[key])
            return inputs
    return wrapped_transform


class WarpOnlineSupervisionDatamodule(BaseDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        dataset_config: dict,
        batch_img_keys: Optional[Union[list, tuple]] = None,
        warp_params: Optional[Union[str, dict]] = 'basic',
        warp_scale_factor: float = 1.0,
        warp_scale_interval: Union[int, float] = 2,
        flow_map: Literal['absolute', 'relative'] = 'relative',
        size: Union[int, Tuple[int, int]] = (256, 256),
        crop_scale: float = 1.0,
        normalize: Union[str, list, tuple] = 'imagenet',
        batch_size: int = 64,
        num_workers: int = 6,
        copy_data_local: Optional[str] = None,
        clip_range: Optional[dict[str, float]] = None,
    ):
        self.dataset_config = dataset_config
        self.batch_img_keys = batch_img_keys
        self.warp_params = warp_params
        self.warp_scale_factor = warp_scale_factor
        self.warp_scale_interval = warp_scale_interval
        self.flow_map = flow_map
        self.size = make2(size)
        self.crop_scale = crop_scale
        self.normalize = normalize
        self.clip_range = clip_range
        super().__init__(root, batch_size, num_workers, copy_data_local)

    def post_init(self):
        if isinstance(self.normalize, str) and self.normalize == 'center':
            self.normalize_vals = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        elif isinstance(self.normalize, str) and self.normalize == 'imagenet':
            from src import imagenet_stats
            self.normalize_vals = imagenet_stats
        else:
            self.normalize_vals = self.normalize

        kw = self.warp_params or {}
        if isinstance(kw, str):
            if kw == 'basic':
                kw = {
                    'affine': {},
                    'homography': {},
                    'tps': {},
                    'elastic': {},
                    'transform_src': 0.5,
                }
            elif kw == 'advanced':
                pass

        self.warper = SyntheticPairWarper(**kw)

        self.photo_aug = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.02),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(5, (0.2, 2))], 0.2),
        ])

    def prepare_data(self):
        if self.copy_data_local is not None:
            from src.data.real.datasets.copy_local import get_file_list
            droot, file_list = get_file_list(self.benchmark)
            self.copy_from = self.root + '/' + droot
            self.file_list = file_list
        super().prepare_data()

    def setup(self, stage='fit'):
        name = self.dataset_config.pop('name')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size, antialias=True),
            torchvision.transforms.ToTensor(),
            # self.warper,
            # _wrap(self.photo_aug, [1]),
            # _wrap(torchvision.transforms.Normalize(*self.normalize_vals), [0, 1]),
        ])
        # if stage == 'fit':
        #     self.train_data = load_dataset(name, self.root, transform=transform, **self.dataset_config)
        #             # Setup synthetic training data from config
        if stage == 'fit':
            synth_class_path = self.dataset_config.get('class_path', 'src.data.synth.datamodule.OnlineComponentsDatamodule')
            synth_init_args = self.dataset_config.get('init_args', {})
            
            # Dynamically load synthetic datamodule class
            from importlib import import_module
            module_path, class_name = synth_class_path.rsplit('.', 1)
            module = import_module(module_path)
            SynthDatamoduleClass = getattr(module, class_name)
            
            self.synth_datamodule = SynthDatamoduleClass(
                **synth_init_args
            )
            self.synth_datamodule.post_init()
            self.synth_datamodule.setup('fit')
            self.train_data = self.synth_datamodule.train_data

        val_dataset_config = self.dataset_config.pop('val_dataset_config')
        val_name = val_dataset_config.pop('name')
        val_root = val_dataset_config.pop('root')
        self.val_data = load_dataset(val_name, val_root, val=True, transform=transform, **val_dataset_config)
        

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int):

        suffix = ""
        if self.trainer is not None and self.trainer.training:
            # print("Calling OnlineComponentsDatamodule's on_after_batch_transfer for TRAINING")
            batch = self.synth_datamodule.on_after_batch_transfer(batch, dataloader_idx)
        else:
            suffix = "_img"

        if self.batch_img_keys is None:
            batch = {'src': batch}
            keys = ['src']
        else:
            keys = self.batch_img_keys

        for key in keys:
            key = key
            imgs = batch[key + suffix]
            if self.clip_range is not None:
                imgs = torch.clip((imgs - self.clip_range['min']) / (self.clip_range['max'] - self.clip_range['min']), 0, 1)
                
            src, trg, field = self.warper(imgs)
            src = (src - trg.min()) / (trg.max() - trg.min())


            trg = self.photo_aug(trg)
            if self.flow_map == 'relative':
                field = flow.convert_mapping_to_flow(field, is_normalized=False)
            if self.normalize != "None":
                batch[key] = torchvision.transforms.functional.normalize(src, *self.normalize_vals)
                batch[key + '_warp'] = torchvision.transforms.functional.normalize(trg, *self.normalize_vals)
            else:
                batch[key] = src
                batch[key + '_warp'] = trg
            batch[key + '_flow'] = field
        # batch = {
        #     'src': batch[0],
        #     'src_warp': batch[1],
        #     'src_flow': batch[2],
        # }

        return batch
    
    def adjust_warp_strength(self, batch_idx, epoch, num_batches):
        interval = self.warp_scale_interval
        idx = epoch * num_batches + batch_idx
        update = int(num_batches * interval)
        
        if idx > 0 and idx % update == 0:
            self.warper.homography['sigma'] = min(0.4, self.warper.homography['sigma'] * self.warp_scale_factor)
            self.warper.tps['sigma'] = min(0.2, self.warper.tps['sigma'] * self.warp_scale_factor)
            jitter = self.photo_aug.transforms[0]
            for k in ('brightness', 'contrast', 'saturation'):
                v = getattr(jitter, k)
                v = tuple([(x - 1) * self.warp_scale_factor + 1 for x in v])
                setattr(jitter, k, v)
            # jitter.hue = tuple([x * self.warp_scale_factor for x in jitter.hue])

    def predict_dataloader(self):
        return self.val_dataloader()
    
    def train_dataloader(self):
        train_loader = self.synth_datamodule.train_dataloader()
        self.train_dataloader_instance = train_loader
        return train_loader


