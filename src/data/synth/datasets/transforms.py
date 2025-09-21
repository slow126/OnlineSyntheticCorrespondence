'''Transforms to be applied to a pair of images and their matching points'''
import random
from typing import Callable, List, Union

import numpy as np
import torch
import torchvision
from torchvision.transforms.transforms import _setup_size


_size_err_msg = 'Please provide only two values (H, W) for size'


class Compose(object):
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, *data):
        for t in self.transforms:
            data = t(*data)
        return data


class ToTensor(object):
    def __init__(self, normalize_pixels: bool=True, channels_first: bool=True):
        self.normalize_pixels = normalize_pixels
        self.channels_first = channels_first

    def _convert(self, x):
        if isinstance(x, torch.Tensor):
            return x
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        x = torch.from_numpy(x)
        return x

    def _normalize(self, x):
        return x.mul(1. / 255.)

    def __call__(self, img1, img2, matches):
        img1 = self._convert(img1)
        img2 = self._convert(img2)
        matches = self._convert(matches)

        if self.channels_first:
            img1 = img1.permute(2,0,1)
            img2 = img2.permute(2,0,1)

        if self.normalize_pixels:
            img1 = self._normalize(img1)
            img2 = self._normalize(img2)
        
        return img1, img2, matches


class RandomCrop(object):
    '''Crop images and remap the correspondences. If either point in a pair falls outside the
    cropped area, it is removed.
    
    Expects inputs to be `torch.Tensor`s. The images should have shape[..., H, W] and the
    point matches should have shape[N, 2, 2] (number of points, yx coords, pair).
    '''
    def __init__(self, size: Union[int, list, tuple]):
        if isinstance(size, int):
            size = (size, size)
        self.size = _setup_size(size, _size_err_msg)

    def _crop_values(self, x):
        _, h, w = x.shape
        ch, cw = self.size

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)

        return (top, left, ch, cw)

    def _crop(self, x, values):
        cropped = torchvision.transforms.functional.crop(x, *values)
        return cropped

    def _adjust_keypoints(self, matches, v1, v2):
        dev = matches.device
        shift = torch.tensor([*v1[:2], *v2[:2]], dtype=torch.int64, device=dev).view(1, 2, 2).permute(0, 2, 1)
        cut = torch.tensor([*v1[2:], *v2[2:]], dtype=torch.int64, device=dev).view(1, 2, 2).permute(0, 2, 1)
        matches = matches.sub(shift)

        mask = matches.lt(cut - 1).view(-1, 4).all(1) & matches.ge(0).view(-1, 4).all(1)
        matches = matches[mask].view(-1, 2, 2)
        return matches

    def __call__(self, img1, img2, matches):
        v1 = self._crop_values(img1)
        v2 = self._crop_values(img2)

        img1 = self._crop(img1, v1)
        img2 = self._crop(img2, v2)
        matches = self._adjust_keypoints(matches, v1, v2)

        return img1, img2, matches
    

class Resize(object):
    '''Resize images and remap the correspondences.

    Expects inputs to be `torch.Tensor`s. The images should have shape[..., H, W] and the
    point matches should have shape[N, 2, 2] (number of points, yx coords, pair).
    '''
    def __init__(self, size: Union[int, list, tuple], as_int: bool=False):
        if isinstance(size, int):
            size = (size, size)
        self.size = _setup_size(size, _size_err_msg)
        self.as_int = as_int

    def __call__(self, img1, img2, matches):
        h1, w1 = img1.shape[-2:]
        h2, w2 = img2.shape[-2:]
        img1 = torchvision.transforms.functional.resize(img1, self.size, antialias=True)
        img2 = torchvision.transforms.functional.resize(img2, self.size, antialias=True)

        s = torch.tensor([h1, w1, h2, w2]).view(1, 2, 2)
        r = torch.tensor(self.size, device=matches.device).view(1, 2, 1).expand(-1, -1, 2)
        matches = matches.mul((r - 1) / (s - 1))

        if self.as_int:
            matches = matches.long()

        return img1, img2, matches


class ScalePoints(object):
    def __init__(self, size: Union[int, list, tuple], as_int: bool=False):
        if isinstance(size, int):
            size = (size, size)
        self.size = _setup_size(size, _size_err_msg)
        self.as_int = as_int
        
    def __call__(self, img1, img2, matches):
        h1, w1 = img1.shape[-2:]
        h2, w2 = img2.shape[-2:]

        s = torch.tensor([h1, w1, h2, w2]).view(1, 2, 2)
        r = torch.tensor(self.size, device=matches.device).view(1, 2, 1).expand(-1, -1, 2)
        matches = matches.mul((r - 1) / (s - 1))

        if self.as_int:
            matches = matches.long()

        return img1, img2, matches


class ColorJitter(object):
    '''
    '''
    def __init__(
            self,
            brightness: Union[float, list, tuple]=0.0,
            saturation: Union[float, list, tuple]=0.0,
            contrast: Union[float, list, tuple]=0.0,
            hue: Union[float, list, tuple]=0.0,
        ):
        self.transform = torchvision.transforms.ColorJitter(brightness, saturation, contrast, hue)

    def __call__(self, img1, img2, matches):
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, matches


class Normalize(object):
    '''
    '''
    def __init__(self, shift: Union[float, list, tuple]=0.0, scale: Union[float, list, tuple]=1.0):
        self.shift = [shift] if isinstance(shift, float) else shift
        self.scale = [scale] if isinstance(scale, float) else scale

    def __call__(self, img1, img2, matches):
        img1 = torchvision.transforms.functional.normalize(img1, self.shift, self.scale)
        img2 = torchvision.transforms.functional.normalize(img2, self.shift, self.scale)
        return img1, img2, matches