from typing import Literal

import numpy as np
import torch


class ComponentsBase(object):
    '''Base class for loading and transforming goemetry and normal maps.
    '''
    def __init__(
        self,
        size: int = 256,
        crop: Literal['center', 'none', 'random'] = 'none',
        seed: int = 987654321,
    ):
        self.size = size
        self.cropping = crop

        self.rng_np: np.random.Generator = np.random.default_rng(seed)
        self.rng: torch.Generator = torch.Generator().manual_seed(seed)

    def _sample_crop(self, in_size, out_size):
        h, w = in_size
        ch, cw = out_size
        top = torch.randint(0, h - ch, (1,), generator=self.rng).item()
        left = torch.randint(0, w - cw, (1,), generator=self.rng).item()
        return (top, left, ch, cw)

    def _center_crop(self, in_size, out_size):
        h, w = in_size
        ch, cw = out_size
        top = (h - ch) // 2
        left = (w - cw) // 2
        return (top, left, ch, cw)

    def _crop(self, input, vals):
        y, x, w, h = vals
        return np.ascontiguousarray(input[y:y + h, x:x + w])

    def get_crop(self, img_size):
        crop_ps = None
        if self.cropping != 'none':
            if self.cropping == 'random':
                crop_ps = self._sample_crop(img_size, (self.size, self.size))
            elif self.cropping == 'center':
                crop_ps = self._center_crop(img_size, (self.size, self.size))
        return crop_ps

    def transform(self, data, crop_ps=None, keys=('geometry', 'normals')):
        for k in keys:
            if crop_ps is not None:
                data[k] = self._crop(data[k], crop_ps)
            data[k] = torch.from_numpy(data[k])

        data['camera'] = torch.from_numpy(data['camera'])
        
        if data.get('max_num_objects') is not None:
            data['max_num_objects'] = torch.tensor(data['max_num_objects'])

        return data
    