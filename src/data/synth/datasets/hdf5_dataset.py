from collections import Counter
import json
import math
import mmap
from pathlib import Path
import pickle
from typing import Any, List, Literal, Optional, Union

import blosc
import numpy as np
import torch

from src.data.image_ops import img_from_byte_array
from .base import ComponentsBase


def path_one_of(root, *options):
    for opt in options:
        p = root.joinpath(opt)
        if p.exists():
            return p
    raise RuntimeError(f'None of the files in {options} were located under {str(root)}')


class Hdf5CorrespondenceDataset(object):
    def __init__(self, root, stems=None, transform=None, obj_limit=None, imgs_per_obj=None, offset=0):
        self.root = Path(root)

        self.data_files = sorted(self.root.glob('*.hdf5'))
        if stems is not None:
            self.data_files = [x for x in self.data_files if x.stem in stems]
        self.meta_files = [x.with_suffix('.json') for x in self.data_files]

        if len(self.data_files) == 0:
            raise RuntimeError(f'No hdf5 files found at {root}')
        if any(not x.is_file() for x in self.meta_files):
            raise RuntimeError(f'Not all hdf5 files have a corresponding json meta file')

        obj_limit = obj_limit or int(1e9)
        per_obj = imgs_per_obj or int(1e9)

        obj_count = Counter()

        self.pairs = []
        self.samples = {}
        for i, meta in enumerate(self.meta_files):
            j = json.load(meta.open('r'))
            for k in j['images']:
                obj = k[:k.find('/')]
                n = obj_count[obj]
                obj_count.update((obj,))
                if len(obj_count) <= offset or len(obj_count) > offset + obj_limit: continue
                if n >= per_obj: continue
                self.samples[k] = [i, *j['images'][k]]

            for k in j['flow']:
                obj, kk = k.split('/')
                id1, id2 = [f'{obj}/{x}.jpg' for x in kk.split('-')]
                if id1 in self.samples and id2 in self.samples:
                    self.pairs.append([i, k, *j['flow'][k]])

        self._len = len(self.pairs)

        self.transform = transform

    def __len__(self):
        return self._len

    def fetch_from_file(self, idx):
        # fetch data from an hdf5 file using memory mapping to grab the necessary bytes without
        # going through the hdf5 interface (more effecient, no issues when reading from multiple processes)
        pair = self.pairs[idx]
        name = pair[1]
        obj, pname = name.split('/')
        id1, id2 = pname.split('-')

        data = []
        h5_path = self.data_files[pair[0]]
        with h5_path.open('rb') as fp:
            fileno = fp.fileno()
            mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
            # load images
            for id in (id1, id2):
                sample = self.samples[f'{obj}/{id}.jpg']
                offset = sample[1]
                shape = sample[2]
                dtype = sample[3]
                img = np.frombuffer(mapping, dtype=dtype, count=np.prod(shape), offset=offset)
                # images are stored in file as jpeg encoded bytes
                img = img_from_byte_array(img)
                data.append(img)
            # load point correspondences
            offset = pair[2]
            shape = pair[3]
            dtype = pair[4]
            matches = np.frombuffer(mapping, dtype=dtype, count=np.prod(shape), offset=offset)
            # matches are stored in file as compressed numpy array bytes
            matches = blosc.unpack_array(matches.tobytes())
            data.append(matches)

        return data

    def __getitem__(self, idx):
        data = self.fetch_from_file(idx)
        if self.transform:
            data = self.transform(*data)
        return data
    

class HDF5ImageGeometryDataset(object):
    def __init__(
        self,
        root,
        size: int = 256,
    ):
        self.root = Path(root)
        self.size = size

        self.data_file = self.root.joinpath('data.hdf5')
        self.meta_file = self.data_file.with_suffix('.json')
        self.pairs_file = self.root.joinpath('pairs.pkl')

        if not self.data_file.exists():
            raise RuntimeError(f'No hdf5 data file found at {root}')
        if not self.meta_file.exists():
            raise RuntimeError(f'No metadata json file found at {root}')
        if not self.pairs_file.exists():
            raise RuntimeError(f'No pairs pickle file found at {root}')

        self.pairs = pickle.load(open(self.pairs_file, 'rb'))
        self.meta = json.load(open(self.meta_file))
        
    def __len__(self):
        return len(self.pairs)

    def center_crop(self, x: np.array):
        p = (x.shape[0] - self.size) // 2
        return np.ascontiguousarray(x[p:p+self.size, p:p+self.size])

    def _from_file(self, file_map, sample, is_jpg):
        # file_map is a memory mapped file pointer
        # load image and geometry
        offset = sample[0]
        shape = sample[1]
        dtype = sample[2]
        arr = np.frombuffer(file_map, dtype=dtype, count=math.prod(shape), offset=offset)
        if is_jpg:
            arr = np.asarray(img_from_byte_array(arr))
        else:
            arr = blosc.unpack_array(arr.tobytes())
        return arr
    
    def fetch_from_file(self, idx):
        pair = self.pairs[idx]
        obj = pair['obj']
        id1 = pair['id1']
        id2 = pair['id2']

        data_pair = []
        with self.data_file.open('rb') as fp:
            fileno = fp.fileno()
            mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
            for id in (id1, id2):
                data = {}
                img_sample = self.meta['images'][f'c_{obj}/{id}.jpg']
                geo_sample = self.meta['geometry'][f'c_{obj}/{id}']
                for sample, key in zip((img_sample, geo_sample), ('image', 'geometry')):
                    is_jpg = key == 'image'
                    arr = self._from_file(mapping, sample, is_jpg)
                    arr = self.center_crop(arr)
                    data[key] = arr
                data_pair.append(data)
        return data_pair
    
    def __getitem__(self, idx):
        pair = self.fetch_from_file(idx)
        return pair


class HDF5ComponentsBase(ComponentsBase):
    '''Base class for loading and transforming goemetry and normal maps.
    '''
    def _from_file(self, file_map, sample):
        # file_map is a memory mapped file pointer
        # load geometry, normals, camera
        d = {}
        for key in ('geometry', 'normals'):
            offset = sample[key][0]
            shape = sample[key][1]
            dtype = sample[key][2]
            arr = np.frombuffer(file_map, dtype=dtype, count=math.prod(shape), offset=offset)
            arr = blosc.unpack_array(arr.tobytes())
            d[key] = arr
        d['camera'] = sample['camera']

        return d


class Hdf5ComponentsDataset(HDF5ComponentsBase):
    '''Dataset for loading single samples of geometry and normals.
    '''
    def __init__(
        self,
        root,
        size: int = 256,
        crop: Literal['center', 'none', 'random'] = 'center',
        seed: int = 987654321,
    ):
        super().__init__(size, crop, seed)
        self.root = root

        self.data_file = self.root.joinpath('data.hdf5')
        self.meta_file = self.data_file.with_suffix('.json')

        if not self.data_file.exists():
            raise RuntimeError(f'No hdf5 data file found at {root}')
        if not self.meta_file.exists():
            raise RuntimeError(f'No metadata json file found at {root}')

        obj_meta = pickle.load(path_one_of(self.root, 'metadata.pkl', 'metadata.obj').open('rb'))

        self.objects = []
        self.samples = []
        j = json.load(self.meta_file.open('r'))
        for k in j['geometry']:
            obj, view = k.split('/')
            m = obj_meta['objects'][obj][view]
            cam = m['camera_position']
            self.samples.append({
                'geometry': j['geometry'][k],
                'normals': j['normals'][k],
                'camera': np.array(cam, dtype=np.float32),
            })
            self.objects.append(k)

    def __len__(self):
        return len(self.samples)

    def transform(self, data):
        img_size = data['geometry'].shape[:2]
        crop_ps = self.get_crop(img_size)

        super().transform(data, crop_ps)
        
        return data
    
    def fetch_from_file(self, idx):
        sample = self.samples[idx]

        with self.data_file.open('rb') as fp:
            fileno = fp.fileno()
            mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
            # load geometry, normals, camera
            sample = self.samples[idx]
            data = self._from_file(mapping, sample)

        return data

    def __getitem__(self, idx):
        data = self.samples[idx]
        data = self.fetch_from_file(idx)
        data = self.transform(data)
        return data


class Hdf5CorrespondenceComponentsDataset(HDF5ComponentsBase):
    '''Dataset for loading paired sets of geometry/normals'''
    def __init__(
        self,
        root: Union[str, Path],
        overlap_sampling: Literal['uniform', 'normal'] = 'uniform',
        overlap_params: Union[list, tuple] = (0.4, 1.0),
        complexity_params: Optional[Union[list, tuple]] = None,
        schedule: Optional[dict] = None,
        num_samples: Optional[int] = None,
        size: int = 256,
        crop: Literal['random', 'center'] = 'random',
        random_flip: Optional[float] = None,
        random_swap: bool = True,
        textures: Any = None,
        seed: int = 987654321,
    ):
        super().__init__(size, crop, seed)
        self.root = Path(root)
        self.size = size
        self.textures = textures
        self.cropping = crop
        self.random_swap = random_swap
        self.random_flip = random_flip

        self.data_files = sorted(self.root.glob('*.hdf5'))
        self.meta_files = [x.with_suffix('.json') for x in self.data_files]
        self.obj_meta_files = [
            x.with_stem(x.stem.replace('data', 'metadata')).with_suffix('.pkl')
            for x in self.data_files
        ]
        self.pair_files = [
            x.with_stem(x.stem.replace('metadata', 'overlaps'))
            for x in self.obj_meta_files
        ]

        if len(self.data_files) == 0:
            raise RuntimeError(f'No hdf5 files found at {root}')
        if any(not x.is_file() for x in self.meta_files):
            raise RuntimeError(f'Not all hdf5 files have a corresponding json meta file')
        if any(not x.is_file() for x in self.obj_meta_files):
            raise RuntimeError(f'Not all hdf5 files have a corresponding metadata pkl file')
        if any(not x.is_file() for x in self.pair_files):
            raise RuntimeError(f'Not all hdf5 files have a corresponding overlaps pkl file')

        if complexity_params is not None:
            self.complexity_files = [
                x.with_stem(x.stem.replace('metadata', 'curvature_allimages'))
                for x in self.obj_meta_files
            ]
            if any(not x.is_file() for x in self.complexity_files):
                raise RuntimeError(f'Not all hdf5 files have a corresponding curvature pkl file')

            # NOTE: ONLY WORKS WITH A SINGLE FILE RIGHT NOW
            self.obj_complexity = ObjectComplexity(
                self.complexity_files[0], complexity_params, num_samples, self.rng_np, schedule=schedule
            )
            self.complexity_params = complexity_params
        else:
            self.obj_complexity = None

        self.view_overlaps = ViewOverlap(
            self.pair_files, overlap_sampling, overlap_params, num_samples, self.rng_np, schedule=schedule
        )

        self.overlap_sampling = overlap_sampling
        self.overlap_params = overlap_params
        self.set_sample_pairs()

        self.num_samples = num_samples if num_samples is not None else self.view_overlaps.num_samples

        self.samples = []
        for i, meta_file in enumerate(self.meta_files):
            meta = json.load(meta_file.open('rb'))
            obj_meta = pickle.load(self.obj_meta_files[i].open('rb'))

            self.samples.append({})
            for k in meta['geometry']:
                obj, view = k.split('/')
                m = obj_meta['objects'][obj][view]
                cam = m['camera_position']
                self.samples[i][k] = {
                    'geometry': meta['geometry'][k],
                    'normals': meta['normals'][k],
                    'camera': np.array(cam, dtype=np.float32),
                }

    def __len__(self):
        return len(self.pair_index)

    def set_sample_pairs(self, epoch=None):
        if self.obj_complexity is not None:
            # Using object complexity
            self.obj_complexity.set_sample_pairs(self.complexity_params)
            self.pairs = self.obj_complexity.pairs
            self.pair_index = self.obj_complexity.pair_index
        else:
            self.view_overlaps.set_sample_pairs(self.overlap_params)
            self.pairs = self.view_overlaps.pairs
            self.pair_index = self.view_overlaps.pair_index

    def fetch_from_file(self, idx):
        # fetch data from an hdf5 file using memory mapping to grab the necessary bytes without
        # going through the hdf5 interface (more effecient, no issues when reading from multiple processes)
        pair = self.pairs[idx]
        file_idx = pair[0]
        obj = pair[1]
        id1, id2 = pair[2].split('-')

        data = []
        with self.data_files[file_idx].open('rb') as fp:
            fileno = fp.fileno()
            mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
            # load geometry, normals, camera
            for id in (id1, id2):
                sample = self.samples[file_idx][f'{obj}/{id}']
                d = self._from_file(mapping, sample)
                data.append(d)

        return data

    def transform(self, data):
        img_size = data[0]['geometry'].shape[:2]
        crop_ps = self.get_crop(img_size)

        for i in range(len(data)):
            super().transform(data[i], crop_ps)

        if self.random_flip and torch.rand(1, generator=self.rng) < self.random_flip:
            # random horizontal flip
            for k in ('geometry', 'normals'):
                data[1][k] = data[1][k].flip(dims=(1,)).contiguous()

        # randomly swap from (src, trg) to (trg, src)
        if self.random_swap and torch.rand(1, generator=self.rng) < 0.5:
            data[0], data[1] = data[1], data[0]
        
        return data

    def __getitem__(self, idx):
        idx = self.pair_index[idx % self.num_samples]
        data = self.fetch_from_file(idx)
        data = self.transform(data)
        return data


class SampleRangeSchedule(object):
    def __init__(
        self,
        steps: int,
        type: Literal['one-sided', 'two-sided'] = 'one-sided',
        val_range: Union[list, tuple] = (0, 1),
        overlap: int = 1,
        low_to_high: bool = True,
    ):
        low, high = val_range
        if type == 'one-sided':
            if low_to_high:
                start = low + (overlap + 1) * (high - low) / steps
                step_vals = np.linspace(start, high, steps)
                self.step_schedule = [(low, v) for v in step_vals]
            else:  # high to low
                start = high - (overlap + 1) * (high - low) / steps
                step_vals = np.linspace(start, low, steps)
                self.step_schedule = [(v, high) for v in step_vals]
        elif type == 'two-sided':
            if low_to_high:
                step_vals = np.linspace(low, high, steps + overlap + 1)
                self.step_schedule = list(zip(step_vals[:steps], step_vals[overlap + 1:]))
            else:  # high to low
                step_vals = np.linspace(high, low, steps + overlap + 1)
                self.step_schedule = list(zip(step_vals[overlap + 1:], step_vals[:steps]))
        else:
            raise RuntimeError(f'Invalid schedule type: {type}')
        
        self.idx = 0

    def next_step(self):
        vals = self[self.idx]
        self.idx += 1
        if self.idx == len(self):
            self.idx -= 1
            print('Reached the end of the sample range schedule')
        return vals

    def __len__(self):
        return len(self.step_schedule)
        
    def __getitem__(self, idx):
        return self.step_schedule[idx]


class ViewOverlap(object):
    def __init__(
        self,
        file_paths: List[Path],
        overlap_sampling: str,
        overlap_params: Union[list, tuple],
        num_samples: int,
        rng: np.random.Generator,
        schedule: Optional[dict] = None,
    ):
        self.overlap_sampling = overlap_sampling
        self.overlap_params = overlap_params

        self.pairs = []
        for i, f in enumerate(file_paths):
            pairs = pickle.load(f.open('rb'))
            pairs = [(i, *x) for x in pairs] 
            self.pairs.append(pairs)
        if len(self.pairs) > 1:
            import itertools
            self.pairs = sorted(itertools.chain(*self.pairs), key=lambda x: x[-1], reverse=True)
        else:
            self.pairs = self.pairs[0]

        if schedule is not None:
            self.schedule = SampleRangeSchedule(val_range=overlap_params, **schedule)
        else:
            self.schedule = None

        self.num_samples = num_samples
        self.rng = rng

    def set_sample_pairs(self, overlap_params=None, step=None):
        if self.schedule is not None:
            if step is None:
                overlap_params = self.schedule.next_step()
            else:
                overlap_params = self.schedule[step]
        elif overlap_params is None:
            overlap_params = self.overlap_params

        # NOTE: we search with negative values because the overlaps are sorted in reverse order
        if self.overlap_sampling == 'uniform':
            import bisect
            low, high = overlap_params
            if low >= high:
                raise ValueError(f'In overlap range, found low > high ({low} > {high})')
            idx_start = bisect.bisect_right(self.pairs, -high, key=lambda x: -x[-1])
            idx_end = bisect.bisect_left(self.pairs, -low, key=lambda x: -x[-1])
            if self.num_samples is None:
                self.num_samples = idx_end - idx_start
            idx = range(idx_start, idx_end)
            if len(idx) < self.num_samples:
                self.pair_index = self.rng.choice(idx, self.num_samples, replace=True).tolist()
            elif len(self.pairs) > self.num_samples:
                self.pair_index = self.rng.choice(idx, self.num_samples, replace=False).tolist()
            else:
                self.pair_index = list(idx)
        elif self.overlap_sampling == 'normal':
            # sample pairs based on a truncated normal distribution of overlaps
            from scipy.stats import truncnorm
            if self.num_samples is None:
                raise ValueError('Must provide a value for num_samples when overlap_sampling="normal"')
            mu, sig = overlap_params
            a, b = (-1 + mu) / sig, (0 + mu) / sig
            rvals = truncnorm.rvs(a, b, loc=-mu, scale=sig, size=self.num_samples, random_state=self.rng_np)
            vals = np.array([-x[-1] for x in self.pairs])
            self.pair_index = np.searchsorted(vals, rvals, side='left').tolist()


class ObjectComplexity(object):
    def __init__(
        self,
        file_path: Path,
        quantile_range: Union[list, tuple],
        num_samples: int,
        rng: np.random.Generator,
        schedule: Optional[dict] = None,
    ):
        comp = pickle.load(file_path.open('rb'))
        comp = [(x[0], x[1], x[2][0]) for x in comp]
        
        by_obj = {}
        for x in comp:
            if x[0] not in by_obj:
                by_obj[x[0]] = []
            by_obj[x[0]].append(x)

        n_obj = len(by_obj)
        n_img = len(comp) // n_obj
        m2 = np.array([x[2] for x in comp])
        m3 = m2.reshape(n_obj, n_img).mean(1)
        idx = m3.argsort()

        quantiles = np.quantile(m3, np.linspace(0, 1, 100))
        self.quantile_idx = np.searchsorted(m3[idx], quantiles)

        self.by_obj = by_obj
        self.obj_names = list(by_obj.keys())

        self.comp = comp
        self.m3 = m3
        self.idx = idx

        if schedule is not None:
            self.schedule = SampleRangeSchedule(val_range=quantile_range, **schedule)
        else:
            self.schedule = None

        self.quantile_range = quantile_range
        self.num_samples = num_samples
        self.rng = rng

    def set_sample_pairs(self, quantile_range=None, step=None):
        if self.schedule is not None:
            if step is None:
                quantile_range = self.schedule.next_step()
            else:
                quantile_range = self.schedule[step]
        elif quantile_range is None:
            quantile_range = self.quantile_range
        # print(f'QUANTILE RANGE {quantile_range}')

        min_quantile, max_quantile = quantile_range
        min_quantile = int(min_quantile)
        max_quantile = min(int(max_quantile), len(self.quantile_idx) - 1)

        low_idx = self.quantile_idx[min_quantile]
        high_idx = self.quantile_idx[max_quantile]
        low = self.m3[self.idx[low_idx]]
        high = self.m3[self.idx[high_idx]]

        pairs = []
        for i in self.idx[low_idx:high_idx]:
            valid = []
            for x in self.by_obj[self.obj_names[i]]:
                if low <= x[2] <= high:
                    valid.append(x)
            vp = []
            for j in range(len(valid) - 1):
                for k in range(j + 1, len(valid)):
                    # NOTE: hardcoding the file index for now, but THIS SHOULD BE FIXED
                    vp.append((0, valid[j][0], f'{valid[j][1]}-{valid[k][1]}'))
            pairs.extend(vp)
        
        n = len(pairs)

        if n > self.num_samples:
            idx = self.rng.choice(n, self.num_samples, replace=False)
        elif n < self.num_samples:
            idx = self.rng.choice(n, self.num_samples, replace=True)
        
        self.pairs = pairs
        self.pair_index = idx