r"""SPair-71k dataset"""
import json
import os

from PIL import Image
import torch
import tqdm

from .dataset import CorrespondenceDataset
from .transforms import random_crop


__all__ = [
    'SPairDataset',
]


class SPairDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(
            self,
            benchmark,
            datapath,
            thres,
            split,
            augmentation=True,
            feature_size=64,
            receptive_field_size=11,
            bidirectional_flows=False,
            normalize='imagenet',
        ):
        r"""SPair-71k dataset constructor"""
        super(SPairDataset, self).__init__(
            benchmark, datapath, thres, split, augmentation, feature_size, receptive_field_size, normalize=normalize
        )

        self.train_data = open(self.spt_path).read().split('\n')
        self.train_data = self.train_data[:len(self.train_data) - 1]
        self.src_imnames = list(map(lambda x: x.split('-')[1] + '.jpg', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[0] + '.jpg', self.train_data))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()

        annos = json.load(open(os.path.join(self.ann_path, 'pair_annotations.json')))
        anntn_files = []
        for data_name in self.train_data:
            anntn_files.append(annos[data_name])

        def _make(k, files, proc='tensor', transpose=True, verbose=False):
            if proc is None: proc = lambda x: x
            elif proc == 'tensor': proc = lambda x: torch.tensor(x)
            if transpose: post = lambda x: x.t().float()
            else: post = lambda x: x
            m = map(lambda x: post(proc(x[k])), files)
            if verbose:
                m = tqdm.tqdm(m, total=len(files), desc=k)
            return list(m)

        for k in 'src_kps trg_kps src_bndbox trg_bndbox'.split():
            setattr(self, k.replace('bndbox', 'bbox'), _make(k, anntn_files, proc='tensor'))
        self.cls_ids = _make('category', anntn_files, proc=lambda x: self.cls.index(x), transpose=False)

        ks = {
            'viewpoint_variation': 'vpvar',
            'scale_variation': 'scvar',
            'truncation': 'trncn',
            'occlusion': 'occln',
        }
        for k, kk in ks.items():
            setattr(self, kk, _make(k, anntn_files, proc='tensor', transpose=False))

        self.bidirectional_flows = False
    
    def __getitem__(self, idx):
        r"""Constructs and return a batch for SPair-71k dataset"""
        batch = super(SPairDataset, self).__getitem__(idx)

        if self.split == 'trn' and self.augmentation:
            # seems to just be resizing without cropping (by passing p=-1)
            batch['src_img'], batch['src_kps'] = random_crop(
                batch['src_img'], batch['src_kps'], self.src_bbox[idx].clone(), size=(self.imside,)*2, p=-1.0)
            batch['trg_img'], batch['trg_kps'] = random_crop(
                batch['trg_img'], batch['trg_kps'], self.trg_bbox[idx].clone(), size=(self.imside,)*2, p=-1.0)

        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize'])
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize'])
        batch['pckthres'] = self.get_pckthres(batch, batch['src_imsize'])

        batch['vpvar'] = self.vpvar[idx]
        batch['scvar'] = self.scvar[idx]
        batch['trncn'] = self.trncn[idx]
        batch['occln'] = self.occln[idx]

        if self.bidirectional_flows:
            flow = self.kps_to_flow(batch['src_kps'], batch['trg_kps'], batch['n_pts'])
            flow[:, flow.eq(0).all(0)] = float('inf')
            batch['trg_flow'] = flow

            flow = self.kps_to_flow(batch['trg_kps'], batch['src_kps'], batch['n_pts'])
            flow[:, flow.eq(0).all(0)] = float('inf')
            batch['src_flow'] = flow
        else:
            flow = self.kps_to_flow(batch['src_kps'], batch['trg_kps'], batch['n_pts'])
            flow[:, flow.eq(0).all(0)] = float('inf')
            batch['flow'] = flow

        return batch
    
    def get_image(self, img_names, idx):
        r"""Returns image tensor"""
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])

        return Image.open(path).convert('RGB')

    def get_bbox(self, bbox_list, idx, imsize):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (self.imside / imsize[0])
        bbox[1::2] *= (self.imside / imsize[1])
        return bbox


    


    

