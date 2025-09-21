import os
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image


class AWADataset(Dataset):
    def __init__(self, datapath, thres, split='random', size=256, normalize='imagenet'):
        
        super(AWADataset, self).__init__()

        self.datapath = datapath
        self.thres = thres

        if split == 'random':
            pair_path = os.path.join(datapath, 'random_pairs.txt')
        elif split == 'class':
            pair_path = os.path.join(datapath, 'class_pairs.txt')
        else:
            raise Exception('Invalid pair type: %s' % split)
 
        with open(pair_path) as file:
            pairs = file.readlines()
            pairs = [pair.rstrip().split(' ') for pair in pairs]
            self.pairs = pairs[0:10000]
        
        self.cls = os.listdir(os.path.join(datapath,'Annotations'))
        
        self.data = {}
        for cls in self.cls:
            cls_path = os.path.join(datapath, 'Annotations', cls)
            if not os.path.isdir(cls_path): continue
            # cls_annotations = os.listdir(cls_path)
            # cls_annotations = sorted(Path(cls_path).glob('*.pickle'))
            cls_annotations = pickle.load(open(os.path.join(cls_path, 'all_annos.pkl'), 'rb'))
            for ann in cls_annotations:
                # with open(os.path.join(datapath, 'Annotations', cls, ann), 'rb') as f:
                # with open(ann, 'rb') as f:
                #     d = pickle.load(f)
                # key = '{}/{}'.format(cls, ann.replace('pickle','jpg'))
                d = cls_annotations[ann]
                key = f'{cls}/{ann}.jpg'
                self.data[key] = {}
                
                parts = list(d['a1'].keys())
                parts = [p for p in parts if not p=='bbox']
                self.data[key]['kps'] = {p:d['a1'][p] for p in parts}
                
                if not 'bbox' in d['a1']:
                    self.data[key]['bbox'] = [0,0,0,0]
                else:
                    self.data[key]['bbox'] = d['a1']['bbox']

        self.max_kp = max(len(v['kps']) for v in self.data.values())

        self.size = size

        if normalize == 'imagenet':
            from src import imagenet_stats
            normalize = imagenet_stats
        elif normalize == True:
            normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize = normalize

    def __getitem__(self, idx):
        
        sample = dict() 
        
        sample['datalen'] = len(self.pairs)
       
        sample['src_imname'] = self.pairs[idx][0]
        sample['trg_imname'] = self.pairs[idx][1]

        src, src_size = self.get_image(sample['src_imname'])
        trg, trg_size = self.get_image(sample['trg_imname'])
        sample['src_img'] = src
        sample['trg_img'] = trg

        sample['src_bbox'] = np.array(self.data[sample['src_imname']]['bbox']) * self.size / np.tile(src_size, 2)
        sample['trg_bbox'] = np.array(self.data[sample['trg_imname']]['bbox']) * self.size / np.tile(trg_size, 2)

        sample['pckthres'] = self.get_pckthres(sample)
        
        src_kps, trg_kps, common_joints = self.get_kps(sample, src_size, trg_size)
        sample['src_kps'], sample['trg_kps'] = src_kps, trg_kps
        sample['n_pts'] = common_joints.shape[0]
        # sample['common_joints'] = common_joints
        sample['pair_class'] = sample['src_imname'].split('/')[0]

        return sample

    def __len__(self):
        return len(self.pairs)

    def get_image(self, img_name):
        
        img_name = os.path.join(self.datapath, 'JPEGImages', img_name)
        # image = self.get_imarr(img_name)
        image = Image.open(img_name).convert('RGB')
        size = np.array(image.size)
        image = F.resize(image, (self.size, self.size), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        image = F.to_tensor(image)
        image = F.normalize(image, *self.normalize)
        # image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image, size
    
    # def get_imarr(self, path):
    #     r"""Read a single image file as numpy array from path"""
    #     return np.array(Image.open(path).convert('RGB'))

    def get_pckthres(self, sample):
        
        if self.thres == 'bbox':
            bbox = sample['trg_bbox']
            if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                return torch.tensor(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            else:
                return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))

        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_kps(self, sample, src_size, trg_size):
        
        src_k = list(self.data[sample['src_imname']]['kps'].keys())
        #trg_kps = list(self.data[sample['trg_imname']]['kps'].keys())
        
        #common_kps = np.intersect1d(src_kps, trg_kps)
        src_kps = []
        trg_kps = []
        joints = []
        for kp in src_k:
            if kp in self.data[sample['trg_imname']]['kps']:
                src_kps.append(self.data[sample['src_imname']]['kps'][kp]) 
                trg_kps.append(self.data[sample['trg_imname']]['kps'][kp])
                joints.append(kp)

        src_kps = np.array(src_kps)
        trg_kps = np.array(trg_kps)
        common_joints = np.where(np.logical_and(src_kps[:, 0] != -1, trg_kps[:, 1] != -1))[0]

        src_kps = self.resize_keypoints(src_kps[common_joints, :2], src_size).T
        trg_kps = self.resize_keypoints(trg_kps[common_joints, :2], trg_size).T
        
        n = self.max_kp
        k = common_joints.shape[0]
        src_kps = np.pad(src_kps, ((0, 0), (0, n - k)), constant_values=-1)
        trg_kps = np.pad(trg_kps, ((0, 0), (0, n - k)), constant_values=-1)
        
        return src_kps, trg_kps, common_joints

    def resize_keypoints(self, kp, img_size):
        scale = self.size / img_size
        return kp * scale