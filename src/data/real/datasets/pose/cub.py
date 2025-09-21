import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image


class CUBDataset(Dataset):
    def __init__(self, datapath, thres, split='random', size=256, normalize='imagenet'):
        
        super(CUBDataset, self).__init__()

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
        
        with open(os.path.join(datapath, 'images.txt')) as file:
            images = file.readlines()
            images = [image.rstrip() for image in images]
            self.images = {image.split(' ')[0]:image.split(' ')[1] for image in images}
       
        with open(os.path.join(datapath, 'image_class_labels.txt')) as file:
            classes = file.readlines()
            classes = [cls.rstrip() for cls in classes]
            self.classes = {cls.split(' ')[0]:cls.split(' ')[1] for cls in classes}
        
        with open(os.path.join(datapath, 'bounding_boxes.txt')) as file:
            boxes = file.readlines()
            boxes = [box.rstrip() for box in boxes]
            self.boxes = {box.split(' ')[0]: [float(b) for b in box.split(' ')[1:]] for box in boxes}

        
        with open(os.path.join(datapath, 'parts', 'part_locs.txt')) as file:
            parts = file.readlines()
            parts = [part.rstrip() for part in parts]
            self.kps = {part.split(' ')[0]: [] for part in parts}
            for part in parts:
                p = part.split(' ')
                kp = [float(c) for c in p[2:]]
                self.kps[p[0]].append(kp)
               
        self.cls = os.listdir(os.path.join(datapath, 'images'))
        self.cls.sort()

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
        src_idx, trg_idx = self.pairs[idx]
       
        sample['src_imname'] = self.images[src_idx]
        sample['trg_imname'] = self.images[trg_idx]

        src, src_size = self.get_image(self.images, src_idx)
        trg, trg_size = self.get_image(self.images, trg_idx)
        sample['src_img'] = src
        sample['trg_img'] = trg

        sample['src_bbox'] = np.array(self.boxes[src_idx]) * self.size / np.tile(src_size, 2)
        sample['trg_bbox'] = np.array(self.boxes[trg_idx]) * self.size / np.tile(trg_size, 2)
        
        sample['pckthres'] = self.get_pckthres(sample)
        
        src_kps, trg_kps, common_joints = self.get_kps(idx, src_size, trg_size)
        sample['src_kps'], sample['trg_kps'] = src_kps, trg_kps
        sample['n_pts'] = common_joints.shape[0]
        # sample['common_joints'] = common_joints
        sample['pair_class'] = sample['src_imname'].split('/')[0]

        return sample

    def __len__(self):
        return len(self.pairs)

    def get_image(self, img_names, idx):
        
        img_name = os.path.join(self.datapath, 'images', img_names[idx])
        # image = self.get_imarr(img_name)
        image = Image.open(img_name).convert('RGB')
        size = np.array(image.size)  # (width, height)
        image = F.resize(image, (self.size, self.size), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        image = F.to_tensor(image)
        image = F.normalize(image, *self.normalize)
        # image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image, size
    
    # def get_imarr(self, path):
    #     r"""Read a single image file as numpy array from path"""
    #     return np.array(Image.open(path).convert('RGB'))

    def get_pckthres(self, sample):
        
        trg_bbox = sample['trg_bbox']
        if self.thres == 'bbox':
            return torch.tensor(max(trg_bbox[2], trg_bbox[3]))

        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_kps(self, idx, src_size, trg_size):
        
        src_idx, trg_idx = self.pairs[idx]

        src_kps = np.array(self.kps[src_idx])
        trg_kps = np.array(self.kps[trg_idx])
        common_joints = np.where(src_kps[:,2] * trg_kps[:,2] == 1)[0]
        n = src_kps.shape[0]
        k = common_joints.shape[0]

        src_kps = self.resize_keypoints(src_kps[common_joints, :2], src_size).T
        trg_kps = self.resize_keypoints(trg_kps[common_joints, :2], trg_size).T

        src_kps = np.pad(src_kps, ((0, 0), (0, n - k)), constant_values=-1)
        trg_kps = np.pad(trg_kps, ((0, 0), (0, n - k)), constant_values=-1)
        
        return src_kps, trg_kps, common_joints

    def resize_keypoints(self, kp, img_size):
        scale = self.size / img_size
        return kp * scale