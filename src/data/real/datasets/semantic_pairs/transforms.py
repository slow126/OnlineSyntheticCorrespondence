import random

import torch
from torchvision import transforms


class Photometric(object):
    def __init__(self, p=0.2):
        self.p = p

        self.transforms = [
            transforms.RandomGrayscale(p),
            transforms.RandomPosterize(4, p),
            transforms.RandomEqualize(p),
            transforms.RandomAdjustSharpness(1.25, p),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p),
            transforms.RandomSolarize(128, p),
        ]

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def resize(img, kps, size=(256, 256)):
    _, h, w = img.shape
    resized_img = transforms.functional.resize(img, size, antialias=True)
    
    kps = kps.t()
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = kps[:, 0] * (size[1] / w)
    resized_kps[:, 1] = kps[:, 1] * (size[0] / h)
    
    return resized_img, resized_kps.t()


def random_crop(img, kps, bbox, size=(256, 256), p=0.5):
    if random.uniform(0, 1) > p:
        return resize(img, kps, size)
    _, h, w = img.shape
    kps = kps.t()
    left = random.randint(0, bbox[0])
    top = random.randint(0, bbox[1])
    height = random.randint(bbox[3], h) - top
    width = random.randint(bbox[2], w) - left
    resized_img = transforms.functional.resized_crop(
        img, top, left, height, width, size=size, antialias=True)
    
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = (kps[:, 0] - left) * (size[1] / width)
    resized_kps[:, 1] = (kps[:, 1] - top) * (size[0] / height)
    resized_kps = torch.clamp(resized_kps, 0, size[0] - 1)
    
    return resized_img, resized_kps.t()