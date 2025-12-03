import re
import os
import shutil

import torch
import torch.nn.functional as F
import numpy as np

r'''
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def save_checkpoint(state, is_best, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))

r'''
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            best_val=checkpoint['best_loss']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, best_val


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def log_args(args):
    r"""Log program arguments"""
    print('\n+================================================+')
    for arg_key in args.__dict__:
        print('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
    print('+================================================+\n')


def parse_list(list_str):
    r"""Parse given list (string -> int)"""
    return list(map(int, re.findall(r'\d+', list_str)))


def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


def flow2kps(trg_kps, flow, n_pts, upsample_size=(512, 512)):
    _, _, h, w = flow.size()
    flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
    
    src_kps = []
    for trg_kps_batch, flow_batch, n_pts_batch in zip(trg_kps.long(), flow, n_pts):
        size = trg_kps_batch.size(1)

        kp = torch.clamp(trg_kps_batch.narrow_copy(1, 0, n_pts_batch), 0, upsample_size[0] - 1)
        
        # Sample flow at keypoint locations
        # flow_batch is [2, H, W], kp is [2, N] where kp[0] is x and kp[1] is y
        sampled_flow = flow_batch[:, kp[1, :], kp[0, :]]  # [2, N]
        
        # Check for invalid flow values (inf or both components exactly 0)
        flow_mag = sampled_flow.norm(dim=0)  # [N]
        is_valid = flow_mag.isfinite() & (flow_mag > 1e-6)  # Valid if finite and non-zero
        
        # Compute estimated keypoints
        estimated_kps = kp + sampled_flow
        
        # Mark invalid keypoints (where flow was invalid) with -1
        estimated_kps[:, ~is_valid] = -1
        
        # Pad remaining keypoints with -1
        estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts_batch, device=estimated_kps.device, dtype=estimated_kps.dtype) * -1), dim=1)
        src_kps.append(estimated_kps)

    return torch.stack(src_kps)