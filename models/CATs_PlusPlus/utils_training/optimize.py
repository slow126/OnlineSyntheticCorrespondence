import time
import numpy as np
from itertools import islice
from tqdm import tqdm
import torch
import torch.nn.functional as F
from models.CATs_PlusPlus.utils_training.utils import flow2kps
from models.CATs_PlusPlus.utils_training.evaluation import Evaluator

# Import flow filter
try:
    from src.data.synth.datasets.flow_filter import FlowLengthFilter
except ImportError:
    # Fallback for different import paths
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from src.data.synth.datasets.flow_filter import FlowLengthFilter

r'''
    loss function implementation from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)


def train_epoch(net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer,
                steps_per_epoch = None,
                flow_filter = None):
    """
    Training epoch with optional flow filtering.
    
    Args:
        flow_filter: FlowLengthFilter instance to filter flow vectors during training.
            If None, no filtering is applied. Only used during training, not validation.
    """
    n_iter = epoch*len(train_loader)
    
    net.train()
    running_total_loss = 0
    
    if steps_per_epoch == 0:
        return 0
    elif steps_per_epoch is not None and steps_per_epoch < len(train_loader):
        train_steps = steps_per_epoch
    else:
        train_steps = len(train_loader)
    
    pbar = tqdm(islice(enumerate(train_loader), train_steps), total=train_steps, position=0, leave=True)
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        
        # Transfer entire batch to GPU once at the start (efficient - single operation)
        # Use explicit device comparison to handle "cuda" vs "cuda:0" properly
        gpu_batch = {}
        for key, value in mini_batch.items():
            if isinstance(value, torch.Tensor):
                # Explicit device comparison: check type and index separately
                value_device = value.device
                needs_transfer = (
                    value_device.type != device.type or
                    (value_device.index if value_device.index is not None else 0) != 
                    (device.index if device.index is not None else 0)
                )
                if needs_transfer:
                    gpu_batch[key] = value.to(device, non_blocking=True)
                else:
                    gpu_batch[key] = value  # Already on correct device
            else:
                gpu_batch[key] = value  # Keep non-tensors as-is
        
        # Ensure all async transfers complete before using tensors
        if device.type == 'cuda' and any(isinstance(v, torch.Tensor) and v.device.type == 'cuda' for v in gpu_batch.values()):
            torch.cuda.synchronize(device)
        
        # Use flow_downsampled if available (CorrespondenceDataset), otherwise use flow (old datasets)
        if 'flow_downsampled' in gpu_batch:
            flow_gt_key = 'flow_downsampled'
        else:
            flow_gt_key = 'flow'
        
        # Apply flow filtering if specified (only during training)
        if flow_filter is not None and flow_gt_key in gpu_batch:
            gpu_batch[flow_gt_key] = flow_filter.filter_batch_flow(gpu_batch[flow_gt_key])
        
        flow_gt = gpu_batch[flow_gt_key]

        pred_flow = net(gpu_batch['trg_img'], gpu_batch['src_img'])
        
        Loss = EPE(pred_flow, flow_gt) 
        Loss.backward()
        optimizer.step()

        running_total_loss += Loss.item()
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
                'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))

    running_total_loss /= train_steps
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   epoch):
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
        pck_array = []
        for i, mini_batch in pbar:
            # Transfer entire batch to GPU once at the start (efficient - single operation)
            # Use explicit device comparison to handle "cuda" vs "cuda:0" properly
            gpu_batch = {}
            for key, value in mini_batch.items():
                if isinstance(value, torch.Tensor):
                    # Explicit device comparison: check type and index separately
                    value_device = value.device
                    needs_transfer = (
                        value_device.type != device.type or
                        (value_device.index if value_device.index is not None else 0) != 
                        (device.index if device.index is not None else 0)
                    )
                    if needs_transfer:
                        gpu_batch[key] = value.to(device, non_blocking=True)
                    else:
                        gpu_batch[key] = value  # Already on correct device
                else:
                    gpu_batch[key] = value  # Keep non-tensors as-is
            
            # Ensure all async transfers complete before using tensors
            if device.type == 'cuda' and any(isinstance(v, torch.Tensor) and v.device.type == 'cuda' for v in gpu_batch.values()):
                torch.cuda.synchronize(device)
            
            # Use flow_downsampled if available (CorrespondenceDataset), otherwise use flow (old datasets)
            if 'flow_downsampled' in gpu_batch:
                flow_gt = gpu_batch['flow_downsampled']
            else:
                flow_gt = gpu_batch['flow']
            pred_flow = net(gpu_batch['trg_img'], gpu_batch['src_img'])

            estimated_kps = flow2kps(gpu_batch['trg_kps'], pred_flow, gpu_batch['n_pts'])

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), gpu_batch)
            
            Loss = EPE(pred_flow, flow_gt) 

            pck_array += eval_result['pck']

            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

    return running_total_loss / len(val_loader), mean_pck