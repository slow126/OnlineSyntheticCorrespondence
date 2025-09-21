from typing import List, Optional

import torch

from src.flow import sparse_downsample


__all__ = [
    'endpoint_error',
    'loss_dynamic_selection',
    'EndpointErrorLoss',
    'EndpointErrorPseudoFilteredLoss',
    'MultiscaleEndpointError',
]


class EndpointErrorLoss(torch.nn.Module):
    def __init__(self, sparse: bool=True, reduction: str='mean'):
        super().__init__()
        self.sparse = sparse
        self.reduction = reduction

    def forward(self, pred_flow, target_flow):
        return endpoint_error(pred_flow, target_flow, self.sparse, self.reduction)


class EndpointErrorPseudoFilteredLoss(torch.nn.Module):
    def __init__(self, kernel_size: int=7, pseudo_weight: float=10.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.pseudo_weight = pseudo_weight
        
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        self.register_buffer('kernel', kernel)

    def forward(self, pred_flow_1, pred_flow_2, target_flow, remember_rate):
        # ---------------------------
        # loss on ground-truth points
        # ---------------------------
        gt_loss_1 = endpoint_error(pred_flow_1, target_flow, reduction='mean')
        gt_loss_2 = endpoint_error(pred_flow_2, target_flow, reduction='mean')

        loss = gt_loss_1 + gt_loss_2

        if self.pseudo_weight > 0:
            # ---------------------------
            # loss on pseudo targets
            # ---------------------------
            target_flow_1 = pred_flow_2.detach().clone()
            target_flow_2 = pred_flow_1.detach().clone()

            epe_1 = torch.norm(target_flow_1 - pred_flow_1, 2, 1)
            epe_2 = torch.norm(target_flow_2 - pred_flow_2, 2, 1)

            # invalid flow is defined with both flow coordinates set to inf
            mask = target_flow.ne(float('inf')).all(dim=1) # (b, 16, 16)
            mask_labeled = mask.float().unsqueeze_(1) # (b, 1, h, w)

            # dilate sparse label masks 
            pad = self.kernel_size // 2
            mask_dilate = torch.nn.functional.conv2d(mask_labeled, weight=self.kernel, padding=pad)
            mask_dilate = mask_dilate.gt(0).squeeze_(1)

            epe_1 = epe_1[mask_dilate] # (n, )
            epe_2 = epe_2[mask_dilate] # (n, )
            epe_1, epe_2 = loss_dynamic_selection(epe_1, epe_2, remember_rate)
            
            # ---------------------------
            # combined loss
            # ---------------------------
            loss = loss + self.pseudo_weight * (epe_1 + epe_2)

        return loss


class MultiscaleEndpointError(EndpointErrorLoss):
    '''Weighted combination of endpoint errors for a spatial pyramid of predicted flows.

    Adapted from GLU-Net.
    '''
    def __init__(self, weights: List[float]=None, sparse: bool=True, reduction: str='mean'):
        super().__init__(sparse, reduction)
        self.weights = weights

    def forward(
        self,
        pred_flows: List[torch.Tensor],
        target_flow: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        weights = weights or self.weights
        if weights is None:
            # assumes the order of pred_flows is from coarsest to finest resolution
            weights = [0.32, 0.08, 0.02, 0.01]
        if len(weights) < len(pred_flows):
            weights += [0.005] * (len(pred_flows) - len(weights))

        loss = 0
        for pred, weight in zip(pred_flows, weights):
            target = sparse_downsample(target_flow, pred.shape[-2:])
            err = endpoint_error(pred, target, reduction=self.reduction)
            loss = loss + weight * err

        return loss


def endpoint_error(pred_flow: torch.Tensor, target_flow: torch.Tensor, sparse: bool=True, reduction: str='mean'):
    '''Cacluate endpoint error.

    The endpoint error is the distance between the target and predicted flow vectors at every
    valid location. Invalid locations have target flow set to inf.

    Args:
        pred_flow (Tensor): predicted flow field, with shape (B, 2, H, W)
        target_flow (Tensor): target flow field, with shape (B, 2, H, W). Invalid flow regions
            should be indicated by inf values.
        sparse (bool): whether to mask out invalid regions. Should be set to True if the target
            contains any invalid regions.
        reduction (str): can be "sum" or "mean" to reduce across full batch, or "batch_sum" to get
            the summed error averaged over each element in the batch; otherwise, the average
            error is returned separately for each item in the batch.
    '''
    if sparse:
        target_flow = target_flow.moveaxis(1, -1)
        pred_flow = pred_flow.moveaxis(1, -1)
        # invalid flow is defined with both flow coordinates set to inf
        mask = target_flow.ne(float('inf')).all(dim=-1, keepdim=True).expand_as(pred_flow)
        pred_flow = pred_flow[mask].reshape(-1, 2)
        target_flow = target_flow[mask].reshape(-1, 2)

    error = torch.norm(target_flow - pred_flow, p=2, dim=1)

    if reduction == 'mean':
        return error.mean()
    elif reduction == 'sum':
        return error.sum()
    elif reduction == 'batch_sum':
        return error.sum() / pred_flow.shape[0]
    else:
        m = mask[..., 0]
        img_idx = m.nonzero(as_tuple=True)[0]
        count = m.flatten(1).count_nonzero(1)
        per_img = pred_flow.new_zeros(mask.shape[0])
        per_img.index_put_((img_idx,), error, accumulate=True).div_(count)
        return per_img


def loss_dynamic_selection(loss_1: torch.Tensor, loss_2: torch.Tensor, remember_rate: float):
    ''' Select small-loss samples on pixel-level, modified from
    https://github.com/bhanML/Co-teaching/blob/master/loss.py
    '''
    ind_1_sorted = loss_1.detach().argsort()
    ind_2_sorted = loss_2.detach().argsort()

    num_remember = int(remember_rate * len(ind_1_sorted))
    if num_remember < 1: 
        num_remember = 1

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # exchange
    loss_1_update = loss_1[ind_2_update]
    loss_2_update = loss_2[ind_1_update]

    # final loss
    loss_1_final = torch.sum(loss_1_update) / num_remember
    loss_2_final = torch.sum(loss_2_update) / num_remember
   
    return loss_1_final, loss_2_final