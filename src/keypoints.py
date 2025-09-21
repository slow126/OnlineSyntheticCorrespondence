from typing import Optional

import torch
import torchmetrics


def eval_pck(
    pred_kps: torch.Tensor,
    target_kps: torch.Tensor,
    num_points: torch.Tensor,
    pck_dim: torch.Tensor,
    alpha: float=0.1
):
    '''Compute percentage of correct key-points for a batch of predictions and targets.
    
    PCK is defined as the number of predicted keypoint locations within fixed radius of the
    corresponding ground truth keypoint, divided by the total number of keypoints for the sample.
    To be considered correct, the distance between the predicted and target keypoints must be less
    than or equal to `pck_dim * alpha`.

    Keypoints (`pred_kps` and `target_kps`) should have shape (B, 2, N), where B is the batch size
    and N is the max possible number of keypoints. `num_points` has shape (B,), and indicates the
    number of valid keypoints for each element of the batch; so the first `num_points` values are
    valid keypoints and the remaining are just padding.

    Args:
        pred_kps: torch.Tensor of shape (B, 2, N), the predicted keypoint locations.
        target_kps: torch.Tensor of shape (B, 2, N), the target keypoint locations (ground truth).
        num_points: torch.Tensor of shape (B,), the number of valid keypoints for each row of the batch.
        pck_dim: torch.Tensor of shape (B,) or shape (1,), containing the spatial extent used for PCK calculation.
        alpha: float in (0, 1], which, in conjuction with pck_dim, defines the radius for correct predictions.

    Returns:
        pck: torch.Tensor of shape (B,), with the PCK for each row of the batch.
    '''
    nmax = num_points.max()
    # valid keypoints (1) or simply padding (0)
    mask = torch.arange(nmax, device=pred_kps.device).view(1, -1).lt(num_points.view(-1, 1))
    pred = pred_kps[..., :nmax]
    target = target_kps[..., :nmax]
    distance = pred.sub(target).norm(2, dim=1)
    threshold = pck_dim * alpha
    correct = distance.le(threshold.view(-1, 1))
    # pck = number_of_correct_predictions / number_of_valid_keypoints
    pck = correct.mul(mask).sum(1).div(num_points.clamp_min(1)) * 100
    return pck


def eval_dense_pck(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float=0.05,
    average: bool=True,
):
    '''
    '''
    size = target.shape[-2:]
    if alpha < 1:
        thresh = alpha * max(size)
    else:
        thresh = alpha

    mask = target.isfinite().all(1)
    
    if pred.shape[-2:] != size:
        pred = torch.nn.functional.interpolate(pred, size, mode='bilinear', align_corners=False)

    dist = torch.pairwise_distance(pred.moveaxis(1, -1), target.moveaxis(1, -1))
    pck = dist.le(thresh).mul(mask)
    if average:
        pck = pck.sum((1, 2)).div(mask.sum((1, 2))) * 100
    else:
        return pck.sum(), mask.sum()

    return pck


class PercentCorrectKeypoints(torchmetrics.Metric):

    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: Optional[bool] = False

    def __init__(
        self,
        pck_dim: int=256,
        alpha: float=0.1,
        num_classes: int=1,
        dense: bool=False,
        level: str='image',
    ):
        super().__init__()

        self.pck_dim = torch.tensor([pck_dim])
        self.alpha = alpha
        self.dense = dense
        self.level = level

        self.add_state('pck_sum', default=torch.zeros(num_classes, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('total', default=torch.zeros(num_classes, dtype=torch.float64), dist_reduce_fx='sum')

    def update(
        self,
        pred_kps: torch.Tensor,
        target_kps: torch.Tensor,
        num_points: Optional[torch.Tensor] = None,
        pck_dim: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if self.dense:
            if self.level == 'image':
                pck = eval_dense_pck(pred_kps, target_kps, self.alpha)
                pck = pck.to(self.pck_sum)
                if labels is None:
                    labels = torch.zeros_like(pck, dtype=torch.int64)
                self.pck_sum.index_put_((labels, ), pck, accumulate=True)
                self.total.index_put_((labels, ), torch.ones_like(pck), accumulate=True)
            else:
                pck, count = eval_dense_pck(pred_kps, target_kps, self.alpha, average=False)
                self.pck_sum += pck
                self.total += count
        else:
            if num_points is None:
                raise RuntimeError('Must provide num_points for sparse PCK')
            pck_dim = pck_dim if pck_dim is not None else self.pck_dim
            pck_dim = pck_dim.to(pred_kps)
            pck = eval_pck(pred_kps, target_kps, num_points, pck_dim, self.alpha)
            self.pck_sum += pck.sum()
            self.total += pck.numel()

    def compute(self):
        # average pck over the (batch/dataset)
        return (self.pck_sum.sum() / self.total.sum()).float()

    def compute_by_class(self):
        return (self.pck_sum / self.total).float()


class KittiF1(torchmetrics.Metric):

    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: Optional[bool] = False

    def __init__(
        self,
    ):
        super().__init__()

        self.add_state('sum', default=torch.zeros(1, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('total', default=torch.zeros(1, dtype=torch.float64), dist_reduce_fx='sum')

    def update(
        self,
        endpoint_errors: torch.Tensor,
        magnitudes: torch.Tensor,
    ):
        outliers = endpoint_errors.gt(3.0) & endpoint_errors.div(magnitudes).gt(0.05)
        self.sum += outliers.count_nonzero()
        self.total += len(outliers)

    def compute(self):
        return (100.0 * self.sum / self.total).float()