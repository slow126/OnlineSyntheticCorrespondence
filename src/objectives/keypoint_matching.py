import torch
from torch.nn.functional import grid_sample, pairwise_distance


class KeypointMatchingLoss(torch.nn.Module):
    '''Keypoint matching loss with subpixel ground-truth keypoints.

    For each keypoint in the source image, computes the distance between the corresponding ground-truth
    target keypoint and the predicted keypoint. The predicted keypoint is bilinearly interpolated based
    on the subpixel location of the source keypoint. The loss is the average distance over all keypoint
    pairs.

    TODO: example
    '''
    def __init__(self):
        super().__init__()

    def forward(self, kp_preds: torch.Tensor, kp_pairs: torch.Tensor):
        '''
        Args:
            kp_preds (Tensor): shape (B, H, W, 2) containing yx predicted target locations for each
                (i, j) source location.
            kp_pairs (Tensor): shape (B, N, 2, 2) containing N corresponding pairs of yx keypoints.
                Dimension 2 is [y, x] and dimension 3 is [source, target].
        '''
        src = kp_pairs[..., 0]
        trg = kp_pairs[..., 1]
        preds = kp_preds.permute(0, 3, 1, 2)

        # normalize to range [-1, 1] and flip yx -> xy for grid_sample
        hw = torch.tensor(tuple(preds.shape[-2:]), device=preds.device)
        p = src.div(0.5 * (hw - 1)).sub(1).flip(-1).unsqueeze(2)  # xy points (B, N, 1, 2)

        # get interpolated predictions based on subpixel source keypoint
        # align_corners=True gives us the interpolation we want
        loc = grid_sample(preds, p, align_corners=True).squeeze(-1).transpose(-1, -2)

        # distance between subpixel target keypoint and interpolated keypoint prediction
        loss = pairwise_distance(loc, trg, p=2).mean()

        return loss