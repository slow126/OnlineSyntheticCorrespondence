from typing import Optional, Union

import torch

from . import ancnet
from src.model.base import BaseCorrespondenceModel
from src.objectives import KeypointMatchingLoss


class ANCNet(BaseCorrespondenceModel):
    def __init__(
        self,
        model_kwargs: Optional[dict] = None,
        checkpoint: Optional[str] = None,
        finetune: Optional[Union[bool, str]] = 'auto',
        covis_loss: str = 'covisibility_mean_loss',
        **kwargs,
    ):
        super().__init__(**kwargs)

        model_kwargs = model_kwargs or dict()
        self.model = ancnet.ANCNet(**model_kwargs)
        self.loss_fn = KeypointMatchingLoss()
        self.covis_loss = globals()[covis_loss]

        if checkpoint is not None:
            self.load_model_weights(checkpoint)
        
        self.finetune_param_names = []
        self._set_finetuning(model_kwargs, checkpoint, finetune)

    def forward(self, src, trg):
        corr = self.model(src, trg)
        pred_kp = self.model.forward_matches(corr)
        covisible = self.model.forward_covisible(corr)
        return corr, pred_kp, covisible
    
    def step(self, batch):
        src = batch['src_img']
        trg = batch['trg_img']
        keypoints = batch['points']
        corr, pred_kp, match_mask = self(src, trg)
        return corr, pred_kp, match_mask, keypoints

    def training_step(self, batch, batch_idx):
        corr, pred_kp, covisible, keypoints = self.step(batch)
        closs, vis, nonvis = self.covis_loss(corr, keypoints)
        loss = self.loss_fn(pred_kp, keypoints)
        loss = loss + closs

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/cmax', corr.detach().max(), prog_bar=True)
        self.log('train/vis', vis, prog_bar=True)
        self.log('train/nonvis', nonvis, prog_bar=True)

        # for n, p in list(self.model.spatial_context.named_parameters('ctx')):
        #     self.log(f'param/{n}', p.abs().mean())
        
        # for n, p in list(self.model.neighbor_consensus.named_parameters('nbor')):
        #     self.log(f'param/{n}', p.abs().mean())

        return loss

    def validation_step(self, batch, batch_idx):
        corr, pred_kp, covisible, keypoints = self.step(batch)
        loss = self.loss_fn(pred_kp, keypoints)

        self.log('val/loss', loss, prog_bar=True)

        return loss

    def keypoints_from_corr(self, corr, kp):
        b, h, w = corr.shape[:3]
        probs = corr.flatten(-2).softmax(-1).view(b, h, w, h, w)
        grid = torch.meshgrid(
            torch.arange(corr.shape[-2], dtype=corr.dtype, device=corr.device),
            torch.arange(corr.shape[-1], dtype=corr.dtype, device=corr.device),
            indexing='ij',
        )
        grid = torch.stack(grid, -1)

        pkp = probs.unsqueeze(-1).mul(grid).sum((3, 4))

        yx = kp[..., 0].long()
        idx = (torch.arange(b, device=corr.device)[:, None], yx[..., 0], yx[..., 1])

        return pkp[idx]

    def keypoints_from_pred(self, pred, kp):
        b = pred.shape[0]
        yx = kp[..., 0].long()
        idx = (torch.arange(b, device=pred.device)[:, None], yx[..., 0], yx[..., 1])
        return pred[idx]


def covisibility_mean_loss(corr: torch.Tensor, kp: torch.Tensor):
    ab = corr.sum((3, 4))
    ba = corr.sum((1, 2))
    pred = torch.stack((ab, ba), -1)

    kp = kp.long()
    i = torch.arange(pred.shape[0], device=pred.device)
    index = (i.view(-1, 1, 1), kp[:, :, 0, :], kp[:, :, 1, :], i[:2].view(1, 1, 2))
    val = pred.new_full((1,), True, dtype=torch.bool)
    mask = torch.zeros_like(pred, dtype=torch.bool).index_put_(index, val)
    vis = pred[mask]
    nonvis = pred[~mask]
    loss = nonvis.mean() + (1 - vis.mul(3).tanh()).mean()
    return loss, vis.detach().mean(), nonvis.detach().mean()


def covisibility_softmax_loss(corr: torch.Tensor, kp: torch.Tensor):
    # NOTE: this one isn't working
    ab = corr.sum((3, 4))
    ba = corr.sum((1, 2))
    pred = torch.stack((ab, ba), -1)

    abmax = corr.detach().flatten(3, 4).max(-1, keepdim=True)[0].unsqueeze(-1).clamp_min(.1)
    bamax = corr.detach().flatten(1, 2).max(1, keepdim=True)[0].unsqueeze(1).clamp_min(.1)
    s = corr.flatten(3, 4).flatten(1, 2)
    ab = corr.div(abmax).mul(s.softmax(2).reshape(corr.shape)).sum((3, 4))
    ba = corr.div(bamax).mul(s.softmax(1).reshape(corr.shape)).sum((1, 2))
    soft_pred = torch.stack((ab, ba), -1)

    kp = kp.long()
    i = torch.arange(pred.shape[0], device=pred.device)
    index = (i.view(-1, 1, 1), kp[:, :, 0, :], kp[:, :, 1, :], i[:2].view(1, 1, 2))
    val = pred.new_full((1,), True, dtype=torch.bool)
    mask = torch.zeros_like(pred, dtype=torch.bool).index_put_(index, val)
    vis = soft_pred[mask]
    nonvis = pred[~mask]
    loss = nonvis.mean() + (1 - vis).mean()
    return loss, vis.detach().mean(), nonvis.detach().mean()