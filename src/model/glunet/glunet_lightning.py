from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from . import sem_glunet
from src.flow import flow_to_keypoints
from src.keypoints import PercentCorrectKeypoints, KittiF1
from src.model.base import BaseCorrespondenceModel
from src.objectives import MultiscaleEndpointError, endpoint_error


class GLUNet(BaseCorrespondenceModel):
    def __init__(
        self,
        model_kwargs: Optional[dict] = None,
        checkpoint: Optional[str] = None,
        finetune: Optional[Union[bool, str]] = 'auto',
        src_key: str = 'src',
        trg_key: str = 'trg',
        val_src_key: str = 'src_img',
        val_trg_key: str = 'trg_img',
        flow_key: str = 'flow',
        bidirectional_eval: bool = False,
        eval_metric: Optional[str] = None,
        per_image_metrics: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        model_kwargs = model_kwargs or dict()
        self.model = sem_glunet.SemanticGLUNet(**model_kwargs)
        self.loss_fn = MultiscaleEndpointError(reduction='batch_sum')

        self.src_key = src_key
        self.trg_key = trg_key
        self.val_src_key = val_src_key
        self.val_trg_key = val_trg_key
        self.flow_key = flow_key
        self.bidirectional_eval = bidirectional_eval

        if checkpoint is not None:
            self.load_model_weights(checkpoint)
        
        self.finetune_param_names = []
        self._set_finetuning(model_kwargs, checkpoint, finetune)

        if eval_metric is None:
            self.eval_metric = None
            self.eval_type = 'epe'
        elif eval_metric == 'kitti_f1':
            self.eval_metric = KittiF1()
            self.eval_type = 'kitti_f1'
        elif eval_metric == 'sintel_pck':
            self.eval_metric = PercentCorrectKeypoints(alpha=1.0, dense=True, level='pixel')
            self.eval_type = 'sintel_pck'
        elif eval_metric.startswith('pck'):
            alpha = float(eval_metric.split('@')[1])
            self.eval_metric = PercentCorrectKeypoints(alpha=alpha)
            self.eval_type = 'pck'
        elif eval_metric == 'tss_pck':
            self.eval_metric = PercentCorrectKeypoints(alpha=0.05, num_classes=3, dense=True)
            self.names = ['FG3DCar', 'JODS', 'PASCAL']
            self.eval_type = 'dense_pck'

        self.per_image_metrics = per_image_metrics

    def on_train_epoch_start(self) -> None:
        if self.current_epoch > 0:
            if hasattr(self.trainer.datamodule.train_data, 'set_sample_pairs'):
                self.trainer.datamodule.train_data.set_sample_pairs()
        if hasattr(self.trainer.datamodule, 'on_train_epoch_start'):
            self.trainer.datamodule.on_train_epoch_start(self.trainer)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        if hasattr(self.trainer.datamodule, 'adjust_warp_strength'):
            self.trainer.datamodule.adjust_warp_strength(
                batch_idx, self.trainer.current_epoch, self.trainer.num_training_batches
            )

    def _make_bidirectional(self, batch, src_key, trg_key):
        sk, tk = src_key, trg_key
        fsk, ftk = sk + '_flow', tk + '_flow'
        combined = {
            sk: torch.cat((batch[sk], batch[tk]), 0),
            tk: torch.cat((batch[tk], batch[sk]), 0),
            self.flow_key: torch.cat((batch[ftk], batch[fsk])),
        }
        if 'src_kps' in batch:
            src_kps = batch.pop('src_kps')
            trg_kps = batch.pop('trg_kps')
            combined['src_kps'] = torch.cat((src_kps, trg_kps), 0)
            combined['trg_kps'] = torch.cat((trg_kps, src_kps), 0)
        for k in (sk, tk, fsk, ftk):
            batch.pop(k)
        for k in batch:
            combined[k] = batch[k].repeat(2)
        return combined

    def forward(self, src, trg):
        return self.model(src, trg)

    def step(self, batch, src_key, trg_key, flow_key):        
        
        src, trg = batch[src_key], batch[trg_key]

        # src = (src - src.min()) / (src.max() - src.min())
        # trg = (trg - trg.min()) / (trg.max() - trg.min())

        
        preds = self(src, trg)
        with torch.no_grad():
            full_pred = self.model.resize(preds['level3'], src.shape[-2:])
            preds['full'] = full_pred
        if flow_key is not None:
            flow = batch[flow_key]
            loss = self.loss_fn(list(preds.values()), flow)
            with torch.no_grad():
                reduct = None if self.per_image_metrics else 'mean'
                full_err = endpoint_error(full_pred, flow, reduction=reduct)
        else:
            loss = -1
            full_err = -1
            flow = None

        
        # self.debug_grid_plot(src[0:5,:,:,:], trg[0:5,:,:,:], flow[0:5,:,:,:], full_pred[0:5,:,:,:])

        return src, trg, flow, preds, loss, full_err

    def training_step(self, batch, batch_idx):
        loss, full_err = self.step(batch, self.src_key, self.trg_key, self.flow_key)[-2:]

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/err', full_err, prog_bar=True, sync_dist=True)

        # Log sample image pairs periodically
        if batch_idx % 15 == 0:
            # Get all image pairs from batch
            src_imgs = batch[self.src_key]
            trg_imgs = batch[self.trg_key]
            
            # Create text labels
            b, c, h, w = src_imgs.shape
            src_label = torch.ones((b, c, h//8, w), device=src_imgs.device) * 0.8
            trg_label = torch.ones((b, c, h//8, w), device=trg_imgs.device) * 0.2
            
            # Stack labels on top of images
            src_with_label = torch.cat([src_label, src_imgs], dim=2)
            trg_with_label = torch.cat([trg_label, trg_imgs], dim=2)
            
            # Stack source and target side by side for each pair
            img_grid = torch.cat([src_with_label, trg_with_label], dim=3)
            
            # Log to tensorboard
            self.logger.experiment.add_images(
                'train/image_pairs',
                img_grid,
                self.global_step
            )

        return loss

    def on_validation_epoch_start(self):
        if self.per_image_metrics:
            self.error_list = []

    def validation_step(self, batch, batch_idx):
        if self.bidirectional_eval:
            batch = self._make_bidirectional(batch, self.val_src_key, self.val_trg_key)

        out = self.step(batch, self.val_src_key, self.val_trg_key, self.flow_key)
        flow, pred, loss, full_err = out[-4:]

        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

        if self.per_image_metrics:
            self.error_list.append(full_err)
        else:
            self.log('val/err', full_err, prog_bar=True, sync_dist=True)

        pred = pred['full']
        if self.eval_type == 'pck':
            num_points = batch['n_pts']
            pck_dim = batch['pckthres']
            pred = flow_to_keypoints(batch['trg_kps'], pred, num_points, pred.shape[-2:])
            acc = self.eval_metric(pred, batch['src_kps'], num_points=num_points, pck_dim=pck_dim)
            self.log('val/acc', acc, sync_dist=True)
        elif self.eval_type == 'dense_pck':
            labels = batch.get('label')
            acc = self.eval_metric(pred, flow, labels=labels)
            self.log('val/acc', acc, sync_dist=True)
        elif self.eval_type == 'sintel_pck':
            valid = batch['correspondence_mask'][:, None].expand_as(flow)
            flow[~valid] = torch.inf
            acc = self.eval_metric(pred, flow)
            self.log('val/acc', acc, sync_dist=True)
        elif self.eval_type == 'kitti_f1':
            flow = flow.moveaxis(1, -1)
            pred = pred.moveaxis(1, -1)
            # invalid flow is defined with both flow coordinates set to inf
            mask = flow.isfinite().all(dim=-1, keepdim=True).expand_as(pred)
            pred = pred[mask].reshape(-1, 2)
            flow = flow[mask].reshape(-1, 2)

            epe = flow.sub(pred).norm(dim=-1)
            mag = flow.norm(dim=-1)
            acc = self.eval_metric(epe, mag)
            self.log('val/acc', acc, sync_dist=True)

        return loss
    
    def on_validation_epoch_end(self):
        if self.eval_type == 'dense_pck':
            accs = self.eval_metric.compute_by_class()
            if len(accs) > 1:
                for n, v in zip(self.names, accs):
                    self.log(f'val/{n}-acc', v, sync_dist=True)

        if self.per_image_metrics:
            errs = torch.cat(self.error_list).cpu()
            f = Path(self.logger.log_dir, 'per_img_errors.pth')
            torch.save(errs, f)

    def predict_step(self, batch, batch_idx):
        src, trg, flow, preds, loss, full_err = self.step(batch)

        self.src_imgs.append(src.mul(self.std).add(self.mean).mul(256).byte().cpu())
        self.trg_imgs.append(trg.mul(self.std).add(self.mean).mul(256).byte().cpu())
        self.gt_flows.append(flow.cpu().float())
        self.pred_flows.append(preds['level3'].detach().cpu().float())
    
    def on_predict_epoch_start(self):
        self.src_imgs = []
        self.trg_imgs = []
        self.gt_flows = []
        self.pred_flows = []

        self.mean = torch.tensor(
            self.trainer.datamodule.normalize_vals[0],
            device=self.device,
        ).view(1, -1, 1, 1)

        self.std = torch.tensor(
            self.trainer.datamodule.normalize_vals[1],
            device=self.device,
        ).view(1, -1, 1, 1)

    def on_predict_epoch_end(self):
        import torchvision
        import os

        dir = os.path.join(self.trainer.log_dir, 'samples')
        os.makedirs(dir, exist_ok=True)
        for src, trg, flow, pred in zip(self.src_imgs, self.trg_imgs, self.gt_flows, self.pred_flows):
            for i in range(src.shape[0]):
                ii = str(i).zfill(4)
                torchvision.io.write_jpeg(src[i], os.path.join(dir, f'src_{ii}.jpeg'))
                torchvision.io.write_jpeg(trg[i], os.path.join(dir, f'trg_{ii}.jpeg'))
                torch.save({'flow': flow[i], 'pred': pred[i]}, os.path.join(dir, f'flow_{ii}.pt'))

    
    def debug_grid_plot(self, src_imgs, trg_imgs, flows, pred_flows, save_path="debug/images/flow_grid.png"):
        """Create a grid visualization showing source images, target images, ground truth flows and predicted flows.
        
        Args:
            src_imgs: Tensor of source images [B,C,H,W] 
            trg_imgs: Tensor of target images [B,C,H,W]
            flows: Tensor of ground truth flows [B,2,H,W]
            pred_flows: Tensor of predicted flows [B,2,H,W]
            save_path: Optional path to save the visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path

        batch_size = src_imgs.shape[0]
        fig, axes = plt.subplots(batch_size, 4, figsize=(40, 10*batch_size))
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)

        for i in range(batch_size):
            # Get images and flows for this batch item
            src_img = src_imgs[i].cpu().permute(1,2,0).numpy()
            trg_img = trg_imgs[i].cpu().permute(1,2,0).numpy()
            flow = flows[i].cpu().numpy()
            pred_flow = pred_flows[i].cpu().numpy()

            # Create coordinate grid
            h, w = flow.shape[1:]
            y, x = np.mgrid[0:h:1, 0:w:1]

            # Plot source image
            axes[i,0].imshow(src_img)
            axes[i,0].set_title('Source Image')
            axes[i,0].axis('off')

            # Plot target image  
            axes[i,1].imshow(trg_img)
            axes[i,1].set_title('Target Image')
            axes[i,1].axis('off')

            # Plot ground truth flow
            axes[i,2].imshow(src_img)
            valid_flow = ~np.isinf(flow).any(axis=0)
            x_valid = x[valid_flow]
            y_valid = y[valid_flow]
            u = flow[0][valid_flow]
            v = flow[1][valid_flow]
            
            stride = 10
            n_arrows = len(x_valid[::stride])
            colors = np.random.rand(n_arrows, 3)
            
            axes[i,2].quiver(x_valid[::stride], y_valid[::stride],
                           u[::stride], v[::stride],
                           color=colors, angles='xy', scale_units='xy', scale=1,
                           width=0.002, headwidth=3)
            axes[i,2].set_title('Ground Truth Flow')
            axes[i,2].axis('off')

            # Plot predicted flow
            axes[i,3].imshow(src_img)
            valid_pred = ~np.isinf(pred_flow).any(axis=0)
            x_valid = x[valid_pred]
            y_valid = y[valid_pred]
            u = pred_flow[0][valid_pred]
            v = pred_flow[1][valid_pred]
            
            n_arrows = len(x_valid[::stride])
            colors = np.random.rand(n_arrows, 3)
            
            axes[i,3].quiver(x_valid[::stride], y_valid[::stride],
                           u[::stride], v[::stride],
                           color=colors, angles='xy', scale_units='xy', scale=1,
                           width=0.002, headwidth=3)
            axes[i,3].set_title('Predicted Flow')
            axes[i,3].axis('off')

        plt.tight_layout()

        if save_path is not None:
            save_dir = Path(save_path).parent
            save_dir.mkdir(exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()