from pathlib import Path

import torch

from src.flow import flow_to_keypoints
from src.keypoints import PercentCorrectKeypoints, KittiF1


class Evaluation(object):
    def __init__(
        self,
        eval_metric: str,
        per_image_metrics: bool,
    ):
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

    @staticmethod
    def make_bidirectional(batch, src_key, trg_key, flow_key):
        sk, tk = src_key, trg_key
        fsk, ftk = sk + '_flow', tk + '_flow'

        combined = {
            sk: torch.cat((batch[sk], batch[tk]), 0),
            tk: torch.cat((batch[tk], batch[sk]), 0),
            flow_key: torch.cat((batch[ftk], batch[fsk])),
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
    
    def start(self):
        if self.per_image_metrics:
            self.error_list = []

    def log_errs(self, err):
        logs = {}

        if self.per_image_metrics:
            self.error_list.append(err)
        else:
            logs['val/err'] = err

        return logs

    def evaluate(self, batch: dict, pred: torch.Tensor, flow: torch.Tensor = None):
        met = {}

        if self.eval_type == 'pck':
            num_points = batch['n_pts']
            pck_dim = batch['pckthres']
            pred = flow_to_keypoints(batch['trg_kps'], pred, num_points, pred.shape[-2:])
            met['val/acc'] = self.eval_metric(pred, batch['src_kps'], num_points=num_points, pck_dim=pck_dim)
        elif self.eval_type == 'dense_pck':
            labels = batch.get('label')
            met['val/acc'] = self.eval_metric(pred, flow, labels=labels)
        elif self.eval_type == 'sintel_pck':
            valid = batch['correspondence_mask'][:, None].expand_as(flow)
            flow[~valid] = torch.inf
            met['val/acc'] = self.eval_metric(pred, flow)
        elif self.eval_type == 'kitti_f1':
            flow = flow.moveaxis(1, -1)
            pred = pred.moveaxis(1, -1)
            # invalid flow is defined with both flow coordinates set to inf
            mask = flow.isfinite().all(dim=-1, keepdim=True).expand_as(pred)
            pred = pred[mask].reshape(-1, 2)
            flow = flow[mask].reshape(-1, 2)

            epe = flow.sub(pred).norm(dim=-1)
            mag = flow.norm(dim=-1)
            met['val/acc'] = self.eval_metric(epe, mag)

        return met

    def end(self, logger=None):
        accs = {}
        if self.eval_type == 'dense_pck':
            accs = self.eval_metric.compute_by_class()
            if len(accs) > 1:
                for n, v in zip(self.names, accs):
                    accs[f'val/{n}-acc'] = v

        if self.per_image_metrics:
            errs = torch.cat(self.error_list).cpu()
            f = Path(logger.log_dir, 'per_img_errors.pth')
            torch.save(errs, f)

        return accs