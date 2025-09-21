from typing import Optional, Union

from . import scorrsan
from ..base import BaseCorrespondenceModel
from ..components.objectives import endpoint_error, EndpointErrorLoss, EndpointErrorPseudoFilteredLoss
from ..utils import PercentCorrectKeypoints
from src.data.flow_utils import flow2kps


class SCorrSAN(BaseCorrespondenceModel):
    def __init__(
        self,
        model_params: Optional[dict] = None,
        checkpoint: Optional[str] = None,
        finetune: Optional[Union[bool, str]] = 'auto',
        evaluate_pck: bool = False,
        pck_alpha: float = 0.1,
        lr: float = 1e-3,
        ft_lr_scale: float = 0.01, 
        warmup: float = 0.05,
        weight_decay: float = 1e-4,
    ):
        super().__init__(lr, ft_lr_scale, warmup, weight_decay)

        model_params = model_params or dict()
        self.model = scorrsan.SCorrSAN(**model_params)
        self.loss_fn = EndpointErrorLoss()

        if checkpoint is not None:
            self.load_model_weights(checkpoint)

        self.finetune_param_names = []
        self._set_finetuning(model_params, checkpoint, finetune)

        self.evaluate_pck = evaluate_pck
        if evaluate_pck:
            self.pck_metric = PercentCorrectKeypoints(alpha=pck_alpha)

    def flow_step(self, batch):
        img_source = batch['src_img']
        img_target = batch['trg_img']
        flow_target = batch['flow']
        flow_pred = self.model(img_target, img_source)
        return flow_pred, flow_target

    def training_step(self, batch, *args, **kwargs):
        B = batch['src_img'].shape[0]
        flow_pred, flow_target = self.flow_step(batch)

        # calculate and log loss
        loss = self.loss_fn(flow_pred, flow_target)
        self.log('train/loss', loss, batch_size=B)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        B = batch['src_img'].shape[0]
        flow_pred, flow_target = self.flow_step(batch)

        # calculate and log loss
        loss = self.loss_fn(flow_pred, flow_target)
        self.log('val/loss', loss, batch_size=B, prog_bar=True)

        # calculate pck
        if self.evaluate_pck:
            estimated_kps = flow2kps(batch['trg_kps'], flow_pred, batch['n_pts'])
            target_kps = batch['src_kps']
            pck = self.pck_metric(estimated_kps, target_kps, batch['n_pts'], batch['pckthres'])
            alpha = self.pck_metric.alpha
            self.log(f'val/pck@{alpha:.2f}', pck, prog_bar=True, sync_dist=True)

        return loss


class SCorrSANStudentTeacher(BaseCorrespondenceModel):
    def __init__(
        self,
        model_params: Optional[dict] = None,
        checkpoint: Optional[str] = None,
        finetune: Optional[Union[bool, str]] = 'auto',
        pseudo_loss_kernel_size: int = 7,
        pseudo_loss_weight: float = 10.0,
        remember_rate: float = 0.2,
        remember_warmup: float = 0.1,
        pseudo_loss_start: int = 4,
        evaluate_pck: bool = False,
        pck_alpha = 0.1,
        lr: float = 1e-3,
        ft_lr_scale: float = 1e-2,
        warmup: float = 0.1,
        weight_decay = 1e-4,
    ):
        super().__init__(lr, ft_lr_scale, warmup, weight_decay)

        model_params = model_params or dict()
        self.student = scorrsan.SCorrSAN(**model_params)
        self.teacher = scorrsan.SCorrSAN(**model_params)
        
        # pseudo weight starts at 0 (turned off) initially, and gets turned on at epoch=pseudo_loss_start
        self.loss_fn = EndpointErrorPseudoFilteredLoss(pseudo_loss_kernel_size, pseudo_weight=0)

        self.ft_param_names = []
        self._set_finetuning(model_params, checkpoint, finetune)

        self.evaluate_pck = evaluate_pck
        if evaluate_pck:
            self.pck_metric_st = PercentCorrectKeypoints(alpha=pck_alpha)
            self.pck_metric_tc = PercentCorrectKeypoints(alpha=pck_alpha)

        self.pseudo_loss_weight = pseudo_loss_weight
        self.pseudo_loss_start = pseudo_loss_start

        self.remember_rate_base = remember_rate
        self.remember_warmup = remember_warmup

        self.lr = lr
        self.ft_lr_scale = ft_lr_scale
        self.warmup = warmup
        self.weight_decay = weight_decay

    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch
        # start using pseudo loss after pseudo_loss_start epochs
        if epoch == self.pseudo_loss_start:
            self.loss_fn.pseudo_weight = self.pseudo_loss_weight

        # remember_rate schedule
        # "remember_warmup" is called "num_gradual" in the original implementation
        cutoff = int(self.remember_warmup * self.trainer.max_epochs)
        if epoch < self.pseudo_loss_start:
            self.remember_rate = self.remember_rate_base
        elif epoch < cutoff + self.pseudo_loss_start:
            t = (epoch - self.pseudo_loss_start) / cutoff
            base = self.remember_rate_base
            self.remember_rate = base + t * (0.9 - base)
        else:
            self.remember_rate = 0.9

    def flow_step(self, batch):
        img_source = batch['src_img']
        img_target = batch['trg_img']
        flow_target = batch['flow']
        flow_pred_student = self.student(img_target, img_source)
        flow_pred_teacher = self.teacher(img_target, img_source)
        return flow_pred_student, flow_pred_teacher, flow_target

    def training_step(self, batch, *args, **kwargs):
        flow_pred_student, flow_pred_teacher, flow_target = self.flow_step(batch)

        # calculate and log loss
        loss = self.loss_fn(flow_pred_student, flow_pred_teacher, flow_target, self.remember_rate)
        self.log('train/loss', loss)

        # print GPU memory usage once at begining of training
        if self.trainer.global_step == 2:
            self.print_gpu_memory()

        return loss

    def validation_step(self, batch, *args, **kwargs):
        flow_pred_student, flow_pred_teacher, flow_target = self.flow_step(batch)

        # calculate and log loss
        loss_student = endpoint_error(flow_pred_student, flow_target)
        loss_teacher = endpoint_error(flow_pred_teacher, flow_target)
        self.log('val/loss_st', loss_student, prog_bar=True)
        self.log('val/loss_tc', loss_teacher, prog_bar=True)

        # calculate pck
        if self.evaluate_pck:
            alpha = self.pck_metric_st.alpha
            target_kps, n_pts, pckthres = batch['src_kps'], batch['n_pts'], batch['pckthres']
            estimated_kps_student = flow2kps(batch['trg_kps'], flow_pred_student, batch['n_pts'])
            estimated_kps_teacher = flow2kps(batch['trg_kps'], flow_pred_teacher, batch['n_pts'])
            pck_student = self.pck_metric_st(estimated_kps_student, target_kps, n_pts, pckthres)
            pck_teacher = self.pck_metric_tc(estimated_kps_teacher, target_kps, n_pts, pckthres)
            self.log(f'val/st_pck@{alpha:.2f}', pck_student, prog_bar=True, sync_dist=True)
            self.log(f'val/tc_pck@{alpha:.2f}', pck_teacher, prog_bar=True, sync_dist=True)

        return loss_student, loss_teacher