import pytorch_lightning as pl
import torch


__all__ = [
    'BaseCorrespondenceModel',
]


class BaseCorrespondenceModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        ft_lr_scale: float = 0.01, 
        warmup: float = 0.05,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.lr = lr
        self.ft_lr = lr * ft_lr_scale
        self.warmup = warmup
        self.weight_decay = weight_decay

    def load_model_weights(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['state_dict']
        self.load_state_dict(state)

    def _set_finetuning(self, model_params, checkpoint=None, finetune=None):
        ft_backbone = False
        if finetune is not None:
            if isinstance(finetune, bool):
                ft_backbone = finetune
            elif finetune == 'auto':
                ft_backbone = model_params.get('weights', None) is not None or checkpoint is not None

        if finetune == True:
            for name, _ in self.named_parameters():
                self.finetune_param_names.append(name)
        elif ft_backbone:
            # fine-tuning pre-trained feature extractor
            base_key = 'model.feature_extractor.backbone'
            backbone = self.model.feature_extractor.backbone
            for name, _ in backbone.named_parameters(prefix=base_key):
                self.finetune_param_names.append(name)

        # Sanity check for bugs
        param_names = set([n for n, _ in self.named_parameters()])
        if not all(n in param_names for n in self.finetune_param_names):
            print([n for n in self.finetune_param_names if n not in param_names])
            raise RuntimeError('Some of the finetune param names are wrong!')

    def print_gpu_memory(self):
        avail, total = torch.cuda.mem_get_info()
        mem_used = 100 * (1 - (avail / total))
        gb = 1024**3
        self.print(f'GPU memory used: {(total-avail)/gb:.2f} of {total/gb:.2f} GB ({mem_used:.2f}%)')

    def on_train_epoch_start(self) -> None:
        if self.current_epoch > 0:
            if hasattr(self.trainer.datamodule.train_data, 'set_sample_pairs'):
                self.trainer.datamodule.train_data.set_sample_pairs()
        if hasattr(self.trainer.datamodule, 'on_train_epoch_start'):
            self.trainer.datamodule.on_train_epoch_start(self.trainer)

    def on_train_batch_start(self, batch, batch_idx: int):
        if hasattr(self.trainer.datamodule, 'adjust_warp_strength'):
            self.trainer.datamodule.adjust_warp_strength(
                batch_idx, self.trainer.current_epoch, self.trainer.num_training_batches
            )

    def on_train_batch_end(self, *args, **kwargs):
        # print GPU memory usage once at begining of training
        if self.trainer.global_step == 2:
            self.print_gpu_memory()

    def on_validation_model_train(self):
        super().on_validation_model_train()
        for m in self.modules():
            ps = list(m.parameters(recurse=False))
            if len(ps) and not any(p.requires_grad for p in ps):
                m.eval()

    def configure_optimizers(self):
        # create parameter groups such that:
        # 1. parameters whos names show up in the (optional) list self.finetune_param_names have a learning
        #    rate scaled down by 10x
        # 2. all layer norm and bias parameters have no weight decay applied to them
        verbose = True
        ft_param_names = set(getattr(self, 'finetune_param_names', []))
        param_groups = {}
        for key in ('scratch', 'finetune'):
            lr = self.lr
            wd = getattr(self, 'weight_decay', 1e-3)
            if key == 'finetune':
                lr = self.ft_lr
            param_groups[key] = {
                'decay': {'params': [], 'weight_decay': wd, 'lr': lr},
                'no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr},
            }
        for modname, module in self.named_modules():
            for name, param in module.named_parameters(prefix=modname, recurse=False):
                if not param.requires_grad: continue
                if name in ft_param_names:
                    k1 = 'finetune'
                else:
                    k1 = 'scratch'
                k2 = 'decay'
                if 'Norm' in module.__class__.__name__ or name.endswith('bias'):
                    k2 = 'no_decay'
                elif hasattr(module, 'no_decay'):
                    if name[name.rfind('.')+1:] in module.no_decay:
                        k2 = 'no_decay'
                if verbose:
                    self.print(k1, k2, name)
                param_groups[k1][k2]['params'].append(param)

        param_groups = [x for g1 in param_groups.values() for x in g1.values()]
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr = [g['lr'] for g in optimizer.param_groups],
                total_steps = self.trainer.estimated_stepping_batches,
                pct_start = self.warmup,
            ),
            'interval': 'step',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}