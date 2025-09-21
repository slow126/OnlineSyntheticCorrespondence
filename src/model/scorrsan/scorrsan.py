'''
Code modified and adapted from https://github.com/shuaiyihuang/SCorrSAN
'''
from typing import Iterable, Optional, Union

import torch
from torch.nn.functional import normalize, pad

from ..components import correlate, upsample_correlation, mutual_nn
from ..components import EfficientSpatialContext, FeatureExtractor, KernelSoftArgmax
from ..utils import unnormalise_and_convert_mapping_to_flow


def default_layer_id(model_name):
    if model_name == 'resnet50':
        return ['layer3.5.bn3']
    elif model_name == 'resnet101':
        return ['layer3.22.bn3']
    else:
        raise RuntimeError(f'No default layer specified for model {model_name}')


class SCorrSAN(torch.nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet50',
        weights: Optional[str] = None,
        layer_ids: Optional[Union[str, Iterable]] = None,
        sce_ksize: int = 7,
        sce_outdim: int = 2048,
        feature_size: int = 64,
        freeze: bool = False,
        use_spatial_context: bool = True,
        use_kernel_softargmax: bool = False,
        learn_sigma: bool = False,
        learn_beta: bool = False,
    ):
        super().__init__()
        # Note that feature_size=64 is stride4 for input resolution 256x256
        if layer_ids is None:
            layer_ids = default_layer_id(model_name)
        elif isinstance(layer_ids, str):
            layer_ids = [layer_ids]
        self.feature_size = feature_size
        
        self.feature_extractor = FeatureExtractor(model_name, layer_ids, weights, freeze) 

        # quick forward pass with dummy input to determine output channels for each layer_id
        with torch.no_grad():
            _x = torch.rand(1, 3, 40, 40)
            feats = self.feature_extractor(_x)
            channels = [x.shape[1] for x in feats.values()]

        self.spatial_context_encoders = None
        if use_spatial_context:
            self.spatial_context_encoders = torch.nn.ModuleList([
                EfficientSpatialContext(kernel_size=sce_ksize, input_channel=c, output_channel=sce_outdim)
                for c in channels
            ])

        self.soft_argmax = KernelSoftArgmax(
            self.feature_size, apply_kernel=use_kernel_softargmax, learn_beta=learn_beta, learn_sigma=learn_sigma)
        self.mutual_nn_filter = mutual_nn
    
    def forward(self, target, source):
        src_feats: dict = self.feature_extractor(source)
        tgt_feats: dict = self.feature_extractor(target)

        # correlation maps
        corrs = []
        for i, (src, tgt) in enumerate(zip(src_feats.values(), tgt_feats.values())):
            if self.spatial_context_encoders is not None:
                src = self.spatial_context_encoders[i](src) 
                tgt = self.spatial_context_encoders[i](tgt)

            corr = correlate(src, tgt)
            corr = self.mutual_nn_filter(corr)
            corrs.append(corr)
        
        # resize correlation maps
        corrs_resize = []
        for i, corr in enumerate(corrs):
            corr_resize = upsample_correlation(corr, self.feature_size)
            corrs_resize.append(corr_resize)

        if len(corrs_resize) > 1:
            corrs_resize = torch.stack(corrs_resize, dim=1) # (B, nlayer, Lsf, hf, wf)
            refined_corr = torch.mean(corrs_resize, dim=1) # refined_corr shape: (b, hf, wf)
        else:
            refined_corr = corrs_resize[0]

        grid_x, grid_y = self.soft_argmax(refined_corr) # grid_x/y (b,1,hf,wf)

        flow_norm = torch.cat((grid_x, grid_y), dim=1) # (B, 2, ht, wt)
        flow = unnormalise_and_convert_mapping_to_flow(flow_norm)

        return flow