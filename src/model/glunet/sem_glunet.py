from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.functional import grid_sample, interpolate, normalize, relu

from ..components import correlate, local_correlate, mutual_nn
from ..components import FeatureExtractor
from ..components import NeighborhoodConsensus
from src.flow import convert_mapping_to_flow


default_layers = {
    'resnet50': ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
    'resnet101': ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
    'convnext_small': ('features.0', 'features.1', 'features.3', 'features.5', 'features.7'),
    'convnext_base': ('features.0', 'features.1', 'features.3', 'features.5', 'features.7'),
}


class SemanticGLUNet(torch.nn.Module):
    '''Semantic GLU-Net.

    Based on the model from Truong, et al., proposed in
        GLU-Net: Global-Local Universal Network for Dense Flow and Correspondence (CVPR 2020).

    Args:
        model_name (str): name of torchvision model for backbone feature extraction.
        layer_names (optional list of str): identifiers for intermediate layers of the backbone
            model where features will be extracted. If None, will attempt to use default settings
            for common backbones.
        nc_channels (list of int): number of channels for 4D convolution in the neighborhood
            consensus network.
        local_window_size (int): window size for local neighborhood correlation.
        weights (str): specifies pre-trained weights for the backbone model.
        freeze (bool): whether to freeze the backbone model weights.
    '''
    def __init__(
        self,
        model_name: str = 'resnet50',
        layer_names: Optional[Union[str, List[str], Tuple[str]]] = None,
        nc_channels: List[int] = (1, 10, 10, 1),
        local_window_size: int = 9,
        decoder_dense_connect: bool = True,
        weights: Optional[str] = None,
        model_weights: Optional[str] = None,
        freeze: bool = False,
        standardize: bool = True,
    ):
        super().__init__()
        self.local_window_size = local_window_size

        if layer_names is None:
            layer_names = default_layers.get(
                 model_name, ('relu', 'layer1', 'layer2', 'layer3', 'layer4'),
            )

        # Feature extractor backbone
        weights_path = None
        if weights is not None and Path(weights).exists():
            weights_path = weights
            weights = None
        self.feature_extractor = FeatureExtractor(
            layer_names=layer_names,
            model_name=model_name,
            weights=weights,
            weights_path=weights_path,
            freeze=freeze
        )
        self.layer_names = self.feature_extractor.layer_names
        num_layers = len(self.layer_names)

        # Neighborhood consensus network
        self.neighbor_consensus = NeighborhoodConsensus(channels=nc_channels)

        # Flow decoders for each level of the pyramid
        c = 2 + local_window_size**2
        self.decoders = torch.nn.ModuleList()
        self.decoders.append(ConvFlowDecoder(256, use_batch_norm=True))
        for i in range(1, num_layers - 1):
            # last decoder has two extra input channels from upsampled and projected hidden state
            cc = c if i < num_layers - 2 else c + 2
            self.decoders.append(
                ConvFlowDecoder(cc, use_batch_norm=True, dense_connect=decoder_dense_connect)
            )

        # Flow refinement networks
        self.coarse_flow_refiner = FlowRefinementNet(self.decoders[1].final_hidden_size)
        self.feature_projection = torch.nn.Conv2d(self.decoders[1].final_hidden_size, 2, 3, padding=1)
        self.fine_flow_refiner = FlowRefinementNet(self.decoders[-1].final_hidden_size)

        self.resize = partial(interpolate, mode='bilinear', align_corners=False, antialias=True)
        self.standardize = standardize

        if model_weights is not None:
            # load model weights from a trained checkpoint
            ckpt = torch.load(model_weights)['state_dict']
            ckpt = {k[len('model.'):]: v for k, v in ckpt.items()}
            self.load_state_dict(ckpt)

    def _extract_features(self, src_img, trg_img):
        src_feats = self.feature_extractor(src_img)
        trg_feats = self.feature_extractor(trg_img)
        if self.standardize:
            for f in (src_feats, trg_feats):
                for k in f:
                    f[k] = torch.nn.functional.instance_norm(f[k])
        return src_feats, trg_feats

    def _upsample_and_concat(self, x_high: torch.Tensor, *x_low):
        '''Resize each *x_low to the same size as x_high, and concatenate them all together along
        the channel dimension.
        '''
        s = x_high.shape[-2:]
        x_up = [self.resize(x, size=s) for x in x_low]
        return torch.cat((x_high, *x_up), dim=1)
    
    def _consensus(self, corr: torch.Tensor):
        '''Neighborhood consensus network with pre and post-processing using mutual nearest
        neighbor filtering.
        '''
        corr = mutual_nn(corr)
        corr = self.neighbor_consensus(corr.unsqueeze(1)).squeeze(1)
        corr = mutual_nn(corr)
        return corr
    
    def _warp(self, x: torch.Tensor, flow: torch.Tensor):
        '''Warp a feature tensor according to the given flow.
        '''
        h, w = x.shape[-2:]
        d = dict(device=x.device, dtype=x.dtype)

        s = x.new_tensor([w - 1, h - 1]).view(2, 1, 1)

        if not hasattr(self, 'grids'):
            self.grids = {}

        if (h, w) in self.grids:
            grid = self.grids[(h, w)]
        else:
            grid = torch.stack(torch.meshgrid(
                torch.arange(w, **d), torch.arange(h, **d), indexing='xy'
            ), dim=0)
            self.grids[(h, w)] = grid

        # convert to normalized coordinates [-1, 1] for grid_sample
        grid = grid + flow
        grid = grid.mul(2 / s).sub(1)
        grid = grid.moveaxis(1, -1)

        # using align_corners = True to match the original implementation
        warped = grid_sample(x, grid, align_corners=True)

        return warped

    def pyramid_level(
            self,
            level: int,
            src: torch.Tensor,
            trg: torch.Tensor,
            flow_upsample: torch.Tensor,
            ratio: float,
            extra_state=None
        ):
        '''Calculate the predicted flow at one level of the feature pyramid.

        Args:
            level (int): numeric level of the pyramid in [1, 2, 3].
            src (Tensor): source image features for this level, with shape (B, N, H, W).
            trg (Tensor): target image features for this level, with shape (B, N, H, W).
            flow_upsample (Tensor): upsampled predicted flow from previous level, with
                shape (B, 2, H, W).
            ratio (float): ratio between spatial dimensions at the current level versus
                the image dimensions.
            extra_state (optional Tensor): an extra set of features to concatenate with
                the correlation tensor before decoding the flow, with shape (B, M, H, W).
                Used in the last layer of the pyramid.

        Returns:
            (Tensor) The predicted flow.
            (Tensor) The final hidden state of the flow decoder.
        '''
        # 1. warp source features to target locations using predicted flow
        src_warp = self._warp(src, flow_upsample * ratio)

        # 2. local correlation
        corr = local_correlate(src_warp, trg, self.local_window_size)
        corr = corr.flatten(1, 2)
        corr = torch.nn.functional.leaky_relu(corr, 0.1)

        # 3. concatenate correlation with previous flow and decode
        corr_state = (corr, flow_upsample)
        if extra_state is not None:
            corr_state = corr_state + (extra_state, )
        corr = torch.cat(corr_state, 1)
        flow, hidden_state = self.decoders[level](corr)

        # 4. add residual flow
        flow = flow + flow_upsample

        return flow, hidden_state

    def forward(self, src_img: torch.Tensor, trg_img: torch.Tensor):
        img_h, img_w = src_img.shape[-2:]

        two_stage = img_h > 256 or img_w > 256

        if two_stage:
            src_img_sm = self.resize(src_img, size=(256, 256))
            trg_img_sm = self.resize(trg_img, size=(256, 256))
        else:
            src_img_sm = src_img
            trg_img_sm = trg_img

        src_feats, trg_feats = self._extract_features(src_img_sm, trg_img_sm)

        # layer names in order from coarse resolution (deeper layers) to fine
        lvls = self.layer_names[::-1]

        ### Bottom of the pyramid
        # upsample lowest level and concat with next up
        s = self._upsample_and_concat(src_feats[lvls[1]], src_feats[lvls[0]])
        t = self._upsample_and_concat(trg_feats[lvls[1]], trg_feats[lvls[0]])
        # correlation and neighborhood consensus
        corr = correlate(s, t)
        corr = self._consensus(corr)
        # decode flow: predicted flow trg->src for each target image location
        corr = corr.flatten(1, 2)
        corr = relu(corr)  # NOTE: this is reduntant, since the consensus network ends with relu
        corr = normalize(corr, p=2, dim=1)
        base_flow, _ = self.decoders[0](corr)
        base_flow = convert_mapping_to_flow(base_flow) * (256 / 16)

        base_flow_upsample = self.resize(base_flow, scale_factor=2)

        ### Pyramid level 1
        s = self._upsample_and_concat(src_feats[lvls[2]], src_feats[lvls[1]], src_feats[lvls[0]])
        t = self._upsample_and_concat(trg_feats[lvls[2]], trg_feats[lvls[1]], trg_feats[lvls[0]])
        level1_flow, hidden_state = self.pyramid_level(1, s, t, base_flow_upsample, 32 / 256)
        residual_refinement_flow = self.coarse_flow_refiner(hidden_state)
        level1_flow = level1_flow + residual_refinement_flow


        if two_stage:
            # only need to extract features from the first few layers (1/8 and 1/4 resolution)
            with self.feature_extractor.halting(self.feature_extractor.layer_names[2]):
                src_feats, trg_feats = self._extract_features(src_img, trg_img)

            img_dims = src_img.new_tensor([img_h, img_w]).view(1, 2, 1, 1)
            # s = (int(img_h / 8), int(img_w / 8))
            s = src_feats[lvls[2]].shape[-2:]
            flow_upsample = self.resize(level1_flow, size=s) * (img_dims / 256)
        else:
            flow_upsample = level1_flow

        # NOTE: currently not implementing the adaptive resolution refinement strategy (see last
        # paragraph of Section 3.3 in the paper); it would go here, between level 1 and level 2

        ### Pyramid level 2
        s, t = src_feats[lvls[2]], trg_feats[lvls[2]]
        level2_flow, hidden_state = self.pyramid_level(2, s, t, flow_upsample, 1 / 8)

        # flow_upsample = self.resize(level2_flow, scale_factor=2)
        size = src_feats[lvls[3]].shape[-2:]
        flow_upsample = self.resize(level2_flow, size=size)
        state_upsample = self.resize(self.feature_projection(hidden_state), size=size)

        ### Final pyramid level
        s, t = src_feats[lvls[3]], trg_feats[lvls[3]]
        level3_flow, hidden_state = self.pyramid_level(3, s, t, flow_upsample, 1 / 4, state_upsample)
        residual_refinement_flow = self.fine_flow_refiner(hidden_state)
        level3_flow = level3_flow + residual_refinement_flow

        return {
            'base': base_flow,
            'level1': level1_flow,
            'level2': level2_flow,
            'level3': level3_flow,
        }


class ConvFlowDecoder(torch.nn.Module):
    '''ConvFlowDecoder for decoding dense flow from correlation volume.

    Replaces CMDTop and OpticalFlowEstimator modules from the original repo.

    Args:
        in_channels (int): number of input feature channels.
        hidden_channels (list of int): number of hidden feature channels for each intermediate
            convolution layer. Default [128, 128, 96, 64, 32], follows the original implementation.
        use_batch_norm (bool): whether to use batch normalization in each layer.
        dense_connect (bool): whether to use dense connections between each convolution layer,
            DenseNet style. Default True.
    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = (128, 128, 96, 64, 32),
        use_batch_norm: bool = False,
        dense_connect: bool = False,
    ):
        super().__init__()

        self.dense_connect = dense_connect
        self.hidden_channels = hidden_channels

        self.layers = torch.nn.ModuleList()

        channels = in_channels
        for c in hidden_channels:
            layer = ConvLayer(channels, c, 3, use_batch_norm)
            self.layers.append(layer)
            if dense_connect:
                channels += c
            else:
                channels = c
        self.head = torch.nn.Conv2d(channels, 2, 3, padding=1, bias=True)

        self.final_hidden_size = channels

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            y = layer(x)
            if self.dense_connect:
                x = torch.cat((x, y), 1)
            elif x.shape[1] == y.shape[1]:
                x = x + y
            else:
                x = y
        flow = self.head(x)
        return flow, x
    

class FlowRefinementNet(torch.nn.Module):
    '''FlowRefinementNet for refining dense flow predictions; see Section 3.4 in the paper.

    In the original repo, the refinement network layers were named `dc_conv*` and `l_dc_conv*`.

    Args:
        in_channels (int): number of input feature channels.
        hidden_channels (list of int): number of hidden feature channels for each intermediate
            convolution layer. Default [128, 128, 128, 96, 64, 32], follows the original
            implementation.
        dilations (list of int): dilation factor for each convolution layer. Default
            [1, 2, 4, 8, 16, 1], follows the original implementation.
        use_batch_norm (bool): whether to use batch normalization in each layer.
    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = (128, 128, 128, 96, 64, 32),
        dilations: List[int] = (1, 2, 4, 8, 16, 1),
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        channels = in_channels
        for c, d in zip(hidden_channels, dilations):
            layer = ConvLayer(channels, c, 3, use_batch_norm, padding=d, dilation=d)
            self.layers.append(layer)
            channels = c
        self.head = torch.nn.Conv2d(channels, 2, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        return x


class ConvLayer(torch.nn.Module):
    '''ConvLayer, encapsulating Conv -> BatchNorm (optional) -> ReLU.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_batch_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        pad = kwargs.pop('padding', kernel_size // 2)
        bias = not use_batch_norm

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias, **kwargs)
        if use_batch_norm:
            self.norm = torch.nn.BatchNorm2d(out_channels)
        else:
            self.norm = torch.nn.Identity()
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x