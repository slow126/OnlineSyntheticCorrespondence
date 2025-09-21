from typing import List, Optional, Union

import torch
from torch.nn.functional import normalize

from ..components import correlate, mutual_nn, upsample_correlation
from ..components import Conv4d
from ..components import FeatureExtractor
from ..components import KernelSoftArgmax
from ..components import MultiScaleSpatialContext

from ..components.conv_fft import Conv4d_fft


def create_conv_4d(k1, k2, channels):
    ks = (k1, k1, k2, k2)
    # padding = tuple(k // 2 for k in ks)
    return torch.nn.Sequential(
        Conv4d_fft(channels[0], channels[1], ks, padding='same'),
        torch.nn.ReLU()
    )


class NonIsotropicNC(torch.nn.Module):
    def __init__(self, symmetric_mode=True):
        super().__init__()
        self.symmetric_mode = symmetric_mode
        self.conv00 = create_conv_4d(3, 5, [1, 8])
        self.conv01 = create_conv_4d(5, 5, [1, 8])
        self.conv10 = create_conv_4d(3, 5, [16, 8])
        self.conv11 = create_conv_4d(5, 5, [16, 8])
        self.conv2 = create_conv_4d(5, 5, [16, 1])

    @staticmethod
    def _symmetric_conv(layer: torch.nn.Module, x: torch.Tensor):
        pmute = (0, 1, 4, 5, 2, 3)
        return layer(x) + layer(x.permute(pmute)).permute(pmute)

    def forward(self, x: torch.Tensor):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to
            # add together. Because of the ReLU layers in between linear layers, this operation is different than
            # convolving a single time with the filters+filters^T and therefore it makes sense to do this.
            x0 = self._symmetric_conv(self.conv00, x)
            x1 = self._symmetric_conv(self.conv01, x)
            x = torch.cat((x0, x1), 1)
            x0 = self._symmetric_conv(self.conv10, x)
            x1 = self._symmetric_conv(self.conv11, x)
            x = torch.cat((x0, x1), 1)
            x = self._symmetric_conv(self.conv2, x)
        else:
            x0 = self.conv00(x)
            x1 = self.conv01(x)
            x = torch.cat((x0, x1), 1)
            x0 = self.conv10(x)
            x1 = self.conv11(x)
            x = torch.cat((x0, x1), 1)
            x = self.conv2(x)
        return x


def max_pool4d(x: torch.Tensor, k: int=4):
    '''Max pooling with kernel size k and stride k.
    '''
    b, s = x.shape[:2]
    d = s // k

    shape1 = (b, ) + (d, k) * 4
    pmute = (0, ) + tuple(range(1, 9, 2)) + tuple(range(2, 9, 2))

    vals, idx = x.reshape(shape1).permute(pmute).flatten(-4).max(-1)

    k2 = k * k
    k3 = k2 * k
    max_i = idx.floor_divide(k3)
    max_j = idx.fmod(k3).floor_divide_(k2)
    max_m = idx.fmod(k2).floor_divide_(k)
    max_n = idx.fmod(k)

    return vals, max_i, max_j, max_m, max_n


class ANCNet(torch.nn.Module):
    """ANCNet: Adaptive Neighborhood Consensus Networks

    Args:
        model_name (str): name of the backbone network.
        layer_names (str | list of str): name(s) of layer(s) where features will be extracted from.
        nc_channels (list of int): number of channels for 4D neighborhood consensus convolutions.
            Default: [1, 16, 16, 1].
        relocalization_k_size (int): kernel size for relocalization. Default: 0.
        freeze (bool): whether to freeze the backbone model parameters during training.
    """
    def __init__(
        self,
        model_name: str="resnet101",
        layer_names: Union[str, List[str]]='layer3.22.bn3',
        corr_size: int=16,
        relocalization_k_size: int=0,
        weights: Optional[str] = None,
        freeze: bool=False,
    ):
        super().__init__()
        self.relocalization_k_size = relocalization_k_size

        self.feature_extractor = FeatureExtractor(layer_names, model_name, weights=weights, freeze=freeze)
        self.layer_names = self.feature_extractor.layer_names

        self.neighbor_consensus = NonIsotropicNC()
        self.spatial_context = MultiScaleSpatialContext(3, output_channel=32)

        self.soft_argmax = KernelSoftArgmax(16, apply_kernel=True, normalized=False)

        if freeze:
            self.feature_extractor.eval()

        self.corr_size = corr_size

    def ncnet(self, corr: torch.Tensor):
        corr = mutual_nn(corr)
        corr = corr.unsqueeze(1)
        corr = self.neighbor_consensus(corr)
        corr = corr.squeeze(1)
        corr = mutual_nn(corr)
        return corr

    def forward_corr(self, src_img, trg_img):
        layer = self.layer_names[0]
        src_feat = self.feature_extractor(src_img)[layer] # (B, C, H, W)
        trg_feat = self.feature_extractor(trg_img)[layer] # (B, C, H, W)

        # feature correlation
        corr = correlate(src_feat, trg_feat)
        # corr = upsample_correlation(corr, self.corr_size)
        corr = self.ncnet(corr)

        # self-similarity feature correlation
        src_ss = self.spatial_context(src_feat)
        trg_ss = self.spatial_context(trg_feat)
        corr_ss = correlate(src_ss, trg_ss)
        # corr_ss = upsample_correlation(corr, self.corr_size)
        corr_ss = self.ncnet(corr_ss)

        corr = 0.5 * (corr + corr_ss)
        return corr

    def forward_matches(self, corr: torch.Tensor):
        # NOTE: would it be better to do bidirectional??
        match_grid = self.soft_argmax(corr, as_tuple=False).flip(-1) # xy -> yx
        return match_grid

    def forward_covisible(self, corr: torch.Tensor):
        # ab = normalize(corr.flatten(3, 4), p=2, dim=-1)
        # ba = normalize(corr.flatten(1, 2), p=2, dim=1)
        # ab = corr.sum((3, 4)).mul(3.5).tanh()
        # ba = corr.sum((1, 2)).mul(3.5).tanh()
        ab = corr.sum((3, 4))
        ba = corr.sum((1, 2))
        return torch.stack((ab, ba), -1)

    def forward(self, src_img, trg_img):
        return self.forward_corr(src_img, trg_img)