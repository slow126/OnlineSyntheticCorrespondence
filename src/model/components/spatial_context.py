from typing import Callable, Union

import torch
from torch.nn.functional import normalize, pad


__all__ = [
    'SpatialContext',
    'MultiScaleSpatialContext',
    'EfficientSpatialContext',
]


def get_activation_func(activation):
    if isinstance(activation, str):
        activation = getattr(torch.nn.functional, activation)
    return activation


class SpatialContext(torch.nn.Module):
    def __init__(
        self,
        context_size: int=5,
        input_channel: int=1024,
        output_channel: int=1024,
        activation: Union[str, Callable]='relu',
    ):
        super().__init__()
        self.context_size = context_size
        self.pad = context_size // 2

        inc = input_channel + self.context_size**2
        self.conv = torch.nn.Conv2d(inc, output_channel, 1, bias=True, padding_mode='zeros')

        self.activation = get_activation_func(activation)

        nb = torch.arange(context_size)
        self.register_buffer('neighborhood', nb)

    def self_sim(self, feature: torch.Tensor):
        device = feature.device
        normalized = normalize(feature, p=2, dim=1)
        padded = pad(normalized, (self.pad, ) * 4, 'constant', 0)

        h, w = normalized.shape[-2:]
        pw = padded.shape[-1]

        normalized = normalized.flatten(2).transpose(1, 2)
        padded = padded.flatten(2).transpose(1, 2)

        idx = torch.arange(h * w, device=device).unsqueeze_(1)
        i = idx.div(w, rounding_mode='floor')
        j = idx.fmod(w)

        kidx = self.neighborhood.mul(pw).unsqueeze_(1).add(self.neighborhood).flatten(0)
        select = i.mul(pw).add_(j).add(kidx)

        nel = self.context_size * self.context_size
        output = padded[:, select].matmul(normalized.unsqueeze(-1)).squeeze_(-1)
        output = output.transpose(1, 2).reshape(-1, nel, h, w)
        return output

    def forward(self, feature: torch.Tensor):
        b, c, h, w = feature.shape
        feature_normalized = normalize(feature, p=2, dim=1)
        feature_pad = pad(feature_normalized, (self.pad, ) * 4, 'constant', 0)

        # TODO: remove the loops
        output = torch.zeros(
            [self.context_size * self.context_size, b, h, w],
            dtype=feature.dtype,
            device=feature.device,
            requires_grad=feature.requires_grad,
        )
        for c in range(self.context_size):
            for r in range(self.context_size):
                output[c * self.context_size + r] = (
                    feature_pad[:, :, r : (h + r), c : (w + c)] * feature_normalized
                ).sum(1)

        output = output.transpose(0, 1).contiguous()
        output = torch.cat((feature, output), 1)
        output = self.conv(output)
        output = self.activation(output)
        return output


class MultiScaleSpatialContext(SpatialContext):
    # ANC-Net
    def __init__(
        self,
        context_size=5,
        output_channel=128,
        activation: Union[str, Callable]='relu',
    ):
        torch.nn.Module.__init__(self)
        self.context_size = context_size
        self.pad = context_size // 2

        inc = self.context_size * self.context_size
        self.conv1 = torch.nn.Conv2d(inc, output_channel * 2, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(output_channel * 2, output_channel, 3, padding=1)

        self.activation = get_activation_func(activation)

        nb = torch.arange(context_size)
        self.register_buffer('neighborhood', nb)

    def forward(self, feature: torch.Tensor):
        feature_normalized = normalize(feature, p=2, dim=1)
        ss = self.self_sim(feature_normalized)

        ss1 = self.activation(self.conv1(ss))
        ss2 = self.activation(self.conv2(ss1))
        output = torch.cat((ss, ss1, ss2), 1)
        return output


class EfficientSpatialContext(torch.nn.Module):
    '''Efficient Spatial Context.
    
    Computes a self-similarity vector, using a sparse kernel (4 * K) instead of a dense one (K^2), which
    is processed jointly with the features.

    Huang, et al., "Learning Semantic Correspondence with Sparse Annotations", ECCV 2022.

    Modified implementation, based on
    github.com/ShuaiyiHuang/SCorrSAN/blob/bc06425a3f1af4c0d7c878bed5f42ff9d468fbab/models/model/scorrsan.py#L128
    '''
    def __init__(
        self,
        kernel_size: int=5,
        input_channel: int=1024,
        output_channel: int=1024,
        activation: Union[str, Callable]='relu',
    ):
        super(EfficientSpatialContext, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            input_channel + 4 * (self.kernel_size - 1),
            output_channel,
            1,
            bias=True,
            padding_mode="zeros",
        )

        self.activation = get_activation_func(activation)

        nb = torch.arange(kernel_size-1)
        nb[kernel_size//2:].add_(1)
        self.register_buffer('neighborhood', nb)

    def self_sim(self, normalized, padded):
        # NOTE: this is equivalent to the original implementation, but is more compact and efficient.
        # Additionally, it doesn't compute the central self-similarity (similarity between the feature
        # with itself, which is always 1), which was computed 4x in the original implementation.
        h, w = normalized.shape[-2:]
        pw = padded.shape[-1]

        # flatten spatial dimensions and move feature dimension to end
        normalized = normalized.flatten(2).permute(0, 2, 1)
        padded = padded.flatten(2).permute(0, 2, 1)

        idx = torch.arange(h * w, device=normalized.device).unsqueeze_(1)
        i = idx.div(w, rounding_mode='floor')
        j = idx.fmod(w)

        ks = self.kernel_size

        d = {'dtype': torch.int64, 'device': normalized.device}
        # the order goes:
        # 1) diagonal top-left to bottom right
        # 2) center top to bottom (column)
        # 3) diagonal top right to bottom left
        # 4) center left to right (row)
        ipw = i.mul(pw).unsqueeze_(-1).expand(-1, 4, 1)  # (h * w, 4, 1)
        base_offset = j.add(torch.tensor([0, ks // 2, ks - 1, (ks // 2) * pw], **d)).view(-1, 4, 1)
        step_offset = torch.tensor([pw + 1, pw, pw - 1, 1], **d).view(1, 4, 1).mul(self.neighborhood)

        # index into `padded`, where each row contains 4 * (kernel_size - 1) indices that will select the
        # features to be compared to for each central feature (from `normalized`)
        nel = 4 * (ks - 1)
        select = (ipw + base_offset + step_offset).view(-1, nel)  # (h * w, 4 * kernel_size)

        # compute dot products as batched matrix multiply: (b, hw, 4k, c) x (b, hw, c, 1) => (b, hw, 4k)
        output = padded[:, select, :].matmul(normalized.unsqueeze(-1)).squeeze_(-1)
        output = output.permute(0, 2, 1).reshape(-1, nel, h, w)
        return output

    def forward(self, feature, v=1):
        normalized = normalize(feature, p=2, dim=1)
        padded = pad(normalized, (self.pad,) * 4, 'constant', 0)

        output = self.self_sim(normalized, padded)
        output = torch.cat((feature, output), 1)
        output = self.conv(output)
        output = self.activation(output)

        return output