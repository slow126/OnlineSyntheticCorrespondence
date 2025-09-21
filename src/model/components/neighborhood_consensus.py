from typing import List

import torch

from .conv_fft import Conv4d_fft


all = [
    'NeighborhoodConsensus',
]


class NeighborhoodConsensus(torch.nn.Module):
    '''Neighborhood consensus network using 4D convolutions.

    First proposed in "Rocco, et al. Neighborhood Consensus Networks. NeurIPS 2018".
    
    This implementation uses FFT for calculating the 4D convolutions.
    '''
    def __init__(
        self,
        kernel_size: List[int] = [3, 3, 3],
        channels: List[int] = [1, 16, 16, 1],
        symmetric_mode: bool = True,
        inplace_relu: bool = False,
    ):
        super().__init__()
        self.symmetric_mode = symmetric_mode

        layers = []
        for inc, outc, k in zip(channels[:-1], channels[1:], kernel_size):
            ks = [k] * 4
            conv = Conv4d_fft(inc, outc, ks, padding='same')
            relu = torch.nn.ReLU(inplace=inplace_relu)
            layers.extend([conv, relu])

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        if self.symmetric_mode:
            # Apply to x and x transpose (swap source and target order in correlation tensor) and sum the result
            # Not equivalent to convolving once with filters and filters.T due to activation functions
            p = (0, 1, 4, 5, 2, 3)
            x = self.layers(x) + self.layers(x.permute(p)).permute(p)
        else:
            x = self.layers(x)

        return x