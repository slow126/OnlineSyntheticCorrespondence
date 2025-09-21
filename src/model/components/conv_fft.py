# Differentiable FFT Conv Layer with Dense Color Channels
# Copyright 2022
# released under MIT license
# https://github.com/jkuli-net/ConvFFT/blob/main/ConvFFTTorch1.py

# this is meant to be a drop in replacement for torch.conv
# functional_conv1d_fft  replaces  torch.nn.functional.conv1d
# Conv1d_fft             replaces  torch.nn.Conv1d
# supports 1d, 2d and 3d convolution

# api is not exactly matching yet
# unsupported:  stride, dilation, groups, etc

# This version modified from the original by Connor Anderson, 2023

import math
from typing import List, Optional, Tuple, Union

import torch


class conv_fft_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, bias=None, padding = 'same', fft_dim = 1):   
          
        # channel first format only
        
        # if these dims are missing, need to skip the sum_reduce
        if x.dim() < fft_dim + 2:
            raise NotImplementedError(
                'vector input to conv_fft expected to have shape '
                '(batch, in_channels, data_dim0, ..., data_dimN)')
        if k.dim() < fft_dim + 2:
            raise NotImplementedError(
                'kernel input to conv_fft expected to have shape '
                '(out_channels, in_channels, data_dim0, ..., data_dimN)')
            
        # the axes where fft is calculated
        fft_axes = list(range(-fft_dim, 0))
        
        # kernel size along fft_axes
        kernel_size = k.shape[-fft_dim:]

        # input, padded, and output sizes along fft_axes, padded is the size used for fft
        input_size = x.shape[-fft_dim:]
        padded_size = list(x.shape[-fft_dim:])
        output_size = x.shape[-fft_dim:]             

        if padding=='valid':
            output_size = [input_size[i] - (kernel_size[i] - 1) for i in range(fft_dim)]
        elif padding=='same':
            padded_size = [input_size[i] + (kernel_size[i] // 2) for i in range(fft_dim)]
        elif isinstance(padding, int):
            padded_size = [input_size[i] + padding * 2 for i in range(fft_dim)]
            output_size = [padding * 2 + input_size[i] - (kernel_size[i] - 1) for i in range(fft_dim)] 

        # the kernel needs to be rolled, all other data are aligned to zero
        kernel_roll =   [-((size - 1) // 2) for size in kernel_size]      
        kernel_unroll = [((size - 1) // 2) for size in kernel_size] 

        # corrections to padding
        #  padded_size will be the size of the fft, any larger paddings should work, other sizes might be faster
        # 'valid' and other strange paddings cause a correction to kernel_roll, other data remain aligned to zero
        
        for i in range(fft_dim):
            # for example, if you only want even size fft
            # if padded_size[i] & 1:
            #    padded_size[i] = padded_size[i] + 1

            # pads to next largest multiple of 32
            # if padding != 'roll':
            #     padded_size[i] = padded_size[i] + 31 & ~31

            if padding == 'valid':
                offset = (min(kernel_size[i], input_size[i]) - 1) // 2
                kernel_roll[i] = kernel_roll[i] + offset
                kernel_unroll[i] = kernel_unroll[i] - offset

            if isinstance(padding, int):
                offset = (min(kernel_size[i], input_size[i]) - 1) // 2 - padding
                kernel_roll[i] = kernel_roll[i] + offset
                kernel_unroll[i] = kernel_unroll[i] - offset

        # the kernel gets padded up to padded_size before being rolled, slightly inefficient
        kernel_padding = [n for k in range(1, fft_dim + 1) for n in (0, padded_size[-k] - kernel_size[-k])]
        
        # outputs will be trimmed by these slices
        b_slice_size = [...] + [slice(0, output_size[i]) for i in range(fft_dim)]
        x_slice_size = [...] + [slice(0, input_size[i]) for i in range(fft_dim)]
        k_slice_size = [...] + [slice(0, kernel_size[i]) for i in range(fft_dim)]

        # force fft and complex number ops to run in single precision instead of half
        with torch.autocast(x.device.type, enabled=False):
            k = k.float()
            x = x.float()
            bias = bias.float()
            k_pad = torch.nn.functional.pad(k, kernel_padding)
            k_fft = torch.fft.rfftn(torch.roll(k_pad, kernel_roll, fft_axes), dim=fft_axes)
            x_fft = torch.fft.rfftn(x, dim=fft_axes, s=padded_size)
            y_fft = torch.einsum('bc...,oc...->bo...', x_fft, k_fft.conj())
        
            # bias is added to zero bin of fft, it needs scaled by prod(padded_size)
            if bias != None:
                y_fft[(..., ) + (0, ) * fft_dim] += bias * math.prod(padded_size)
                
            y = torch.fft.irfftn(y_fft, dim=fft_axes, s=padded_size)[b_slice_size]
                                    
        ctx.save_for_backward(x_fft, k_fft)
        ctx.my_saved_variables = [
            bias, 
            fft_dim,
            padded_size,
            kernel_unroll,
            fft_axes, 
            x_slice_size,
            k_slice_size
        ]

        return y 

    @staticmethod
    def backward(ctx, dz_dy):
        x_fft, k_fft = ctx.saved_tensors
        bias, fft_dim, padded_size, kernel_unroll, fft_axes, x_slice_size, k_slice_size = ctx.my_saved_variables

        with torch.autocast(dz_dy.device.type, enabled=False):
            dz_dy = dz_dy.float()
            dz_dy_fft = torch.fft.rfftn(dz_dy.float(), dim=fft_axes, s=padded_size).unsqueeze(2)
            
            # the zero freq dc bin of an fft ... is the sum of the signal ...
            # so dz_dbias[out_channel] = dz_db_fft[out_channel, 0, 0].real
            if bias != None:
                # this should instead sum all leading axes
                dz_dbias = torch.sum(dz_dy_fft[ (..., 0) + (0,) * fft_dim ], dim=0).real # sum along batch dim(s)
            else:
                dz_dbias = None
            
            dz_dx_fft = torch.sum(dz_dy_fft * k_fft, dim=-(fft_dim + 2)) # sum along out_channels dim
            dz_dx = torch.fft.irfftn(dz_dx_fft, dim=fft_axes, s=padded_size)[x_slice_size]
            
            # this should instead sum all leading axes
            # reshape(-1, out_c, in_c, *fft_size)
            # if i wanted broadcasted conv k=(extradim1, out, in, kernelsize), x=(extradim0, extradim1, in, kernelsize)
            # sum pre-channel axes (size>1) in dz_da_fft that are 1 or missing in k_fft.shape, keepdim if 1 is present
            dz_dk_fft = x_fft.unsqueeze(1).mul(dz_dy_fft.conj()).sum(dim=0)      # sum along batch dim(s)
            dz_dk = torch.roll(torch.fft.irfftn(dz_dk_fft, dim=fft_axes, s=padded_size), kernel_unroll, fft_axes)[k_slice_size]
        
        return dz_dx, dz_dk, dz_dbias, None, None


class Conv_fft(torch.nn.Module):
    '''
    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (list of int): kernel size in each dimension.
        bias (bool): whether to add a bias parameter.
        padding (int or str): amount or type of padding.
        device (str or torch.device): device to put the parameters on. Default is None.
        dtype (torch.dtype): data type for the parameters. Default is torch.float32.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple, List],
        bias: bool = True,
        padding: Union[int, str] = 0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(Conv_fft, self).__init__()

        self.padding = padding
                     
        weight = torch.zeros((out_channels, in_channels, *kernel_size), dtype=dtype, device=device)
        self.weight = torch.nn.Parameter(weight)
        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        
        if bias:
            bias = torch.zeros((out_channels,), dtype=dtype, device=device)
            self.bias = torch.nn.Parameter(bias)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = None

    def __repr__(self):
        s = (
            f'{self.__class__.__name__}('
            f'{self.weight.shape[1]}, {self.weight.shape[0]}, '
            f'kernel_size={tuple(self.weight.shape[2:])}, padding={self.padding})'
        )
        return s


class Conv1d_fft(Conv_fft):
    def __init__(self, *args, **kwargs):
        super(Conv1d_fft, self).__init__(*args, **kwargs)
                
    def forward(self, x):
        return conv_fft_function.apply(x, self.weight, self.bias, self.padding, 1)


class Conv2d_fft(Conv_fft):
    def __init__(self, *args, **kwargs):
        super(Conv2d_fft, self).__init__(*args, **kwargs)
                
    def forward(self, x):
        return conv_fft_function.apply(x, self.weight, self.bias, self.padding, 2)


class Conv3d_fft(Conv_fft):
    def __init__(self, *args, **kwargs):
        super(Conv3d_fft, self).__init__(*args, **kwargs)
                
    def forward(self, x):
        return conv_fft_function.apply(x, self.weight, self.bias, self.padding, 3)


class Conv4d_fft(Conv_fft):
    def __init__(self, *args, **kwargs):
        super(Conv4d_fft, self).__init__(*args, **kwargs)
                
    def forward(self, x):
        return conv_fft_function.apply(x, self.weight, self.bias, self.padding, 4)


def functional_conv1d_fft(x, k, bias=None, padding='valid'):
    return conv_fft_function.apply(x, k, bias, padding, 1)


def functional_conv2d_fft(x, k, bias=None, padding='valid'):
    return conv_fft_function.apply(x, k, bias, padding, 2)


def functional_conv3d_fft(x, k, bias=None, padding='valid'):
    return conv_fft_function.apply(x, k, bias, padding, 3)


def functional_conv4d_fft(x, k, bias=None, padding='valid'):
    return conv_fft_function.apply(x, k, bias, padding, 4)


if __name__ == '__main__':
    torch.manual_seed(123456789)

    for i in range(1, 4):
        print(f'Conv{i}d')
        print('--------')
        x = torch.normal(0, 1, (4, 32) + (16,) * i, device='cuda').requires_grad_(True)
        k = torch.normal(0, 0.1, (48, 32) + (3,) * i, device='cuda').requires_grad_(True)
        b = torch.normal(0, 0.1, (48,), device='cuda').requires_grad_(True)

        conv = getattr(torch.nn.functional, f'conv{i}d')
        conv_fft = globals()[f'functional_conv{i}d_fft']

        y1 = conv(x, k, b, padding=1)
        y1.mean().backward()
        g1 = []
        for t in [x, k, b]:
            g1.append(t.grad.clone())
            t.grad.zero_()

        y2 = conv_fft(x, k, b, padding='same')
        y2.mean().backward()
        g2 = []
        for t in [x, k, b]:
            g2.append(t.grad.clone())
            t.grad.zero_()
        
        y1 = y1.detach()
        y2 = y2.detach()

        ydiff = y1.sub(y2).abs()
        xgdiff = g1[0].sub(g2[0]).abs()
        kgdiff = g1[1].sub(g2[1]).abs()
        bgdiff = g1[2].sub(g2[2]).abs()

        def show_diff(a, b, label):
            s = '{:.2e}'
            ss = '{:4.2f}'
            amag = a.abs().mean().item()
            am = s.format(amag)
            bm = s.format(b.abs().mean().item())
            diff = a.sub(b).abs()
            dmin = diff.min().item()
            davg = diff.mean().item()
            dmax = diff.max().item()
            dmin = f'{s.format(dmin)} ({ss.format(dmin/amag*100)})'
            davg = f'{s.format(davg)} ({ss.format(davg/amag*100)})'
            dmax = f'{s.format(dmax)} ({ss.format(dmax/amag*100)})'

            print(f'{label+":":<12}  {am}  {bm}  {dmin}  {davg}  {dmax}')

        show_diff(y1, y2, 'Output')
        show_diff(g1[0], g2[0], 'Input grad')
        show_diff(g1[1], g2[1], 'Kernel grad')
        show_diff(g1[2], g2[2], 'Bias grad')
        print()