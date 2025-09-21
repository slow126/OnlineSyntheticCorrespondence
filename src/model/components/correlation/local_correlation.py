import numba
from numba import cuda
import torch


__all__ = [
    'local_correlate',
]


class LocalCorrelation(torch.autograd.Function):
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, k: int):
        x1 = x1.moveaxis(1, -1).float().contiguous()
        x2 = x2.moveaxis(1, -1).float().contiguous()

        ctx.save_for_backward(x1, x2)
        ctx.k = k

        b, h, w, c = x1.shape
        corr = x1.new_zeros(b, k, k, h, w)

        # based on some limited testing, 64 threads seems to offer pretty good comparative performance
        blocks = (w, h, b)
        threads = (min(64, c), 1, 1)
        stream = cuda.external_stream(torch.cuda.current_stream(x1.device).cuda_stream)
        mem = 4 * (c + threads[0])
        compute_local_correlation[blocks, threads, stream, mem](
            cuda.as_cuda_array(corr),
            cuda.as_cuda_array(x1),
            cuda.as_cuda_array(x2),
            k
        )

        return corr

    def backward(ctx, grad_output):
        x1, x2 = ctx.saved_tensors
        k = ctx.k

        b, h, w, c = x1.shape

        blocks = (w, h, b)
        threads = (min(64, c), 1, 1)
        stream = cuda.external_stream(torch.cuda.current_stream(x1.device).cuda_stream)
        mem = 4 * c

        grad_x1 = torch.zeros_like(x1)
        grad_x2 = torch.zeros_like(x2)
        grad_output = cuda.as_cuda_array(grad_output)

        local_correlation_grad_x1[blocks, threads, stream, mem](
            cuda.as_cuda_array(grad_x1),
            grad_output,
            cuda.as_cuda_array(x2),
            k
        )

        local_correlation_grad_x2[blocks, threads, stream, mem](
            cuda.as_cuda_array(grad_x2),
            grad_output,
            cuda.as_cuda_array(x1),
            k
        )

        grad_x1 = grad_x1.moveaxis(-1, 1).contiguous()
        grad_x2 = grad_x2.moveaxis(-1, 1).contiguous()

        return grad_x1, grad_x2, None


def local_correlate(x1: torch.Tensor, x2: torch.Tensor, k: int):
    '''
    '''
    if not x1.is_cuda or not x2.is_cuda:
        raise RuntimeError('The function local_correlate requires all tensors to be cuda tensors')
    
    corr = LocalCorrelation.apply(x1, x2, k)

    return corr


@cuda.jit
def compute_local_correlation(corr, x1, x2, k):
    '''CUDA kernel for computing local correlation between x1 and x2.

    corr has shape (b, k, k, h, w)
    x1 and x2 have shape (b, h, w, c)
    k is the local window size

    Approach:
        We have a thread block for each spatial location in the target image. Each thread block
        has a number of threads that together calculate the k^2 correlation values for that
        location.
        We load the target vector for this location into shared memory, since it will be reused
        multiple times.
    '''
    # dynamic shared memory, size gets passed as part of kernel launch
    # size should be (number of channels + number of threads)
    trg_vector = cuda.shared.array(0, dtype=numba.float32)
    # last (number of threads) elements are for storing intermediate calculations of the inner
    # product
    inner_prod = trg_vector[x1.shape[-1]:]

    x = cuda.blockIdx.x  # x location
    y = cuda.blockIdx.y  # y location
    b = cuda.blockIdx.z  # batch index

    thread = cuda.threadIdx.x  # thread index

    ### Load target vector into shared memory
    for c in range(thread, x2.shape[-1], cuda.blockDim.x):
        trg_vector[c] = x2[b, y, x, c]
    
    cuda.syncthreads()

    h, w = x1.shape[1], x1.shape[2]

    ### Compute the inner product
    # First, each thread computes one or more of the single-element products (for a particular
    #   feature channel) between the target  vector and a source vector in the local neighborhood,
    #   summing up locally for each thread
    # Then, a reduction is performed to add all the partial sums

    # looping over the local neighborhood in x1
    for i in range(k):
        row = y + i - k // 2
        if row < 0 or row >= h: continue
        for j in range(k):
            col = x + j - k // 2
            if col < 0 or col >= w: continue
            # compute partial sums for each thread
            inner_prod[thread] = 0.0
            for c in range(thread, x1.shape[-1], cuda.blockDim.x):
                inner_prod[thread] += trg_vector[c] * x1[b, row, col, c]
    
            cuda.syncthreads()

            # reduce the partial sums to a single value (the inner product)
            n = cuda.blockDim.x
            while n > 2:
                n //= 2
                if thread < n:
                    inner_prod[thread] += inner_prod[n + thread]
                cuda.syncthreads()

            # store in the output correlation array
            if thread == 0:
                corr[b, i, j, y, x] = (inner_prod[0] + inner_prod[1]) / x1.shape[-1]

            cuda.syncthreads()


@cuda.jit
def local_correlation_grad_x1(grad_out, grad_in, other, k):
    '''CUDA kernel to compute the gradient of local correlation with respect to input x1.

    grad_out and other have shape (b, h, w, c)
    grad_in has shape (b, k, k, h, w)
    '''
    out_vector = cuda.shared.array(0, dtype=numba.float32)

    x = cuda.blockIdx.x  # x location
    y = cuda.blockIdx.y  # y location
    b = cuda.blockIdx.z  # batch index

    thread = cuda.threadIdx.x  # thread index

    for c in range(thread, other.shape[-1], cuda.blockDim.x):
        out_vector[c] = 0
    
    cuda.syncthreads()

    h, w = other.shape[1], other.shape[2]
    k2 = k // 2

    for i in range(k):
        row = y + i - k2
        if row < 0 or row >= h: continue
        for j in range(k):
            col = x + j - k2
            if col < 0 or col >= w: continue

            for c in range(thread, other.shape[-1], cuda.blockDim.x):
                ii = k - 1 - i
                jj = k - 1 - j
                out_vector[c] += grad_in[b, ii, jj, row, col] * other[b, row, col, c]
            
            cuda.syncthreads()
    
    for c in range(thread, other.shape[-1], cuda.blockDim.x):
        grad_out[b, y, x, c] = out_vector[c] / other.shape[-1]


@cuda.jit
def local_correlation_grad_x2(grad_out, grad_in, other, k):
    '''CUDA kernel to compute the gradient of local correlation with respect to input x2.

    grad_out and other have shape (b, h, w, c)
    grad_in has shape (b, k, k, h, w)
    '''
    out_vector = cuda.shared.array(0, dtype=numba.float32)

    x = cuda.blockIdx.x  # x location
    y = cuda.blockIdx.y  # y location
    b = cuda.blockIdx.z  # batch index

    thread = cuda.threadIdx.x  # thread index

    for c in range(thread, other.shape[-1], cuda.blockDim.x):
        out_vector[c] = 0
    
    cuda.syncthreads()

    h, w = other.shape[1], other.shape[2]
    k2 = k // 2

    for i in range(k):
        row = y + i - k2
        if row < 0 or row >= h: continue
        for j in range(k):
            col = x + j - k2
            if col < 0 or col >= w: continue

            for c in range(thread, other.shape[-1], cuda.blockDim.x):
                out_vector[c] += grad_in[b, i, j, y, x] * other[b, row, col, c]
            
            cuda.syncthreads()
    
    for c in range(thread, other.shape[-1], cuda.blockDim.x):
        grad_out[b, y, x, c] = out_vector[c] / other.shape[-1]


if __name__ == '__main__':
    torch.manual_seed(999)
    k = 9
    b, c, h, w = 2, 256, 32, 32
    x1 = torch.rand(b, c, h, w).cuda().requires_grad_(True)
    x2 = torch.rand(b, c, h, w).cuda().requires_grad_(True)

    grad_in = x1.new_zeros(b, k, k, h, w).normal_(0, 0.25).requires_grad_(False)

    unfold = torch.nn.functional.unfold

    ### Test correctness of compute_local_correlation kernel
    print('--- Compute local correlation ---')
    
    corr = x1.new_zeros(b, k, k, h, w).requires_grad_(False)

    blocks = (w, h, b)
    threads = (min(128, c), 1, 1)
    memsize = 4 * (c + threads[0])
    compute_local_correlation[blocks, threads, 0, memsize](
        cuda.as_cuda_array(corr),
        cuda.as_cuda_array(x1.detach().moveaxis(1, -1).contiguous()),
        cuda.as_cuda_array(x2.detach().moveaxis(1, -1).contiguous()),
        k
    )

    compare = unfold(x1, k, padding=k // 2).reshape(b, c, -1, h, w)
    compare = torch.einsum('bcnhw,bcmhw->bnmhw', compare, x2.unsqueeze(2)).reshape(b, k, k, h, w) / x1.shape[1]

    compare.backward(grad_in)

    grad_x1 = x1.grad.clone()
    grad_x2 = x2.grad.clone()

    print('Average error: ', compare.detach().sub(corr).abs().mean().item(), '\n')

    del compare, corr


    ### Test correctness of local_correlation_grad_x2
    print('--- Local correlation grad x2 ---')

    grad_out = torch.zeros_like(x2).moveaxis(1, -1)

    blocks = (w, h, b)
    threads = (min(128, c), 1, 1)
    memsize = 4 * c
    local_correlation_grad_x2[blocks, threads, 0, memsize](
        cuda.as_cuda_array(grad_out),
        cuda.as_cuda_array(grad_in),
        cuda.as_cuda_array(x1.detach().moveaxis(1, -1).contiguous()),
        k
    )
    grad_out = grad_out.moveaxis(-1, 1)

    # compare = unfold(x1, k, padding=k // 2).reshape(b, c, k, k, h, w)
    # compare = compare.mul(grad_in.unsqueeze(1)).sum((2, 3)) / x1.shape[1]
    compare = grad_x2

    print('Average error: ', compare.sub(grad_out).abs().mean().item(), '\n')

    del compare, grad_out


    ### Test correctness of local_correlation_grad_x1
    print('--- Local correlation grad x1 ---')

    grad_out = torch.zeros_like(x1).moveaxis(1, -1)

    blocks = (w, h, b)
    threads = (min(128, c), 1, 1)
    memsize = 4 * c
    local_correlation_grad_x1[blocks, threads, 0, memsize](
        cuda.as_cuda_array(grad_out),
        cuda.as_cuda_array(grad_in),
        cuda.as_cuda_array(x2.detach().moveaxis(1, -1).contiguous()),
        k
    )
    grad_out = grad_out.moveaxis(-1, 1)
        
    # kk = k // 2
    # dc = grad_in.permute(0, 3, 1, 4, 2).reshape(b, 1, h * k, w * k)[..., kk:-kk, kk:-kk]
    # dc = unfold(dc, k, dilation=k - 1, padding=kk * (k - 1), stride=k).reshape(b, 1, k, k, h, w)
    # compare = unfold(x2, k, padding=kk).reshape(b, c, k, k, h, w).mul(dc).sum((2, 3)) / x1.shape[1]
    compare = grad_x1

    print('Average error: ', compare.sub(grad_out).abs().mean().item(), '\n')