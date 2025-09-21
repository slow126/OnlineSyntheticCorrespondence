from typing import Any, Optional, Tuple, Union
import torch


def fftind(*size, device='cpu'):
    """ Returns a tensor of shifted Fourier coordinates.
    """
    d = {'device': device}
    k_ind = torch.meshgrid(*[torch.arange(s, **d) - (s + 1) // 2 for s in size], indexing='ij')
    k_ind = torch.fft.fftshift(torch.stack(k_ind, 0), tuple(range(1, len(size) + 1)))
    return k_ind


def multivariate(covariance, rng, device):
    # needs to be 64-bit float during computation, otherwise numerical errors break it
    cov = torch.empty(3, 3, dtype=torch.float64, device=device)
    cov[:, :2].normal_(generator=rng)
    torch.nn.functional.normalize(cov[:, :2], dim=0, out=cov[:, :2])
    # construct orthonormal eigenvector basis
    cov[:, 2] = cov[:, 0].cross(cov[:, 1], dim=0)
    cov[:, 1] = cov[:, 2].cross(cov[:, 0], dim=0)
    torch.nn.functional.normalize(cov, dim=0, out=cov)
    # sample positive eigenvalues with some buffer to avoid numerical issues
    eig = cov.new_empty(3).uniform_(max(0.1, covariance[0]), covariance[1], generator=rng)
    cov = (cov * eig) @ cov.T
    cov = cov.float()

    dist = torch.distributions.MultivariateNormal(cov.new_zeros(3), cov)
    return dist


def normalize(x, rand_mean=None, rand_std=None, rng=None):
    b = x.shape[0]
    dims = tuple(range(1, x.ndim))
    # standardize
    std, mean = torch.std_mean(x, dim=dims, keepdim=True)
    x -= mean
    x /= std
    if rand_std is not None:
        x *= torch.empty_like(std).uniform_(*rand_std, generator=rng)
    if rand_mean is not None:
        x += torch.empty_like(mean).uniform_(*rand_mean, generator=rng)
    # sigmoid
    x = x.sigmoid_()
    return x


def gaussian_random_field(
    size: Tuple[int] = (128, 128),
    alpha: Union[float, tuple, list] = 3.0,
    c: int = 3,
    batch_size: int = 1,
    method: str = 'real',
    output_uint8: bool = False,
    covariance: Optional[Tuple[float]] = None,
    rand_mean: Optional[Tuple[float]] = None,
    rand_std: Optional[Tuple[float]] = None,
    rng: Optional[torch.Generator] = None,
    device: str = 'cpu',
):
    """
    Args:
        size: the size of the output in each dimension.
        alpha: the power of the power-law momentum distribution.
        c: the number of feature channels. Default 3.
        method: method for converting complex output to real.
        rand_mean: optional range for sampling a new mean for the output.
        rand_std: optional range for sampling a new standard deviation for the output.
        rng: optional torch.Generator for random sampling.
        device: torch device (cpu, cuda)

    Returns:
        field: gaussian random field tensor of shape (*size, c)
    """
    assert method in ('real', 'imag', 'angle', 'magnitude')
    
    nd = len(size)
    d = {'device': device}

    if isinstance(alpha, (tuple, list)):
        alpha = torch.empty(batch_size, **d).uniform_(*alpha, generator=rng)
        alpha = alpha.view(-1, *[1 for _ in range(nd)])
    else:
        alpha = torch.full([batch_size] + [1 for _ in range(nd)], alpha, **d)
    
    # Defines momentum indices
    k_idx = fftind(*size, device=device)

    # Defines the amplitude as a power law 1/|k|^(alpha/4)
    amplitude = k_idx.square().sum(0).add(1e-10).pow(-alpha / 4.0)
    amplitude[(range(batch_size),) + tuple(0 for _ in range(nd))] = 0
    
    if covariance is not None:
        # Complex gaussian random noise with covariance structure
        mnorm = multivariate(covariance, rng, **d)
        samples = (batch_size,) + tuple(size)
        noise = torch.complex(mnorm.sample(samples), mnorm.sample(samples)).moveaxis(-1, 1)
    else:
        noise = torch.empty(batch_size, c, *size, dtype=torch.complex64, **d).normal_(generator=rng)
    noise = noise.mul_(amplitude.unsqueeze(1))
    
    # To real space
    field = torch.fft.ifftn(noise, dim=tuple(range(2, 2 + len(size))))
    if method in ('real', 'imag'):
        field = getattr(field, method)
    elif method == 'magnitude':
        field = field.abs()
    elif method == 'angle':    
        field = field.angle()
    field = field.moveaxis(1, -1)
    
    field = normalize(field, rand_mean, rand_std, rng=rng)

    if output_uint8:
        field *= 255
        field = field.byte()
        
    return field


if __name__ == '__main__':
    gaussian_random_field(covariance=(0.0, 1.0))