from typing import Dict, List, Optional, Tuple, Union

import torch

from . import grf


texture_kw = {
    'grf': dict(
        covariance=(0.0, 5.0),
        rand_mean=(-1.0, 1.0),
        rand_std=(0.3, 1.2),
    ),
}

texture_fns = {
    'grf': grf.gaussian_random_field,
}

class SolidColorSampler(object):
    def __init__(self, color=(0, 0, 0)):
        self.color = torch.tensor(color)

    def sample(self, n: int, size: Union[List, Tuple], rng: torch.Generator):
        col = torch.zeros(n, *size, 3, device=rng.device)
        return col, col

class TextureSampler(object):
    def __init__(
        self,
        texture_type: Union[str, List, Tuple],
        texture_prob: float = 1.0,
        matching_prob: float = 1.0,
        texture_scale: Optional[Dict] = None,
    ):
        self.texture_type = texture_type
        self.texture_prob = texture_prob
        self.matching_prob = matching_prob
        self.texture_scale = texture_scale

    def set_probs(self, probs: Dict=None, **kwargs):
        if probs is None: probs = {}
        probs.update(**kwargs)
        for k in ('texture_prob', 'matching_prob', 'texture_scale'):
            if k in probs:
                setattr(self, k, probs[k])

    def sample(self, n: int, size: Union[List, Tuple], rng: torch.Generator):
        if isinstance(self.texture_type, str):
            texture_type = self.texture_type
        else:
            i = torch.randint(0, len(self.texture_type), (1,), generator=rng).item()
            texture_type = self.texture_type[i]
        kw = texture_kw[texture_type]
        fn = texture_fns[texture_type]

        is_matching = bernoulli(n, self.matching_prob, rng)
        is_textured = bernoulli(n, self.texture_prob, rng)

        # textures for first image in each pair
        a = torch.empty(n, *size, 3, device=rng.device)

        tex_idx = is_textured.nonzero().squeeze()
        num_tex = tex_idx.nelement()
        if num_tex > 0:
            tex = fn(size, batch_size=num_tex, rng=rng, device=rng.device, **kw, **self.texture_scale)
            a[tex_idx] = tex

        idx = is_textured.logical_not().nonzero().squeeze()
        num_solid = idx.nelement()
        if num_solid > 0:
            solid = uniform((num_solid, *(1,) * (a.ndim - 2), 3), 0, 1, rng)
            a[idx] = solid

        # textures for second image in each pair
        b = torch.empty(n, *size, 3, device=rng.device)
        match_idx = is_matching.nonzero().squeeze()
        b[match_idx] = a[match_idx]

        tex_idx = is_textured.logical_and(~is_matching).nonzero().squeeze()
        num_tex = tex_idx.nelement()
        if num_tex > 0:
            tex = fn(size, batch_size=num_tex, rng=rng, device=rng.device, **kw, **self.texture_scale)
            b[tex_idx] = tex

        idx = is_textured.logical_not().logical_and(~is_matching).nonzero().squeeze()
        num_solid = idx.nelement()
        if num_solid > 0:
            solid = uniform((num_solid, *(1,) * (b.ndim - 2), 3), 0, 1, rng)
            b[idx] = solid

        return a, b

    def __repr__(self):
        return '\n'.join((
            'TextureSampler()',
            f'  texture_type: {self.texture_type}',
            f'  texture_prob: {self.texture_prob}',
            f'  matching_prob: {self.matching_prob}',
            f'  texture_scale: {self.texture_scale}'
        ))


def bernoulli(n, p, rng):
    return torch.empty(n, dtype=torch.bool, device=rng.device).bernoulli_(p, generator=rng)


def uniform(shape, vmin, vmax, rng):
    return torch.empty(shape, device=rng.device).uniform_(vmin, vmax, generator=rng)


def normal(shape, loc, scale, rng):
    return torch.empty(shape, device=rng.device).normal_(loc, scale, generator=rng)