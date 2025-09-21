import textwrap
from typing import Dict, List, Union
import yaml

import numpy as np
from scipy.special import erf
from scipy.stats import distributions


class Sampler(object):
    '''Random sampler for sampling values from a distribution or mixture of
    distributions.

    Args:
        components (dict or list of dicts): each component is a dictionary specifying
            the type of distribution, its parameters, and its weight (only applicable
            for mixtures). The distribution name and parameters should match
            scipy.stats.distributions. Components can either be a single dict or a list
            of dicts.
        seed (int): seed for the random number generator.
    '''
    def __init__(
        self,
        components: Union[Dict, List[Dict]],
        seed: int,
    ):
        self.rng = np.random.default_rng(seed)
        self.set_probs(components)

    @staticmethod
    def _make_distribution(component: dict):
        dist = component.pop('distribution')
        dist = getattr(distributions, dist)(**component)
        return dist

    @staticmethod
    def _make_mixture(components: list):
        dists = []
        weights = []
        for component in components:
            weight = component.pop('weight', 1.0)
            dist = Sampler._make_distribution(component)
            dists.append(dist)
            weights.append(weight)

        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        return dists, weights

    def set_probs(self, components):
        self.dist_params = components
        if isinstance(components, dict):
            self.dist = Sampler._make_distribution(components)
            self.weights = None
        else:
            self.dist, self.weights = Sampler._make_mixture(components)

    def sample(self, n=1):
        if self.weights is not None:
            idx = self.rng.choice(len(self.dist), n, replace=True, p=self.weights)
            vs = np.empty(n)
            for i in range(n):
                vs[i] = self.dist[idx[i]].rvs(1)
        else:
            vs = self.dist.rvs(n)
        return vs
    
    def __repr__(self):
        return (
            self.__class__.__name__ + '()\n' +
            textwrap.indent(yaml.dump(self.dist_params), '  ')
        )


class TwoSampler(Sampler):
    '''Random sampler for sampling values from a pair of distributions, either of which
    could be a mixture model.

    Args:
        components1 (dict or list of dicts): components of a probability distribution.
            Each component is a dictionary specifying the type of distribution, its parameters,
            and its weight (only applicable for mixtures). The distribution name and parameters
            should match scipy.stats.distributions. Components can either be a single dict or a
            list of dicts.
        components2 (dict or list of dicts): second distribution, specified in the same way as
            components1.
        seed (int): seed for the random number generator.
    '''
    def __init__(
        self,
        components1: Dict,
        components2: Dict,
        seed: int,
    ):
        self.rng = np.random.default_rng(seed)
        self.set_probs(components1, components2)

    def set_probs(self, comps1=None, comps2=None):
        if comps1 is not None:
            if isinstance(comps1, dict):
                self.dist1 = Sampler._make_distribution(comps1)
                self.weights1 = None
            else:
                self.dist1, self.weights1 = Sampler._make_mixture(comps1)

        if comps2 is not None:
            if isinstance(comps2, dict):
                self.dist2 = Sampler._make_distribution(comps2)
                self.weights2 = None
            else:
                self.dist2, self.weights2 = Sampler._make_mixture(comps2)

    def sample(self, n=1):
        if self.weights1 is not None:
            idx = self.rng.choice(len(self.dist1), n, replace=True, p=self.weights1)
            vs1 = np.empty(n)
            for i in range(n):
                vs1[i] = self.dist1[idx[i]].rvs(1)
        else:
            vs1 = self.dist1.rvs(n)
        
        if self.weights2 is not None:
            idx = self.rng.choice(len(self.dist2), n, replace=True, p=self.weights2)
            vs2 = np.empty(n)
            for i in range(n):
                vs2[i] = self.dist2[idx[i]].rvs(1)
        else:
            vs2 = self.dist2.rvs(n)
        
        return vs1, vs2

    
class ScaleSampler(TwoSampler):
    '''Random sampler for sampling a pair of dependent scale parameters.
    
    Uses two different distributions:
    1) an "absolute" distribution, which samples a base absolute scale A.
    2) a "relative" distribution, which samples a second scale R relative to the absolue one.

    The values that are returned are (A, A + R)

    Args:
        abs_components (dict or list of dicts): components of a probability distribution for
            sampling absolute scale. Each component is a dictionary specifying the type of
            distribution, its parameters, and its weight (only applicable for mixtures).
            The distribution name and parameters should match scipy.stats.distributions.
            Components can either be a single dict or a list of dicts.
        rel_components (dict or list of dicts): components of a probability distribution for
            sampling relative scale, specified in the same way as abs_components.
        seed (int): seed for the random number generator.
    '''
    def __init__(
        self,
        abs_components: Dict,
        rel_components: Dict,
        seed: int,
    ):
        self.dist_params = {}
        super().__init__(abs_components, rel_components, seed)

    def set_probs(self, abs_components=None, rel_components=None):
        super().set_probs(abs_components, rel_components)
        if abs_components: self.dist_params['abs_components'] = abs_components
        if rel_components: self.dist_params['rel_components'] = rel_components

    def sample(self, n=1):
        abs_vs, rel_vs = super().sample(n)
        vs = np.stack([abs_vs, abs_vs + rel_vs], 1)
        if n == 1: vs = vs[0]
        return vs


# NOTE: need to make sure this is thought through correctly.
# - if only positive angular separation is specified, and the same ordering is
#   is always used when sampling, then that would introduce a bias in the data
#   that is probably unwanted. So either the distributions should always contain
#   both positive and negative values (probably symetric around 0), or else the 
#   order needs to be randomly adjusted.
class AngleSampler(TwoSampler):
    '''Random sampler for sampling a pair of view angles. Each item in the pair consists of an
    x-angle (rotation in the x-z plane) and a y-angle (rotation in the y-z plane). Angles are
    specified proportional to PI radians.
    
    Uses two different distributions, one for x-angle and one for y-angle.

    When sampling N points, an array A of shape (N, 2, 2) is returned. A[:, 0] are the angles
    for the first item of each pair, and A[:, 1] are the angles of the second items. The last
    axis contains the x and y angles.

    Args:
        x_components (dict or list of dicts): components of a probability distribution for
            sampling absolute scale. Each component is a dictionary specifying the type of
            distribution, its parameters, and its weight (only applicable for mixtures).
            The distribution name and parameters should match scipy.stats.distributions.
            Components can either be a single dict or a list of dicts.
        y_components (dict or list of dicts): components of a probability distribution for
            sampling relative scale, specified in the same way as abs_components.
        seed (int): seed for the random number generator.
    '''
    def __init__(
        self,
        x_components: Union[Dict, List[Dict]],
        y_components: Union[Dict, List[Dict]],
        seed: int,
        bounds: Union[list, tuple] = (0.5, 0.25),
    ):
        self.dist_params = {}
        self.bounds = bounds
        super().__init__(x_components, y_components, seed)

    def set_probs(self, x_components=None, y_components=None):
        super().set_probs(x_components, y_components)
        if x_components: self.dist_params['x_components'] = x_components
        if y_components: self.dist_params['y_components'] = y_components

    def sample(self, n=1):
        vx, vy = super().sample(n)
        vx /= 2
        vy /= 2

        vxr = self.bounds[0] - vx
        center_x = self.rng.uniform(-vxr, vxr)

        vxy = self.bounds[1] - vy
        center_y = self.rng.uniform(-vxy, vxy)

        out = np.empty((n, 2, 2))
        out[:, :, 0] = center_x[:, None]
        out[:, :, 1] = center_y[:, None]
        out[:, 0, 0] -= vx
        out[:, 0, 1] -= vy
        out[:, 1, 0] += vx
        out[:, 1, 1] += vy

        if n == 1: out = out[0]
        return out


class CurvatureMapSampler(Sampler):
    def __init__(
        self,
        curvature_map: np.ndarray,
        grid_points: np.ndarray,
        dist: Dict,
        seed: int,
    ):
        self.dist_params = {}
        cmap = curvature_map.reshape(-1, 5)
        self.curvature_map = self.convert_curvature(cmap[:, 0], cmap[:, 1], cmap[:, -1])
        self.grid_points = grid_points.reshape(-1, 4)
        self.radius = 0.5 * (self.grid_points[1, -1] - self.grid_points[0, -1])
        self.rng = np.random.default_rng(seed)
        self.set_probs(dist)

    @staticmethod
    def convert_curvature(x, y, r):
        # harmonic mean of (average mean curvature and std of mean curvature)
        # scaled by ratio of number of internal surface points to "edge" surface points
        score = 2 * ((x * y) / (x + y)) * (1 + r)
        return score

    def set_probs(self, dist: dict=None, **kwargs):
        if dist is None: dist = {}
        dist.update(**kwargs)
        self.dist_params.update(**dist)
        type = dist['distribution']
        loc, scale = dist['loc'], dist['scale']
        if type == 'norm':
            alpha = dist.get('alpha', 0)
            x = self.curvature_map
            self.pmap = np.exp(-0.5 * np.square((x - loc) / scale))
            if alpha != 0:
                # skewnormal distribution with skew parameter alpha
                self.pmap *= (1 + erf(alpha * (x - loc) / (scale * 1.414213562373)))
            self.pmap[np.isnan(self.pmap)] = 0
        else:
            self.pmap = (self.curvature_map >= loc) & (self.curvature_map <= loc + scale)
            self.pmap = self.pmap.astype(np.float64)
        self.pmap /= self.pmap.sum()

    def sample(self, n=1):
        idx = self.rng.choice(len(self.pmap), n, replace=True, p=self.pmap)
        offset = self.rng.uniform(-self.radius, self.radius, (n, 4))
        points = self.grid_points[idx] + offset
        if n == 1: points = points[0]
        return points