from typing import Dict, Optional, Tuple

import numpy as np


class MandelbulbPowerSampler(object):
    def __init__(
        self,
        power_range: Dict = {
            'min': 1.0,
            'offset': 10.0,
        },
        debug: bool = False,
    ):
        self.power_range = power_range
        self.debug = debug
        self.rng_np = np.random.RandomState()

    def set_probs(self, probs: Dict=None, **kwargs):
        if probs is None: probs = {}
        probs.update(**kwargs)

    def sample(self):
        """Sample a single Mandelbulb power parameter
        
        Returns:
            Float containing power parameter
        """
        if self.debug:
            # Return midpoint of range
            return (self.power_range['min'] + self.power_range['offset'] + self.power_range['min']) / 2
        else:
            # Sample random power from range
            return self.rng_np.uniform(self.power_range['min'], self.power_range['offset'] + self.power_range['min'])

    def __repr__(self):
        return '\n'.join((
            'MandelbulbPowerSampler()',
            f'  power_range: {self.power_range}',
            f'  debug: {self.debug}'
        ))