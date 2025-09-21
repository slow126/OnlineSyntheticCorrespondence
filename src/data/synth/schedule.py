import math
from typing import Dict

from numpy import isin


class Scheduler(object):
    '''Implements a scheduling policy for samplers.

    The schedule is a function that determines an interpolation between sampling parameters with
    "start" and "end" values at each step.

    The schedule is specified as a dictionary with the following form:
    {
        "schedule": { "type": "_name_", **params },
        "complexity": { "start": {**params}, "end": {**params} }
        "angle": { "start": {**params}, "end": {**params} }
        "scale": { "start": {**params}, "end": {**params} }
        "texture": { "start": {**params}, "end": {**params} }
    }

    The "schedule" sub-dict may have 0 or more parameters depending on the schedule type.
    Any of the samplers (complexity, angle, scale, texture) can be left out, in which case no
    schedule will be applied to that sampler. The params dict for "start" and "end" in each
    sampler contains the same configuration as required for the respective sampler class.
    '''
    def __init__(
        self,
        schedule: Dict,
        # end: int,
        complexity_sampler = None,
        angle_sampler = None,
        scale_sampler = None,
        texture_sampler = None,
    ):
        if schedule['schedule']['type'] not in (
            'linear',
            'poly',
            'sin',
            'exp',
        ):
            raise ValueError(f'"{schedule["type"]}" is not a supported schedule type')
        self.schedule = schedule
        # self.end = end
        self.complexity_sampler = complexity_sampler
        self.angle_sampler = angle_sampler
        self.scale_sampler = scale_sampler
        self.texture_sampler = texture_sampler

    @staticmethod
    def interpolated(d1, d2, x):
        if isinstance(d1, dict):
            out = {}
            for k in d1:
                out[k] = Scheduler.interpolated(d1[k], d2[k], x)
        elif isinstance(d1, (list, tuple)):
            out = d1.__class__((Scheduler.interpolated(v1, v2, x) for v1, v2 in zip(d1, d2)))
        elif isinstance(d1, str):
            return d1
        else:
            return (1 - x) * d1 + x * d2
        return out

    def step(self, step: int, end: int):
        sched = self.schedule['schedule']
        t = sched['type']
        x = float(step) / float(end)
        if t == 'linear':
            x = linear(x)
        elif t == 'poly':
            x = poly(x, sched['exp'])
        elif t == 'exp':
            x = exp(x, sched['alpha'])
        elif t == 'sin':
            x = sin(x, sched['periods'], sched.get('shape', None))

        for k in ('complexity', 'angle', 'scale', 'texture'):
            if getattr(self, k + '_sampler') is not None and self.schedule.get(k, None) is not None:
                start, end = self.schedule[k]['start'], self.schedule[k]['end']
                probs = self.interpolated(start, end, x)
                getattr(self, k + '_sampler').set_probs(**probs)


def linear(x):
    return x


def poly(x, exp):
    return x**exp


def exp(x, alpha):
    return (math.exp(alpha * x) - 1) / (math.exp(alpha) - 1)


def sin(x, periods, shape: str=None):
    # starts at 0 and ends at 1, completing (periods + 1/2) cycles
    y = 0.5 * (1 + math.sin(math.pi * ((2 * periods + 1) * x - 0.5)))
    if shape is not None:
        if shape == 'linear':
            a = x
        elif shape.startswith('pow'):
            exp = float(shape[3:])
            a = poly(x, exp)
        elif shape.startswith('root'):
            exp = 1 / float(shape[4:])
            a = poly(x, exp)
        elif shape.startswith('exp'):
            alpha = float(shape[3:])
            a = exp(x, alpha)
        else:
            raise ValueError(f'Unsupported shape parameter "{shape}"')
        y *= a
    return y