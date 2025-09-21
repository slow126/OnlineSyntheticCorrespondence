'''Custom resolvers that can be used in configuration files through OmegaConf.
'''
import os
import re

from omegaconf import OmegaConf


def register(cache=False):
    def decorator_register(func):
        OmegaConf.register_new_resolver(func.__name__, func, use_cache=cache)
        return func
    return decorator_register


@register(cache=True)
def next_run(root, wandb=True):
    '''Determine the name of the next run and setup the run folder.
    '''
    path = _next_run_path(root)
    os.makedirs(path)
    if wandb:
        os.makedirs(os.path.join(path, 'wandb'))
    return path


@register(cache=False)
def path_seg(path, seg_idx=-1):
    '''Given a path made up of segments separated by "/", return the segment at seg_idx.
    '''
    segments = str(path).split('/')
    return segments[seg_idx]


@register(cache=False)
def linear_scale_factor(bs, base_bs, nodes=1, gpus_per_node=1):
    '''Compute a linear scaling factor for the learning rate based on the ratio of the batch size to
    a base batch size. Batch size is given in terms of a single GPU, so scaling needs to take into
    consideration the total number of distributed processes.
    '''
    return (bs / base_bs) * nodes * gpus_per_node

OmegaConf.register_new_resolver("mul", lambda x, y: x*y)


def _next_run_path(root):
    run = 0
    if os.path.exists(root):
        runs = [x for x in os.listdir(root) if re.match(r'run-\d+', x)]
        if len(runs) > 0:
            runs = sorted(int(x.split('-')[-1]) for x in runs)
            run = runs[-1] + 1
    path = os.path.join(root, f'run-{run}')
    return path