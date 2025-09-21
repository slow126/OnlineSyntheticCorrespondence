from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torchvision


__all__ = [
    'FeatureExtractor',
]


class HaltForwardPass(Exception):
    '''Exception thrown to halt execution of the forward pass after reaching a certain layer.
    '''
    pass


def get_layer(net: torch.nn.Module, layer_path: str):
    '''Returns the torch.nn.Module for a layer in the network, described by `layer_path`.
    `layer_path` follows the naming convention used by torch.nn.Module.state_dict.
    '''
    parts = layer_path.split('.')
    layer = net
    for p in parts:
        if p.isdigit():
            layer = layer[int(p)]
        else:
            layer = getattr(layer, p)
    return layer


def layer_output_hook(module, input, output, output_dict, key, halt=False):
    '''When attached as a forward hook to a module, saves a clone of the output of the module's
    forward function.
    '''
    output_dict[key] = output.clone()
    if halt:
        raise HaltForwardPass(f'Forward pass halted with key {key}')


class FeatureExtractor(torch.nn.Module):
    '''Feature extraction module. Allows to flexibly extract the intermediate features from any
    layer(s) of the backbone network.

    Args:
        layer_names (list of str): names of layers where features will be extracted, specified
            as a dot path; for example, see the output of torch.nn.Module.named_modules.
        model_name (str): name of a torchvision model which will be instantiated as the backbone.
            model_name and backbone arguments are mutually exclusive.
        backbone (torch.nn.Module): an instantiated torch.nn.Module that will serve as the
            backbone model. model_name and backbone arguments are mutually exclusive.
        weights (str or Enum): torchvision pre-trained weights to be loaded when the backbone
            is instantiated through model_name; passed directly to the constructor as
            weights=weights.
        weights_path (str or Path): path to file containing pre-trained weights as a model state-
            dict that will be loaded by the backbone.
        freeze (bool): whether to freeze the backbone model or allow it to be
            finetuned. (Default: False).
        post_proc (callable): function that will be called after the forward pass completes. The
            function should accept a single argument: the dictionary containing the features that
            were saved during the forward pass. This allows custom logic to be applied to the
            features before they are passed on.
    '''
    def __init__(
        self,
        layer_names: Union[str, List[str], Tuple[str]],
        model_name: Optional[str] = None,
        backbone: Optional[torch.nn.Module] = None,
        weights: Optional[str] = None,
        weights_path: Optional[Union[str, Path]] = None,
        freeze: bool = False,
        post_proc: Optional[Callable] = None,
    ):
        super().__init__()

        if model_name is None and backbone is None:
            raise ValueError('Must supply either model_name or backbone')
        elif model_name is not None and backbone is not None:
            raise ValueError('`model_name` and `backbone` are mutually exclusive')
        elif backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = getattr(torchvision.models, model_name)(weights=weights)

        if weights_path:
            print(f'Loading weights from {weights_path}')
            state = torch.load(weights_path, map_location='cpu')
            self.backbone.load_state_dict(state)

        self.backbone.requires_grad_(not freeze)

        self.post_proc = post_proc

        if isinstance(layer_names, str):
            layer_names = [layer_names]
        self.layer_names = layer_names
        self.layer_outputs = OrderedDict()
        self.hook_handles = {}
        self.halt_hook = None
        self.register_hooks(self.layer_names)

    def __enter__(self):
        self._prev_halt = self.halt_hook
        self.set_halt(self._halt_layer)

    def __exit__(self, *args, **kwargs):
        self.remove_halt()
        self.halt_hook = self._prev_halt
        del self._prev_halt
        del self._halt_layer

    def halting(self, layer_name):
        self._halt_layer = layer_name
        return self

    def register_hooks(self, layer_names):
        # register hooks to collect the output of layers specified in layer_names
        self.hook_handles.clear()
        for i, layer_path in enumerate(layer_names):
            halt = i == len(layer_names) - 1
            hook = partial(layer_output_hook, output_dict=self.layer_outputs, key=layer_path, halt=halt)
            layer = get_layer(self.backbone, layer_path)
            handle = layer.register_forward_hook(hook)
            self.hook_handles[layer_path] = handle

    def clear_hooks(self):
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()

    def set_halt(self, layer_name):
        def halt(*args, **kwargs):
            raise HaltForwardPass(f'Forward pass halted with key {layer_name}')
        
        if self.halt_hook is not None:
            self.halt_hook.remove()
        self.halt_hook = get_layer(self.backbone, layer_name).register_forward_hook(halt)

    def remove_halt(self):
        if self.halt_hook is not None:
            self.halt_hook.remove()
        self.halt_hook = None

    def forward(self, x):
        try:
            self.backbone(x)
        except HaltForwardPass as halting:
            pass
        
        outputs = {k: v for k, v in self.layer_outputs.items()}
        self.layer_outputs.clear()

        if self.post_proc is not None:
            outputs = self.post_proc(outputs)

        return outputs
