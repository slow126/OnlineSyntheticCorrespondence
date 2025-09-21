from pathlib import Path
from typing import Union

import numpy as np


def read_flo_file(path: Union[str, Path]):
    fp = open(path, 'rb')

    tag = fp.read(4)
    if tag != b'PIEH':
        raise AssertionError('Incorrect tag when attempting to read flo file: {path}')

    w, h = np.frombuffer(fp.read(8), np.int32)
    data = np.frombuffer(fp.read(h * w * 8), np.float32)
    flow = data.copy().reshape(h, w, 2)

    return flow