import os
from pathlib import Path
import subprocess
import time


def copy_data(copy_from, copy_to, datadir, file_list=None, pattern=None, verbose=True):
    if file_list is not None and pattern is not None:
        raise RuntimeError(f'`file_list` and `pattern` are mutually exclusive')

    src_root = Path(copy_from)
    src = str(src_root) + '/'

    ### setup destination dir
    dst_root = Path(copy_to)
    dst = dst_root.joinpath(src_root.relative_to(datadir))
    if not dst.is_dir():
        dst.mkdir(parents=True)
    dst = str(dst) + '/'

    ### transfer files
    t = time.time()
    print(f'Copying data from {src} to {dst}...')

    rsync_args = ['rsync', '-a']
    if verbose:
        rsync_args.append('-v')

    if file_list is not None:
        rsync_args.append('--relative')
        for f in file_list:
            s = os.path.join(src, './', f)
            cmd = ' '.join(rsync_args + [s, dst])
            subprocess.run(cmd, shell=True)
    else:
        if pattern is not None:
            src = src + pattern
        cmd = ' '.join(rsync_args + [src, dst])
        subprocess.run(cmd, shell=True)

    t = time.time() - t
    print(f'Finished copying in {t:.5f} seconds')
    
    return dst