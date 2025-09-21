import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('root')

args = parser.parse_args()
root = Path(args.root)

remap = {
    'world_coordinates.npz': 'coords.npz',
    'surface_normals.npz': 'normals.npz',
}

for obj_dir in list(root.iterdir()):
    if not obj_dir.is_dir(): continue
    for name in ('world_coordinates.npz', 'surface_normals.npz'):
        p = obj_dir.joinpath(name)
        if p.exists():
            p.rename(p.with_name(remap[name]))
    pre, n = obj_dir.name.split('_')
    n = n.zfill(3)
    n_dir = obj_dir.with_name(f'{pre}_{n}')
    if n_dir.name != obj_dir.name:
        obj_dir.rename(n_dir)