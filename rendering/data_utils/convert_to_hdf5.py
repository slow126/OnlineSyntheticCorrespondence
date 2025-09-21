import argparse
import json
from pathlib import Path

import blosc
import h5py
import numpy as np
import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--flow', action=argparse.BooleanOptionalAction)
    parser.add_argument('--normals', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    root = Path(args.root)

    meta = {'images': {}, 'geometry': {}}
    if args.normals:
        meta['normals'] = {}
    if args.flow:
        meta['flow'] = {}
    meta['key'] = {'offset': 0, 'shape': 1, 'dtype': 2}

    with h5py.File(root.joinpath('data.hdf5'), 'w') as h5:
        for subdir in tqdm.tqdm(sorted(root.iterdir())):
            if not subdir.is_dir(): continue

            img_path = subdir.joinpath('images')
            if not img_path.is_dir():
                img_path = subdir
            g = h5.create_group(f'{subdir.name}/images')
            for img_p in sorted(img_path.iterdir()):
                if not img_p.suffix.lower() in ('.jpg', '.jpeg', '.png'): continue
                data = np.frombuffer(img_p.open('rb').read(), dtype=np.uint8)
                d = g.create_dataset(img_p.name, data=data)
                meta['images'][str(img_p.relative_to(root))] = [d.id.get_offset(), d.shape, d.dtype.char]

            g = h5.create_group(f'{subdir.name}/geometry')
            geo_path = subdir / 'coords.npz'
            if not geo_path.exists():
                geo_path = subdir / 'world_coordinates.npz'
                if not geo_path.exists():
                    raise RuntimeError(f'{str(subdir)} is missing coords.npz or world_coordinates.npz')
            geo = np.load(geo_path)
            for k in geo.keys():
                arr = geo[k]
                data = np.frombuffer(blosc.pack_array(arr, cname='lz4hc'), dtype=np.uint8)
                d = g.create_dataset(k, data=data)
                meta['geometry'][f'{subdir.name}/{k}'] = [d.id.get_offset(), d.shape, d.dtype.char]

            if args.normals:
                g = h5.create_group(f'{subdir.name}/normals')
                normals_path = subdir / 'normals.npz'
                if not normals_path.exists():
                    normals_path = subdir / 'surface_normals.npz'
                    if not normals_path.exists():
                        raise RuntimeError(f'{str(subdir)} is missing normals.npz or surface_normals.npz')
                nrml = np.load(normals_path)
                for k in nrml.keys():
                    arr = nrml[k]
                    data = np.frombuffer(blosc.pack_array(arr, cname='lz4hc'), dtype=np.uint8)
                    d = g.create_dataset(k, data=data)
                    meta['normals'][f'{subdir.name}/{k}'] = [d.id.get_offset(), d.shape, d.dtype.char]

            if args.flow:
                g = h5.create_group(f'{subdir.name}/flow')
                flow = np.load(subdir / 'flows.npz')
                for k in flow.keys():
                    arr = flow[k]
                    data = np.frombuffer(blosc.pack_array(arr, cname='lz4hc'), dtype=np.uint8)
                    d = g.create_dataset(k, data=data)
                    meta['flow'][f'{subdir.name}/{k}'] = [d.id.get_offset(), d.shape, d.dtype.char]

    json.dump(meta, root.joinpath('data.json').open('w'))
