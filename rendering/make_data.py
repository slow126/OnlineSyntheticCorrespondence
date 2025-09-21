from pathlib import Path
import textwrap

from cv2 import cvtColor, COLOR_HSV2RGB
import numpy as np
from omegaconf import OmegaConf
import torch
import tqdm

import julia3d_shader as jsh
import textures
import textures.grf
import textures.texture_sampler


def sample_colors(n, rng=None):
    rng: np.random.Generator = rng or np.random.default_rng()
    colors = np.empty((1, n, 3), dtype=np.uint8)
    colors[..., 0] = rng.integers(0, 180, n)
    colors[..., 1:] = rng.beta(3.0, 1.4, (n, 2)) * 255
    colors = cvtColor(colors, COLOR_HSV2RGB)[0] / 255.0
    return colors


def sample_julia_sets(n, rng):
    '''Sample parameters of a Quaternion Julia set.
    The shape of the set is defined by a single Quaternion (4 values), the number "c"
    in the equation z = z^2 + c, which is used to compute the set.
    This function samples a set of Julia set parameters be first choosing a direction
    in 4-D space uniformly (more or less), and then scaling it. The Julia sets become
    sparse and then dissapear altogether as c gets further from the origin. At the origin
    the set is a sphere, and the sets geometry varies smoothly with c.
    '''
    rng: np.random.Generator = rng or np.random.default_rng()
    d = rng.uniform(-1, 1, (n, 4))
    d /= np.linalg.norm(d, ord=2, axis=1, keepdims=True)
    m = rng.beta(12, 4, (n, 1))
    c = d * m
    return c


def set_light_source(imgs_per_set, light_range, camera=None, rng=None):
    rng: np.random.Generator = rng or np.random.default_rng()
    v = rng.uniform(-1., 1., (imgs_per_set, 2)) * light_range
    if camera is not None:
        v = np.clip(camera + v, -1, 1)
    return v


def convert_location(loc, dist):
    loc = np.multiply(loc, [np.pi, -np.pi])
    c = np.cos(loc)
    s = np.sin(loc)
    return dist * np.stack([s[0] * c[1], -s[1], c[0] * c[1]])


def render_images(
        sets,
        dest,
        imgs_per_set = 20,
        resolution = 320,
        view_range = [1., 0.5],
        zoom_range = [1., 4.],
        light_range = [1., 0.5],
        ambient_range = [0.25, 0.45],
        add_texture = True,
        rng = None,
        cam_grid = (0, 0),
    ):
    rng: np.random.Generator = rng or np.random.default_rng()

    res = (resolution, resolution)
    palette = sample_colors(5000)

    shader_code = jsh.load_shader_code('shaders/quatJulia.c')
    state = jsh.setup_moderngl(*shader_code, res)

    uniforms = {
        'iResolution': tuple(float(x) for x in res) + (1.0,),
        'iViewDistance': 3.0,
        'iAntialias': 4
    }

    rename = {
        'iViewAngleXY': 'camera_position',
        'iFocalPlane': 'focal_plane',
        # 'iMaterial': 'material',
        'iLightSource': 'light_source',
        'iAmbientLight': 'ambient_light',
        'iDiffuseScale': 'diffuse_scale',
        'iSpecularScale': 'specular_scale',
        'iSpecularExp': 'specular_exp',
    }

    dest = Path(dest)

    if cam_grid[0] > 0 and cam_grid[1] > 0:
        sx, sy = cam_grid
        nz = len(zoom_range)
        if sx * sy * nz != imgs_per_set:
            print(f'NOTE: imgs_per_set ({imgs_per_set}) does not match the number of images specified by '
                  f'cam_grid and zoom_range ({sx} * {sy} * {nz} = {sx * sy * nz}). The number of images '
                  f'per set will be {sx * sy * nz}')
            imgs_per_set = sx * sy * nz
    elif cam_grid[0] > 0 or cam_grid[1] > 0:
        raise RuntimeError(f'Both values of cam_grid need to be zero or else both need to be positive')

    for k in tqdm.trange(len(sets)):
        oid = f'c_{k:>05d}'
        subfolder = dest.joinpath(oid)
        subfolder.mkdir(exist_ok=True)
        coordinates = {}
        normal_maps = {}
        
        with open(dest.joinpath('metadata.yaml'), 'a') as fp:
            fp.write(f'  {oid}:\n')

        c = sets[k]
        uniforms['iJuliaC'] = c
    
        bg = None
        fg = None

        if add_texture:
            torch_rng = torch.Generator().manual_seed(int(rng.integers(2**16, 2**32)))
            fg = textures.grf.gaussian_random_field(
                (80, 80, 80), (3, 4.5), rand_mean=(-1, 1), rand_std=(0.5, 1), output_uint8=True, rng=torch_rng
            ).numpy()
            uniforms['iUseObjTexture'] = True
            uniforms['iObjTexture'] = 0

        if cam_grid[0] > 0 and cam_grid[1] > 0:
            sx, sy = cam_grid
            x, y = view_range
            view = np.flip(np.moveaxis(np.mgrid[-y:y:1j*sy, -x:x:1j*sx], 0, -1).reshape(-1, 2), -1)
            view = np.repeat(view, len(zoom_range), 0)
            zoom = np.tile(zoom_range, sx * sy)
        else:
            view = rng.uniform(-1., 1., (imgs_per_set, 2)) * view_range
            zoom = rng.uniform(*zoom_range, imgs_per_set)

        # sampling ranges were chosen based on a little bit of experimentation
        mat = palette[rng.integers(0, len(palette), imgs_per_set)]
        light = set_light_source(imgs_per_set, light_range, view)
        ambient = rng.uniform(*ambient_range, imgs_per_set)
        diffuse = rng.uniform(0.5, 0.7, imgs_per_set)
        spec = rng.uniform(0.4, 0.55, imgs_per_set)
        specex = rng.uniform(8., 12., imgs_per_set)

        for i in tqdm.trange(imgs_per_set, leave=False):
            params = dict(
                iViewAngleXY   = view[i],
                iFocalPlane    = zoom[i],
                iMaterial      = mat[i],
                iLightSource   = light[i],
                iAmbientLight  = ambient[i],
                iDiffuseScale  = diffuse[i],
                iSpecularScale = spec[i],
                iSpecularExp   = specex[i]
            )
            uniforms.update(params)
            if add_texture:
                bg = textures.grf.gaussian_random_field(
                    res, (3.5, 5), rand_mean=(-1, 1), rand_std=(0.5, 1), output_uint8=True, rng=torch_rng
                ).numpy()
                uniforms['iUseBgTexture'] = True
                uniforms['iBgTexture'] = 1

            jsh.update_uniforms(state['program'], uniforms)

            img, coord, normals = jsh.get_rendered(state, res, fg, bg)
            img = img.convert('RGB')

            name = f'{i:>04}'
            coordinates[name] = coord
            normal_maps[name] = normals
            img.save(subfolder.joinpath(name).with_suffix('.jpg'), quality=90)

            ps = {}
            for k, v in params.items():
                if k in rename:
                    if type(v).__module__ == 'numpy':
                        v = v.tolist()
                    ps[rename[k]] = v
            # convert from angular offsets to 3D location
            ps['camera_position'] = convert_location(ps['camera_position'], 3.0).tolist()
            ps['light_source'] = convert_location(ps['light_source'], 6.0).tolist()

            # write to metadata file
            s = textwrap.indent(OmegaConf.to_yaml({name: ps}, sort_keys=True), '    ')
            with open(dest.joinpath('metadata.yaml'), 'a') as fp:
                fp.write(s)

        np.savez_compressed(subfolder.joinpath('coords.npz'), **coordinates)
        np.savez_compressed(subfolder.joinpath('normals.npz'), **normal_maps)


def compute_flow(c1, c2, m1, m2, index, threshold=5e-5):
    '''Computes the flow between two sets of coordinates by finding nearest neighbors and
    thresholding based on distance. The flow is returned as an (N, 2, 2) array, with N
    matching points (dimension 0), the yx pixel coordinates (dimension 1) for the two
    images (dimension 2). So flow[0, :, 0] is a yx pixel coordinate in image1, and flow[0, :, 1]
    is the corresponding pixel coordinate in image2.
    '''
    # mask out points not on the object surface
    c1 = c1[m1]
    c2 = c2[m2]
    
    # find correspondences by nearest neighbor search
    dist, idx1 = index.search(c2, 1)
    dist, idx1 = dist.ravel(), idx1.ravel()
    
    # threshold based on distance
    mask = dist <= threshold
    idx1 = idx1[mask]
    
    # corresponding point sets (yx in pixel space)
    pts1 = m1.nonzero()[idx1]
    pts2 = m2.nonzero()[mask]
    return pts1, pts2


def pairwise_flows(coords):
    import faiss
    import faiss.contrib.torch_utils
    import torch

    gres = faiss.StandardGpuResources()
    index: faiss.GpuIndex = faiss.index_cpu_to_gpu(gres, 0, faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 32))

    n = len(coords)
    keys = list(coords.keys())

    coords = [torch.from_numpy(v).cuda(non_blocking=True) for v in coords.values()]

    flows = {}
    pbar = tqdm.tqdm(total=n * (n - 1) // 2, leave=False)
    for i in range(n - 1):
        c1 = coords[i]
        m1 = c1.ne(0).any(-1)
        index.reset()
        index.train(c1[m1])
        index.add(c1[m1])
        for j in range(i + 1, n):
            c2 = coords[j]
            m2 = c2.ne(0).any(-1)
            p1, p2 = compute_flow(c1, c2, m1, m2, index)
            flow = torch.stack((p1, p2), 2)
            k = '-'.join((keys[i], keys[j]))
            flows[k] = flow.to(torch.int16)
            pbar.update()
    pbar.close()
    return flows


def dataset_flows(root, force=False):
    root = Path(root)
    subdirs = sorted(x for x in root.iterdir() if x.is_dir())
    for subdir in tqdm.tqdm(subdirs):
        path = subdir.joinpath('flows.npz')
        if path.exists() and not force: continue
        coords = np.load(subdir.joinpath('coords.npz'))
        flows = pairwise_flows(coords)
        flows = {k: v.to('cpu', non_blocking=True).numpy() for k, v in flows.items()}
        np.savez_compressed(path, **flows)


def measure_overlaps(root):
    import pickle
    import faiss
    import faiss.contrib.torch_utils
    import torch

    root = Path(root)
    subdirs = sorted(x for x in root.iterdir() if x.is_dir())

    gres = faiss.StandardGpuResources()
    index: faiss.GpuIndex = faiss.index_cpu_to_gpu(gres, 0, faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 32))

    overlaps = {}
    for subdir in tqdm.tqdm(subdirs):
        name = subdir.name
        overlaps[name] = {}
        
        coords_path = subdir.joinpath('coords.npz')
        coords_path = coords_path if coords_path.exists() else subdir.joinpath('world_coordinates.npz')
        coords = np.load(coords_path)
        n = len(coords)
        keys = list(coords.keys())

        coords = [torch.from_numpy(v).cuda(non_blocking=True) for v in coords.values()]
        masks = [x.ne(0).any(-1) for x in coords]
        counts = torch.as_tensor([x.count_nonzero() for x in masks])
        by_count = counts.argsort(descending=True)

        pbar = tqdm.tqdm(total=n * (n - 1) // 2, leave=False)
        for i in range(n - 1):
            ii = by_count[i]
            c1 = coords[ii]
            m1 = masks[ii]
            s1 = counts[ii]

            index.reset()
            index.train(c1[m1])
            index.add(c1[m1])
            for j in range(i + 1, n):
                jj = by_count[j]
                c2 = coords[jj]
                m2 = masks[jj]
                s2 = counts[jj]

                p1, _ = compute_flow(c1, c2, m1, m2, index)
                overlap = p1.shape[0] / (0.5 * (s1 + s2))

                k = '-'.join((keys[ii], keys[jj]))
                overlaps[name][k] = overlap.item()
                pbar.update()
        pbar.close()

    # convert to a sorted list of pairs, sorted in descending order based on overlap
    overlaps = sorted(
        [(k1, k2, v2) for k1, v1 in overlaps.items() for k2, v2 in v1.items()],
        key=lambda x: x[-1],
        reverse=True
    )
    
    pickle.dump(overlaps, root.joinpath('overlaps.pkl').open('wb'))


def dataset_render(root, num_sets, imgs_per_set, seed, **kwargs):
    dest = Path(root)
    dest.mkdir(exist_ok=True, parents=True)
    meta_file = dest.joinpath('metadata.yaml')
    params = dict(kwargs, num_sets=num_sets, imgs_per_set=imgs_per_set, seed=seed)
    with open(meta_file, 'w') as fp:
        meta = OmegaConf.to_yaml({'params': params})
        fp.write(meta)
        fp.write('rotation_units: normalized\n')
        fp.write('camera_distance: 3.0\n')
        fp.write('objects:\n')
    rng = np.random.default_rng(seed)
    sets = sample_julia_sets(num_sets, rng)
    np.save(str(dest.joinpath('julia_sets.npy')), sets)
    render_images(sets, root, imgs_per_set, rng=rng, **kwargs)
    
    import yaml
    import pickle
    meta = yaml.unsafe_load(meta_file.open())
    pickle.dump(meta, meta_file.with_suffix('.pkl').open('wb'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True, help='Path to root directory for saving')
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser('render')
    p.add_argument('-n', '--num_sets', type=int, default=100)
    p.add_argument('-i', '--imgs_per_set', type=int, default=100)
    p.add_argument('-v', '--view_range', nargs=2, type=float, default=(0.5, 0.25))
    p.add_argument('-z', '--zoom_range', nargs='+', type=float, default=(0.75, 3.0))
    p.add_argument('-l', '--light_range', nargs=2, type=float, default=(0.22, 0.17))
    p.add_argument('-a', '--ambient_range', nargs=2, type=float, default=(0.25, 0.45))
    p.add_argument('-t', '--add_texture', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('-s', '--seed', type=int, default=987654321)
    p.add_argument('-g', '--cam_grid', nargs=2, type=int, default=(0, 0))
    p.set_defaults(func=dataset_render)

    p = subparsers.add_parser('flow')
    p.add_argument('-f', '--force', action=argparse.BooleanOptionalAction)
    p.set_defaults(func=dataset_flows)

    p = subparsers.add_parser('overlap')
    p.set_defaults(func=measure_overlaps)

    args = parser.parse_args()

    args = args.__dict__
    func = args.pop('func')
    func(**args)