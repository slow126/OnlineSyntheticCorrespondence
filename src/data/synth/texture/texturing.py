# TODO:
#   1. Using 2D textures for objects -- maybe something like environment mapping?
import math
from typing import Union

import numba
from numba import cuda
import numpy as np
import torch


def apply_texture(
    geometry: torch.Tensor,
    normals: torch.Tensor,
    texture: torch.Tensor,
    background: torch.Tensor,
    camera: torch.Tensor,
    light: torch.Tensor,
    c_min: Union[float, torch.Tensor] = -1.5,
    c_max: Union[float, torch.Tensor] = 1.5,
    ambient: Union[float, torch.Tensor] = 0.4,
    diffuse: Union[float, torch.Tensor] = 0.6,
    specular: Union[float, torch.Tensor] = 0.45,
    specular_exp: Union[float, torch.Tensor] = 10.0
):
    if geometry.device.type == 'cpu':
        textured = apply_texture_cpu(
            geometry.numpy(),
            normals.numpy(),
            texture.numpy(),
            background.numpy(),
            camera.numpy(),
            light.numpy(),
            c_min,
            c_max,
            ambient,
            diffuse,
            specular,
            specular_exp
        )
        textured = torch.from_numpy(textured).moveaxis(-1, 0).contiguous()
    else:
        # make sure everything is float32, because otherwise the kernel might still run
        # but the output will be wrong
        geometry = geometry.to(torch.float32)
        normals = normals.to(torch.float32)
        texture = texture.to(torch.float32)
        background = background.to(torch.float32)
        camera = camera.to(torch.float32)
        light = light.to(torch.float32)

        b = geometry.shape[0]

        args = wrap_scalars(
            c_min, c_max, ambient, diffuse, specular, specular_exp,
            batch_size=b, device=geometry.device
        )

        threads = (16, 16)
        blocks = tuple([int(math.ceil(geometry.shape[i + 1] / threads[i])) for i in range(2)] + [b])
        textured = torch.zeros_like(geometry)

        # launch kernel
        apply_texture_cuda[blocks, threads](
            cuda.as_cuda_array(textured),
            cuda.as_cuda_array(geometry),
            cuda.as_cuda_array(normals),
            cuda.as_cuda_array(texture),
            cuda.as_cuda_array(background),
            cuda.as_cuda_array(camera),
            cuda.as_cuda_array(light),
            *args
        )

        textured = textured.moveaxis(-1, 1).contiguous()

    return textured


def wrap_scalars(*args, batch_size=1, device='cuda'):
    result = []
    for x in args:
        if isinstance(x, float):
            x = torch.tensor([x] * batch_size, device=device, dtype=torch.float32)
        result.append(cuda.as_cuda_array(x.to(torch.float32)))
    return result


### CPU VERSION (COMPILED WITH NUMBA) ###

@numba.njit
def _normalize(v, out):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    n = n if n > 0 else np.float32(1.0)
    out[0] = v[0] / n
    out[1] = v[1] / n
    out[2] = v[2] / n
    return out


@numba.njit
def _minmax(arr):
    mn = np.inf
    mx = -np.inf
    arr = arr.ravel()
    for i in range(arr.shape[0]):
        v = arr[i]
        mn = min(mn, v)
        mx = max(mn, v)
    return mn, mx


@numba.njit('float32(float32,float32,float32)')
def _interp(u, v, a):
    return (1.0 - a) * u + a * v


@numba.njit
def _texture2d_interp(out, texture2d, x, y):
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = int(math.ceil(x)), int(math.ceil(y))
    
    dx = x - np.float32(x0)
    dy = y - np.float32(y0)
    
    v0 = texture2d[y0, x0]
    v1 = texture2d[y0, x1]
    v2 = texture2d[y1, x0]
    v3 = texture2d[y1, x1]
    
    out[0] = _interp(_interp(v0[0].item(), v1[0].item(), dx), _interp(v2[0].item(), v3[0].item(), dx), dy)
    out[1] = _interp(_interp(v0[1].item(), v1[1].item(), dx), _interp(v2[1].item(), v3[1].item(), dx), dy)
    out[2] = _interp(_interp(v0[2].item(), v1[2].item(), dx), _interp(v2[2].item(), v3[2].item(), dx), dy)


@numba.njit
def _texture3d_interp(out, texture3d, xyz):
    x0, y0, z0 = int(math.floor(xyz[0])), int(math.floor(xyz[1])), int(math.floor(xyz[2]))
    x1, y1, z1 = int(math.ceil(xyz[0])), int(math.ceil(xyz[1])), int(math.ceil(xyz[2]))
    
    dx = xyz[0] - np.float32(x0)
    dy = xyz[1] - np.float32(y0)
    dz = xyz[2] - np.float32(z0)
    
    v0 = texture3d[z0, y0, x0]
    v1 = texture3d[z0, y0, x1]
    v2 = texture3d[z0, y1, x0]
    v3 = texture3d[z0, y1, x1]
    v4 = texture3d[z1, y0, x0]
    v5 = texture3d[z1, y0, x1]
    v6 = texture3d[z1, y1, x0]
    v7 = texture3d[z1, y1, x1]
    
    for i in range(3):
        out[i] = _interp(
            _interp(_interp(v0[i].item(), v1[i].item(), dx), _interp(v2[i].item(), v3[i].item(), dx), dy),
            _interp(_interp(v4[i].item(), v5[i].item(), dx), _interp(v6[i].item(), v7[i].item(), dx), dy),
            dz
        )


@numba.njit
def _dotprod(u, v):
    return (u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2])


@numba.njit
def _vsub(u, v, out):
    out[0] = u[0] - v[0]
    out[1] = u[1] - v[1]
    out[2] = u[2] - v[2]


@numba.njit
def apply_texture_cpu(
    geo,
    normals,
    obj_texture,
    background,
    camera,
    light,
    c_min = -1.5,
    c_max = 1.5,
    ambient = np.float32(0.4),
    diffuse = np.float32(0.6),
    specular = np.float32(0.45),
    specular_exp = np.float32(10.0)
):
    c_scale = (np.array(obj_texture.shape[:3], dtype=np.float32) - 1) / (c_max - c_min)
    
    h, w = geo.shape[:2]
    shaded = np.empty((h, w, 3), dtype=np.float32)

    buf = np.empty(3, dtype=np.float32)
    E = np.empty(3, dtype=np.float32)
    
    for i in range(h):
        y = 2 * i / h - 1
        for j in range(w):
            coord = geo[i, j]

            if coord[0] != 0 or coord[1] != 0 or coord[2] != 0:
                x = 2 * j / w - 1
                
                _vsub(light, coord, buf)
                _normalize(buf, buf)

                L_N = _dotprod(buf, normals[i, j])

                _vsub(camera, coord, E)
                _normalize(E, E)

                buf[0] = 0.5 * (E[0] + buf[0])
                buf[1] = 0.5 * (E[1] + buf[1])
                buf[2] = 0.5 * (E[2] + buf[2])
                _normalize(buf, buf)
                
                dif = max(L_N, 0.0) * diffuse
                spec = pow(max(_dotprod(buf, normals[i, j]), 0.0), specular_exp) * specular
                shade = ambient + dif + spec
                
                buf[0] = (coord[0] - c_min) * c_scale[0]
                buf[1] = (coord[1] - c_min) * c_scale[1]
                buf[2] = (coord[2] - c_min) * c_scale[2]
                _texture3d_interp(buf, obj_texture, buf)

                shaded[i, j, 0] = pow(min(max(0, shade * buf[0]), 1), 1.0 / 2.2)
                shaded[i, j, 1] = pow(min(max(0, shade * buf[1]), 1), 1.0 / 2.2)
                shaded[i, j, 2] = pow(min(max(0, shade * buf[2]), 1), 1.0 / 2.2)
            elif background.shape[0] == geo.shape[0] and background.shape[1] == geo.shape[1]:
                shaded[i, j] = background[i, j]
            elif background.shape[0] == 1 or background.shape[1] == 1:
                shaded[i, j] = background[0, 0]
            else:
                x = j * ((background.shape[1] - 1) / (geo.shape[1] - 1))
                y = i * ((background.shape[0] - 1) / (geo.shape[0] - 1))
                _texture2d_interp(shaded[i, j], background, x, y)
            
    return shaded

### END CPU CODE ###

### CUDA VERSION (COMPILED WITH NUMBA) ###

@cuda.jit(device=True)
def _normalize_dv(vec):
    n = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    n = 1 / n if n > 0 else 1.0
    vec[0] *= n
    vec[1] *= n
    vec[2] *= n


@cuda.jit(device=True)
def _interp_dv(u, v, a):
    return (1.0 - a) * u + a * v


@cuda.jit(device=True)
def _texture2d_interp_dv(out, texture2d, x, y):
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = int(math.ceil(x)), int(math.ceil(y))
    
    dx = x - float(x0)
    dy = y - float(y0)
    
    v0 = texture2d[y0, x0]
    v1 = texture2d[y0, x1]
    v2 = texture2d[y1, x0]
    v3 = texture2d[y1, x1]
    
    out[0] = _interp_dv(_interp_dv(v0[0], v1[0], dx), _interp_dv(v2[0], v3[0], dx), dy)
    out[1] = _interp_dv(_interp_dv(v0[1], v1[1], dx), _interp_dv(v2[1], v3[1], dx), dy)
    out[2] = _interp_dv(_interp_dv(v0[2], v1[2], dx), _interp_dv(v2[2], v3[2], dx), dy)

    
@cuda.jit(device=True)
def _texture3d_interp_dv(out, texture, xyz):
    x0, y0, z0 = int(math.floor(xyz[0])), int(math.floor(xyz[1])), int(math.floor(xyz[2]))
    x1, y1, z1 = int(math.ceil(xyz[0])), int(math.ceil(xyz[1])), int(math.ceil(xyz[2]))
    
    dx = xyz[0] - float(x0)
    dy = xyz[1] - float(y0)
    dz = xyz[2] - float(z0)
    
    v0 = texture[z0, y0, x0]
    v1 = texture[z0, y0, x1]
    v2 = texture[z0, y1, x0]
    v3 = texture[z0, y1, x1]
    v4 = texture[z1, y0, x0]
    v5 = texture[z1, y0, x1]
    v6 = texture[z1, y1, x0]
    v7 = texture[z1, y1, x1]
    
    for i in range(3):
        out[i] = _interp_dv(
            _interp_dv(_interp_dv(v0[i], v1[i], dx), _interp_dv(v2[i], v3[i], dx), dy),
            _interp_dv(_interp_dv(v4[i], v5[i], dx), _interp_dv(v6[i], v7[i], dx), dy),
            dz
        )


@cuda.jit
def _dotprod_dv(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@cuda.jit
def _vsub_dv(out, u, v):
    out[0] = u[0] - v[0]
    out[1] = u[1] - v[1]
    out[2] = u[2] - v[2]


@cuda.jit(device=True)
def _phong_shade_dv(
    buf, coord, nrml, light, camera,
    ambient, diffuse, specular, specular_exp
):
    eye = cuda.local.array(3, numba.types.float32)
    _vsub_dv(eye, camera, coord)
    _normalize_dv(eye)

    _vsub_dv(buf, light, coord)
    _normalize_dv(buf)

    # light source and normal
    L_N = _dotprod_dv(buf, nrml)

    buf[0] = 0.5 * (eye[0] + buf[0])
    buf[1] = 0.5 * (eye[1] + buf[1])
    buf[2] = 0.5 * (eye[2] + buf[2])
    _normalize_dv(buf)

    # half-vector and normal
    H_N = _dotprod_dv(buf, nrml)

    dif = max(L_N, 0.0) * diffuse
    spec = pow(max(H_N, 0.0), specular_exp) * specular
    shade = ambient + dif + spec
    
    return shade


@cuda.jit(device=True)
def _texture3d_dv(buf, coord, texture, shade, c_min, c_max):
    buf[0] = (coord[0] - c_min) * (texture.shape[0] - 1) / (c_max - c_min)
    buf[1] = (coord[1] - c_min) * (texture.shape[1] - 1) / (c_max - c_min)
    buf[2] = (coord[2] - c_min) * (texture.shape[2] - 1) / (c_max - c_min)

    _texture3d_interp_dv(buf, texture, buf)

    buf[0] = pow(min(max(0, buf[0] * shade), 1), 1.0 / 2.2)
    buf[1] = pow(min(max(0, buf[1] * shade), 1), 1.0 / 2.2)
    buf[2] = pow(min(max(0, buf[2] * shade), 1), 1.0 / 2.2)


@cuda.jit(
    "void(float32[:,:,:,:],float32[:,:,:,:],float32[:,:,:,:],float32[:,:,:,:,:],"
    "float32[:,:,:,:],float32[:,:],float32[:,:],float32[:],float32[:],float32[:],"
    "float32[:],float32[:],float32[:])"
)
def apply_texture_cuda(
    output,
    geo,
    normals,
    obj_texture3d,
    background,
    camera,
    light,
    c_min,
    c_max,
    ambient,
    diffuse,
    specular,
    specular_exp,
):
    h, w = geo.shape[1:3]
    z = cuda.blockIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if i < h and j < w:
        coord = geo[z, i, j]
        # buffer used to store intermediate vector results
        buf = cuda.local.array(3, numba.types.float32)
        
        if coord[0] != 0 or coord[1] != 0 or coord[2] != 0:
            
            nrml = normals[z, i, j]

            shade = _phong_shade_dv(
                buf, coord, nrml, light[z], camera[z], ambient[z],
                diffuse[z], specular[z], specular_exp[z]
            )
            _texture3d_dv(buf, coord, obj_texture3d[z], shade, c_min[z], c_max[z])
        elif h == background.shape[1] and w == background.shape[2]:
            # background resolution matches image resolution
            buf[0] = background[z, i, j, 0]
            buf[1] = background[z, i, j, 1]
            buf[2] = background[z, i, j, 2]
        elif background.shape[1] == 1 or background.shape[2] == 1:
            # single buf per image
            buf[0] = background[z, 0, 0, 0]
            buf[1] = background[z, 0, 0, 1]
            buf[2] = background[z, 0, 0, 2]
        else:
            # interpolate background with different resolution
            x = j * ((background.shape[2] - 1.0) / (w - 1.0))
            y = i * ((background.shape[1] - 1.0) / (h - 1.0))
            _texture2d_interp_dv(buf, background[z], x, y)
        
        output[z, i, j, 0] = buf[0]
        output[z, i, j, 1] = buf[1]
        output[z, i, j, 2] = buf[2]

### END CUDA CODE ###


def test():
    from pathlib import Path
    import pickle

    try:
        from . import grf
    except:
        import grf

    obj = 'c_00013'

    root = Path.home().joinpath('compute/julia3d/v1/')
    geometry = np.load(root.joinpath(f'{obj}/coords.npz'))
    normals = np.load(root.joinpath(f'{obj}/normals.npz'))
    meta = pickle.load(root.joinpath('metadata.pkl').open('rb'))
    m = meta['objects'][obj]

    names = ['0032', '0082']
    geometry = torch.from_numpy(np.stack([geometry[i] for i in names], 0)).cuda()
    normals = torch.from_numpy(np.stack([normals[i] for i in names], 0)).cuda()
    view = torch.from_numpy(np.array([m[i]['camera_position'] for i in names], dtype=np.float32)).cuda()
    light = view * 2
    # light = (view + torch.normal(0, 0.08, view.shape, device=view.device)).mul_(2)
    # light = torch.from_numpy(np.array([m[i]['light_source'] for i in names], dtype=np.float32) * math.pi).cuda()

    t_kw = dict(
        alpha=(3.0, 4.5),
        batch_size=geometry.shape[0],
        rand_mean=(-0.5, 0.5),
        rand_std=(0.6, 1.4),
        device=geometry.device
    )
    texture3d = grf.gaussian_random_field((48, 48, 48), **t_kw).contiguous()
    background = grf.gaussian_random_field((320, 320), **t_kw).contiguous()

    img = apply_texture(
        geometry,
        normals,
        texture3d,
        background,
        view,
        light,
        c_min = geometry.flatten(1).min(1)[0],
        c_max = geometry.flatten(1).max(1)[0],
    )

    from torchvision import io
    for i in range(img.shape[0]):
        io.write_jpeg(img[i].mul(255).byte().cpu(), f'test{i}.jpg')

    img = apply_texture(
        geometry[0].cpu(),
        normals[0].cpu(),
        texture3d[0].cpu(),
        background[0].cpu(),
        view[0].cpu(),
        light[0].cpu(),
        c_min = geometry[0].min().cpu().item(),
        c_max = geometry[0].max().cpu().item(),
    )

    io.write_jpeg(img.mul(255).byte().cpu(), f'test.jpg')

if __name__ == '__main__':
    test()