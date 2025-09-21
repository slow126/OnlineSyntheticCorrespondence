import moderngl
import numpy as np
from PIL import Image


def get_vertices():
    vertices = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            -1.0,  1.0,
            1.0, -1.0,
            1.0,  1.0,
            -1.0,  1.0,
        ], dtype=np.float32,
    )
    return vertices

def load_shader_code(frag, vert='shaders/vertex.c'):
    # Shader code
    vertex_shader = open(vert).read()
    fragment_shader = open(frag).read()
    return vertex_shader, fragment_shader


def setup_moderngl(vertex_shader, fragment_shader, resolution):
    ctx = moderngl.create_standalone_context(backend='egl', require=440, device_index=0)

    rbuf_col = ctx.renderbuffer(resolution, components=4)
    rbuf_coord = ctx.renderbuffer(resolution, components=3, dtype='f4')
    rbuf_normal = ctx.renderbuffer(resolution, components=3, dtype='f4')
    fbo = ctx.framebuffer([rbuf_col, rbuf_coord, rbuf_normal])
    fbo.use()

    compiled_prog = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=(fragment_shader),
    )
    vao = ctx.vertex_array(
        compiled_prog, [
            (ctx.buffer(get_vertices()), '2f', 'in_vert'),
        ]
    )

    return {'ctx': ctx, 'fbo': fbo, 'program': compiled_prog, 'vao': vao}


def update_uniforms(program, uniforms=None, **kw):
    if uniforms is None:
        uniforms = {}
    uniforms.update(kw)
    for k, v in uniforms.items():
        if program.get(k, None) is not None:
            program[k] = v


def make_textures(ctx, fg=None, bg=None):
    textures = []
    if fg is not None:
        texture_fg = ctx.texture3d(fg.shape[1:-1], 3, fg.tobytes(), dtype='f1')
        texture_fg.filter = (moderngl.LINEAR,) * 2
        texture_fg.use(0)
        textures.append(texture_fg)

    if bg is not None:
        texture_bg = ctx.texture(bg.shape[1:-1], 3, bg.tobytes(), dtype='f1')
        texture_bg.filter = (moderngl.LINEAR,) * 2
        texture_bg.use(1)
        textures.append(texture_bg)

    return textures


def render(vao):
    vao.render(mode=moderngl.TRIANGLES)


def get_img_and_buffers(fbo, resolution):
    img = fbo.read(components=4)
    img = Image.frombytes('RGBA', resolution, img)
    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    coord = fbo.read(attachment=1, dtype='f4')
    coord = np.frombuffer(coord, dtype=np.float32).reshape(*resolution, 3)
    coord = np.ascontiguousarray(coord[::-1])

    normals = fbo.read(attachment=2, dtype='f4')
    normals = np.frombuffer(normals, dtype=np.float32).reshape(*resolution, 3)
    normals = np.ascontiguousarray(normals[::-1])

    return img, coord, normals


def get_rendered(state, res, fg=None, bg=None, **kw):
    textures = make_textures(state['ctx'], fg, bg)
    update_uniforms(state['program'], **kw)
    render(state['vao'])
    res = (int(res[0]), int(res[1]))
    rendered = get_img_and_buffers(state['fbo'], res)
    for t in textures: t.release() # prevent GPU memory leaks
    return rendered


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', type=float, default=0.0)
    parser.add_argument('-r', '--resolution', type=int, default=256)
    parser.add_argument('-v', '--view', nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument('-d', '--dist', type=float, default=5.0)
    args = parser.parse_args()

    resolution = args.resolution

    uniforms = {
        'iResolution': (float(resolution), ) * 3,
        'iTime': args.time,
        'iViewAngleXY': args.view,
        'iViewDistance': args.dist,
    }

    shader_code = load_shader_code('rendering/shaders/quatJulia.c', vert='rendering/shaders/vertex.c')
    state = setup_moderngl(*shader_code, (resolution,) * 2)
    update_uniforms(state['program'], uniforms)
    render(state['vao'])

    img, coord = get_img_and_buffers(state['fbo'], (resolution,) * 2)
    img.save('frame.png')
    np.save('coord.npy', coord)
