from importlib.metadata import distribution
from typing import Dict, Literal

from PIL import Image
from more_itertools import distribute
import numpy as np
import torch

from .base import ComponentsBase
from .computed.curvature import load_curvature
from . import sampler
from ..texture import grf
from .mandelbulb import mandelbulb_param_sampler
debug_renders = False

from .samplers.default_samplers import default_julia_sampler, default_angle_sampler, default_scale_sampler, default_mandelbulb_sampler
import torch.profiler

class OnlineGeometryDataset(ComponentsBase):
    '''Dataset that renders Julia set sample pairs on the fly, returning the geometry and normal maps.

    Args:
        shader_code_path (str): path to file containing the OpenGL rendering code for the Julia sets.
        antialias (int): number of antialias passes for rendering.
        num_samples (int): size of the dataset (number of pairs).
        random_flip (float): proportion of pairs where one of the images is mirrored from the other.
        random_swap (bool): whether to randomly swap the source and target images within the pair.
        julia_sampler (Dict): specification of the sampling distribution for the Julia set parameters.
            See sampler.CurvatureMapSampler. This argument is passed directly as the `dist` parameter
            of the constructor.
        angle_sampler (Dict): specification of the sampling distributions for view angles. See
            sampler.AngleSampler. This argument should contain sub-dicts for the `x_components` and
            `y_components` parameters of the constructor.
        scale_sampler (Dict): specification of the sampling distributions for scale (zoom). See
            sampler.ScaleSampler. This argument should contain sub-dicts for the `abs_components` and
            `rel_components` parameters of the constructor.
        size (int): size in pixels of the images (image will be square).
        crop (str): cropping method, one of "center", "none" or "random".
        single_process (bool): whether the dataset will be running in a single process or multiprocess
            (i.e. dataloader workers) environment. This is needed for properly setting up OpenGL.
        seed (int): seed for random number generator.
    '''
    def __init__(
        self,
        antialias: int = 3,
        num_samples: int = 1000,
        random_flip: float = 0.0,
        random_swap: bool = True,
        julia_sampler: Dict = None,
        angle_sampler: Dict = None,
        scale_sampler: Dict = None,
        mandelbulb_sampler: Dict = None,
        size: int = 256,
        crop: Literal['center', 'none', 'random'] = 'none',
        single_process: bool = True,
        seed: int = 987654321,
        shaders: Dict = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        super().__init__(size, crop, seed)
        self.size = size
        self.num_samples = num_samples
        self.random_flip = random_flip
        self.random_swap = random_swap
        self.shaders = shaders

        self.res = (int(1.25 * size),) * 2

        # samplers
        if julia_sampler is None:
            julia_sampler = default_julia_sampler()
        if angle_sampler is None:
            angle_sampler = default_angle_sampler()
        if scale_sampler is None:
            scale_sampler = default_scale_sampler()
        if mandelbulb_sampler is None:
            mandelbulb_sampler = default_mandelbulb_sampler()

        self.julia_sampler = sampler.CurvatureMapSampler(*load_curvature(), julia_sampler, seed=seed + 1)
        self.angle_sampler = sampler.AngleSampler(**angle_sampler, seed=seed + 2)
        self.scale_sampler = sampler.ScaleSampler(**scale_sampler, seed=seed + 3)

        self.shaders = {k: v for k, v in self.shaders.items() if v is not None}

        self.mandelbulb_sampler = mandelbulb_param_sampler.MandelbulbPowerSampler(**mandelbulb_sampler)
        self.mandel_exists = any('mandelbulb' in shader_path.lower() for shader_path in self.shaders.values())

        self.uniforms = {
            'iResolution': tuple(float(x) for x in self.res) + (1.0,),
            'iViewDistance': 5.0,
            'iAntialias': antialias, 
        }

        self.t_kw = dict(
            alpha=(3.0, 4.5),
            covariance=(0.0, 5.0),
            rand_mean=(-1.0, 1.0),
            rand_std=(0.3, 1.2),
            output_uint8=True,
        )

        # Device management - defaults to CPU for flexibility
        self._device = torch.device('cpu')

        program_count = sum(1 for key in self.shaders if key.startswith('program'))
        if single_process:
            self.setup()

    def setup(self):
        self.mgl_state = setup_moderngl_multi(
            shader_paths=self.shaders, 
            resolution=self.res
        )


    def __len__(self):
        return self.num_samples

    def sample_light_and_material(self, view_angle):
        # standard deviation approximately 15 degrees
        light_angle = self.rng_np.normal(view_angle*-4, 0.088)

        ambient = self.rng_np.uniform(0.1, 0.4, 2)
        diffuse = self.rng_np.uniform(0.2, 0.4, 2)
        specular = self.rng_np.uniform(0.7, 0.99, 2)
        specular_exp = self.rng_np.integers(32, 75, 2)

        return [
            {
                'iLightSource': light_angle[i],
                'iAmbientLight': ambient[i],
                'iDiffuseScale': diffuse[i],
                'iSpecularScale': specular[i],
                'iSpecularExp': specular_exp[i],
            }
            for i in range(2)
        ]

    def convert_location(self, loc, dist=1):
        loc = np.multiply(loc, [np.pi, -np.pi])
        c = np.cos(loc)
        s = np.sin(loc)
        return dist * np.stack([s[:, 0] * c[:, 1], -s[:, 1], c[:, 0] * c[:, 1]], 1)

    def transform(self, data):
        keys = ('geometry', 'normals', 'object_id')
        crop_ps = self.get_crop(self.res)
        for i in range(len(data)):
            data[i] = super().transform(data[i], crop_ps, keys)

        if self.random_flip and torch.rand(1, generator=self.rng) < self.random_flip:
            # random horizontal flip
            for k in keys:
                if k in data[1]:
                    data[1][k] = data[1][k].flip(dims=(1,)).contiguous()

        # randomly swap from (src, trg) to (trg, src)
        if self.random_swap and torch.rand(1, generator=self.rng) < 0.5:
            data[0], data[1] = data[1], data[0]

        return data

    @torch.profiler.record_function("OnlineDataset::__getitem__")
    def __getitem__(self, idx):
        # for now do the simplest thing and just return geometry, normals etc.
        #    later it might be worth looking at moving the texturing in here as well

        # sample a julia set based on a distribution over parameters
        # breakpoint()

        # Spencer note: This currently gets overwritten by a loop that generates one for each object. 
        # This will most likely be deleted later. 
        # TODO: Double check if i need this anymore.
        c = self.julia_sampler.sample()
        self.uniforms['iJuliaC'] = c

        # c = self.rng_np.uniform(-1, 1, 4)
        # c /= np.linalg.norm(c, ord=2)
        # m = self.rng_np.beta(12, 4)
        # c = c * m

        # sample a pair of rendering parameters based on distributions
        #  - view angle, scale, etc...
        #  - there's an unconditional distribution for the first image, and a conditional one for the second
        # view_angle = self.rng_np.uniform(-1, 1, (1, 2)) * (0.5, 0.25)
        # view_angle = np.repeat(view_angle, 2, axis=0)
        # view_angle[1] += self.rng_np.uniform(-0.2, 0.2, (2,))
        view_angle = self.angle_sampler.sample()
        tmp_distance = self.uniforms['iViewDistance']
        camera = self.convert_location(view_angle, tmp_distance)

        scale = self.scale_sampler.sample()
        # convert from scale to focal plane shader argument: focal_plane == 1 -> scale == 0.36
        zoom = scale / 0.36
        # zoom = self.rng_np.uniform(2.5, 2.1)
        # zoom = np.array([zoom, zoom + self.rng_np.uniform(-0.25, 0.25)])

        other_params = self.sample_light_and_material(view_angle)

        if self.mgl_state.num_objects > 2:
            # Create array of possible offsets
            possible_offsets = [
            (0,0,0), 
            (1,0,0), (-1,0,0),
            (0,1,0), (0,-1,0),
            (0,0,1), (0,0,-1),
            (1,1,0), (-1,-1,0),
            (1,0,1), (-1,0,-1),
            (0,1,1), (0,-1,-1),
            (-1,1,0), (1,-1,0),
            (-1,0,1), (1,0,-1),
            (-1,1,1), (1,-1,-1),
            (1,1,1), (-1,-1,-1),
        ]

            # Randomly permute all offsets
            offsets = self.rng_np.permutation(possible_offsets)
            offsets_scale = np.random.rand(self.mgl_state.num_objects) * 0.5 + 1.0

        else:
            offsets = [(0,0,0), (0,0,0)]
            offsets_scale = [1, 1]

        if debug_renders:
            offsets = []
            for i in range(self.mgl_state.num_objects):
                offsets.append((i,i,i))
            # offsets = np.array([(0,0,0), (-2,-1,-1), (2,1,1)])

        # Convert to numpy array
        offsets = np.array(offsets)

        julia_c_list = []
        mandelbulb_p_list = []
        for i in range(self.mgl_state.num_objects):
            c = self.julia_sampler.sample()
            p = self.mandelbulb_sampler.sample()
            julia_c_list.append(c)
            mandelbulb_p_list.append(p)

        # run the shader and return the data
        data = []
        # print("idx: ", idx)
        # mandelbulb_p_list = [1.5 + idx / 39]

        for i in range(2):
            self.uniforms.update(
                iViewAngleXY = view_angle[i],
                iFocalPlane = zoom[i],
                **other_params[i]
            )

            for j in range(self.mgl_state.num_objects):  

                # c = self.julia_sampler.sample()
                self.uniforms['iJuliaC'] = julia_c_list[j]
                if self.mandel_exists:  
                    self.uniforms['iMandelbulbP'] = mandelbulb_p_list[j]
                obj = getattr(self.mgl_state, f'object{j}')
                self.uniforms['iObjectID'] = j
                self.uniforms['iObjectOffset'] = offsets[j] * offsets_scale[j]# np.array([1, 1, 1]) * i 

                # TODO: combine the render and merge so that you don't have to fetch buffers twice.
                # [coord, normals], object_id = get_multirendered(obj, self.res, **self.uniforms)
                if get_multirendered_gpu(obj, self.res, **self.uniforms) == 1:
                    print("Error: Failed to render")
                    continue


            # Merge the renders
            if self.mgl_state.num_objects > 1:
                [coord, normals], object_id = merge_renders(self.mgl_state, self.res)
            else:
                rendered = get_img_and_buffers(self.mgl_state.object0.fbo, self.res)
                coord = rendered[0]
                normals = rendered[1]
                object_id = np.zeros(self.res, dtype=np.int32)

            data.append({
                'geometry': coord, 
                'normals': normals, 
                'camera': camera[0],
                'object_id': object_id,
                'max_num_objects': self.mgl_state.num_objects,
            })


        # Transform with device-aware tensor creation
        data = self.transform(data)

        return data

    def debug_view(self, idx, save_path=None):
        """
        Debug helper to visualize a sample.
        """
        data = self[idx]

        import matplotlib
        import matplotlib.pyplot as plt

        # Check if we're in a headless environment
        is_headless = not hasattr(plt, 'get_backend') or plt.get_backend() == 'agg'

        if is_headless:
            matplotlib.use('Agg')

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Show geometry
        axes[0,0].imshow(data[0]['geometry'])
        axes[0,0].set_title('Source Geometry')
        axes[0,1].imshow(data[1]['geometry'])
        axes[0,1].set_title('Target Geometry')

        # Show normals
        axes[1,0].imshow((data[0]['normals'] + 1) / 2)  # normalize to 0-1
        axes[1,0].set_title('Source Normals')
        axes[1,1].imshow((data[1]['normals'] + 1) / 2)
        axes[1,1].set_title('Target Normals')

        plt.tight_layout()

        if is_headless:
            if save_path is None:
                save_path = f'debug_view_{idx}.png'
            plt.savefig(save_path)
            plt.close()
            print(f"Saved debug view to: {save_path}")
        else:
            plt.show()

def update_uniforms(program, uniforms=None, **kw):
    if uniforms is None:
        uniforms = {}
    uniforms.update(kw)
    for k, v in uniforms.items():
        if program.get(k, None) is not None:
            program[k] = v

def make_textures(ctx, fg=None, bg=None):
    LINEAR = 9729  # = moderngl.LINEAR
    textures = []
    if fg is not None:
        if isinstance(fg, torch.Tensor): fg = fg.numpy()
        texture_fg = ctx.texture3d(fg.shape[:-1], 3, fg.tobytes(), dtype='f1')
        texture_fg.filter = (LINEAR,) * 2
        texture_fg.use(0)
        textures.append(texture_fg)

    if bg is not None:
        if isinstance(bg, torch.Tensor): bg = bg.numpy()
        texture_bg = ctx.texture(bg.shape[:-1], 3, bg.tobytes(), dtype='f1')
        texture_bg.filter = (LINEAR,) * 2
        texture_bg.use(1)
        textures.append(texture_bg)

    return textures

@torch.profiler.record_function("OnlineDataset::get_img_and_buffers")
def get_img_and_buffers(fbo, resolution):
    # img = fbo.read(components=4)
    # img = Image.frombytes('RGBA', resolution, img).convert('RGB')
    # img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    coord = fbo.read(attachment=1, dtype='f4')
    coord = np.frombuffer(coord, dtype=np.float32).reshape(*resolution, 3)
    coord = np.ascontiguousarray(coord[::-1])

    normals = fbo.read(attachment=2, dtype='f4')
    normals = np.frombuffer(normals, dtype=np.float32).reshape(*resolution, 3)
    normals = np.ascontiguousarray(normals[::-1])

    debug = fbo.read(attachment=3, dtype='f4')
    debug = np.frombuffer(debug, dtype=np.float32).reshape(*resolution, 3)
    debug = np.ascontiguousarray(debug[::-1])

    # return coord, normals, depth
    # return img, coord, normals
    return coord, normals

@torch.profiler.record_function("OnlineDataset::get_multirendered_gpu")
def get_multirendered_gpu(state, res, fg=None, bg=None, **kw):
    try:
        state.fbo.use()
        state.ctx.clear(depth=1.0, color=(0.0, 0.0, 0.0, 1.0))

        TRIANGLES = 4  # = moderngl.TRIANGLES
        update_uniforms(state.program, **kw)
        state.vao.render(mode=TRIANGLES) # PAPAYA: This is where rendering happens
        
        # Synchronize GPU without reading buffers
        # sync_gpu(state.ctx)

        return 0
    except:
        return 1

@torch.profiler.record_function("OnlineDataset::merge_renders")
def merge_renders(state, res):
    """Merge multiple pre-rendered FBOs based on depth"""

    # Set up single pass merge program and return early for testing
    if state.merge_name == 'merge_single_pass':
        merge_prog = state.merge.program
        bind_merge_single_pass(state, merge_prog, state.num_objects)
        state.merge.fbo.use()
        TRIANGLES = 4
        state.merge.vao.render(mode=TRIANGLES)
        rendered = get_img_and_buffers(state.merge.fbo, res)
        object_id = state.merge.fbo.read(attachment=5, components=1, dtype='i4')
        object_id = np.frombuffer(object_id, dtype=np.int32).reshape(res)
        object_id = np.ascontiguousarray(object_id[::-1])
        return rendered, object_id

    else:
        # Get depth buffers for debugging if needed

        # Bind the merge textures

        # Start with first object's FBO

        current_fbo = state.object0.fbo

        if debug_renders:
            depths = []
            for i in range(state.num_objects):
                depth = np.frombuffer(getattr(state, f'object{i}').fbo.color_textures[4].read(), 
                                    dtype=np.float32).reshape(res)
                depths.append(depth)

            
            t = get_img_and_buffers(current_fbo, res)
            debug_show_image(t[0], 'geometry' + str(0))
            debug_show_image(t[1], 'normals' + str(0))
            depth = np.frombuffer(current_fbo.color_textures[4].read(), dtype=np.float32).reshape(res)
            depth = np.ascontiguousarray(depth[::-1])
            debug_show_image(depth, 'depth' + str(0))

        # Iteratively merge with remaining objects
        for i in range(1, state.num_objects):
            next_fbo = getattr(state, f'object{i}').fbo

            if debug_renders:
                t = get_img_and_buffers(next_fbo, res)
                debug_show_image(t[0], 'geometry' + str(i))
                debug_show_image(t[1], 'normals' + str(i))  
                depth = np.frombuffer(next_fbo.color_textures[4].read(), dtype=np.float32).reshape(res)
                depth = np.ascontiguousarray(depth[::-1])
                debug_show_image(depth, 'depth' + str(i))   


            # Alternate between merge and current_fbo
            merge = state.merge

            # Bind the current and next FBOs for merging
            bind_merge(current_fbo, next_fbo, merge.program)
            
            # Set the merge FBO as the render target
            merge.fbo.use()

            # Render merged result
            TRIANGLES = 4
            merge.vao.render(mode=TRIANGLES)

            # Synchronize GPU without reading buffers
            sync_gpu(merge.ctx)

            # Update current_fbo to be the merged result for next iteration
            current_fbo = merge.fbo

            if debug_renders:
                t = get_img_and_buffers(merge.fbo, res)
                debug_show_image(t[0], 'merge_geometry' + str(i))
                debug_show_image(t[1], 'merge_normals' + str(i))

                depth = np.frombuffer(merge.fbo.color_textures[4].read(), dtype=np.float32).reshape(res)
                depth = np.ascontiguousarray(depth[::-1])
                debug_show_image(depth, 'merge_depth' + str(i))

        # Get final object IDs and render results
        res = (int(res[0]), int(res[1]))
        
        if hasattr(state, 'anti_alias') and state.anti_alias is not None:    
            bind_anti_alias(current_fbo, state.anti_alias.program)
            state.anti_alias.fbo.use()
            state.anti_alias.vao.render(mode=TRIANGLES)
            # Make sure GPU has finished rendering before reading data
            sync_gpu(state.anti_alias.ctx)
            rendered = get_img_and_buffers(state.anti_alias.fbo, res)
        else:
            # Make sure GPU has finished rendering before reading data
            rendered = get_img_and_buffers(current_fbo, res)

        object_id = current_fbo.read(attachment=5, components=1, dtype='i4')
        object_id = np.frombuffer(object_id, dtype=np.int32).reshape(res)
        object_id = np.ascontiguousarray(object_id[::-1])

        return rendered, object_id

def bind_merge_single_pass(state, program, num_objects):
    # Set the number of active objects
    program['num_active_objects'] = num_objects
    
    # Bind textures to sequential texture units
    # Only bind actual objects (num_objects)
    for i in range(num_objects):
        obj_fbo = getattr(state, f'object{i}').fbo
        obj_fbo.color_textures[0].use(i)
        obj_fbo.color_textures[1].use(i + 10)  # Use fixed offsets based on array size
        obj_fbo.color_textures[2].use(i + 20)
        obj_fbo.color_textures[3].use(i + 30)
        obj_fbo.color_textures[4].use(i + 40)
        obj_fbo.color_textures[5].use(i + 50)
        obj_fbo.color_textures[6].use(i + 60)
    
    # Always set full 10-element arrays regardless of actual object count
    # Use 0 for unused array elements
    color_array = [i if i < num_objects else 0 for i in range(10)]
    coord_array = [i+10 if i < num_objects else 0 for i in range(10)]
    normal_array = [i+20 if i < num_objects else 0 for i in range(10)]
    debug_array = [i+30 if i < num_objects else 0 for i in range(10)]
    depth_array = [i+40 if i < num_objects else 0 for i in range(10)]
    object_id_array = [i+50 if i < num_objects else 0 for i in range(10)]
    distance_field_array = [i+60 if i < num_objects else 0 for i in range(10)]
    
    # Set full arrays
    program['color_textures'] = color_array
    program['coord_textures'] = coord_array
    program['normal_textures'] = normal_array
    program['debug_textures'] = debug_array
    program['depth_textures'] = depth_array
    program['object_id_textures'] = object_id_array
    program['distance_field_textures'] = distance_field_array

@torch.profiler.record_function("OnlineDataset::bind_merge")
def bind_merge(fbo1, fbo2, merge_prog):
    # Bind the FBO textures
    fbo1.color_textures[0].use(0)  # color
    fbo1.color_textures[1].use(1)  # coord
    fbo1.color_textures[2].use(2)  # normal
    fbo1.color_textures[3].use(3)     # debug   
    fbo1.color_textures[4].use(4)     # depth
    fbo1.color_textures[5].use(5)     # object_id
    fbo1.color_textures[6].use(6)     # distance_field

    fbo2.color_textures[0].use(7)  # color
    fbo2.color_textures[1].use(8)  # coord
    fbo2.color_textures[2].use(9)  # normal
    fbo2.color_textures[3].use(10)     # debug   
    fbo2.color_textures[4].use(11)     # depth
    fbo2.color_textures[5].use(12)     # object_id
    fbo2.color_textures[6].use(13)     # distance_field

    # Set uniforms for the merge shader
    # merge_prog = state.merge.program
    merge_prog['fbo1_color'].value = 0
    merge_prog['fbo1_coord'].value = 1
    merge_prog['fbo1_normal'].value = 2
    merge_prog['fbo1_debug'].value = 3
    merge_prog['fbo1_depth'].value = 4
    merge_prog['fbo1_object_id'].value = 5
    merge_prog['fbo1_distance_field'].value = 6

    merge_prog['fbo2_color'].value = 7
    merge_prog['fbo2_coord'].value = 8
    merge_prog['fbo2_normal'].value = 9
    merge_prog['fbo2_debug'].value = 10
    merge_prog['fbo2_depth'].value = 11
    merge_prog['fbo2_object_id'].value = 12
    merge_prog['fbo2_distance_field'].value = 13

    return 

def bind_anti_alias(fbo, anti_alias_prog):
    # Bind the FBO textures
    fbo.color_textures[0].use(0)
    fbo.color_textures[1].use(1)
    fbo.color_textures[2].use(2)
    fbo.color_textures[4].use(4)
    fbo.color_textures[5].use(5)
    fbo.color_textures[6].use(6)

    anti_alias_prog['fbo_color'].value = 0
    anti_alias_prog['fbo_coord'].value = 1
    anti_alias_prog['fbo_normal'].value = 2
    anti_alias_prog['fbo_depth'].value = 3
    anti_alias_prog['fbo_object_id'].value = 4
    anti_alias_prog['fbo_distance_field'].value = 6
    
    return

def setup_moderngl_multi(shader_paths, resolution):
    import moderngl
    
    # Create single context to be shared
    ctx = moderngl.create_standalone_context(backend='egl', require=430, device_index=0)

    # Common vertex shader code...
    vertex_shader = (
        '#version 430\n'
        'in vec2 in_vert;\n'
        'in vec2 in_texcoord;\n'
        'uniform int iObjectID;\n'

        'out vec2 v_texcoord;\n'
        'out flat int v_object_id;\n'

        'void main() {\n'
        '    gl_Position = vec4(in_vert, 0.0, 1.0);\n'
        '    v_texcoord = in_texcoord;\n'
        '    v_object_id = iObjectID;\n'
        '}\n'
    )

    # Add texture coordinates
    vertices = np.array([
    -1.0, -1.0,  0.0, 0.0,  # Bottom-left
     1.0, -1.0,  1.0, 0.0,  # Bottom-right
    -1.0,  1.0,  0.0, 1.0,  # Top-left
     1.0,  1.0,  1.0, 1.0,  # Top-right
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

    # Helper class definition remains the same...
    class RenderContext:
        def __init__(self, ctx, fbo, program, vao):
            self.ctx = ctx
            self.fbo = fbo
            self.program = program 
            self.vao = vao

    # Create contexts dictionary...
    contexts = {}

    # Create program and VAO for each shader
    program_id = 0
    merge_name = "merge"
    for name, shader_path in shader_paths.items():
        with open(shader_path) as f:
            fragment_shader = f.read()

        program = ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        if 'merge_single_pass' in shader_path:
            merge_name = 'merge_single_pass'
            # Pre-allocate texture arrays by accessing them (prevents optimization out)
            for i in range(10):  # For up to 10 objects
                # This will create the uniform locations even if optimized
                try:
                    program[f'color_textures[{i}]'] = 0
                    program[f'coord_textures[{i}]'] = 0
                    program[f'normal_textures[{i}]'] = 0
                    program[f'debug_textures[{i}]'] = 0
                    program[f'depth_textures[{i}]'] = 0
                    program[f'object_id_textures[{i}]'] = 0
                    program[f'distance_field_textures[{i}]'] = 0
                except KeyError:
                    pass  # Ignore if uniform doesn't exist

        vbo = ctx.buffer(vertices)
        ibo = ctx.buffer(indices)
        vao = ctx.vertex_array(
            program,
            [(vbo, '2f 2f', 'in_vert', 'in_texcoord')],
            ibo
        )

        # Create a new framebuffer for each context
        fbo = create_framebuffer(ctx, resolution)
        contexts[name] = RenderContext(ctx, fbo, program, vao)
        program_id += 1
    # Create and return namespace object...
    attrs = {}
    object_id = 0
    
    for i, (name, ctx) in enumerate(contexts.items()):
        if name.startswith('merge'):
            attrs[name] = ctx
            merge_name = merge_name
        elif name.startswith('anti_alias'):
            attrs[name] = ctx
        else:
            attrs[f'object{object_id}'] = ctx
            object_id += 1
    result = type('RenderContexts', (), attrs)()
    # Add number of objects as attribute
    result.num_objects = object_id
    result.merge_name = merge_name

    return result

# Function to create framebuffer with textures directly bound
def create_framebuffer(ctx, resolution):
    # Create textures for direct rendering and sampling
    # PAPAYA: output from shaders goes directly to these textures
    tex_col = ctx.texture(resolution, components=4)
    tex_coord = ctx.texture(resolution, components=3, dtype='f4')
    tex_normal = ctx.texture(resolution, components=3, dtype='f4')
    tex_debug = ctx.texture(resolution, components=3, dtype='f4')
    tex_depth = ctx.texture(resolution, components=1, dtype='f4')
    tex_object_id = ctx.texture(resolution, components=1, dtype='i4')
    tex_distance_field = ctx.texture(resolution, components=1, dtype='f4')

    # Create framebuffer with textures directly as color attachments
    fbo = ctx.framebuffer(
        color_attachments=[tex_col, tex_coord, tex_normal, tex_debug, tex_depth, tex_object_id, tex_distance_field],
    )

    # Store reference to color textures for easier access in merge operations
    fbo.color_textures = [tex_col, tex_coord, tex_normal, tex_debug, tex_depth, tex_object_id, tex_distance_field]
    
    return fbo

@torch.profiler.record_function("OnlineDataset::sync_gpu")
def sync_gpu(ctx):
    """Synchronize GPU operations without reading buffers back to CPU.
    This ensures all pending rendering operations are completed before proceeding."""
    # Wait for all GPU commands to complete
    ctx.finish()
    
    # Alternative (less blocking) option if available:
    # ctx.flush()  # Submit commands but don't wait

def debug_show_image(data, title=None, save_path=None):
    """
    Display image data during debugging.
    Works with numpy arrays and PyTorch tensors.
    In headless environments, saves to file instead of displaying.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    # Check if we're in a headless environment
    is_headless = not hasattr(plt, 'get_backend') or plt.get_backend() == 'agg'

    if is_headless:
        matplotlib.use('Agg')  # Use non-interactive backend
    else:
        try:
            matplotlib.use('TkAgg')  # Try interactive backend
        except ImportError:
            try:
                matplotlib.use('Qt5Agg')  # Try alternative interactive backend
            except ImportError:
                matplotlib.use('Agg')  # Fall back to non-interactive
                is_headless = True

    # Convert PyTorch tensor to numpy if needed
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    # Handle different data types
    if data.dtype == np.float32 or data.dtype == np.float64:
        # Normalize float data to 0-1 range
        data = (data - data.min()) / (data.max() - data.min())

    plt.figure(figsize=(10, 10))
    if title:
        plt.title(title)
    plt.imshow(data)
    plt.axis('off')

    if is_headless:
        # Save to file in headless mode
        if save_path is None:
            save_path = f'debug_{title if title else "image"}.png'
        save_path = os.path.join('./debug/images', save_path)
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
        print(f"Saved debug image to: {save_path}")
    else:
        # Display in interactive mode
        plt.show()