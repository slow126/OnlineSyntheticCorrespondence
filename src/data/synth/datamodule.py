from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union
import warnings
import pdb

import torch
from torchvision.transforms.functional import normalize, center_crop

import os
print(f"current working directory: {os.getcwd()}")

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.joinpath('../../../').resolve()))

from src.data.copy_data import copy_data
from src.data.base import BaseDatamodule, make2
from src.data.synth.datasets import hdf5_dataset, transforms
# from src.data.synth.texture import texturing, texture_sampler
from src.data.synth.texture import texture_sampler
from src.data.synth.texture import multi_texturing as texturing
from src.flow import corr2flow, sample_kps, flow_by_coordinate_matching, convert_mapping_to_flow
from src.warp import warp
from src.data.synth.texture import video_sampler
from src.data.synth.texture import worley_sampler
import copy
from src.data.synth.texture import bg_color_sampler
import random
# faiss causes pytorch to raise a warning about TypedStorage deprecation, which doesn't seem
# to cause any problems (faiss == 1.7.4, torch == 2.0.1)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


class BaseComponentsDatamodule(BaseDatamodule):
    '''Datamodule for loading paired images. Handles rendering images based
    on geometry and normals.
    '''
    def __init__(
        self,
        root: str,
        num_samples: Optional[int] = None,
        num_samples_val: Optional[int] = None,
        img_res: int = 256,
        bg_color_sampler_dict: Optional[Dict] = None,
        background_sampler: Optional[Dict] = None,
        foreground_sampler: Optional[Dict] = None,
        use_video_sampler: bool = False,
        use_worley_sampler: bool = False,
        use_grf_sampler: bool = False,
        worley_sampler_dict: Optional[Dict] = None,
        background_flow: float = 0.0,
        random_flip: Optional[float] = None,
        random_swap: bool = True,
        subsample_flow: Optional[float] = None,
        normalize: Union[bool, str, tuple] = 'imagenet',
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
        seed: int = 987654321,
        texture_resolution: int = 64,
    ):
        self.num_samples = num_samples
        self.num_samples_val = num_samples_val
        self.img_res = img_res
        self.background_flow = background_flow
        self.random_flip = random_flip
        self.random_swap = random_swap
        self.subsample_flow = subsample_flow

        self.texture_resolution = texture_resolution
        self.bg_sampler = texture_sampler.TextureSampler(**self.get_bg_sampler(background_sampler))
        self.fg_sampler = texture_sampler.TextureSampler(**self.get_fg_sampler(foreground_sampler))

        self.use_worley_sampler = use_worley_sampler
        if use_worley_sampler:
            self.worley_sampler = worley_sampler.WorleyParamSampler(**worley_sampler_dict)
        else:
            self.worley_sampler = worley_sampler.WorleyParamSampler(**self.get_worley_sampler(worley_sampler_dict))

        self.object_texture_samplers = []
        self.use_grf_sampler = use_grf_sampler

        if use_video_sampler:
            self.videosampler = video_sampler.VideoSampler()
            self.object_texture_samplers = [self.fg_sampler, self.videosampler]
        else:
            self.videosampler = None
            self.object_texture_samplers = [self.fg_sampler]
        
        if bg_color_sampler_dict is not None:
            self.bg_color_sampler = bg_color_sampler.BGColorSampler(**bg_color_sampler_dict)
        else:
            self.bg_color_sampler = None

        

        import faiss
        import faiss.contrib.torch_utils
        self.faiss = faiss
        self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 32)

        self.transferred = False

        self.rng = torch.Generator().manual_seed(seed)
        self.rng_gpu = None

        if normalize == 'imagenet':
            from src import imagenet_stats
            normalize = imagenet_stats
        elif normalize == True:
            normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize = normalize

        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local=copy_data_local)

    @staticmethod
    def get_bg_sampler(opts):
        base = dict(
            texture_type = 'grf',
            texture_prob = 1.0,
            matching_prob = 0.0, 
            texture_scale = dict(alpha=(3.0, 5.0)),
        )
        if opts is not None:
            base.update(opts)

        return base
    
    @staticmethod
    def get_fg_sampler(opts):
        base = dict(
            texture_type = 'grf',
            texture_prob = 1.0,
            matching_prob = 1.0,
            texture_scale = dict(alpha=(3.0, 5.0)),
        )
        if opts is not None:
            base.update(opts)

        return base
    
    @staticmethod
    def get_worley_sampler(opts):
        base = dict(
            texture_type = 'worley',
            matching_prob = 1.0,
            texture_scale0 = dict(alpha=(3.0, 5.0)),
            texture_scale1 = dict(alpha=(3.0, 5.0)),
            mixing_param0 = dict(alpha=(0.0, 1.0)),
            mixing_param1 = dict(alpha=(0.0, 1.0)),
        )
        if opts is not None:
            base.update(opts)

        return base
    
    def sample_render_params(self, b, device):
        return dict(
            ambient = torch.empty(b, device=device).uniform_(0.3, 0.5, generator=self.rng_gpu),
            diffuse = torch.empty(b, device=device).uniform_(0.5, 0.7, generator=self.rng_gpu),
            specular = torch.empty(b, device=device).uniform_(0.2, 0.7, generator=self.rng_gpu),
            specular_exp = torch.randint(9, 12, (b,), device=device, generator=self.rng_gpu).float(),
        )

    def setup_for_gpu_once(self, device):
        if not self.transferred:
            # setup GPU-specific things: faiss index, random generator
            idx = device.index
            self.index = self.faiss.index_cpu_to_gpu(self.faiss.StandardGpuResources(), idx, self.index)
            seed = torch.empty(1, dtype=torch.int64).random_(generator=self.rng).item()
            self.rng_gpu = torch.Generator(device=device).manual_seed(seed)
            self.transferred = True

    def render(self, **kwargs):
        geo: torch.Tensor = kwargs['geometry']
        normal: torch.Tensor = kwargs['normals']
        cam: torch.Tensor = kwargs['camera']

        worley_params = kwargs.get('worley_params', None)
        
        # Get or generate shader background
        shader_background = kwargs.get('shader_background', None)

        light = kwargs.get('light', None)
        if light is None:
            light = cam.add(torch.normal(0, 0.20, cam.shape, device=cam.device, generator=self.rng_gpu)).mul_(2)

        foreground = kwargs['foreground']
        background = kwargs['background']

        c_min = geo.flatten(1).min(1)[0]
        c_max = geo.flatten(1).max(1)[0]
        kw = self.sample_render_params(geo.shape[0], geo.device)

        object_id = kwargs.get('object_id', None)
        kw.update(
            object_id=object_id,
            worley_params=worley_params,
            use_worley=self.use_worley_sampler,
            use_grf=self.use_grf_sampler,
            bg_solid_color=kwargs.get('bg_solid_color', None),
        )

        # render foreground


        fg_img = texturing.apply_texture(
            geo, normal, foreground, background, cam, light, c_min, c_max, **kw
        )

        # # If we have a shader background, combine it with the foreground
        # # Assuming black (0,0,0) represents transparent pixels in fg_img
        # if shader_background is not None:
        #     mask = (fg_img.sum(dim=1, keepdim=True) != 0)  # Create mask where foreground exists
        #     img = shader_background * (~mask) + fg_img * mask
        # else:
        #     img = fg_img
        img = fg_img

        return img
    

    def visualize_volumetric_textures(self, texture, name):
        import numpy as np
        slices = []
        max_z = texture.shape[1]
        for z in range(0, max_z, 8):  # Sample every 8th slice for visualization
            slice_img = texture[0, z, :, :, :]  # Take first batch item
            if isinstance(slice_img, torch.Tensor):
                slice_img = slice_img.cpu().numpy()
            slices.append(slice_img)
        grid = np.hstack(slices)

        # Save debug image
        from src.data.synth.datasets.online_dataset import debug_show_image
        debug_show_image(grid, f"volume_vis_{name}")
        return grid


    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int=0):
        device = batch[0]['geometry'].device
        batch_size = batch[0]['geometry'].shape[0]
        size = batch[0]['geometry'].shape[1:3]

        max_num_objects = batch[0]['max_num_objects'][0]

        self.setup_for_gpu_once(device)

        # PAPAYA: This is where texturing happens.
        # Sample first object texture

        # post_batch = {}
        # post_batch['src'] = torch.zeros(batch_size, 3, size[0], size[1]).to(device)
        # post_batch['trg'] = torch.zeros(batch_size, 3, size[0], size[1]).to(device)
        # post_batch['flow'] = torch.zeros(batch_size, 2, size[0], size[1]).to(device)
        # return post_batch


        # worley_texture = self.worley_sampler.sample((batch_size, max_num_objects)) * 100 + 1
        # worley_scale = self.worley_scale_sampler.sample((batch_size, max_num_objects)) * 100 + 1
        # worley_texture = worley_texture.to(device)
        # worley_scale = worley_scale.to(device)

        # worley_params = torch.cat([worley_texture.unsqueeze(-1), worley_scale.unsqueeze(-1)], dim=-1)
        
        # # Copy worley textures for src and tgt
        # worley_params_tuple = (worley_params, worley_params.clone())
        worley_params = self.worley_sampler.sample(batch_size, max_num_objects, self.rng_gpu)

        if self.use_grf_sampler:
            sampler_idx = torch.randint(0, len(self.object_texture_samplers), (1,)).item()
            foreground = self.object_texture_samplers[sampler_idx].sample(batch_size, (self.texture_resolution, self.texture_resolution, self.texture_resolution), self.rng_gpu)
            foreground = (foreground[0].unsqueeze(1), foreground[1].unsqueeze(1))

            # Sample remaining object textures
            for object_itr in range(batch[0]['max_num_objects'][0]):
                sampler_idx = torch.randint(0, len(self.object_texture_samplers), (1,)).item()
                foreground_tmp = self.object_texture_samplers[sampler_idx].sample(batch_size, (self.texture_resolution, self.texture_resolution, self.texture_resolution), self.rng_gpu)
                foreground_tmp = (foreground_tmp[0].unsqueeze(1), foreground_tmp[1].unsqueeze(1))
                foreground = (torch.cat([foreground[0], foreground_tmp[0]], dim=1), torch.cat([foreground[1], foreground_tmp[1]], dim=1))
        else:
            # Don't waste memmory generating grf we don't use.
            foreground = (torch.zeros(batch_size, 1, 1, 1, 1, 1).to(device), torch.zeros(batch_size, 1, 1, 1, 1, 1).to(device))


        background = self.bg_sampler.sample(batch_size, size, self.rng_gpu)
        if self.bg_color_sampler is not None:
            bg_solid_color = self.bg_color_sampler.sample(batch_size, self.rng_gpu)
        else:
            bg_solid_color = (torch.zeros(batch_size, 3).to(device), torch.ones(batch_size, 3).to(device))


        post_batch = {}
        for i, k in enumerate(('src', 'trg')):
            img = self.render(**batch[i], foreground=foreground[i], background=background[i], worley_params=worley_params[i], bg_solid_color=bg_solid_color[i])

            # normalize images
            img = normalize(img, *self.normalize)

            post_batch[k] = img

        # calculate ground-truth flow from geometry
        flow = flow_by_coordinate_matching(batch[0]['geometry'], batch[1]['geometry'], self.index)

        # subsampling flow (for ablation experiments)
        has_trainer = self.trainer is not None
        do_subsample = (has_trainer and self.trainer.state.stage == 'train') or (not has_trainer)
        if self.subsample_flow is not None and do_subsample:
            # randomly keep an approximate percentage of flow
            # also ensure there are at least a few points preserved for each pair
            valid = flow.isfinite().all(1)
            nz = valid.nonzero(as_tuple=True)
            counts = torch.bincount(nz[0])[:, None]
            offset = counts.cumsum(0).roll(1, 0)
            offset[0] = 0
            idx = flow.new_zeros(flow.shape[0], 10).uniform_(0, 1).mul(counts).floor().long().add_(offset).ravel()
            mask = valid.mul(1 - self.subsample_flow)
            mask[nz[0][idx], nz[1][idx], nz[2][idx]] = 0.0
            mask = mask.bernoulli(generator=self.rng_gpu)
            mask = mask.bool().unsqueeze(1).expand_as(flow)
            flow[mask] = torch.inf

        post_batch['flow'] = flow

        return post_batch



class OnlineComponentsDatamodule(BaseComponentsDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        val_root: Union[str, Path],
        num_samples: Optional[int] = None,
        num_samples_val: Optional[int] = None,
        img_res: int = 256,
        bg_color_sampler_dict: Optional[Dict] = None,
        background_sampler: Optional[Dict] = None,
        foreground_sampler: Optional[Dict] = None,
        julia_sampler: Optional[dict] = None,
        angle_sampler: Optional[dict] = None,
        scale_sampler: Optional[dict] = None,
        use_video_sampler: bool = False,
        use_worley_sampler: bool = False,
        use_grf_sampler: bool = False,
        worley_sampler_dict: Optional[Dict] = None,
        mandelbulb_sampler: Optional[Dict] = None,
        schedule: Optional[dict] = None,
        random_flip: Optional[float] = 0.0,
        random_swap: bool = True,
        crop: Literal['center', 'none', 'random'] = 'random',
        subsample_flow: Optional[float] = None,
        normalize: Union[bool, str, tuple] = 'imagenet',
        batch_size: int = 32,
        num_workers: int = 1,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
        seed: int = 987654321,
        shaders: Optional[Dict] = None,
        texture_resolution: int = 64,
        
    ):
        self.save_hyperparameters()
        self.val_root = val_root
        self.crop = crop
        self.julia_sampler = julia_sampler
        self.angle_sampler = angle_sampler
        self.scale_sampler = scale_sampler
        self.schedule = schedule
        self.shaders = shaders
        self.mandelbulb_sampler = mandelbulb_sampler
        self.bg_color_sampler_dict = bg_color_sampler_dict
        super().__init__(
            root,
            num_samples = num_samples,
            num_samples_val = num_samples_val,
            img_res = img_res,
            bg_color_sampler_dict = bg_color_sampler_dict,
            background_sampler = background_sampler,
            foreground_sampler = foreground_sampler,
            use_video_sampler = use_video_sampler,
            use_worley_sampler = use_worley_sampler,
            use_grf_sampler = use_grf_sampler,
            worley_sampler_dict = worley_sampler_dict,
            random_flip = random_flip,
            random_swap = random_swap,
            subsample_flow = subsample_flow,
            normalize = normalize,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = shuffle,
            copy_data_local=copy_data_local,
            seed = seed,
            texture_resolution = texture_resolution,
        )

    def post_init(self):
        from src.data.synth.datasets import online_dataset
        from src.data.synth.schedule import Scheduler
        self.dataset_class = online_dataset.OnlineGenerationDataset
        self.scheduler = Scheduler
    
    def setup(self, stage='fit'):
        self.val_julia_sampler = copy.deepcopy(self.julia_sampler)
        self.val_angle_sampler = copy.deepcopy(self.angle_sampler)
        self.val_scale_sampler = copy.deepcopy(self.scale_sampler)

        single_process = self.num_workers == 0
        if stage == 'fit':
            self.train_data = self.dataset_class(
                shader_code_path = self.root,
                num_samples = self.num_samples,
                size = self.img_res,
                crop = self.crop,
                random_flip = self.random_flip,
                random_swap = self.random_swap,
                julia_sampler = self.julia_sampler,
                angle_sampler = self.angle_sampler,
                scale_sampler = self.scale_sampler,
                mandelbulb_sampler = self.mandelbulb_sampler,
                single_process = single_process,
                seed = torch.empty(1, dtype=torch.int64).random_(generator=self.rng).item(),
                shaders = self.shaders,
            )
            if self.schedule is not None:
                self.scheduler = self.scheduler(
                    self.schedule,
                    complexity_sampler = self.train_data.julia_sampler,
                    angle_sampler = self.train_data.angle_sampler,
                    scale_sampler = self.train_data.scale_sampler,
                    texture_sampler = self.fg_sampler,
                )

        # TODO: FIX VAL DATA. Right now its the same as train data and is generated on the fly. This needs to be fixed.
        self.val_data = self.dataset_class(
                shader_code_path = self.root,
                num_samples = self.num_samples_val,
                size = self.img_res,
                crop = self.crop,
                random_flip = self.random_flip,
                random_swap = self.random_swap,
                julia_sampler = self.val_julia_sampler,
                angle_sampler = self.val_angle_sampler,
                scale_sampler = self.val_scale_sampler,
                mandelbulb_sampler = self.mandelbulb_sampler,
                single_process = single_process,
                seed = torch.empty(1, dtype=torch.int64).random_(generator=self.rng).item(),
                shaders = self.shaders,
            )

        # self.val_data = hdf5_dataset.Hdf5CorrespondenceComponentsDataset(
        #     root = self.val_root, 
        #     overlap_sampling = 'uniform',
        #     overlap_params = (0.3, 1.0),
        #     num_samples = self.num_samples_val,
        #     size = self.img_res,
        #     crop = 'center',
        #     seed = torch.empty(1, dtype=torch.int64).random_(generator=self.rng).item()
        # )

        

    def on_train_epoch_start(self, trainer):
        if self.schedule is not None:
            epoch = trainer.current_epoch
            total = trainer.max_epochs - 1
            self.scheduler.step(epoch, total)

    def get_worker_init(self):
        def _worker_init(worker_id):
            info = torch.utils.data.get_worker_info()
            if hasattr(info.dataset, 'setup'):
                info.dataset.setup()
        
        return _worker_init


if __name__ == '__main__':
    from pathlib import Path
    import os
    import dotenv
    import torchvision
    from torch.utils.data import DataLoader
    path = Path(__file__).parent.joinpath('../../../').resolve()
    dotenv.load_dotenv(path.joinpath('.env'))


    root = Path('/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c')

    val_root = Path(os.environ['DATADIR']).joinpath('synthetic/')
    shaders = {
        'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/terrain_sdf.glsl',
        'program1': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
        'program2': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
        # 'program1': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/mandelbulb.glsl',
        # 'program2': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/sphere.glsl',
        'program3': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/mandelbulb.glsl',
        'program4': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/deformed_sphere.glsl',
        'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
    }

    worley_sampler_dict = {
        'matching_prob': 0.50,
        'terrain_matching_prob': 0.50,
        'texture_scale0': {
            'alpha': {
                'offset': 0.5,
                'scale': 7.5,
            },
        },
        'texture_scale1': {
            'alpha': {
                'offset': 20.0,
                'scale': 80.0,
            },
        },
        'mixing_param0': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.0,
            },
        },
        'mixing_param1': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.0,
            },
        },
        'terrain_texture_scale0': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.5,
            },
        },
        'terrain_texture_scale1': {
            'alpha': {
                'offset': 10.0,
                'scale': 40.0,
            },
        },
        'terrain_mixing_param0': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.0,
            },
        },
        'terrain_mixing_param1': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.0,
            },
        },
    }

    angle_sampler_dict = {
        'x_components': {
            'distribution': 'uniform',
            'loc': -0.16667,
            'scale': 0.33333,
        },
        'y_components': {
            'distribution': 'uniform',
            'loc': -0.16667,
            'scale': 0.33333,
        },
    }

    mandelbulb_sampler_dict = {
        'power_range': {
            'min': 1.5,
            'offset': 2.5,
        },
    }

    bg_color_sampler_dict = {
        'matching_prob': 0.5,
        'hue_offset': 0.0,
        'hue_scale': 1.0,
        'saturation_offset': 0.0,
        'saturation_scale': 0.5,
        'value_offset': 0.25,
        'value_scale': 0.75,
    }

    # dm = ComponentsDatamodule(root, batch_size=4, num_workers=0, shuffle=False)
    dm = OnlineComponentsDatamodule(root, val_root=val_root, batch_size=32, 
                                    num_workers=0, shuffle=False, shaders=shaders, 
                                    use_worley_sampler=True, worley_sampler_dict=worley_sampler_dict,
                                    mandelbulb_sampler=mandelbulb_sampler_dict,
                                    bg_color_sampler_dict=bg_color_sampler_dict, use_grf_sampler=False, num_samples=1000)
    dm.post_init()
    dm.setup('fit')

    profiling = False
    length = len(dm.train_data)

    if profiling: 
        import cProfile
        import pstats
        import io
        # Profile multiple iterations to get an average cost
        num_iterations = 10
        pr = cProfile.Profile()
        pr.enable()
        
        for i in range(num_iterations):
            b = dm.train_data.__getitem__(i % length)
            b = [{k: v.unsqueeze(0).cuda() for k, v in x.items()} for x in b]
        
            b = dm.on_after_batch_transfer(b, 0)
        
        pr.disable()
        
        # Print profiling results
        s = io.StringIO()
        sortby = 'cumtime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        
        # Calculate and print average times
        print(f"Average times over {num_iterations} iterations:")
        print(f"Total time: {ps.total_tt / num_iterations:.4f} seconds per iteration")
        print(s.getvalue())
        # Save profiling stats to a file for later analysis
        profile_path = Path('datamodule_profile.txt')
        with profile_path.open('w') as f:
            ps = pstats.Stats(pr, stream=f)
            ps.sort_stats(sortby)
            ps.print_stats()
        
        print(f"Profiling stats saved to {profile_path}")
        profile_binary_path = Path('datamodule_profile.prof')
        pr.dump_stats(profile_binary_path)
        
        exit()        

    
    # b = [{k: v.cuda() for k, v in x.items()} for x in next(iter(dm.train_dataloader()))]
    single_image = True

    # Dictionary mapping shader filenames to human-readable labels
    shader_to_label = {
        'terrain_sdf.glsl': 'FBM',
        'quatJulia.c': 'Julia',
        'mandelbulb.glsl': 'Mandelbulb',
        'deformed_sphere.glsl': 'Radial Disp.',
    }
    
    # Helper function to get label from shader path
    def get_shader_label(shader_path):
        filename = os.path.basename(shader_path)
        return shader_to_label.get(filename, filename)

    def get_example_image(shader_list):
        src_images = []
        trg_images = []
        
        for shader_set in shader_list:
            dm = OnlineComponentsDatamodule(root, val_root=val_root, batch_size=32, 
                                num_workers=0, shuffle=False, shaders=shader_set, 
                                use_worley_sampler=True, worley_sampler_dict=worley_sampler_dict,
                                mandelbulb_sampler=mandelbulb_sampler_dict, 
                                bg_color_sampler_dict=bg_color_sampler_dict, use_grf_sampler=False, 
                                seed=random.randint(0, 10000))
            dm.post_init()
            dm.setup('fit')
            b = dm.train_data[0]
            b = [{k: v.unsqueeze(0).cuda() for k, v in x.items()} for x in b]
            b = dm.on_after_batch_transfer(b, 0)
            src = b['src'].squeeze(0).cpu()
            trg = b['trg'].squeeze(0).cpu()

            # Normalize the images to be between 0 and 1
            src = (src - src.min()) / (src.max() - src.min())
            trg = (trg - trg.min()) / (trg.max() - trg.min())

            src_images.append(src)
            trg_images.append(trg)
        return src_images, trg_images


    if single_image:
        b = dm.train_data[0]
        b = [{k: v.unsqueeze(0).cuda() for k, v in x.items()} for x in b]
        
        b = dm.on_after_batch_transfer(b, 0)

        from src.data.synth.datasets.online_dataset import debug_show_image
        debug_show_image(b['src'].squeeze(0).permute(1,2,0), 'src')
        debug_show_image(b['trg'].squeeze(0).permute(1,2,0), 'trg')

        shader_list = [
        {
            'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/terrain_sdf.glsl',
            'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
        }, 
        {
            'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
            'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
        }, 
        {
            'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/mandelbulb.glsl',
            'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
        }, 
        {
            'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/deformed_sphere.glsl',
            'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
        },
        {
            'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/terrain_sdf.glsl',
            'program1': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
            'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
        },
        {
            'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/terrain_sdf.glsl',
            'program1': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
            'program2': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/mandelbulb.glsl',
            'program3': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/deformed_sphere.glsl',
            'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
        },
        ]
        src_images, trg_images = get_example_image(shader_list)

        from torchvision.utils import save_image
        grid = torchvision.utils.make_grid(src_images, nrow=8, padding=2, pad_value=0.5)
        save_image(grid, 'debug/images/src_images_grid.png')

        grid_trg = torchvision.utils.make_grid(trg_images, nrow=8, padding=2, pad_value=0.5)
        save_image(grid_trg, 'debug/images/trg_images_grid.png')

        shader_labels = []
        for shader_config in shader_list:
            labels = []
            for key in ['program0', 'program1', 'program2']: # Check for up to program2
                if key in shader_config:
                    path = shader_config[key]
                    label = get_shader_label(path)
                    labels.append(label)
            shader_labels.append('+'.join(labels))

        # Create a figure with matplotlib to display src and trg images with labels
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
        from pathlib import Path
        
        # Create debug directory if it doesn't exist
        debug_dir = Path('debug/images/matplotlib')
        debug_dir.mkdir(exist_ok=True, parents=True)
        
        # Number of shader examples to display
        num_shaders = len(shader_list)
        
        # Create a figure with 2 rows (src and trg) and num_shaders columns
        fig = plt.figure(figsize=(num_shaders * 3, 8))
        gs = gridspec.GridSpec(2, num_shaders, figure=fig)
        
        # Add row labels on the left
        fig.text(0.02, 0.75, 'Source', rotation=90, fontsize=22, va='center')
        fig.text(0.02, 0.25, 'Target', rotation=90, fontsize=22, va='center')
        
        # Plot each image with its shader label
        for i in range(num_shaders):
            # Convert PyTorch tensors to numpy arrays for matplotlib
            src_img = src_images[i].permute(1, 2, 0).cpu().numpy()
            trg_img = trg_images[i].permute(1, 2, 0).cpu().numpy()
            
            # Source image
            ax_src = fig.add_subplot(gs[0, i])
            ax_src.imshow(src_img)
            ax_src.set_title(shader_labels[i], fontsize=20)
            ax_src.axis('off')
            
            # Target image
            ax_trg = fig.add_subplot(gs[1, i])
            ax_trg.imshow(trg_img)
            ax_trg.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.05)  # Make room for row labels
        
        # Save the figure
        plt.savefig(debug_dir / 'shader_comparison_grid.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Matplotlib grid saved to {debug_dir / 'shader_comparison_grid.png'}")
        
    else:

        # Create a list to store 'src' images
        from torchvision.utils import save_image
        src_images = []
        trg_images = []
        # Sample 40 images from the dataset
        # sample_indices = torch.randint(0, len(dm.train_data), (40,))
        for i in range(40):
            b = dm.train_data[i]
            b = [{k: v.unsqueeze(0).cuda() for k, v in x.items()} for x in b]
            b = dm.on_after_batch_transfer(b, 0)
            src_images.append(b['src'].squeeze(0).cpu())
            trg_images.append(b['trg'].squeeze(0).cpu())
        # print(len(src_images))
        # import numpy as np
        # tmp = np.stack(src_images, axis=0)
        # print(tmp.shape)
        # np.save('src_images.npy', tmp)

        # check = np.load('src_images.npy')
        # print(check.shape)

        # Create a grid of 8x5 'src' images
        grid = torchvision.utils.make_grid(src_images, nrow=8, padding=2, pad_value=0.5)
        max_brightness = max(img.max().item() for img in src_images)
        min_brightness = min(img.min().item() for img in src_images)

        # Normalize each individual image uniformly so the new min of each individual image is 0 and the new max is 1
        # for i in range(len(src_images)):
        #     img = src_images[i]
        #     img_min = img.min()
        #     img_max = img.max()
        #     src_images[i] = (img - img_min) / (img_max - img_min)

        # # Create a grid of 8x5 'src' images
        # grid = torchvision.utils.make_grid(src_images, nrow=8, padding=2, pad_value=0.5)

        # Normalize all images uniformly so the new min across all images is 0 and the new max is 1
        # grid = (grid - min_brightness) / (max_brightness - min_brightness)

        # Scale pixel intensities with a sigmoid to be between 0 and 1
        # grid = torch.relu(grid)

        # Save the grid as a single image
        save_image(grid, 'debug/images/src_images_grid.png')

        grid_trg = torchvision.utils.make_grid(trg_images, nrow=8, padding=2, pad_value=0.5)
        max_brightness = max(img.max().item() for img in trg_images)
        min_brightness = min(img.min().item() for img in trg_images)
        # grid_trg = (grid_trg - min_brightness) / (max_brightness - min_brightness)
        save_image(grid_trg, 'debug/images/trg_images_grid.png')

    
