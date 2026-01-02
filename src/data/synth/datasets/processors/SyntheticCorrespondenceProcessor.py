import torch
from torchvision.transforms.functional import normalize, center_crop
from src.data.synth.texture import texture_sampler
from src.data.synth.texture import multi_texturing as texturing
from src.flow import corr2flow, sample_kps, flow_by_coordinate_matching, convert_mapping_to_flow
from src.warp import warp
from src.data.synth.texture import video_sampler
from src.data.synth.texture import worley_sampler
import copy
from src.data.synth.texture import bg_color_sampler
import random
from src.data.synth.texture import pbr_sampler
from src.data.synth.texture import uv_texturing


class SyntheticCorrespondenceProcessor:
    def __init__(self,
                 seed=987654321, 
                 normalize='imagenet', 
                 use_worley_sampler=True, 
                 use_grf_sampler=False, 
                 use_pbr_sampler=False,
                 worley_sampler_dict=None,
                 bg_color_sampler_dict=None,
                 grf_sampler=None,
                 pbr_sampler_dict=None,
                 subsample_flow=None,
                 subsample_during_train=True,
                 subsample_during_val=False,
                 subsample_without_trainer=False,
                 downsample_for_cats=False,
                 cats_feat_size=32,
                 faiss_index_type="ivf",
                 faiss_nlist=32,
                 faiss_nprobe=2,
                 ):
        super().__init__()

        # Import faiss locally to avoid storing module reference (unpickleable)
        import faiss
        import faiss.contrib.torch_utils
        self.faiss_index_type = str(faiss_index_type).lower()
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        if self.faiss_index_type == "flat":
            self.index = faiss.IndexFlatL2(3)
        elif self.faiss_index_type == "ivf":
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, faiss_nlist)
            self.index.nprobe = faiss_nprobe
        else:
            raise ValueError(f"Unknown faiss_index_type: {faiss_index_type}")

        self.transferred = False

        self.subsample_flow = subsample_flow
        self.subsample_during_train = subsample_during_train
        self.subsample_during_val = subsample_during_val
        self.subsample_without_trainer = subsample_without_trainer
        
        # DEPRECATED: downsample_for_cats is no longer used when using CorrespondenceDataset.
        # Downsampling is now handled uniformly in CorrespondenceDataset.collate_fn via ensure_flow_and_kps().
        # Kept for backward compatibility with OnlineCorrespondenceDataset (old dataset class).
        self.downsample_for_cats = downsample_for_cats
        self.cats_feat_size = cats_feat_size
        if downsample_for_cats:
            import warnings
            warnings.warn(
                "downsample_for_cats is deprecated when using CorrespondenceDataset. "
                "Downsampling is now handled automatically in the collate pipeline. "
                "This parameter is only used with the legacy OnlineCorrespondenceDataset.",
                DeprecationWarning,
                stacklevel=2
            )

        self.use_worley_sampler = use_worley_sampler
        self.use_grf_sampler = use_grf_sampler
        self.use_pbr_sampler = use_pbr_sampler
        self.bg_color_sampler_dict = bg_color_sampler_dict
        self.worley_sampler_dict = worley_sampler_dict
        self.pbr_sampler_dict = pbr_sampler_dict
        self.bg_color_sampler_dict = bg_color_sampler_dict

        if self.use_pbr_sampler:
            self.pbr_sampler = pbr_sampler.PBRSampler(**pbr_sampler_dict)

        if use_worley_sampler:
            self.worley_sampler = worley_sampler.WorleyParamSampler(**worley_sampler_dict)
        
        if bg_color_sampler_dict is not None:
            self.bg_color_sampler = bg_color_sampler.BGColorSampler(**bg_color_sampler_dict)
        else:   
            self.bg_color_sampler = None

        self.rng = torch.Generator().manual_seed(seed)
        self.rng_gpu = None 

        if normalize == 'imagenet':
            from src import imagenet_stats
            # Store the tuple values, not the module reference
            normalize = imagenet_stats  # This is a tuple, not a module
        elif normalize == True:
            normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif normalize == False:
            normalize = None
        self.normalize = normalize

        # Device management - defaults to GPU for compatibility with texturing kernel
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dummy_background_sampler = texture_sampler.TextureSampler(**self.get_bg_sampler(None))

    @property
    def device(self):
        """Get the current device"""
        return self._device

    def to(self, device):
        """Move processor to specified device"""
        new_device = torch.device(device)
        # Only reset if device actually changed
        if self._device != new_device:
            self._device = new_device
            self.transferred = False
        return self

    def cuda(self, device=None):
        """Move processor to CUDA device"""
        if device is None:
            current_device = torch.cuda.current_device() if torch.cuda.is_available() else 0
            new_device = torch.device(f'cuda:{current_device}')
        else:
            # Handle both integer (device index) and string (full device name) inputs
            if isinstance(device, int):
                new_device = torch.device(f'cuda:{device}')
            else:
                new_device = torch.device(device)
        # Only reset if device actually changed
        if self._device != new_device:
            self._device = new_device
            self.transferred = False
        return self

    def cpu(self):
        """Move processor to CPU"""
        new_device = torch.device('cpu')
        # Only reset if device actually changed
        if self._device != new_device:
            self._device = new_device
            self.transferred = False
        return self

    def setup_for_gpu_once(self, device):
        if self.device.type == 'cuda':
            if not self.transferred:
                # Import faiss locally to avoid storing module reference (unpickleable)
                import faiss
                # setup GPU-specific things: faiss index, random generator
                # Handle case where device.index might not exist (e.g., torch.device('cuda'))
                if hasattr(device, 'index') and device.index is not None:
                    idx = device.index
                else:
                    # Use current CUDA device if no index specified
                    idx = torch.cuda.current_device()
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), idx, self.index)
                if self.faiss_index_type == "ivf":
                    self.index.nprobe = self.faiss_nprobe
                seed = torch.empty(1, dtype=torch.int64).random_(generator=self.rng).item()
                self.rng = torch.Generator(device=device).manual_seed(seed)
                self.transferred = True

    def render(self,  **kwargs):
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

    def sample_render_params(self, b, device):
        return dict(
            ambient = torch.empty(b, device=device).uniform_(0.3, 0.5, generator=self.rng_gpu),
            diffuse = torch.empty(b, device=device).uniform_(0.5, 0.7, generator=self.rng_gpu),
            specular = torch.empty(b, device=device).uniform_(0.2, 0.7, generator=self.rng_gpu),
            specular_exp = torch.randint(9, 12, (b,), device=device, generator=self.rng_gpu).float(),
        )
    
    def batch_to_device(self, batch, device):
        """Move batch to device, with multi-GPU support for PyTorch Lightning.
        
        Uses torch.cuda.current_device() to ensure compatibility with Lightning's
        DDP strategy where each process is assigned to a specific GPU.
        """
        # Use the device that Lightning assigns (via torch.cuda.current_device())
        # This ensures it works with multi-GPU DDP
        if device.type == 'cuda' and torch.cuda.is_available():
            # Ensure we're using the GPU that Lightning assigned to this process
            actual_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            actual_device = device
        
        batch = [{k: v.to(actual_device) for k, v in x.items()} for x in batch]
        return batch

    def process_scene(self, batch):
        device = batch[0]['geometry'].device
        batch_size = batch[0]['geometry'].shape[0]
        size = batch[0]['geometry'].shape[1:3]

        max_num_objects = batch[0]['max_num_objects'][0]

        self.setup_for_gpu_once(device)
        if self.use_pbr_sampler:
            # Sample textures with matching probability
            src_textures, trg_textures = self.pbr_sampler.sample_textures(
                num_objects=max_num_objects,
                matching_prob=0.7,  # 70% matching
                target_device=device
            )
            
            # Apply UV texturing to both src and trg
            post_batch = {}
            for i, k in enumerate(('src_img', 'trg_img')):
                if i == 0:
                    texture_dict = src_textures
                else:
                    texture_dict = trg_textures
                
                camera = batch[i]['camera']  # (B, 3)
                light = camera.add(torch.normal(0, 0.20, camera.shape, device=device, generator=self.rng)).mul_(2)
                # Apply UV texturing
                img = uv_texturing.apply_uv_texture_batched(
                    batch[i]['geometry'],      # (B, H, W, 3)
                    batch[i]['normals'],       # (B, H, W, 3) 
                    batch[i]['object_id'],    # (B, H, W)
                    texture_dict,              # Dict mapping object_id to texture
                    camera,        # (B, 3)
                    light,         # (B, 3)
                    projection='rounded_cube'
                )
                
                # Convert from (B, H, W, 3) to (B, 3, H, W) for consistency
                img = img.permute(0, 3, 1, 2)
                
                # Normalize images
                if self.normalize is not None:
                    img = normalize(img, *self.normalize)
                
                post_batch[k] = img
        
        else:     
            worley_params = self.worley_sampler.sample(batch_size, max_num_objects, self.rng)

            if self.use_grf_sampler:
                sampler_idx = torch.randint(0, len(self.object_texture_samplers), (1,)).item()
                foreground = self.object_texture_samplers[sampler_idx].sample(batch_size, (self.texture_resolution, self.texture_resolution, self.texture_resolution), self.rng)
                foreground = (foreground[0].unsqueeze(1), foreground[1].unsqueeze(1))

                # Sample remaining object textures
                for object_itr in range(batch[0]['max_num_objects'][0]):
                    sampler_idx = torch.randint(0, len(self.object_texture_samplers), (1,)).item()
                    foreground_tmp = self.object_texture_samplers[sampler_idx].sample(batch_size, (self.texture_resolution, self.texture_resolution, self.texture_resolution), self.rng)
                    foreground_tmp = (foreground_tmp[0].unsqueeze(1), foreground_tmp[1].unsqueeze(1))
                    foreground = (torch.cat([foreground[0], foreground_tmp[0]], dim=1), torch.cat([foreground[1], foreground_tmp[1]], dim=1))
            else:
                # Don't waste memmory generating grf we don't use.
                foreground = (torch.zeros(batch_size, 1, 1, 1, 1, 1).to(device), torch.zeros(batch_size, 1, 1, 1, 1, 1).to(device))


            background = self.dummy_background_sampler.sample(batch_size, size, self.rng)
            if self.bg_color_sampler is not None:
                bg_solid_color = self.bg_color_sampler.sample(batch_size, self.rng)
            else:
                bg_solid_color = (torch.zeros(batch_size, 3).to(device), torch.ones(batch_size, 3).to(device))


            post_batch = {}
            for i, k in enumerate(('src_img', 'trg_img')):
                img = self.render(**batch[i], foreground=foreground[i], background=background[i], worley_params=worley_params[i], bg_solid_color=bg_solid_color[i])

                # normalize images
                if self.normalize is not None:
                    img = normalize(img, *self.normalize)

                post_batch[k] = img

        # calculate ground-truth flow from geometry
        flow = flow_by_coordinate_matching(
            batch[0]['geometry'],
            batch[1]['geometry'],
            self.index,
            train_index=(self.faiss_index_type == "ivf"),
        )
        full_flow = flow.clone()
        # subsampling flow (for ablation experiments)
        has_trainer = hasattr(self, 'trainer') and self.trainer is not None
        if has_trainer:
            stage = self.trainer.state.stage
        else:
            stage = 'Non-Trainer'
        
        do_subsample = (self.subsample_during_train and stage == 'train') or (self.subsample_during_val and stage == 'val') or (self.subsample_without_trainer and stage == 'Non-Trainer')
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
            mask = mask.bernoulli(generator=self.rng)
            mask = mask.bool().unsqueeze(1).expand_as(flow)
            flow[mask] = torch.inf

        # NOTE: Downsampling is now handled by CorrespondenceDataset.collate_fn
        # via ensure_flow_and_kps(). No need to downsample here.
        post_batch['flow'] = flow
        post_batch['flow_full'] = full_flow
        return post_batch

    def downsample_flow(self, flow, feat_size):
        """
        Downsample flow to be compatible with CATS model.
        Converts flow from (B, 2, H, W) to (B, 2, feat_size, feat_size) format expected by CATS.
        Automatically calculates the appropriate downsampling factor for any input size.
        
        Args:
            flow: Input flow tensor of shape (B, 2, H, W)
            feat_size: Target feature size (e.g., 32 for 32x32 output)
        """
        if flow is None:
            return flow
            
        # Get flow dimensions
        B, C, H, W = flow.shape
        
        # Calculate the scale factor for both dimensions
        scale_factor_h = H / feat_size
        scale_factor_w = W / feat_size
        
        # Downsample the flow using masked average pooling
        # We need to handle the case where flow might contain inf values
        # Only average over valid pixels to avoid skewing the result
        
        # Create a mask for valid flow values
        valid_mask = torch.isfinite(flow).all(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Set invalid values to 0 for pooling (temporary, won't affect masked average)
        flow_clean = flow.clone()
        flow_clean[~valid_mask.expand_as(flow_clean)] = 0
        
        # Sum of valid flow values in each pooling region
        flow_sum = torch.nn.functional.adaptive_avg_pool2d(
            flow_clean, (feat_size, feat_size)
        ) * (scale_factor_h * scale_factor_w)  # Multiply back to get sum
        
        # Count of valid pixels in each pooling region
        valid_count = torch.nn.functional.adaptive_avg_pool2d(
            valid_mask.float(), (feat_size, feat_size)
        ) * (scale_factor_h * scale_factor_w)  # Multiply back to get count
        
        # Compute masked average: divide sum by count of valid pixels (not total pixels)
        # Avoid division by zero for regions with no valid pixels
        valid_count_safe = torch.clamp(valid_count, min=1e-8)
        flow_downsampled = flow_sum / valid_count_safe
        
        # Scale the flow values by the average scale factor to maintain proper magnitude
        # Use the average of both scale factors for consistent scaling
        avg_scale_factor = (scale_factor_h + scale_factor_w) / 2
        flow_downsampled = flow_downsampled / avg_scale_factor
        
        # Mark regions with no valid pixels as invalid (set to 0 or inf)
        # Create downsampled mask for invalid regions
        valid_mask_downsampled = valid_count > 0.5  # At least 0.5 valid pixels
        
        # Set invalid regions back to [0, 0]
        flow_downsampled[~valid_mask_downsampled.expand_as(flow_downsampled)] = 0
        # # Set invalid regions back to inf
        # flow_downsampled[~valid_mask_downsampled.expand_as(flow_downsampled)] = float('inf')
        
        
        return flow_downsampled

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





        
