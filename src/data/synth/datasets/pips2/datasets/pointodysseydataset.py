from numpy import random
import torch
import numpy as np
import os
from typing import Optional
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.misc
import utils.improc
import glob
import cv2
import albumentations as A
from functools import partial
import sys

def augment_video(augmenter, **kwargs):
    assert isinstance(augmenter, A.ReplayCompose)
    keys = kwargs.keys()
    for i in range(len(next(iter(kwargs.values())))):
        data = augmenter(**{
            key: kwargs[key][i] if key not in ['bboxes', 'keypoints'] else [kwargs[key][i]] for key in keys
        })
        if i == 0:
            augmenter = partial(A.ReplayCompose.replay, data['replay'])
        for key in keys:
            if key == 'bboxes':
                kwargs[key][i] = np.array(data[key]).reshape(4)
            elif key == 'keypoints':
                kwargs[key][i] = np.array(data[key]).reshape(2)
            else:
                kwargs[key][i] = data[key]
                
class PointOdysseyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='/orion/group/point_odyssey_v1.2',
                 dset='train',
                 use_augs=False,
                 S=8,
                 N=32,
                 strides=[1,2,4],
                 clip_step=2,
                 resize_size=(368+64, 496+64),
                 crop_size=(368, 496),
                 req_full=False,
                 quick=False,
                 max_sequences=None,
                 verbose=False,
                 val_sequence_fraction: Optional[float] = None,
                 use_all_valid=False,
                 disable_motion_filter=False,
    ):
        print('loading pointodyssey dataset...')

        self.S = S
        self.N = N
        self.req_full = req_full
        self.verbose = verbose
        self.use_all_valid = use_all_valid
        self.disable_motion_filter = disable_motion_filter

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.mask_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.full_idxs = []

        self.subdirs = []
        self.sequences = []
        
        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*")):
                if os.path.isdir(seq):
                    seq_name = seq.split('/')[-1]
                    self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        total_sequences = len(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (total_sequences, dataset_location, dset))
        
        print('loading trajectories...')
        
        # Deterministic sequence sampling
        if max_sequences is not None and max_sequences > 0:
            # Sample sequences evenly spaced for better coverage
            if max_sequences >= total_sequences:
                # Use all sequences if max_sequences >= total
                print(f'Using all {total_sequences} sequences')
            else:
                # Sample evenly spaced sequences (deterministic)
                indices = np.linspace(0, total_sequences - 1, max_sequences, dtype=int)
                self.sequences = [self.sequences[i] for i in indices]
                print(f'Using {len(self.sequences)} sequences (sampled from {total_sequences} total)')
        elif quick:
            # Backward compatibility: quick mode uses first sequence
            self.sequences = self.sequences[:1]
            print('Quick mode: using first sequence only')

        # For validation, use only first K% of frames in each sequence to speed up loading
        # Default is 1.0 (full dataset) for val/test, None for train
        if val_sequence_fraction is None:
            val_sequence_fraction = 1.0 if self.dset in ['val', 'test'] else None
        
        for seq in self.sequences:
            
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'rgbs')

            annotations_path = os.path.join(seq, 'anno.npz')
            if os.path.isfile(annotations_path):
                num_frames = len(os.listdir(rgb_path))
                
                # Limit to first 20% of sequence for validation
                if val_sequence_fraction is not None and val_sequence_fraction < 1.0:
                    max_frame_idx = int(num_frames * val_sequence_fraction)
                    # Ensure we have enough frames for at least one clip
                    max_stride = max(strides) if strides else 1
                    max_frame_idx = max(max_frame_idx, self.S * max_stride)
                    if self.verbose:
                        print(f'  Limiting to first {val_sequence_fraction*100:.0f}% of frames: {max_frame_idx}/{num_frames}')
                else:
                    max_frame_idx = num_frames
                
                for stride in strides:
                    # Limit the range to first K% of frames for validation
                    max_start_idx = max_frame_idx - self.S*stride + 1
                    total_possible = num_frames - self.S*stride + 1
                    for ii in range(0, min(max_start_idx, total_possible), clip_step):
                        full_idx = ii + np.arange(self.S)*stride
                        self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                        self.mask_paths.append([os.path.join(seq, 'masks', 'mask_%05d.png' % idx) for idx in full_idx])
                        self.annotation_paths.append(os.path.join(seq, 'anno.npz'))
                        self.full_idxs.append(full_idx)
                    if self.verbose:
                        sys.stdout.write('.')
                        sys.stdout.flush()
            else:
                print('missing annotations:', annotations_path)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

        self.spatial_aug_prob = 0.7
        self.reverse_prob = 0.5

        # occlusion augmentation
        self.eraser_prob = 0.2
        self.eraser_bounds = [20, 300]
        self.eraser_max = 10

        # spatial augmentations
        self.pad_bounds = [0, 64]
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.resize_lim = [0.25, 1.5] # sample resizes from here
        self.resize_delta = 0.1
        self.max_crop_offset = 20
        
        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        self.color_augmenter = A.ReplayCompose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
            ], p=0.2),
            A.RGBShift(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        ], p=0.8)
        
        # Cache for annotation files (per-worker, since each worker has its own dataset instance)
        # This avoids reloading the same annotation file when processing multiple clips from the same sequence
        self._cached_annotation_path = None
        self._cached_annotations = None
        
        # Optimization: Skip edge computation if not needed (for correspondence tasks)
        # Edge computation is only needed for filtering trajectories on segmentation boundaries
        # If disable_motion_filter is True, we're likely doing correspondence and can skip edges
        self.skip_edge_computation = disable_motion_filter

    def getitem_helper(self, index):
        """
        Get a sample from the dataset with optimizations for speed:
        1. Annotation caching: Reuses memory-mapped annotation files across clips from same sequence
        2. Optimized array operations: Uses vectorized numpy operations instead of loops
        3. Edge computation skipping: Skips expensive Canny edge detection when disable_motion_filter=True
        4. Fast image loading: Uses cv2.imread instead of PIL for faster JPEG/PNG loading
        5. Optimized motion filtering: Uses np.diff and vectorized norm computations
        """
        sample = None
        gotit = False

        try:
            rgb_paths = self.rgb_paths[index]
            mask_paths = self.mask_paths[index]
            full_idx = self.full_idxs[index]
            annotations_path = self.annotation_paths[index]
            
            # Use cached annotation if it's the same file
            if annotations_path != self._cached_annotation_path:
                # Load new annotation file
                try:
                    # Try memory mapping first (fastest for large files)
                    annotations = np.load(annotations_path, allow_pickle=True, mmap_mode='r')
                    # Cache the mmap object for reuse (mmap allows concurrent reads)
                    self._cached_annotations = annotations
                    self._cached_annotation_path = annotations_path
                except (OSError, ValueError, TypeError):
                    # Fallback if mmap fails (e.g., compressed .npz files)
                    annotations = np.load(annotations_path, allow_pickle=True)
                    self._cached_annotations = annotations
                    self._cached_annotation_path = annotations_path
            else:
                # Reuse cached annotation
                annotations = self._cached_annotations
            
            # Optimized: Extract slices directly from mmap without intermediate array conversion
            # This avoids unnecessary memory copies when using mmap_mode='r'
            if isinstance(annotations, np.lib.npyio.NpzFile):
                # For .npz files, we need to access the arrays
                trajs_2d = annotations['trajs_2d']
                visibs_arr = annotations['visibs']
                valids_arr = annotations['valids']
                
                # Extract slices - use direct indexing for better performance
                if hasattr(trajs_2d, '__getitem__'):
                    trajs = np.asarray(trajs_2d[full_idx], dtype=np.float32)
                    visibs = np.asarray(visibs_arr[full_idx], dtype=np.float32)
                    valids = np.asarray(valids_arr[full_idx], dtype=np.float32)
                else:
                    # Fallback for non-array objects
                    trajs = np.array(trajs_2d[full_idx]).astype(np.float32)
                    visibs = np.array(visibs_arr[full_idx]).astype(np.float32)
                    valids = np.array(valids_arr[full_idx]).astype(np.float32)
            else:
                # Fallback for other formats
                trajs = np.array(annotations['trajs_2d'][full_idx]).astype(np.float32)
                visibs = np.array(annotations['visibs'][full_idx]).astype(np.float32)
                valids = np.array(annotations['valids'][full_idx]).astype(np.float32)
            # Don't delete annotations here - we're caching it for reuse

            # some data is valid in 3d but invalid in 2d
            # here we will filter to the data which is valid in 2d
            # Optimized: Use vectorized operations instead of np.where
            valids_xy = np.ones_like(trajs, dtype=np.float32)
            # Check for inf/nan in one pass
            invalid_mask = ~np.isfinite(trajs)
            trajs[invalid_mask] = 0
            valids_xy[invalid_mask] = 0
            # Check if both x and y are valid (sum along last axis should be 2)
            inv_idx = np.sum(valids_xy, axis=2) < 2  # S,N boolean mask
            visibs[inv_idx] = 0
            valids[inv_idx] = 0
            
            # ensure that the point is good in frame0
            vis_and_val = valids * visibs
            vis0 = vis_and_val[0] > 0
            trajs = trajs[:,vis0]
            visibs = visibs[:,vis0]
            valids = valids[:,vis0]

            S,N,D = trajs.shape
            assert(D==2)
            assert(S==self.S)

            if self.req_full:
                min_N = self.N//2
            else:
                min_N = self.N

            if N < min_N:
                if self.verbose:
                    print('returning before cropping: N=%d; need at least N=%d' % (N, min_N))
                return None, False

            # Optimized: Use cv2.imread for faster loading (BGR->RGB conversion needed)
            rgbs = []
            for rgb_path in rgb_paths:
                # cv2.imread is typically faster than PIL for JPEG files
                img_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    # Fallback to PIL if cv2 fails
                    with Image.open(rgb_path) as im:
                        img_bgr = np.array(im)[:, :, :3]
                    rgb = img_bgr
                else:
                    # Convert BGR to RGB
                    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                rgbs.append(rgb)  # H,W,3

            H,W,C = rgbs[0].shape
            
            masks = []
            for mask_path in mask_paths:
                # cv2.imread is faster for PNG files too
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    # Fallback to PIL if cv2 fails
                    with Image.open(mask_path) as im:
                        mask = np.array(im)
                # Only do median blur if needed (optimization: check first)
                if np.sum(mask==0) > 128:
                    # fill holes caused by fog/smoke
                    mask_filled = cv2.medianBlur(mask, 7)
                    mask[mask==0] = mask_filled[mask==0]
                masks.append(mask)  # H,W

            # discard pixels that are OOB on frame0
            # Optimized: Vectorize across all frames at once
            oob_mask = ((trajs[:,:,0] < 0) | (trajs[:,:,0] > W-1) | 
                       (trajs[:,:,1] < 0) | (trajs[:,:,1] > H-1))
            visibs[oob_mask] = 0
            vis0 = visibs[0] > 0
            trajs = trajs[:,vis0]
            visibs = visibs[:,vis0]
            valids = valids[:,vis0]

            # compute edge map (skip if not needed for correspondence tasks)
            edges = []
            if not self.skip_edge_computation:
                kernel = np.ones((3,3), np.uint8)
                dilate_iters = max(int(H/self.resize_size[0]),1)
                for si in range(S):
                    edge = cv2.Canny(masks[si], 1, 1)
                    # block apparent edges from fog/smoke
                    keep = 1 - cv2.dilate((masks[si]==0).astype(np.uint8), kernel, iterations=1) 
                    edge = edge * keep
                    edge = cv2.dilate(edge, kernel, iterations=dilate_iters)
                    edges.append(edge)
                
                # discard trajs that begin exactly on segmentation boundaries
                # since their labels are ambiguous
                x0, y0 = trajs[0,:,0].astype(np.int32), trajs[0,:,1].astype(np.int32)
                # Clamp indices to valid range to avoid out-of-bounds access
                x0_clamped = np.clip(x0, 0, edges[0].shape[1] - 1)
                y0_clamped = np.clip(y0, 0, edges[0].shape[0] - 1)
                on_edge = edges[0][y0_clamped, x0_clamped] > 0
                trajs = trajs[:,~on_edge]
                visibs = visibs[:,~on_edge]
                valids = valids[:,~on_edge]
            else:
                # Create dummy edges for compatibility (not used when skip_edge_computation=True)
                edges = [np.zeros((H, W), dtype=np.uint8) for _ in range(S)]
            
            N = trajs.shape[1]


            if N < min_N:
                if self.verbose:
                    print('returning after edge check: N=%d; need at least N=%d' % (N, min_N))
                return None, False
                    
            if self.use_augs:
                rgbs = np.stack(rgbs, 0)
                augment_video(self.color_augmenter, image=rgbs)
                rgbs = [rgb.astype(np.float32) for rgb in rgbs]

                if np.random.rand() < self.eraser_prob:
                    for i in range(1, S):
                        if np.random.rand() < self.eraser_prob:
                            for _ in range(np.random.randint(1, self.eraser_max+1)): # number of times to occlude
                                xc = np.random.randint(0, W)
                                yc = np.random.randint(0, H)
                                dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                                dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                                x0 = np.clip(xc - dx/2, 0, W-1).round().astype(np.int32)
                                x1 = np.clip(xc + dx/2, 0, W-1).round().astype(np.int32)
                                y0 = np.clip(yc - dy/2, 0, H-1).round().astype(np.int32)
                                y1 = np.clip(yc + dy/2, 0, H-1).round().astype(np.int32)
                                mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1,3), axis=0)
                                rgbs[i][y0:y1, x0:x1, :] = mean_color
                                occ_inds = np.logical_and(np.logical_and(trajs[i,:,0] >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
                                visibs[i, occ_inds] = 0
                    rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

                if np.random.rand() < self.reverse_prob:
                    rgbs = np.stack(rgbs, 0)
                    rgbs = np.flip(rgbs, axis=0)
                    rgbs = [rgb for rgb in rgbs]
                    
                    masks = np.stack(masks, 0)
                    masks = np.flip(masks, axis=0)
                    masks = [mask for mask in masks]
                    
                    edges = np.stack(edges, 0)
                    edges = np.flip(edges, axis=0)
                    edges = [mask for mask in edges]
                    
                    trajs = np.flip(trajs, axis=0)
                    visibs = np.flip(visibs, axis=0)

            if self.use_augs and (np.random.rand() < self.spatial_aug_prob):
                rgbs, masks, edges, trajs = self.add_spatial_augs(rgbs, masks, edges, trajs, visibs)
            else:
                # either crop or resize
                if np.random.rand() < 0.5:
                    rgbs, masks, edges, trajs = self.just_crop(rgbs, masks, edges, trajs)
                else:
                    rgbs, masks, edges, trajs = self.just_resize(rgbs, masks, edges, trajs)

            H,W,C = rgbs[0].shape
            assert(C==3)
            
            # update visibility annotations
            # Optimized: Vectorize across all frames at once
            # avoid 1px edge
            oob_inds = ((trajs[:,:,0] < 1) | (trajs[:,:,0] > W-2) |
                       (trajs[:,:,1] < 1) | (trajs[:,:,1] > H-2))
            visibs[oob_inds] = 0

            # when a point moves far oob, don't supervise with it
            very_oob_inds = ((trajs[:,:,0] < -64) | (trajs[:,:,0] > W+64) |
                           (trajs[:,:,1] < -64) | (trajs[:,:,1] > H+64))
            valids[very_oob_inds] = 0

            # ensure that the point is good in frame0
            vis_and_val = valids * visibs
            vis0 = vis_and_val[0] > 0
            trajs = trajs[:,vis0]
            visibs = visibs[:,vis0]
            valids = valids[:,vis0]

            # ensure that the point is good in frame1
            vis_and_val = valids * visibs
            vis1 = vis_and_val[1] > 0
            trajs = trajs[:,vis1]
            visibs = visibs[:,vis1]
            valids = valids[:,vis1]

            # ensure that the point is good in at least sqrt(S) frames
            vis_and_val = valids * visibs
            val_ok = np.sum(vis_and_val, axis=0) >= max(np.sqrt(S),2)
            trajs = trajs[:,val_ok]
            visibs = visibs[:,val_ok]
            valids = valids[:,val_ok]

            # Early return if no trajectories left after filtering
            N = trajs.shape[1]
            if N == 0:
                if self.verbose:
                    print('No trajectories left after val_ok filtering')
                return None, False

            # some of the data is a bit crazy,
            # so we will filter down based on motion
            # Skip motion filtering if disabled (useful for correspondence tasks)
            if self.disable_motion_filter:
                # Skip motion filtering entirely
                mot_ok = np.ones(N, dtype=bool)
            elif self.S > 2:
                # Full motion filtering for longer sequences
                # Optimized: Pre-compute normalization factor once
                norm_factor = max(H, W)
                trajs_norm = trajs / norm_factor
                
                # Compute velocity, acceleration, jerk in one pass where possible
                vel = np.diff(trajs_norm, axis=0)  # (S-1, N, 2) - equivalent to trajs_norm[1:] - trajs_norm[:-1]
                accel = np.diff(vel, axis=0)  # (S-2, N, 2)
                jerk = np.diff(accel, axis=0)  # (S-3, N, 2)
                
                # Optimized: Compute norms more efficiently
                # Use np.linalg.norm with axis=-1 to get per-trajectory norms, then max across time
                vel_norms = np.linalg.norm(vel, axis=-1)  # (S-1, N)
                accel_norms = np.linalg.norm(accel, axis=-1)  # (S-2, N)
                jerk_norms = np.linalg.norm(jerk, axis=-1)  # (S-3, N)
                
                # Filter based on motion smoothness
                vel_ok = np.max(vel_norms, axis=0) < 0.4  # N
                accel_ok = np.max(accel_norms, axis=0) < 0.3  # N
                jerk_ok = np.max(jerk_norms, axis=0) < 0.1  # N
                mot_ok = vel_ok & accel_ok & jerk_ok
            else:
                # For S <= 2, only do basic velocity check if possible
                if self.S == 2:
                    norm_factor = max(H, W)
                    trajs_norm = trajs / norm_factor
                    vel = trajs_norm[1:] - trajs_norm[:-1]  # Shape: (1, N, 2)
                    # Filter out trajectories with excessive velocity (outliers)
                    vel_norms = np.linalg.norm(vel, axis=-1)  # (1, N)
                    vel_ok = np.max(vel_norms, axis=0) < 0.4  # N
                    mot_ok = vel_ok
                else:
                    # S == 1, no motion to filter
                    mot_ok = np.ones(N, dtype=bool)
            # if np.sum(~mot_ok):
            #     print('sum(mot_ok), sum(~mot_ok)', np.sum(mot_ok), np.sum(~mot_ok))
            trajs = trajs[:,mot_ok]
            visibs = visibs[:,mot_ok]
            valids = valids[:,mot_ok]

            N = trajs.shape[1]

            
            if N < min_N:
                if self.verbose:
                    print('returning after cropping: N=%d; need at least N=%d' % (N, min_N))
                return None, False

            # we won't supervise with the extremes, but let's clamp anyway just to be safe
            trajs = np.minimum(np.maximum(trajs, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2
            
            if self.use_all_valid:
                # Return all valid trajectories (no truncation)
                N = trajs.shape[1]
                
                # prep for batching, using actual N (not fixed self.N)
                trajs_full = np.zeros((self.S, N, 2)).astype(np.float32)
                visibs_full = np.zeros((self.S, N)).astype(np.float32)
                valids_full = np.zeros((self.S, N)).astype(np.float32)
                trajs_full[:,:N] = trajs
                visibs_full[:,:N] = visibs
                valids_full[:,:N] = valids
            else:
                # Original behavior: use farthest point sampling to select diverse subset
                if N < self.N:
                    if self.verbose:
                        print('N=%d; ideally we want N=%d, but we will pad' % (N, self.N))

                # even out the distribution, across initial positions and velocities
                # fps based on xy0 and mean motion
                xym = np.concatenate([trajs[0], np.mean(trajs[1:] - trajs[:-1], axis=0)], axis=-1)
                inds = utils.misc.farthest_point_sample_py(xym, self.N)
                trajs = trajs[:,inds]
                visibs = visibs[:,inds]
                valids = valids[:,inds]

                N = trajs.shape[1]
                N_ = min(N, self.N)
                inds = np.random.choice(N, N_, replace=False)

                # prep for batching, by fixing N
                trajs_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
                visibs_full = np.zeros((self.S, self.N)).astype(np.float32)
                valids_full = np.zeros((self.S, self.N)).astype(np.float32)
                trajs_full[:,:N_] = trajs[:,inds]
                visibs_full[:,:N_] = visibs[:,inds]
                valids_full[:,:N_] = valids[:,inds]

            rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2).byte() # S,C,H,W
            masks = torch.from_numpy(np.stack(masks, 0)).unsqueeze(1).byte() # S,C,H,W
            edges = torch.from_numpy(np.stack(edges, 0)).unsqueeze(1).byte() # S,C,H,W
            trajs = torch.from_numpy(trajs_full).float() # S,N,2
            visibs = torch.from_numpy(visibs_full).float() # S,N
            valids = torch.from_numpy(valids_full).float() # S,N

            sample = {
                'rgbs': rgbs,
                'masks': masks,
                'edges': edges,
                'trajs': trajs,
                'visibs': visibs,
                'valids': valids,
            }
            return sample, True
        except Exception as e:
            if self.verbose:
                print(f"Error in getitem_helper: {e}")
            return None, False
    
    def __getitem__(self, index):
        gotit = False
        sample, gotit = self.getitem_helper(index)
        if not gotit:
            # return a fake sample, so we can still collate
            sample = {
                'rgbs': torch.zeros((self.S, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.uint8),
                'trajs': torch.zeros((self.S, self.N, 2), dtype=torch.float32),
                'visibs': torch.zeros((self.S, self.N), dtype=torch.float32),
                'valids': torch.zeros((self.S, self.N), dtype=torch.float32),
            }
        return sample, gotit

    def add_spatial_augs(self, rgbs, masks, edges, trajs, visibs):
        T, N, _ = trajs.shape
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        masks = [mask.astype(np.float32) for mask in masks]
        edges = [edge.astype(np.float32) for edge in edges]

        # start by resizing to resize_size, which should be larger than crop_size
        H_new, W_new = self.resize_size
        H_new = np.clip(H_new, self.crop_size[0]+16, None)
        W_new = np.clip(W_new, self.crop_size[1]+16, None)
        scale_x = W_new/float(W)
        scale_y = H_new/float(H)
        rgbs = [cv2.resize(rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        masks = [cv2.resize(mask, (W_new, H_new), interpolation=cv2.INTER_NEAREST) for mask in masks]
        edges = [cv2.resize(edge, (W_new, H_new), interpolation=cv2.INTER_NEAREST) for edge in edges]
        trajs[:,:,0] *= scale_x
        trajs[:,:,1] *= scale_y
        
        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
        masks = [np.pad(mask, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for mask in masks]
        edges = [np.pad(edge, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for edge in edges]
        trajs[:,:,0] += pad_x0
        trajs[:,:,1] += pad_y0
        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        masks_scaled = []
        edges_scaled = []
        trajs_scaled = []
        
        scales_x = []
        scales_y = []
        for si in range(S):
            if si==1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)*0.1
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)*0.1
            elif si > 1:
                scale_delta_x = scale_delta_x*0.9 + np.random.uniform(-self.resize_delta, self.resize_delta)*0.1
                scale_delta_y = scale_delta_y*0.9 + np.random.uniform(-self.resize_delta, self.resize_delta)*0.1
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y)*0.5
            scale_x = scale_x*0.5 + scale_xy*0.5
            scale_y = scale_y*0.5 + scale_xy*0.5
            
            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)
            
            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, self.crop_size[0]+16, None)
            W_new = np.clip(W_new, self.crop_size[1]+16, None)
            # recompute scale in case we clipped
            scale_x = W_new/float(W)
            scale_y = H_new/float(H)

            rgbs_scaled.append(cv2.resize(rgbs[si], (W_new, H_new), interpolation=cv2.INTER_LINEAR))
            masks_scaled.append(cv2.resize(masks[si], (W_new, H_new), interpolation=cv2.INTER_NEAREST))
            edges_scaled.append(cv2.resize(edges[si], (W_new, H_new), interpolation=cv2.INTER_NEAREST))
            trajs[si,:,0] *= scale_x
            trajs[si,:,1] *= scale_y
        rgbs = rgbs_scaled
        masks = masks_scaled
        edges = edges_scaled
        
        ok_inds = visibs[0,:] > 0
        vis_trajs = trajs[:,ok_inds] # S,?,2
            
        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0,:,0])
            mid_y = np.mean(vis_trajs[0,:,1])
        else:
            mid_y = self.crop_size[0]
            mid_x = self.crop_size[1]
            
        x0 = int(mid_x - self.crop_size[1]//2)
        y0 = int(mid_y - self.crop_size[0]//2)
        
        offset_x = 0
        offset_y = 0
        
        for si in range(S):
            # on each frame, shift a bit more 
            if si==1:
                offset_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                offset_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
            elif si > 1:
                offset_x = int(offset_x*0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)*0.2)
                offset_y = int(offset_y*0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)*0.2)
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[si].shape[:2]
            if H_new==self.crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)
                
            if W_new==self.crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
            rgbs[si] = rgbs[si][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            masks[si] = masks[si][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            edges[si] = edges[si][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            trajs[si,:,0] -= x0
            trajs[si,:,1] -= y0

            
        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                rgbs = [rgb[:,::-1] for rgb in rgbs]
                masks = [mask[:,::-1] for mask in masks]
                edges = [edge[:,::-1] for edge in edges]
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
                masks = [mask[::-1] for mask in masks]
                edges = [edge[::-1] for edge in edges]
        if h_flipped:
            trajs[:,:,0] = W_new - trajs[:,:,0]
        if v_flipped:
            trajs[:,:,1] = H_new - trajs[:,:,1]
            
        return rgbs, masks, edges, trajs

    def just_crop(self, rgbs, masks, edges, trajs):
        T, N, _ = trajs.shape
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        H_new, W_new = self.crop_size[0], self.crop_size[1]

        y0 = np.random.randint(0, H-H_new)
        x0 = np.random.randint(0, W-W_new)
        rgbs = [rgb[y0:y0+H_new, x0:x0+W_new] for rgb in rgbs]
        masks = [mask[y0:y0+H_new, x0:x0+W_new] for mask in masks]
        edges = [edge[y0:y0+H_new, x0:x0+W_new] for edge in edges]
        trajs[:,:,0] -= x0
        trajs[:,:,1] -= y0

        return rgbs, masks, edges, trajs

    def just_resize(self, rgbs, masks, edges, trajs):
        T, N, _ = trajs.shape
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        H_new, W_new = self.crop_size[0], self.crop_size[1]

        sx_ = W_new / W
        sy_ = H_new / H
        rgbs = [cv2.resize(rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        masks = [cv2.resize(mask, (W_new, H_new), interpolation=cv2.INTER_NEAREST) for mask in masks]
        edges = [cv2.resize(edge, (W_new, H_new), interpolation=cv2.INTER_NEAREST) for edge in edges]
        sc_py = np.array([sx_, sy_]).reshape([1,1,2])
        trajs = trajs * sc_py
        
        return rgbs, masks, edges, trajs

    def __len__(self):
        return len(self.rgb_paths)
