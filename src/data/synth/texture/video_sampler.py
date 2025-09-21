import torch
import cv2
import numpy as np
import os

class VideoSampler(object):
    def __init__(self, path="/home/spencer/Deployments/synthetic-correspondence/videos/converted_videos", device='cuda'):
        self.textures = [f for f in os.listdir(path) if f.lower().endswith(('.npy'))]

        # Load and cache all video textures as tensors
        self.videos = []
        for texture_file in self.textures:
            # Load numpy array
            texture_path = os.path.join(path, texture_file)
            texture = np.load(texture_path)
            
            # Convert to tensor and move to device
            texture = torch.from_numpy(texture).float()
            if device is not None:
                texture = texture.to(device)
                
            # Normalize to [0,1] if needed
            if texture.max() > 1.0:
                texture = texture / 255.0
                
            self.videos.append(texture)
            
        print(f"Loaded {len(self.videos)} video textures")

        self.path = path
        self.device = None

    def sample(self, batch_size, shape, rng):
        self.device = rng.device
        
        # Create output tensors for batch
        frames = []
        
        for _ in range(batch_size):
            # Randomly select a video
            video_idx = torch.randint(0, len(self.videos), (1,), device=self.device).item()
            video = self.videos[video_idx]
            
            # Get random starting frame
            start_idx = torch.randint(0, video.shape[0], (1,), generator=rng, device=self.device).item()
            
            # Extract sequence of frames
            indices = torch.arange(start_idx, start_idx + shape[0], device=self.device) % video.shape[0]
            frame_sequence = video[indices]
            
            # Resize frames
            frame_sequence = torch.nn.functional.interpolate(
                frame_sequence.permute(0, 3, 1, 2),  # NHWC -> NCHW
                size=(shape[1], shape[2]),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # NCHW -> NHWC
            
            frames.append(frame_sequence)
            
        # Stack into batch
        frames = torch.stack(frames)
        
        # Return two copies for src/trg pairs
        return (frames, frames.clone())