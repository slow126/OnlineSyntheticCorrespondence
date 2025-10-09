import torch
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class PBRSampler:
    """PBR texture sampler with intelligent caching strategy
    
    This sampler maintains a cache of N textures in memory and refreshes
    the cache every M sampling calls to balance memory usage and performance.
    """
    
    def __init__(self, 
                 texture_dir: str = "pbr_textures/polyhaven",
                 cache_size: int = 20,
                 refresh_frequency: int = 100,
                 texture_resolution: int = 512,
                 device: str = 'cpu'):
        """Initialize PBR sampler
        
        Args:
            texture_dir: Directory containing PBR textures
            cache_size: Number of textures to keep in memory (N)
            refresh_frequency: Number of calls before refreshing cache (M)
            texture_resolution: Resolution to load textures at
            device: Device to load textures on
        """
        self.texture_dir = Path(texture_dir)
        self.cache_size = cache_size
        self.refresh_frequency = refresh_frequency
        self.texture_resolution = texture_resolution
        self.device = device
        
        # Cache management
        self.texture_cache: Dict[str, torch.Tensor] = {}
        self.cached_texture_ids: List[str] = []
        self.call_count = 0
        
        # Load texture metadata
        self.metadata = self._load_metadata()
        self.available_texture_ids = list(self.metadata.keys())
        
        if not self.available_texture_ids:
            print(f"Warning: No textures found in {texture_dir}")
        
        # Initialize cache
        self._refresh_cache()
    
    def _load_metadata(self) -> Dict:
        """Load texture metadata from JSON file"""
        metadata_file = self.texture_dir / "metadata.json"
        
        if not metadata_file.exists():
            print(f"Metadata file not found: {metadata_file}")
            return {}
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Convert to dict with texture_id as key
            metadata_dict = {}
            for texture_info in metadata:
                texture_id = texture_info['id']
                metadata_dict[texture_id] = texture_info
            
            return metadata_dict
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}
    
    def _load_texture_from_disk(self, texture_id: str) -> Optional[torch.Tensor]:
        """Load a single texture from disk"""
        if texture_id not in self.metadata:
            print(f"Texture {texture_id} not found in metadata")
            return None
        
        texture_info = self.metadata[texture_id]
        filepath = Path(texture_info['filepath'])
        
        if not filepath.exists():
            print(f"Texture file not found: {filepath}")
            return None
        
        try:
            # Load image
            image = Image.open(filepath).convert('RGB')
            
            # Resize if needed
            if image.size != (self.texture_resolution, self.texture_resolution):
                image = image.resize((self.texture_resolution, self.texture_resolution), Image.LANCZOS)
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            texture = transform(image).to(self.device)
            return texture
            
        except Exception as e:
            print(f"Error loading texture {texture_id}: {e}")
            return None
    
    def _refresh_cache(self):
        """Refresh the texture cache with new random textures"""
        if not self.available_texture_ids:
            return
        
        # Clear current cache
        self.texture_cache.clear()
        self.cached_texture_ids.clear()
        
        # Select random textures to cache
        num_to_cache = min(self.cache_size, len(self.available_texture_ids))
        selected_ids = random.sample(self.available_texture_ids, num_to_cache)
        
        print(f"Refreshing texture cache with {num_to_cache} textures...")
        
        # Load selected textures
        for texture_id in selected_ids:
            texture = self._load_texture_from_disk(texture_id)
            if texture is not None:
                self.texture_cache[texture_id] = texture
                self.cached_texture_ids.append(texture_id)
        
        print(f"Cache refreshed with {len(self.texture_cache)} textures")
    
    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed"""
        return self.call_count % self.refresh_frequency == 0
    
    def sample_textures(self, num_objects: int, target_device: Optional[str] = None, 
                       matching_prob: float = 0.5) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Sample random textures for the given number of objects with matching probability
        
        Args:
            num_objects: Number of objects that need textures
            target_device: Device to move textures to (e.g., 'cuda:0'). If None, uses cache device
            matching_prob: Probability that src and trg use the same texture for each object
            
        Returns:
            Tuple of (src_texture_dict, trg_texture_dict) mapping object_id to texture tensor
        """
        self.call_count += 1
        
        # Refresh cache if needed
        if self._should_refresh_cache() and self.call_count > 1:
            self._refresh_cache()
        
        if not self.cached_texture_ids:
            print("No textures available in cache")
            return {}
        
        # Sample random textures from cache
        num_to_sample = min(num_objects, len(self.cached_texture_ids))
        src_sampled_ids = random.sample(self.cached_texture_ids, num_to_sample)
        
        # Determine target device
        if target_device is None:
            target_device = self.device
        
        # Create src texture dictionary
        src_texture_dict = {}
        for i, texture_id in enumerate(src_sampled_ids):
            object_id = i  
            texture = self.texture_cache[texture_id]
            # Move to target device if different from cache device
            if target_device != self.device:
                texture = texture.to(target_device)
            src_texture_dict[object_id] = texture
        
        # Create trg texture dictionary with matching probability
        trg_texture_dict = {}
        for i, src_texture_id in enumerate(src_sampled_ids):
            object_id = i
            # Decide if this object should match between src and trg
            should_match = random.random() < matching_prob
            
            if should_match:
                # Use same texture as src
                trg_texture_id = src_texture_id
            else:
                # Sample a different random texture
                available_ids = [tid for tid in self.cached_texture_ids if tid != src_texture_id]
                if available_ids:
                    trg_texture_id = random.choice(available_ids)
                else:
                    trg_texture_id = src_texture_id  # Fallback if no other textures
            
            texture = self.texture_cache[trg_texture_id]
            # Move to target device if different from cache device
            if target_device != self.device:
                texture = texture.to(target_device)
            trg_texture_dict[object_id] = texture
        
        return src_texture_dict, trg_texture_dict
    
    def get_texture_by_id(self, texture_id: str, target_device: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get a specific texture by ID (loads from disk if not in cache)
        
        Args:
            texture_id: ID of the texture to get
            target_device: Device to move texture to. If None, uses cache device
            
        Returns:
            Texture tensor on target device
        """
        if texture_id in self.texture_cache:
            texture = self.texture_cache[texture_id]
        else:
            # Load from disk
            texture = self._load_texture_from_disk(texture_id)
            if texture is not None:
                # Add to cache (evict oldest if cache is full)
                if len(self.texture_cache) >= self.cache_size:
                    oldest_id = self.cached_texture_ids.pop(0)
                    del self.texture_cache[oldest_id]
                
                self.texture_cache[texture_id] = texture
                self.cached_texture_ids.append(texture_id)
        
        # Move to target device if specified and different from cache device
        if texture is not None and target_device is not None and target_device != self.device:
            texture = texture.to(target_device)
        
        return texture
    
    def get_cache_info(self) -> Dict:
        """Get information about current cache state"""
        return {
            'cache_size': len(self.texture_cache),
            'max_cache_size': self.cache_size,
            'cached_texture_ids': self.cached_texture_ids,
            'call_count': self.call_count,
            'next_refresh_at': self.refresh_frequency - (self.call_count % self.refresh_frequency),
            'total_available_textures': len(self.available_texture_ids)
        }
    
    def preload_textures(self, texture_ids: List[str]):
        """Preload specific textures into cache"""
        for texture_id in texture_ids:
            if texture_id not in self.texture_cache:
                texture = self._load_texture_from_disk(texture_id)
                if texture is not None:
                    # Add to cache (evict oldest if cache is full)
                    if len(self.texture_cache) >= self.cache_size:
                        oldest_id = self.cached_texture_ids.pop(0)
                        del self.texture_cache[oldest_id]
                    
                    self.texture_cache[texture_id] = texture
                    self.cached_texture_ids.append(texture_id)
    
    def clear_cache(self):
        """Clear the texture cache"""
        self.texture_cache.clear()
        self.cached_texture_ids.clear()
        print("Texture cache cleared")
    
    def set_device(self, device: str):
        """Move all cached textures to new device"""
        self.device = device
        for texture_id in self.texture_cache:
            self.texture_cache[texture_id] = self.texture_cache[texture_id].to(device)
        print(f"Textures moved to device: {device}")


class AdaptivePBRSampler(PBRSampler):
    """PBR sampler with adaptive cache management"""
    
    def __init__(self, 
                 texture_dir: str = "pbr_textures/polyhaven",
                 initial_cache_size: int = 20,
                 max_cache_size: int = 50,
                 refresh_frequency: int = 100,
                 texture_resolution: int = 256,
                 device: str = 'cpu'):
        """Initialize adaptive PBR sampler
        
        Args:
            texture_dir: Directory containing PBR textures
            initial_cache_size: Initial number of textures to cache
            max_cache_size: Maximum number of textures to cache
            refresh_frequency: Number of calls before refreshing cache
            texture_resolution: Resolution to load textures at
            device: Device to load textures on
        """
        super().__init__(texture_dir, initial_cache_size, refresh_frequency, texture_resolution, device)
        self.max_cache_size = max_cache_size
        self.initial_cache_size = initial_cache_size
        
        # Adaptive parameters
        self.cache_hit_rate = 0.0
        self.recent_requests = []
        self.performance_history = []
    
    def _adaptive_refresh_cache(self):
        """Refresh cache with adaptive sizing based on performance"""
        if not self.available_texture_ids:
            return
        
        # Calculate adaptive cache size based on hit rate
        if self.cache_hit_rate > 0.8:  # High hit rate, increase cache
            self.cache_size = min(self.cache_size + 5, self.max_cache_size)
        elif self.cache_hit_rate < 0.5:  # Low hit rate, decrease cache
            self.cache_size = max(self.cache_size - 5, self.initial_cache_size)
        
        # Clear current cache
        self.texture_cache.clear()
        self.cached_texture_ids.clear()
        
        # Select random textures to cache
        num_to_cache = min(self.cache_size, len(self.available_texture_ids))
        selected_ids = random.sample(self.available_texture_ids, num_to_cache)
        
        print(f"Adaptive cache refresh: {num_to_cache} textures (hit rate: {self.cache_hit_rate:.2f})")
        
        # Load selected textures
        for texture_id in selected_ids:
            texture = self._load_texture_from_disk(texture_id)
            if texture is not None:
                self.texture_cache[texture_id] = texture
                self.cached_texture_ids.append(texture_id)
    
    def sample_textures(self, num_objects: int, target_device: Optional[str] = None, 
                       matching_prob: float = 0.5) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Sample random textures for the given number of objects with adaptive cache management
        
        Args:
            num_objects: Number of objects that need textures
            target_device: Device to move textures to (e.g., 'cuda:0'). If None, uses cache device
            matching_prob: Probability that src and trg use the same texture for each object
            
        Returns:
            Tuple of (src_texture_dict, trg_texture_dict) mapping object_id to texture tensor
        """
        self.call_count += 1
        
        # Track cache hits
        cache_hits = 0
        total_requests = 0
        
        # Refresh cache if needed
        if self._should_refresh_cache() and self.call_count > 1:
            self._adaptive_refresh_cache()
        
        if not self.cached_texture_ids:
            print("No textures available in cache")
            return {}
        
        # Sample random textures from cache
        num_to_sample = min(num_objects, len(self.cached_texture_ids))
        src_sampled_ids = random.sample(self.cached_texture_ids, num_to_sample)
        
        # Count cache hits
        for texture_id in src_sampled_ids:
            total_requests += 1
            if texture_id in self.texture_cache:
                cache_hits += 1
        
        # Update hit rate
        if total_requests > 0:
            self.cache_hit_rate = cache_hits / total_requests
        
        # Determine target device
        if target_device is None:
            target_device = self.device
        
        # Create src texture dictionary
        src_texture_dict = {}
        for i, texture_id in enumerate(src_sampled_ids):
            object_id = i + 1  # Object IDs start from 1 (0 is background)
            texture = self.texture_cache[texture_id]
            # Move to target device if different from cache device
            if target_device != self.device:
                texture = texture.to(target_device)
            src_texture_dict[object_id] = texture
        
        # Create trg texture dictionary with matching probability
        trg_texture_dict = {}
        for i, src_texture_id in enumerate(src_sampled_ids):
            object_id = i + 1
            
            # Decide if this object should match between src and trg
            should_match = random.random() < matching_prob
            
            if should_match:
                # Use same texture as src
                trg_texture_id = src_texture_id
            else:
                # Sample a different random texture
                available_ids = [tid for tid in self.cached_texture_ids if tid != src_texture_id]
                if available_ids:
                    trg_texture_id = random.choice(available_ids)
                else:
                    trg_texture_id = src_texture_id  # Fallback if no other textures
            
            texture = self.texture_cache[trg_texture_id]
            # Move to target device if different from cache device
            if target_device != self.device:
                texture = texture.to(target_device)
            trg_texture_dict[object_id] = texture
        
        return src_texture_dict, trg_texture_dict


# Example usage and testing
if __name__ == "__main__":
    # Test basic sampler
    print("Testing PBR Sampler...")
    
    sampler = PBRSampler(
        texture_dir="pbr_textures/polyhaven",
        cache_size=10,
        refresh_frequency=5,  # Refresh every 5 calls for testing
        device='cpu'
    )
    
    print(f"Cache info: {sampler.get_cache_info()}")
    
    # Test sampling
    for i in range(8):
        print(f"\n--- Sample call {i+1} ---")
        src_textures, trg_textures = sampler.sample_textures(
            num_objects=3, 
            target_device='cuda' if torch.cuda.is_available() else 'cpu',
            matching_prob=0.7
        )
        print(f"Sampled {len(src_textures)} src textures for object IDs: {list(src_textures.keys())}")
        print(f"Sampled {len(trg_textures)} trg textures for object IDs: {list(trg_textures.keys())}")
        if src_textures:
            print(f"Texture device: {next(iter(src_textures.values())).device}")
        print(f"Cache info: {sampler.get_cache_info()}")
    
    # Test adaptive sampler
    print("\n\nTesting Adaptive PBR Sampler...")
    
    adaptive_sampler = AdaptivePBRSampler(
        texture_dir="pbr_textures/polyhaven",
        initial_cache_size=5,
        max_cache_size=15,
        refresh_frequency=3,  # Refresh every 3 calls for testing
        device='cpu'
    )
    
    for i in range(10):
        print(f"\n--- Adaptive Sample call {i+1} ---")
        src_textures, trg_textures = adaptive_sampler.sample_textures(
            num_objects=2, 
            target_device='cuda' if torch.cuda.is_available() else 'cpu',
            matching_prob=0.6
        )
        print(f"Sampled {len(src_textures)} src textures for object IDs: {list(src_textures.keys())}")
        print(f"Sampled {len(trg_textures)} trg textures for object IDs: {list(trg_textures.keys())}")
        if src_textures:
            print(f"Texture device: {next(iter(src_textures.values())).device}")
        cache_info = adaptive_sampler.get_cache_info()
        print(f"Cache size: {cache_info['cache_size']}, Hit rate: {adaptive_sampler.cache_hit_rate:.2f}")
