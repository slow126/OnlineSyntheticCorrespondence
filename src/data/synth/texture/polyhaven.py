import requests
import os
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import List, Dict, Optional
import time
import random


class PolyHavenDownloader:
    """Download and manage textures from Poly Haven"""
    
    def __init__(self, cache_dir: str = "pbr_textures/polyhaven"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_base = "https://api.polyhaven.com"
        self.headers = {
            'User-Agent': 'SyntheticCorrespondenceProcessor/1.0'
        }
        
    def get_available_assets(self, asset_type: str = "textures", limit: int = 10000) -> List[Dict]:
        """Get list of available assets from Poly Haven
        
        Args:
            asset_type: Type of assets ('textures', 'hdris', 'models', 'all')
            limit: Maximum number of assets to fetch
            
        Returns:
            List of asset metadata dictionaries
        """
        try:
            # Get all assets
            response = requests.get(f"{self.api_base}/assets", headers=self.headers)
            response.raise_for_status()
            
            # Get the list of all assets
            all_assets = response.json()
            
            # Filter by type and convert to list
            assets = []
            for asset_id, metadata in all_assets.items():
                if len(assets) >= limit:
                    break
                
                # Filter by type if not 'all'
                if asset_type != "all":
                    # Check if this asset matches the requested type
                    asset_type_num = metadata.get('type', 0)
                    type_mapping = {0: 'hdris', 1: 'textures', 2: 'models'}
                    if type_mapping.get(asset_type_num) != asset_type:
                        continue
                
                metadata['id'] = asset_id
                assets.append(metadata)
            
            return assets
            
        except requests.RequestException as e:
            print(f"Error fetching asset list: {e}")
            print(f"Response status: {response.status_code if 'response' in locals() else 'No response'}")
            return []
    
    def get_available_textures(self, limit: int = 50) -> List[Dict]:
        """Get list of available textures from Poly Haven (backward compatibility)
        
        Args:
            limit: Maximum number of textures to fetch
            
        Returns:
            List of texture metadata dictionaries
        """
        return self.get_available_assets("textures", limit)
    
    def download_asset(self, asset_id: str, asset_type: str = "textures", 
                      resolution: str = "1k", texture_type: str = "diffuse") -> Optional[Path]:
        """Download a specific asset from Poly Haven
        
        Args:
            asset_id: ID of the asset to download
            asset_type: Type of asset ('textures', 'hdris', 'models')
            resolution: Resolution ('1k', '2k', '4k', '8k')
            texture_type: Type of texture ('diffuse', 'normal', 'roughness', 'displacement', 'ao')
            
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            # Get file info
            response = requests.get(f"{self.api_base}/files/{asset_id}", headers=self.headers)
            response.raise_for_status()
            
            file_info = response.json()
            
            if asset_type == "textures":
                # For textures, look for texture type at top level
                texture_type_capitalized = texture_type.capitalize()
                
                if texture_type_capitalized not in file_info:
                    print(f"Texture type {texture_type_capitalized} not available for {asset_id}")
                    return None
                
                if resolution not in file_info[texture_type_capitalized]:
                    print(f"Resolution {resolution} not available for {asset_id} {texture_type_capitalized}")
                    return None
                
                # Get the first available format (prefer jpg)
                formats = file_info[texture_type_capitalized][resolution]
                if "jpg" in formats:
                    download_url = formats["jpg"]["url"]
                elif "png" in formats:
                    download_url = formats["png"]["url"]
                else:
                    # Get the first available format
                    format_name = list(formats.keys())[0]
                    download_url = formats[format_name]["url"]
                    
            elif asset_type == "models":
                # For models, they have the same structure as textures
                texture_type_capitalized = texture_type.capitalize()
                
                if texture_type_capitalized not in file_info:
                    print(f"Texture type {texture_type_capitalized} not available for {asset_id}")
                    return None
                
                if resolution not in file_info[texture_type_capitalized]:
                    print(f"Resolution {resolution} not available for {asset_id} {texture_type_capitalized}")
                    return None
                
                # Get the first available format (prefer jpg)
                formats = file_info[texture_type_capitalized][resolution]
                if "jpg" in formats:
                    download_url = formats["jpg"]["url"]
                elif "png" in formats:
                    download_url = formats["png"]["url"]
                else:
                    # Get the first available format
                    format_name = list(formats.keys())[0]
                    download_url = formats[format_name]["url"]
                
            elif asset_type == "hdris":
                # For HDRIs, look in the hdri section
                if "hdri" not in file_info:
                    print(f"No HDRI files available for {asset_id}")
                    return None
                
                if resolution not in file_info["hdri"]:
                    print(f"Resolution {resolution} not available for {asset_id}")
                    return None
                
                # Get the requested format (default to hdr)
                file_format = texture_type if texture_type in file_info["hdri"][resolution] else "hdr"
                download_url = file_info["hdri"][resolution][file_format]["url"]
            else:
                print(f"Unsupported asset type: {asset_type}")
                return None
            
            # Create filename from URL
            filename = download_url.split("/")[-1]
            filepath = self.cache_dir / filename
            
            # Skip if already downloaded
            if filepath.exists():
                print(f"Asset {filename} already exists, skipping download")
                return filepath
            
            # Download the file
            print(f"Downloading {asset_id} {resolution} {texture_type}...")
            response = requests.get(download_url, stream=True, headers=self.headers)
            response.raise_for_status()
            
            # Save the file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {filepath}")
            return filepath
            
        except requests.RequestException as e:
            print(f"Error downloading {asset_id}: {e}")
            return None
    
    def download_texture(self, texture_id: str, resolution: str = "1k", 
                        texture_type: str = "diffuse") -> Optional[Path]:
        """Download a specific texture from Poly Haven (backward compatibility)
        
        Args:
            texture_id: ID of the texture to download
            resolution: Resolution ('1k', '2k', '4k', '8k')
            texture_type: Type of texture ('diffuse', 'normal', 'roughness', 'displacement', 'ao')
            
        Returns:
            Path to downloaded file, or None if failed
        """
        return self.download_asset(texture_id, "textures", resolution, texture_type)
    
    def download_asset_set(self, asset_id: str, asset_type: str = "textures", 
                          resolution: str = "1k", texture_types: List[str] = None) -> Dict[str, Path]:
        """Download a complete set of assets for a material
        
        Args:
            asset_id: ID of the asset to download
            asset_type: Type of asset ('textures', 'hdris', 'models')
            resolution: Resolution to download
            texture_types: List of texture types to download
            
        Returns:
            Dictionary mapping texture type to file path
        """
        if texture_types is None:
            if asset_type == "hdris":
                texture_types = ["hdr"]
            else:
                texture_types = ["diffuse", "normal", "roughness"]
        
        downloaded = {}
        
        for texture_type in texture_types:
            filepath = self.download_asset(asset_id, asset_type, resolution, texture_type)
            if filepath:
                downloaded[texture_type] = filepath
                time.sleep(0.5)  # Be nice to the API
        
        return downloaded
    
    def download_texture_set(self, texture_id: str, resolution: str = "1k", 
                           texture_types: List[str] = None) -> Dict[str, Path]:
        """Download a complete set of textures for a material (backward compatibility)
        
        Args:
            texture_id: ID of the texture to download
            resolution: Resolution to download
            texture_types: List of texture types to download
            
        Returns:
            Dictionary mapping texture type to file path
        """
        return self.download_asset_set(texture_id, "textures", resolution, texture_types)
    
    def load_texture_as_tensor(self, filepath: Path, device: str = 'cpu') -> torch.Tensor:
        """Load a texture file and convert to PyTorch tensor
        
        Args:
            filepath: Path to texture image
            device: Device to load tensor on
            
        Returns:
            Texture tensor (3, H, W) in range [0, 1]
        """
        try:
            # Load image
            image = Image.open(filepath).convert('RGB')
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            texture = transform(image).to(device)
            return texture
            
        except Exception as e:
            print(f"Error loading texture {filepath}: {e}")
            return None
    
    def get_texture_metadata(self, texture_id: str) -> Optional[Dict]:
        """Get metadata for a specific texture
        
        Args:
            texture_id: ID of the texture
            
        Returns:
            Metadata dictionary or None if failed
        """
        try:
            response = requests.get(f"{self.api_base}/files/textures/{texture_id}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching metadata for {texture_id}: {e}")
            return None


def download_sample_assets(num_assets: int = 10, asset_type: str = "textures", 
                          resolution: str = "1k", cache_dir: str = "pbr_textures/polyhaven", 
                          random_sample: bool = False, limit: int = 10000) -> List[Dict]:
    """Download a sample set of assets for testing
    
    Args:
        num_assets: Number of assets to download
        asset_type: Type of assets ('textures', 'hdris', 'models')
        resolution: Resolution to download
        cache_dir: Directory to cache assets
        
    Returns:
        List of downloaded asset metadata
    """
    downloader = PolyHavenDownloader(cache_dir)
    
    # Get available assets
    print(f"Fetching available {asset_type}...")
    available_assets = downloader.get_available_assets(asset_type, limit=limit)
    
    if not available_assets:
        print(f"No {asset_type} available")
        return []
    
    # Download assets
    downloaded_assets = []
    if random_sample:
        for i in range(num_assets):
            asset_meta = random.choice(available_assets)
            asset_id = asset_meta['id']
            print(f"\nDownloading {asset_type} {i+1}/{num_assets}: {asset_id}")
            
            # Download main asset file
            if asset_type == "hdris":
                filepath = downloader.download_asset(asset_id, asset_type, resolution, "hdr")
            else:
                filepath = downloader.download_asset(asset_id, asset_type, resolution, "diffuse")
            
            if filepath:
                downloaded_assets.append({
                    'id': asset_id,
                    'name': asset_meta.get('name', asset_id),
                    'category': asset_meta.get('categories', ['unknown']),
                    'type': asset_type,
                    'filepath': filepath,
                    'metadata': asset_meta
                })
            
            # Be nice to the API
            time.sleep(1)
    else:
        for i, asset_meta in enumerate(available_assets[:num_assets]):
            asset_id = asset_meta['id']
            print(f"\nDownloading {asset_type} {i+1}/{num_assets}: {asset_id}")
            
            # Download main asset file
            if asset_type == "hdris":
                filepath = downloader.download_asset(asset_id, asset_type, resolution, "hdr")
            else:
                filepath = downloader.download_asset(asset_id, asset_type, resolution, "diffuse")
            
            if filepath:
                downloaded_assets.append({
                    'id': asset_id,
                    'name': asset_meta.get('name', asset_id),
                    'category': asset_meta.get('categories', ['unknown']),
                    'type': asset_type,
                    'filepath': filepath,
                    'metadata': asset_meta
                })
            
            # Be nice to the API
            time.sleep(1)
    
    return downloaded_assets

def download_sample_textures(num_textures: int = 10, resolution: str = "1k", 
                           cache_dir: str = "pbr_textures/polyhaven") -> List[Dict]:
    """Download a sample set of textures for testing (backward compatibility)
    
    Args:
        num_textures: Number of textures to download
        resolution: Resolution to download
        cache_dir: Directory to cache textures
        
    Returns:
        List of downloaded texture metadata
    """
    return download_sample_assets(num_textures, "textures", resolution, cache_dir)


def create_texture_dict_from_downloads(downloaded_textures: List[Dict], 
                                     device: str = 'cpu') -> Dict[int, torch.Tensor]:
    """Create a texture dictionary from downloaded textures
    
    Args:
        downloaded_textures: List of downloaded texture metadata
        device: Device to load tensors on
        
    Returns:
        Dictionary mapping object_id to texture tensor
    """
    downloader = PolyHavenDownloader()
    texture_dict = {}
    
    for i, texture_meta in enumerate(downloaded_textures):
        filepath = texture_meta['filepath']
        texture_tensor = downloader.load_texture_as_tensor(filepath, device)
        
        if texture_tensor is not None:
            # Use object_id starting from 1 (0 is typically background)
            object_id = i + 1
            texture_dict[object_id] = texture_tensor
            print(f"Loaded texture {texture_meta['name']} as object_id {object_id}")
    
    return texture_dict


def save_texture_metadata(downloaded_textures: List[Dict], 
                         metadata_file: str = "pbr_textures/polyhaven/metadata.json"):
    """Save texture metadata to a JSON file
    
    Args:
        downloaded_textures: List of downloaded texture metadata
        metadata_file: Path to save metadata
    """
    metadata_path = Path(metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings for JSON serialization
    serializable_metadata = []
    for texture_meta in downloaded_textures:
        serializable_meta = texture_meta.copy()
        serializable_meta['filepath'] = str(texture_meta['filepath'])
        serializable_metadata.append(serializable_meta)
    
    with open(metadata_path, 'w') as f:
        json.dump(serializable_metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")


def load_texture_metadata(metadata_file: str = "textures/polyhaven/metadata.json") -> List[Dict]:
    """Load texture metadata from JSON file
    
    Args:
        metadata_file: Path to metadata file
        
    Returns:
        List of texture metadata
    """
    metadata_path = Path(metadata_file)
    
    if not metadata_path.exists():
        print(f"Metadata file {metadata_path} not found")
        return []
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Convert string paths back to Path objects
    for texture_meta in metadata:
        texture_meta['filepath'] = Path(texture_meta['filepath'])
    
    return metadata


def test_api_connection() -> bool:
    """Test if the Poly Haven API is accessible
    
    Returns:
        True if API is accessible, False otherwise
    """
    try:
        headers = {'User-Agent': 'SyntheticCorrespondenceProcessor/1.0'}
        response = requests.get("https://api.polyhaven.com/assets", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"API connection successful! Found {len(data)} assets available.")
            return True
        else:
            print(f"API returned status code: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"API connection failed: {e}")
        return False


def create_fallback_textures(num_textures: int = 10, cache_dir: str = "pbr_textures/fallback") -> List[Dict]:
    """Create fallback textures if API is not accessible
    
    Args:
        num_textures: Number of textures to create
        cache_dir: Directory to save textures
        
    Returns:
        List of created texture metadata
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    created_textures = []
    
    for i in range(num_textures):
        # Create a simple procedural texture
        texture_id = f"fallback_{i:02d}"
        
        # Create a simple colored texture
        colors = [
            (0.8, 0.2, 0.2),  # Red
            (0.2, 0.8, 0.2),  # Green
            (0.2, 0.2, 0.8),  # Blue
            (0.8, 0.8, 0.2),  # Yellow
            (0.8, 0.2, 0.8),  # Magenta
            (0.2, 0.8, 0.8),  # Cyan
            (0.5, 0.5, 0.5),  # Gray
            (0.8, 0.5, 0.2),  # Orange
            (0.5, 0.2, 0.8),  # Purple
            (0.2, 0.5, 0.8),  # Light Blue
        ]
        
        color = colors[i % len(colors)]
        
        # Create a simple texture with some variation
        import numpy as np
        size = 256
        texture_array = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Add some noise for variation
        noise = np.random.randint(-30, 30, (size, size, 3))
        base_color = np.array(color) * 255
        texture_array = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
        # Save as image
        filename = f"{texture_id}_1k_diffuse.jpg"
        filepath = cache_path / filename
        
        image = Image.fromarray(texture_array)
        image.save(filepath, 'JPEG')
        
        created_textures.append({
            'id': texture_id,
            'name': f'Fallback Texture {i+1}',
            'category': 'fallback',
            'filepath': filepath,
            'metadata': {'type': 'fallback', 'color': color}
        })
        
        print(f"Created fallback texture: {filename}")
    
    return created_textures


# Example usage
if __name__ == "__main__":
    print("Testing Poly Haven API connection...")
    
    # Test API connection first
    if test_api_connection():
        print("\nDownloading sample assets from Poly Haven...")
        
        # Download 5 textures, 3 HDRIs, and 2 models
        downloaded_textures = download_sample_assets(num_assets=20, asset_type="textures", resolution="1k", random_sample=True, limit=10000)
        downloaded_hdris = download_sample_assets(num_assets=0, asset_type="hdris", resolution="1k")
        downloaded_models = download_sample_assets(num_assets=0, asset_type="models", resolution="1k")
        
        all_downloaded = downloaded_textures + downloaded_hdris + downloaded_models
        
        if all_downloaded:
            print(f"\nSuccessfully downloaded {len(all_downloaded)} assets:")
            print(f"  - {len(downloaded_textures)} textures")
            print(f"  - {len(downloaded_hdris)} HDRIs") 
            print(f"  - {len(downloaded_models)} models")
            
            for asset in all_downloaded:
                print(f"  - {asset['name']} ({asset['id']}) [{asset['type']}]")
        else:
            print("Failed to download assets from API, creating fallback textures...")
            all_downloaded = create_fallback_textures(num_textures=10)
    else:
        print("API not accessible, creating fallback textures...")
        all_downloaded = create_fallback_textures(num_textures=10)
    
    if all_downloaded:
        # Save metadata
        save_texture_metadata(all_downloaded)
        
        # Create texture dictionary for use with UV texturing (only for textures)
        texture_assets = [asset for asset in all_downloaded if asset.get('type') == 'textures']
        if texture_assets:
            texture_dict = create_texture_dict_from_downloads(texture_assets)
            
            print(f"\nCreated texture dictionary with {len(texture_dict)} textures")
            print("Object IDs:", list(texture_dict.keys()))
            
            # Print texture shapes
            for obj_id, texture in texture_dict.items():
                print(f"Object {obj_id}: {texture.shape}")
    else:
        print("No assets were created")
