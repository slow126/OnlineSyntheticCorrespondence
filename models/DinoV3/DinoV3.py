import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import torchvision
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image



class DinoV3:
    def __init__(self, pretrained_model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", resize_size=512):
        self.pretrained_model_name = pretrained_model_name
        self.resize_size = resize_size
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto", 
            dtype=torch.float16,
        )
        self.transform = self.make_transform(resize_size)


    def make_transform(self, resize_size: int = 256):
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])
    
    def forward(self, image):
        image = self.transform(image)
        inputs = self.processor(
            images=image, 
            return_tensors="pt", 
            do_resize=False,
            do_center_crop=False).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        return pooled_output
    
    def _apply_tensor_transforms(self, tensor_batch):
        """Apply the same transforms as self.transform but directly on tensors."""
        # Replicate the transform pipeline: resize, to_float, normalize
        # Assuming input is already in CHW format and on GPU
        
        # Resize to the configured size
        tensor_batch = torch.nn.functional.interpolate(
            tensor_batch, 
            size=(self.resize_size, self.resize_size), 
            mode='bilinear', 
            align_corners=False,
            antialias=True
        )
        
        # Convert to float32 and scale to [0, 1] if needed
        if tensor_batch.dtype != torch.float32:
            tensor_batch = tensor_batch.float()
        
        # Ensure values are in [0, 1] range
        tensor_batch = torch.clamp(tensor_batch, 0.0, 1.0)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_batch.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor_batch.device).view(1, 3, 1, 1)
        tensor_batch = (tensor_batch - mean) / std
        
        return tensor_batch

    def get_spatial_features(self, image):
        # Check if input is a batch of tensors (already on GPU) or a single image
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            # Batch of tensors already on GPU: (batch_size, 3, H, W)
            image = self._apply_tensor_transforms(image)
            
        elif isinstance(image, torch.Tensor) and image.dim() == 3:
            # Single tensor: (3, H, W) - add batch dimension, transform, remove batch dimension
            image = image.unsqueeze(0)  # (1, 3, H, W)
            image = self._apply_tensor_transforms(image)
            image = image.squeeze(0)  # (3, H, W)
            
        else:
            # Single PIL image or numpy array - use existing transform
            image = self.transform(image)
        
        # Process through the model
        inputs = self.processor(
            images=image, 
            return_tensors="pt", 
            do_resize=False,
            do_center_crop=False).to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        last_layer_output = outputs.last_hidden_state
        last_layer_output = last_layer_output[:, 5:, :]
        return last_layer_output

    def dummy_forward(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = load_image(url)
        output = self.get_spatial_features(image)
        print("output shape: ", output.shape)
        self.visualize_spatial_features(image)

    def visualize_spatial_features(self, image, save_path="spatial_features_pca.png", input_image_save_path="input_image.png"):
        """
        Visualizes the spatial features of the image by projecting them to 3D using PCA and saving the result.
        Also saves the (transformed) input image.
        """
        # Save the input image before transformation (if it's a PIL image or numpy array)
        if hasattr(image, "save"):
            # PIL Image
            image.save(input_image_save_path)
        elif isinstance(image, np.ndarray):
            plt.imsave(input_image_save_path, image)
        elif torch.is_tensor(image):
            # If it's a torch tensor, try to convert to numpy and save
            img = image
            if img.dim() == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                plt.imsave(input_image_save_path, img_np)
        # Save the transformed image as well (for clarity, after transform)
        transformed_image = self.transform(image)
        if torch.is_tensor(transformed_image):
            img = transformed_image
            if img.dim() == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                plt.imsave("transformed_" + input_image_save_path, img_np)

        # Get spatial features: (1, num_patches, dim)
        features = self.get_spatial_features(transformed_image)  # shape: (1, num_patches, dim)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        # features: (1, num_patches, dim)
        b, num_patches, dim = features.shape

        # Try to infer the patch grid size (assume square)
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Cannot reshape {num_patches} patches into a square grid.")

        # Reshape to (grid_h, grid_w, dim)
        features_reshaped = features[0].reshape(grid_size, grid_size, dim)

        # Flatten spatial dimensions for PCA: (num_patches, dim)
        features_flat = features_reshaped.reshape(-1, dim).cpu().numpy()

        # Run PCA to reduce to 3D
        pca = PCA(n_components=3, whiten=True)
        features_pca = pca.fit_transform(features_flat)  # (num_patches, 3)

        # Reshape back to (grid_h, grid_w, 3)
        features_pca_img = features_pca.reshape(grid_size, grid_size, 3)
        features_pca_img = torch.from_numpy(features_pca_img)

        # Enhance colors and apply sigmoid for vibrant visualization
        features_pca_img = torch.nn.functional.sigmoid(features_pca_img.mul(2.0)).permute(2, 0, 1)  # (3, H, W)

        # Plot and save
        plt.figure(figsize=(4, 4), dpi=200)
        plt.imshow(features_pca_img.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title("Spatial Features PCA")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def visualize_features_grid(self, features_dict, save_path="models/DinoV3/outputs/feature_grids.png"):
        """
        Create grid visualizations from already processed spatial features and save as images.
        
        Args:
            features_dict: Dictionary with 'src' and 'trg' keys containing spatial features
                          Shape: (batch_size, num_patches, dim)
            save_path: Path to save the grid image (or directory for multiple files)
        """
        import torchvision.utils as vutils
        import os
        
        # Create save directory if it's a directory path
        if save_path.endswith('/') or '.' not in os.path.basename(save_path):
            os.makedirs(save_path, exist_ok=True)
            base_path = save_path
        else:
            base_path = os.path.dirname(save_path)
            if base_path:
                os.makedirs(base_path, exist_ok=True)
        
        # Get batch size from first available features
        batch_size = None
        for key, features in features_dict.items():
            if features is not None:
                batch_size = features.shape[0]
                break
        
        if batch_size is None:
            print("No features found in features_dict")
            return
        
        # Create grids for each key (src, trg)
        for key, features in features_dict.items():
            if features is None:
                continue
                
            batch_size, num_patches, dim = features.shape
            
            # Calculate spatial dimensions based on DinoV3 patch size (16x16)
            # DinoV3 downsamples by factor of 16, so patch grid size = resize_size / 16
            patch_size = 16
            grid_size = self.resize_size // patch_size
            
            if grid_size * grid_size != num_patches:
                print(f"Warning: Expected {grid_size}x{grid_size}={grid_size*grid_size} patches, got {num_patches}")
                # Try to infer grid size from num_patches
                grid_size = int(np.sqrt(num_patches))
                if grid_size * grid_size != num_patches:
                    print(f"Cannot create square grid for {num_patches} patches")
                    continue
            
            # Process all images in batch
            batch_visualizations = []
            for i in range(batch_size):
                img_features = features[i]  # (num_patches, dim)
                
                # Reshape to spatial grid: (H, W, dim)
                spatial_features = img_features.reshape(grid_size, grid_size, dim)
                
                # Apply PCA to reduce to 3 channels for RGB visualization
                features_flat = spatial_features.reshape(-1, dim).cpu().numpy()
                pca = PCA(n_components=3, whiten=True)
                features_pca = pca.fit_transform(features_flat)
                
                # Reshape back to spatial dimensions: (H, W, 3)
                features_pca_img = features_pca.reshape(grid_size, grid_size, 3)
                features_pca_tensor = torch.from_numpy(features_pca_img)
                
                # Normalize and enhance for visualization
                features_pca_tensor = torch.nn.functional.sigmoid(features_pca_tensor.mul(2.0))
                features_pca_tensor = features_pca_tensor.permute(2, 0, 1)  # (3, H, W)
                
                batch_visualizations.append(features_pca_tensor)
            
            # Create grid and save
            if batch_visualizations:
                grid = vutils.make_grid(batch_visualizations, nrow=4, padding=2, pad_value=0.5)
                
                if save_path.endswith('/') or '.' not in os.path.basename(save_path):
                    # Directory path - save with key name
                    grid_path = os.path.join(base_path, f'{key}_features_grid.png')
                else:
                    # Single file path - save with key prefix
                    name, ext = os.path.splitext(save_path)
                    grid_path = f"{name}_{key}{ext}"
                
                vutils.save_image(grid, grid_path)
                print(f"Saved {key} features grid to {grid_path}")
        
        # Create comparison grid (src vs trg side by side)
        if 'src' in features_dict and 'trg' in features_dict:
            src_features = features_dict['src']
            trg_features = features_dict['trg']
            
            if src_features is not None and trg_features is not None:
                comparison_visualizations = []
                
                # Calculate grid size for comparison (use same logic as above)
                patch_size = 16
                grid_size = self.resize_size // patch_size
                
                for i in range(min(batch_size, 8)):  # Limit to 8 for readability
                    # Process src
                    src_img_features = src_features[i]  # (num_patches, dim)
                    src_spatial = src_img_features.reshape(grid_size, grid_size, dim)
                    src_flat = src_spatial.reshape(-1, dim).cpu().numpy()
                    src_pca = PCA(n_components=3, whiten=True)
                    src_features_pca = src_pca.fit_transform(src_flat)
                    src_vis = torch.from_numpy(src_features_pca.reshape(grid_size, grid_size, 3))
                    src_vis = torch.nn.functional.sigmoid(src_vis.mul(2.0)).permute(2, 0, 1)
                    
                    # Process trg
                    trg_img_features = trg_features[i]  # (num_patches, dim)
                    trg_spatial = trg_img_features.reshape(grid_size, grid_size, dim)
                    trg_flat = trg_spatial.reshape(-1, dim).cpu().numpy()
                    trg_pca = PCA(n_components=3, whiten=True)
                    trg_features_pca = trg_pca.fit_transform(trg_flat)
                    trg_vis = torch.from_numpy(trg_features_pca.reshape(grid_size, grid_size, 3))
                    trg_vis = torch.nn.functional.sigmoid(trg_vis.mul(2.0)).permute(2, 0, 1)
                    
                    comparison_visualizations.extend([src_vis, trg_vis])
                
                if comparison_visualizations:
                    comparison_grid = vutils.make_grid(comparison_visualizations, nrow=2, padding=2, pad_value=0.5)
                    
                    if save_path.endswith('/') or '.' not in os.path.basename(save_path):
                        # Directory path
                        comparison_path = os.path.join(base_path, 'src_trg_comparison.png')
                    else:
                        # Single file path
                        name, ext = os.path.splitext(save_path)
                        comparison_path = f"{name}_comparison{ext}"
                    
                    vutils.save_image(comparison_grid, comparison_path)
                    print(f"Saved comparison grid to {comparison_path}")
    

if __name__ == "__main__":
    dino_v3 = DinoV3()
    dino_v3.dummy_forward()
    # Load the test image
    input_image_path = "test_render.png"
    input_image = Image.open(input_image_path).convert("RGB")
    dino_v3.visualize_spatial_features(input_image)




