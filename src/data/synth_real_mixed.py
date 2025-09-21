from pathlib import Path
from typing import Optional, Union, Literal, Dict, Any

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.joinpath('../../').resolve()))

from src.data.base import BaseDatamodule
from src.data.synth.datamodule import OnlineComponentsDatamodule, BaseComponentsDatamodule
from src.data.real.datamodule import KITTIDatamodule, HPatchesDatamodule, SintelDatamodule, AWADatamodule, CUBDatamodule, TSSDatamodule
import torchvision


class SynthRealMixedDatamodule(OnlineComponentsDatamodule):
    def __init__(
        self,
        root: Union[str, Path],
        config: Dict[str, Any],
        copy_data_local: Optional[str] = None,
    ):
        super().__init__(
            root,
            copy_data_local,
        )
        self.config = config
        self.synth_config = config.get('data', {})
        self.real_config = config.get('val_data', {})

    def setup(self, stage='fit'):
        # Setup synthetic training data from config
        if stage == 'fit':
            synth_class_path = self.synth_config.get('class_path', 'src.data.synth.datamodule.OnlineComponentsDatamodule')
            synth_init_args = self.synth_config.get('init_args', {})
            
            # Dynamically load synthetic datamodule class
            from importlib import import_module
            module_path, class_name = synth_class_path.rsplit('.', 1)
            module = import_module(module_path)
            SynthDatamoduleClass = getattr(module, class_name)
            
            self.synth_datamodule = SynthDatamoduleClass(
                **synth_init_args
            )
            self.synth_datamodule.post_init()
            self.synth_datamodule.setup('fit')
            self.train_data = self.synth_datamodule.train_data

        # Setup real validation data from config if specified
        if self.real_config:
            real_class_path = self.real_config.get('class_path', 'src.data.real.datamodule.TSSDatamodule')
            real_init_args = self.real_config.get('init_args', {})
            
            # Dynamically load real datamodule class
            from importlib import import_module
            module_path, class_name = real_class_path.rsplit('.', 1)
            module = import_module(module_path)
            RealDatamoduleClass = getattr(module, class_name)
            
            
            self.real_datamodule = RealDatamoduleClass(
                **real_init_args,
            )
            self.real_datamodule.setup(stage='val')
            self.val_data = self.real_datamodule.val_dataloader().dataset
        else:
            self.val_data = None

    def train_dataloader(self):
        train_loader = self.synth_datamodule.train_dataloader()
        self.train_dataloader_instance = train_loader
        return train_loader

    def val_dataloader(self):
        if self.val_data is None:
            return None
        val_loader = self.real_datamodule.val_dataloader()
        self.val_dataloader_instance = val_loader
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # if dataloader_idx == 0: # Assuming train_dataloader is at index 0
        #     print("Calling OnlineComponentsDatamodule's on_after_batch_transfer for train_dataloader_idx =", dataloader_idx)
        #     return self.synth_datamodule.on_after_batch_transfer(batch, dataloader_idx)
        # else:
        #     print("Using default on_after_batch_transfer (no-op) for val_dataloader_idx =", dataloader_idx)
        #     return batch
        if self.trainer is not None and self.trainer.training:
            # print("Calling OnlineComponentsDatamodule's on_after_batch_transfer for TRAINING")
            return self.synth_datamodule.on_after_batch_transfer(batch, dataloader_idx)
        elif self.trainer is not None and self.trainer.validating:
            # print("Using default on_after_batch_transfer (no-op) for VALIDATION")
            return batch # Or perform different validation-specific batch transfer here
        else:
            # Handle other cases (test, predict, etc.) if needed, or default behavior
            # print("Using default on_after_batch_transfer (no-op) for OTHER stage")
            return batch


if __name__ == '__main__':
    # Get the data directory from .env file
    import os
    from dotenv import load_dotenv
    from torch.utils.data import DataLoader

    path = Path(__file__).parent.joinpath('../../').resolve()
    load_dotenv()
    data_root = os.getenv('DATADIR', '')
    Path(data_root).mkdir(exist_ok=True, parents=True)

    # Load base config
    from omegaconf import OmegaConf
    base_config_path = path.joinpath('src/configs/base_synth_training.yaml')
    base_config = OmegaConf.load(base_config_path)
    base_config_dict = OmegaConf.to_container(base_config, resolve=True)

    print("Testing SynthRealMixedDatamodule as drop-in replacement...")
    
    # Test with full config
    print("\n=== Testing with config-driven setup ===")
    dm = SynthRealMixedDatamodule(
        root=data_root,
        config=base_config_dict,
    )
    dm.setup()
    
    print("\nTrain Data:")
    print(f"Type: {type(dm.train_data)}")
    print(f"Length: {len(dm.train_data)}")
    
    print("\nValidation Data:")
    if dm.val_data:
        print(f"Type: {type(dm.val_data)}")
        print(f"Length: {len(dm.val_data)}")
        batch = next(iter(dm.val_dataloader()))
        print(f"Batch keys: {batch.keys()}")
    else:
        print("No validation data configured")

    # Test training data pipeline
    print("\n=== Testing training data flow ===")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    # batch = dm.train_data[0]
    batch = [{k: v.cuda() for k, v in x.items()} for x in batch]

    batch = dm.on_after_batch_transfer(batch, 0)
    print(f"Batch keys: {batch.keys()}")    
    print(f"Source image shape: {batch['src'].shape}")
    print(f"Target image shape: {batch['trg'].shape}")

    import os
    from pathlib import Path
    import torch
    from torchvision.utils import make_grid

    debug_dir = Path('debug/images')
    debug_dir.mkdir(exist_ok=True, parents=True)

    def save_image_grid(batch, split_name, debug_dir):
        for key in ['src', 'trg']:
            if key in batch:
                images = batch[key]
                grid = make_grid(images, nrow=8)  # Adjust nrow as needed
                # Assuming images are normalized, we can directly save them.
                # If they are not in [0, 1], we might need to scale them.
                filepath = debug_dir.joinpath(f'{split_name}_{key}.png')
                torchvision.utils.save_image(grid, str(filepath))
                print(f"Saved {split_name} {key} grid to {filepath}")

    print("\n=== Saving debug image grids ===")
    # Train data grid
    train_batch = next(iter(train_loader))
    train_batch = [{k: v.cuda() for k, v in x.items()} for x in train_batch]
    # train_batch = dm.synth_datamodule.on_after_batch_transfer(train_batch, 0)
    train_batch = dm.on_after_batch_transfer(train_batch, 0)
    save_image_grid(train_batch, 'train', debug_dir)

    # Validation data grid
    if dm.val_dataloader():
        val_loader = dm.val_dataloader()
        val_batch = next(iter(val_loader))
        save_image_grid(val_batch, 'val', debug_dir)
    else:
        print("No validation dataloader available, skipping val grid.")

    print("Debug image grids saved to debug/images/")
