"""
Config generator for creating training configs from sweep definitions.

Reads a base config template and generates variations for each sweep point.
Supports multiple modes:
1. Single base config: Use 'base_config' field (backward compatible)
2. Multi-dataset mode: Use 'training_datasets' list with 'base_config' per dataset
3. Split config mode: Use 'training_base', 'eval_base', and 'dataset_config' to separate
   training/eval/dataset parameters (prevents conflicts when sweeping parameters)

The split config format is recommended for sweeps because:
- Training parameter sweeps don't affect eval config
- Eval config changes don't affect training config
- Dataset configs only contain dataset-specific parameters

Mixed Datasets:
- Mixed datasets (using MixedCorrespondenceDataset) are fully supported
- Mixed dataset configs should have 'mixed: true' and 'datasets' list
- You can override mixed dataset parameters using dot notation:
  - dataset.percentages: [0.7, 0.3]  # Change mixing ratios
  - dataset.datasets: ['spair', 'synthetic']  # Change datasets (rarely needed)
  - dataset.dataset_overrides.spair.normalize_images: false  # Override sub-dataset params
"""

import os
import yaml
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional
from itertools import product


def resolve_config_path(path: str, sweep_config_dir: Path, project_root: Path) -> str:
    """
    Resolve a config file path.
    
    Paths starting with 'src/' are resolved relative to project root.
    Other relative paths are resolved relative to sweep_config_dir.
    Absolute paths are returned as-is.
    
    Args:
        path: Path string to resolve
        sweep_config_dir: Directory containing the sweep config file
        project_root: Project root directory
        
    Returns:
        Resolved absolute path
    """
    if os.path.isabs(path):
        return path
    
    # Paths starting with 'src/' are relative to project root
    if path.startswith('src/'):
        resolved = project_root / path
        return str(resolved.resolve())
    
    # Other relative paths are relative to sweep_config_dir
    resolved = sweep_config_dir / path
    return str(resolved.resolve())


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple config dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dicts to merge
        
    Returns:
        Merged config dict
    """
    if not configs:
        return {}
    
    result = copy.deepcopy(configs[0])
    
    for config in configs[1:]:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = merge_configs(result[key], value)
            else:
                # Override with new value
                result[key] = copy.deepcopy(value)
    
    return result


def set_nested_value(config: Dict[str, Any], key_path: str, value: Any):
    """
    Set a nested value in config using dot notation.
    
    Args:
        config: Config dict to modify
        key_path: Dot-separated path (e.g., 'training.lr')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent dict
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def generate_config_name(params: Dict[str, Any], name_template: str = None) -> str:
    """
    Generate experiment name from parameters.
    
    Args:
        params: Parameter dict
        name_template: Optional template string with {key} placeholders
        
    Returns:
        Generated experiment name
    """
    if name_template:
        # Simple template replacement
        name = name_template
        for key, value in params.items():
            if isinstance(value, list):
                value_str = '_'.join(str(v) for v in value)
            else:
                value_str = str(value)
            name = name.replace(f'{{{key}}}', value_str)
        return name
    else:
        # Default: join key-value pairs
        parts = []
        for key, value in params.items():
            if isinstance(value, list):
                value_str = '_'.join(str(v) for v in value)
            else:
                value_str = str(value)
            parts.append(f"{key}_{value_str}")
        return '_'.join(parts)


def generate_configs_from_sweep(
    base_config_path: str,
    sweep_config: Dict[str, Any],
    output_dir: str
) -> List[tuple]:
    """
    Generate training configs from sweep definition.
    
    Args:
        base_config_path: Path to base config template
        sweep_config: Sweep config dict with 'parameters' and optional 'name_template'
        output_dir: Directory to save generated configs
        
    Returns:
        List of (config_path, experiment_name) tuples
    """
    # Load base config
    base_config = load_config(base_config_path)
    
    # Get parameters and generate all combinations
    parameters = sweep_config.get('parameters', {})
    if not parameters:
        # No parameters to sweep - just save base config
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        config_name = sweep_config.get('name', 'base')
        config_path = output_dir_path / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        return [(str(config_path), config_name)]
    
    # Get parameter names and values
    param_names = list(parameters.keys())
    param_values = [parameters[name] if isinstance(parameters[name], list) else [parameters[name]] 
                   for name in param_names]
    
    # Generate all combinations
    generated_configs = []
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    name_template = sweep_config.get('name_template', None)
    
    for combination in product(*param_values):
        # Create parameter dict for this combination
        params = dict(zip(param_names, combination))
        
        # Start with base config
        config = copy.deepcopy(base_config)
        
        # Apply parameter overrides
        for key_path, value in params.items():
            set_nested_value(config, key_path, value)
        
        # Generate experiment name
        if name_template:
            exp_name = generate_config_name(params, name_template)
        else:
            exp_name = generate_config_name(params)
        
        # Save config
        config_filename = f"{exp_name}.yaml"
        config_path = output_dir_path / config_filename
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        generated_configs.append((str(config_path), exp_name))
    
    return generated_configs


def generate_all_configs(sweep_config_path: str, output_dir: str = None) -> List[tuple]:
    """
    Generate all configs from a sweep config file.
    
    Supports two modes:
    1. Single base config: Use 'base_config' field (backward compatible)
    2. Multiple training datasets: Use 'training_datasets' list to sweep across datasets
    
    Args:
        sweep_config_path: Path to sweep config YAML file
        output_dir: Optional output directory (defaults to same dir as sweep config)
        
    Returns:
        List of (config_path, experiment_name) tuples
    """
    # Load sweep config
    sweep_config = load_config(sweep_config_path)
    
    # Resolve relative paths
    sweep_config_dir = Path(sweep_config_path).parent.resolve()
    
    # Get project root from config or use sweep_config_dir as fallback
    project_root_str = sweep_config.get('project_root')
    if project_root_str:
        project_root = Path(project_root_str).resolve()
        if not project_root.exists():
            raise ValueError(f"Project root path does not exist: {project_root}")
    else:
        # Fallback: assume sweep config is in slurm/experiment_configs/ and project root is 2 levels up
        if 'slurm' in sweep_config_dir.parts and 'experiment_configs' in sweep_config_dir.parts:
            idx = sweep_config_dir.parts.index('slurm')
            project_root = Path(*sweep_config_dir.parts[:idx])
        else:
            # Last resort: use current working directory
            project_root = Path.cwd()
    
    # Get output directory
    if output_dir is None:
        output_dir = sweep_config.get('output_dir', str(sweep_config_dir / 'generated'))
    
    # Resolve output_dir: if relative and starts with common prefixes, resolve relative to project root
    # Otherwise resolve relative to sweep_config_dir
    if not os.path.isabs(output_dir):
        if output_dir.startswith('slurm/') or output_dir.startswith('experiment_configs/'):
            output_dir = str(project_root / output_dir)
        else:
            output_dir = str(sweep_config_dir / output_dir)
    
    # Check if using multiple training datasets
    training_datasets = sweep_config.get('training_datasets', None)
    
    if training_datasets:
        # Multi-dataset mode: generate configs for each dataset
        return _generate_multi_dataset_configs(
            sweep_config, training_datasets, sweep_config_dir, output_dir, project_root
        )
    else:
        # Single base config mode (backward compatible)
        base_config_path = sweep_config.get('base_config')
        if not base_config_path:
            raise ValueError("sweep config must have either 'base_config' or 'training_datasets' field")
        
        base_config_path = resolve_config_path(base_config_path, sweep_config_dir, project_root)
        
        # Generate configs for each sweep
        all_configs = []
        sweeps = sweep_config.get('sweeps', [])
        
        if not sweeps:
            # No sweeps defined - just generate base config
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            config_name = sweep_config.get('name', 'base')
            config_path = output_dir_path / f"{config_name}.yaml"
            base_config = load_config(base_config_path)
            with open(config_path, 'w') as f:
                yaml.dump(base_config, f, default_flow_style=False)
            return [(str(config_path), config_name)]
        
        for sweep in sweeps:
            sweep_name = sweep.get('name', 'sweep')
            print(f"Generating configs for sweep: {sweep_name}")
            
            # Get sweep-specific output dir
            sweep_output_dir = sweep.get('output_dir', output_dir)
            if not os.path.isabs(sweep_output_dir):
                if sweep_output_dir.startswith('slurm/') or sweep_output_dir.startswith('experiment_configs/'):
                    sweep_output_dir = str(project_root / sweep_output_dir)
                else:
                    sweep_output_dir = str(Path(output_dir) / sweep_name)
            
            configs = generate_configs_from_sweep(
                base_config_path,
                sweep,
                sweep_output_dir
            )
            
            all_configs.extend(configs)
            print(f"  Generated {len(configs)} configs")
        
        return all_configs


def _generate_multi_dataset_configs(
    sweep_config: Dict[str, Any],
    training_datasets: List[Dict[str, Any]],
    sweep_config_dir: Path,
    output_dir: str,
    project_root: Path
) -> List[tuple]:
    """
    Generate configs for multiple training datasets.
    
    Supports two formats:
    1. Old format: dataset has 'base_config' (backward compatible)
    2. New format: dataset has 'dataset_config', and sweep_config has 'training_base' and 'eval_base'
    
    For each dataset and each sweep, generates configs with:
    1. Training base config (if using new format)
    2. Eval base config (if using new format)
    3. Dataset config (or base_config in old format)
    4. Dataset-specific overrides (if any)
    5. Sweep parameter overrides
    
    Args:
        sweep_config: Full sweep config dict
        training_datasets: List of dataset configs
        sweep_config_dir: Directory of sweep config (for resolving relative paths)
        output_dir: Base output directory
        
    Returns:
        List of (config_path, experiment_name) tuples
    """
    all_configs = []
    sweeps = sweep_config.get('sweeps', [])
    
    if not sweeps:
        raise ValueError("When using 'training_datasets', at least one 'sweeps' entry is required")
    
    # Check if using new split config format
    training_base_path = sweep_config.get('training_base', None)
    eval_base_path = sweep_config.get('eval_base', None)
    using_split_configs = training_base_path is not None or eval_base_path is not None
    
    # Resolve base config paths if using split format
    training_base_config = None
    eval_base_config = None
    
    if using_split_configs:
        if training_base_path:
            training_base_path = resolve_config_path(training_base_path, sweep_config_dir, project_root)
            training_base_config = load_config(training_base_path)
            print(f"Using training base config: {training_base_path}")
        
        if eval_base_path:
            eval_base_path = resolve_config_path(eval_base_path, sweep_config_dir, project_root)
            eval_base_config = load_config(eval_base_path)
            print(f"Using eval base config: {eval_base_path}")
    
    for dataset_config in training_datasets:
        dataset_name = dataset_config.get('name')
        if not dataset_name:
            raise ValueError("Each entry in 'training_datasets' must have a 'name' field")
        
        # Get dataset config path (new format) or base_config path (old format)
        dataset_config_path = dataset_config.get('dataset_config') or dataset_config.get('base_config')
        if not dataset_config_path:
            raise ValueError(f"Dataset '{dataset_name}' must have either 'dataset_config' or 'base_config' field")
        
        # Resolve relative paths
        dataset_config_path = resolve_config_path(dataset_config_path, sweep_config_dir, project_root)
        
        # Get dataset-specific overrides
        dataset_overrides = dataset_config.get('overrides', {})
        
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"  Dataset config: {dataset_config_path}")
        if dataset_overrides:
            print(f"  Dataset overrides: {list(dataset_overrides.keys())}")
        
        # Generate configs for each sweep
        for sweep in sweeps:
            sweep_name = sweep.get('name', 'sweep')
            
            # Get sweep-specific output dir
            sweep_output_dir = sweep.get('output_dir', output_dir)
            if not os.path.isabs(sweep_output_dir):
                if sweep_output_dir.startswith('slurm/') or sweep_output_dir.startswith('experiment_configs/'):
                    sweep_output_dir = str(project_root / sweep_output_dir / dataset_name / sweep_name)
                else:
                    sweep_output_dir = str(Path(output_dir) / dataset_name / sweep_name)
            
            # Generate configs for this dataset + sweep combination
            configs = _generate_configs_for_dataset_sweep(
                dataset_config_path,
                dataset_name,
                dataset_overrides,
                sweep,
                sweep_output_dir,
                training_base_config=training_base_config,
                eval_base_config=eval_base_config
            )
            
            all_configs.extend(configs)
            print(f"  Sweep '{sweep_name}': Generated {len(configs)} configs")
    
    return all_configs


def _generate_configs_for_dataset_sweep(
    dataset_config_path: str,
    dataset_name: str,
    dataset_overrides: Dict[str, Any],
    sweep: Dict[str, Any],
    output_dir: str,
    training_base_config: Dict[str, Any] = None,
    eval_base_config: Dict[str, Any] = None
) -> List[tuple]:
    """
    Generate configs for a specific dataset + sweep combination.
    
    Args:
        dataset_config_path: Path to dataset config (or base_config in old format)
        dataset_name: Name of the training dataset
        dataset_overrides: Dataset-specific parameter overrides
        sweep: Sweep definition dict
        output_dir: Output directory for generated configs
        training_base_config: Optional training base config (new format)
        eval_base_config: Optional eval base config (new format)
        
    Returns:
        List of (config_path, experiment_name) tuples
    """
    # Load dataset config
    dataset_config_dict = load_config(dataset_config_path)
    
    # Merge configs in order: training_base -> eval_base -> dataset -> dataset_overrides
    config_parts = []
    
    if training_base_config:
        config_parts.append(training_base_config)
    
    if eval_base_config:
        config_parts.append(eval_base_config)
    
    # Dataset config might be just the 'dataset' section (new format) or full config (old format)
    if 'dataset' in dataset_config_dict and len(dataset_config_dict) == 1:
        # New format: only dataset section
        config_parts.append(dataset_config_dict)
    else:
        # Old format: full config
        config_parts.append(dataset_config_dict)
    
    # Merge all config parts
    config = merge_configs(*config_parts)
    
    # Apply dataset-specific overrides
    for key_path, value in dataset_overrides.items():
        set_nested_value(config, key_path, value)
    
    # Get sweep parameters
    parameters = sweep.get('parameters', {})
    if not parameters:
        # No parameters to sweep - just save config with dataset overrides
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        config_name = f"{dataset_name}_{sweep.get('name', 'base')}"
        config_path = output_dir_path / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return [(str(config_path), config_name)]
    
    # Get parameter names and values
    param_names = list(parameters.keys())
    param_values = [parameters[name] if isinstance(parameters[name], list) else [parameters[name]] 
                   for name in param_names]
    
    # Generate all combinations
    generated_configs = []
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    name_template = sweep.get('name_template', None)
    
    for combination in product(*param_values):
        # Create parameter dict for this combination
        params = dict(zip(param_names, combination))
        
        # Start with base config + dataset overrides
        sweep_config = copy.deepcopy(config)
        
        # Apply sweep parameter overrides (these override dataset overrides)
        for key_path, value in params.items():
            set_nested_value(sweep_config, key_path, value)
        
        # Generate experiment name
        if name_template:
            exp_name = generate_config_name(params, name_template)
        else:
            exp_name = generate_config_name(params)
        
        # Prepend dataset name to experiment name
        exp_name = f"{dataset_name}_{exp_name}"
        
        # Save config
        config_filename = f"{exp_name}.yaml"
        config_path = output_dir_path / config_filename
        
        with open(config_path, 'w') as f:
            yaml.dump(sweep_config, f, default_flow_style=False)
        
        generated_configs.append((str(config_path), exp_name))
    
    return generated_configs


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training configs from sweep definitions')
    parser.add_argument('--sweep_config', type=str, required=True,
                       help='Path to sweep config YAML file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for generated configs (defaults to sweep config dir)')
    
    args = parser.parse_args()
    
    configs = generate_all_configs(args.sweep_config, args.output_dir)
    print(f"\nGenerated {len(configs)} config files:")
    for config_path, exp_name in configs:
        print(f"  {exp_name}: {config_path}")
