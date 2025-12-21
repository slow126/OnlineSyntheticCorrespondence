#!/usr/bin/env python3
"""
Run a unified post-training analysis pipeline from a single YAML config.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def _coerce_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _build_args(args_spec: Any) -> List[str]:
    if args_spec is None:
        return []
    if isinstance(args_spec, list):
        return [str(item) for item in args_spec]
    if not isinstance(args_spec, dict):
        raise ValueError(f"args must be a list or dict, got {type(args_spec).__name__}")

    args: List[str] = []
    for key, value in args_spec.items():
        flag = key if key.startswith("--") else f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            else:
                if flag.startswith("--no-"):
                    args.append(flag)
                else:
                    args.append(f"--no-{flag[2:]}")
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(v) for v in value)
            continue
        args.append(flag)
        args.append(str(value))
    return args


def _run_command(cmd: Iterable[str], dry_run: bool, cwd: Path) -> None:
    cmd_list = [str(c) for c in cmd]
    print("+ " + " ".join(cmd_list))
    if dry_run:
        return
    subprocess.run(cmd_list, check=True, cwd=str(cwd))


def _run_build_coresets(step: Dict[str, Any], python_bin: str, dry_run: bool, cwd: Path) -> None:
    configs = step.get("configs", [])
    for config_path in configs:
        cmd = [python_bin, "scripts/build_coresets.py", "--config", config_path]
        _run_command(cmd, dry_run, cwd)


def _run_calculate_coverage(step: Dict[str, Any], python_bin: str, dry_run: bool, cwd: Path) -> None:
    runs = step.get("runs", [])
    for run in runs:
        args = _build_args(run.get("args", {}))
        cmd = [python_bin, "scripts/calculate_coverage.py"] + args
        _run_command(cmd, dry_run, cwd)


def _run_mmd(step: Dict[str, Any], python_bin: str, dry_run: bool, cwd: Path) -> None:
    configs = step.get("configs", {})
    flow_config = configs.get("flow")
    if flow_config:
        cmd = [python_bin, "scripts/calculate_flow_mmd.py", "--config", flow_config]
        _run_command(cmd, dry_run, cwd)
    feature_config = configs.get("feature")
    if feature_config:
        cmd = [python_bin, "scripts/calculate_feature_mmd.py", "--config", feature_config]
        _run_command(cmd, dry_run, cwd)


def _run_leakage_free_eval(step: Dict[str, Any], python_bin: str, dry_run: bool, cwd: Path) -> None:
    args = _build_args(step.get("args", {}))
    cmd = [python_bin, "scripts/build_leakage_free_eval.py"] + args
    _run_command(cmd, dry_run, cwd)


def _run_select_checkpoints(step: Dict[str, Any], python_bin: str, dry_run: bool, cwd: Path) -> None:
    args = _build_args(step.get("args", {}))
    cmd = [python_bin, "scripts/select_best_checkpoints.py"] + args
    _run_command(cmd, dry_run, cwd)


def _run_summarize(step: Dict[str, Any], python_bin: str, dry_run: bool, cwd: Path) -> None:
    args = _build_args(step.get("args", {}))
    cmd = [python_bin, "scripts/summarize_leakage_free_results.py"] + args
    _run_command(cmd, dry_run, cwd)


STEP_HANDLERS = {
    "build_coresets": _run_build_coresets,
    "calculate_coverage": _run_calculate_coverage,
    "mmd": _run_mmd,
    "leakage_free_eval": _run_leakage_free_eval,
    "select_checkpoints": _run_select_checkpoints,
    "summarize": _run_summarize,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified experiment pipeline.")
    parser.add_argument(
        "--config",
        default="src/configs/pipeline/leakage_free_pipeline.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated list of steps to run (optional).",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated list of steps to skip (optional).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values (key.path=value). Can be repeated.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Override python executable (e.g., python3).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    for override in args.set:
        if "=" not in override:
            raise ValueError(f"Invalid --set '{override}', expected key.path=value")
        key_path, raw_value = override.split("=", 1)
        _set_nested_value(config, key_path.strip(), _coerce_value(raw_value.strip()))

    pipeline_cfg = config.get("pipeline", {})
    python_bin = args.python or pipeline_cfg.get("python", "python3")
    cwd = Path(pipeline_cfg.get("workdir", ".")).resolve()
    dry_run = args.dry_run or pipeline_cfg.get("dry_run", False)

    only_steps = None
    if args.only:
        only_steps = {s.strip() for s in args.only.split(",") if s.strip()}
    skip_steps = set()
    if args.skip:
        skip_steps = {s.strip() for s in args.skip.split(",") if s.strip()}

    steps_cfg = config.get("steps", {})
    for step_name, step in steps_cfg.items():
        if only_steps is not None and step_name not in only_steps:
            continue
        if step_name in skip_steps:
            continue
        if not step.get("enabled", True):
            continue
        handler = STEP_HANDLERS.get(step_name)
        if handler is None:
            raise ValueError(f"Unknown step '{step_name}' in config")
        print(f"\n==> Running step: {step_name}")
        handler(step, python_bin, dry_run, cwd)


if __name__ == "__main__":
    main()
