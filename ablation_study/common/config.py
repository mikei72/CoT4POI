from pathlib import Path

from ablation_study.common.io import load_yaml
from ablation_study.common.paths import CONFIGS_ROOT


def load_stage_config(stage: str, dataset_name: str) -> dict:
    config_path = CONFIGS_ROOT / stage / f"{dataset_name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return load_yaml(config_path)


def load_analysis_config(name: str = "default") -> dict:
    config_path = CONFIGS_ROOT / "analysis" / f"{name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Analysis config not found: {config_path}")
    return load_yaml(config_path)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path
