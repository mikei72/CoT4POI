from __future__ import annotations

from pathlib import Path
from typing import Any

from ablation_study.common.io import load_yaml
from ablation_study.common.paths import CONFIGS_ROOT, PROJECT_ROOT


def _load_external_config() -> dict[str, Any]:
    cfg_path = CONFIGS_ROOT / "external_results.yml"
    if not cfg_path.exists():
        return {"external_results": {"auto_discover": True, "datasets": {}}}
    return load_yaml(cfg_path) or {"external_results": {"auto_discover": True, "datasets": {}}}


def _resolve_optional_path(path_str: str) -> Path | None:
    if not path_str:
        return None
    # Accept both Windows-style "\" and POSIX-style "/" separators in config.
    normalized = str(path_str).replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def _all_present(mapping: dict[str, Path | None]) -> bool:
    return all(mapping.get(k) is not None for k in ["train", "val", "test"])


def _discover_cot_fine_full_files(dataset_name: str) -> dict[str, Path | None]:
    dataset_root = PROJECT_ROOT / "datasets" / dataset_name / "dataIntegration"
    candidates = {
        "train": [
            dataset_root / "train_set_with_fine.json",
            dataset_root / "train_set_with_fines.json",
        ],
        "val": [
            dataset_root / "val_set_with_fine.json",
            dataset_root / "val_set_with_fines.json",
        ],
        "test": [
            dataset_root / "test_set_with_fine.json",
            dataset_root / "test_set_with_fines.json",
        ],
    }
    found: dict[str, Path | None] = {}
    for split, paths in candidates.items():
        hit = None
        for path in paths:
            if path.exists():
                hit = path
                break
        found[split] = hit
    return found


def _discover_cot_macro_full_files(dataset_name: str) -> dict[str, Path | None]:
    dataset_root = PROJECT_ROOT / "datasets" / dataset_name / "dataIntegration"
    candidates = {
        "train": [dataset_root / "train_set_with_macros.json"],
        "val": [dataset_root / "val_set_with_macros.json"],
        "test": [dataset_root / "test_set_with_macros.json"],
    }
    found: dict[str, Path | None] = {}
    for split, paths in candidates.items():
        hit = None
        for path in paths:
            if path.exists():
                hit = path
                break
        found[split] = hit
    return found


def _discover_final_sft_full_files(dataset_name: str) -> dict[str, Path | None]:
    root = PROJECT_ROOT / "datasets" / dataset_name / "preprocessed"
    candidates = {
        "train": [root / "final_sft_train.json"],
        "val": [root / "final_sft_val.json"],
        "test": [root / "final_sft_test.json"],
    }
    found: dict[str, Path | None] = {}
    for split, paths in candidates.items():
        hit = None
        for path in paths:
            if path.exists():
                hit = path
                break
        found[split] = hit
    return found


def resolve_cot_fine_full_files(dataset_name: str) -> dict[str, Path]:
    cfg = _load_external_config().get("external_results", {})
    dataset_cfg = cfg.get("datasets", {}).get(dataset_name, {})
    manual = dataset_cfg.get("cot_fine_full", {}).get("files", {})

    manual_paths: dict[str, Path | None] = {
        "train": _resolve_optional_path(manual.get("train", "")),
        "val": _resolve_optional_path(manual.get("val", "")),
        "test": _resolve_optional_path(manual.get("test", "")),
    }
    if _all_present(manual_paths):
        return {k: v for k, v in manual_paths.items() if v is not None}

    if cfg.get("auto_discover", True):
        discovered = _discover_cot_fine_full_files(dataset_name)
        if _all_present(discovered):
            return {k: v for k, v in discovered.items() if v is not None}

    missing = [k for k, v in manual_paths.items() if v is None]
    raise FileNotFoundError(
        "Failed to resolve external full fine files for dataset "
        f"'{dataset_name}'. Missing splits: {missing}. "
        "Set paths in ablation_study/configs/external_results.yml "
        "or ensure datasets/<dataset>/dataIntegration/*_set_with_fine.json exists."
    )


def resolve_cot_macro_full_files(dataset_name: str) -> dict[str, Path]:
    cfg = _load_external_config().get("external_results", {})
    dataset_cfg = cfg.get("datasets", {}).get(dataset_name, {})
    manual = dataset_cfg.get("cot_macro_full", {}).get("files", {})

    manual_paths: dict[str, Path | None] = {
        "train": _resolve_optional_path(manual.get("train", "")),
        "val": _resolve_optional_path(manual.get("val", "")),
        "test": _resolve_optional_path(manual.get("test", "")),
    }
    if _all_present(manual_paths):
        return {k: v for k, v in manual_paths.items() if v is not None}

    if cfg.get("auto_discover", True):
        discovered = _discover_cot_macro_full_files(dataset_name)
        if _all_present(discovered):
            return {k: v for k, v in discovered.items() if v is not None}

    missing = [k for k, v in manual_paths.items() if v is None]
    raise FileNotFoundError(
        "Failed to resolve external full macro files for dataset "
        f"'{dataset_name}'. Missing splits: {missing}. "
        "Set paths in ablation_study/configs/external_results.yml "
        "or ensure datasets/<dataset>/dataIntegration/*_set_with_macros.json exists."
    )


def resolve_final_sft_full_files(dataset_name: str) -> dict[str, Path]:
    cfg = _load_external_config().get("external_results", {})
    dataset_cfg = cfg.get("datasets", {}).get(dataset_name, {})
    manual = dataset_cfg.get("final_sft_full", {}).get("files", {})

    manual_paths: dict[str, Path | None] = {
        "train": _resolve_optional_path(manual.get("train", "")),
        "val": _resolve_optional_path(manual.get("val", "")),
        "test": _resolve_optional_path(manual.get("test", "")),
    }
    if _all_present(manual_paths):
        return {k: v for k, v in manual_paths.items() if v is not None}

    if cfg.get("auto_discover", True):
        discovered = _discover_final_sft_full_files(dataset_name)
        if _all_present(discovered):
            return {k: v for k, v in discovered.items() if v is not None}

    missing = [k for k, v in manual_paths.items() if v is None]
    raise FileNotFoundError(
        "Failed to resolve external full final_sft files for dataset "
        f"'{dataset_name}'. Missing splits: {missing}. "
        "Set paths in ablation_study/configs/external_results.yml "
        "or ensure datasets/<dataset>/preprocessed/final_sft_{split}.json exists."
    )
