from __future__ import annotations

from pathlib import Path

from ablation_study.common.external_results import (
    resolve_cot_fine_full_files,
    resolve_cot_macro_full_files,
    resolve_final_sft_full_files,
)
from ablation_study.common.paths import OUTPUTS_ROOT

SPLITS = ("train", "val", "test")


def _ensure_exists(mapping: dict[str, Path], label: str) -> dict[str, Path]:
    missing = [split for split in SPLITS if split not in mapping or not mapping[split].exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files for splits: {missing}")
    return mapping


def _ablation_split_files(stage: str, dataset_name: str, variant: str, leaf: str) -> dict[str, Path]:
    base = OUTPUTS_ROOT / "cot" / dataset_name / stage / variant / leaf
    files = {split: base / f"{split}.json" for split in SPLITS}
    return _ensure_exists(files, f"ablation {stage}/{variant}/{leaf}")


def resolve_macro_variant_files(dataset_name: str, variant: str, prefer_external_full: bool = True) -> dict[str, Path]:
    if variant == "full" and prefer_external_full:
        return resolve_cot_macro_full_files(dataset_name)
    return _ablation_split_files(stage="macro", dataset_name=dataset_name, variant=variant, leaf="predictions")


def resolve_fine_variant_files(dataset_name: str, variant: str, prefer_external_full: bool = True) -> dict[str, Path]:
    if variant == "full" and prefer_external_full:
        return resolve_cot_fine_full_files(dataset_name)
    return _ablation_split_files(stage="fine", dataset_name=dataset_name, variant=variant, leaf="fine")


def resolve_final_sft_files(dataset_name: str, prefer_external_full: bool = True) -> dict[str, Path]:
    if prefer_external_full:
        return resolve_final_sft_full_files(dataset_name)
    base = OUTPUTS_ROOT / "final_llm" / dataset_name / "full" / "data"
    files = {split: base / f"final_sft_{split}.json" for split in SPLITS}
    return _ensure_exists(files, f"ablation final_sft full data for {dataset_name}")

