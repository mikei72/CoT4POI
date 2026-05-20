from __future__ import annotations

from pathlib import Path

from ablation_study.common.cot_helpers import process_train_json, process_txt_file
from ablation_study.common.io import dump_json
from ablation_study.common.paths import ensure_dir


def build_processed_splits(dataset_root: Path, output_dir: Path, use_time_discretization: bool) -> dict[str, Path]:
    preprocessed_dir = dataset_root / "preprocessed"
    ensure_dir(output_dir)

    outputs = {
        "train": output_dir / "train_set.json",
        "val": output_dir / "val_set.json",
        "test": output_dir / "test_set.json",
    }

    dump_json(
        process_train_json(preprocessed_dir / "train_qa_pairs_kqt.json", use_time_discretization=use_time_discretization),
        outputs["train"],
    )
    dump_json(
        process_txt_file(preprocessed_dir / "val_qa_pairs_kqt.txt", use_time_discretization=use_time_discretization),
        outputs["val"],
    )
    dump_json(
        process_txt_file(preprocessed_dir / "test_qa_pairs_kqt.txt", use_time_discretization=use_time_discretization),
        outputs["test"],
    )
    return outputs
