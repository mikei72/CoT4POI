import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.external_results import (
    resolve_cot_fine_full_files,
    resolve_cot_macro_full_files,
    resolve_final_sft_full_files,
)
from ablation_study.common.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Show whether external full results can be reused.")
    parser.add_argument("--dataset_name", required=True, choices=["nyc", "tky", "ca"])
    args = parser.parse_args()

    result = {"dataset": args.dataset_name}
    try:
        files = resolve_cot_macro_full_files(args.dataset_name)
        result["cot_macro_full_files"] = {k: str(v.relative_to(PROJECT_ROOT)) for k, v in files.items()}
    except Exception as exc:
        result["cot_macro_full_files_error"] = str(exc)

    try:
        files = resolve_cot_fine_full_files(args.dataset_name)
        result["cot_fine_full_files"] = {k: str(v.relative_to(PROJECT_ROOT)) for k, v in files.items()}
    except Exception as exc:
        result["cot_fine_full_files_error"] = str(exc)

    try:
        sft_files = resolve_final_sft_full_files(args.dataset_name)
        result["final_sft_full_files"] = {k: str(v.relative_to(PROJECT_ROOT)) for k, v in sft_files.items()}
    except Exception as exc:
        result["final_sft_full_files_error"] = str(exc)

    print(result)


if __name__ == "__main__":
    main()
