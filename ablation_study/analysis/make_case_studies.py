import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.config import load_analysis_config
from ablation_study.common.io import dump_json, load_json
from ablation_study.common.paths import OUTPUTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Build case-study artifacts for the ablation report.")
    parser.add_argument("--dataset_name", required=True, choices=["nyc", "tky", "ca"])
    args = parser.parse_args()

    cfg = load_analysis_config("default")["case_study"]
    fine_path = OUTPUTS_ROOT / "cot" / args.dataset_name / "fine" / "full" / "fine" / "test.json"
    rows = load_json(fine_path)
    success_cases = []
    failure_cases = []
    for item in rows:
        true_cat = item["result"]["category_name"]
        predicted = list(item.get("fine", {}).keys())
        if predicted and predicted[0] == true_cat and len(success_cases) < cfg["num_success_cases"]:
            success_cases.append(item)
        elif predicted and predicted[0] != true_cat and len(failure_cases) < cfg["num_failure_cases"]:
            failure_cases.append(item)
        if len(success_cases) >= cfg["num_success_cases"] and len(failure_cases) >= cfg["num_failure_cases"]:
            break
    output = {"dataset": args.dataset_name, "success_cases": success_cases, "failure_cases": failure_cases}
    out_path = OUTPUTS_ROOT / "cases" / f"{args.dataset_name}_case_studies.json"
    dump_json(output, out_path)
    print({"output": str(out_path), "success": len(success_cases), "failure": len(failure_cases)})


if __name__ == "__main__":
    main()
