import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.io import dump_json
from ablation_study.common.paths import OUTPUTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment cost for ablation stages.")
    parser.add_argument("--dataset_name", choices=["nyc", "tky", "ca"])
    parser.add_argument("--preference_hours", type=float, default=0.0)
    parser.add_argument("--fine_hours", type=float, default=5.0)
    parser.add_argument("--final_llm_hours", type=float, default=5.0)
    parser.add_argument("--preference_api_cost", type=float, default=0.0)
    parser.add_argument("--final_llm_gpu_cost", type=float, default=0.0)
    args = parser.parse_args()

    payload = {
        "dataset": args.dataset_name or "all",
        "preference_generation": {
            "hours": args.preference_hours,
            "api_cost": args.preference_api_cost,
        },
        "fine_training": {
            "hours": args.fine_hours,
        },
        "final_llm_training": {
            "hours": args.final_llm_hours,
            "gpu_cost": args.final_llm_gpu_cost,
        },
    }
    out_path = OUTPUTS_ROOT / "tables" / f"cost_summary_{payload['dataset']}.json"
    dump_json(payload, out_path)
    print({"output": str(out_path)})


if __name__ == "__main__":
    main()
