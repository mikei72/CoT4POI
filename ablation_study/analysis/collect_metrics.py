import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.io import dump_csv_rows, dump_json, load_json
from ablation_study.common.paths import OUTPUTS_ROOT

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect metrics from ablation outputs.")
    parser.add_argument("--stage", required=True, choices=["cot", "final_llm"])
    args = parser.parse_args()

    stage_root = OUTPUTS_ROOT / args.stage
    rows = []
    for metrics_path in stage_root.rglob("metrics.json"):
        payload = load_json(metrics_path)
        base = {
            "stage": args.stage,
            "path": str(metrics_path),
            "dataset": payload.get("dataset", ""),
            "variant": payload.get("variant", ""),
        }
        metrics = payload.get("metrics") or {}
        if args.stage == "cot":
            for split, split_metrics in metrics.items():
                row = base | {"split": split}
                row.update(split_metrics)
                rows.append(row)
        else:
            row = base | {"split": "test"}
            row.update(payload.get("eval_metrics", {}))
            rows.append(row)

    output_json = OUTPUTS_ROOT / "tables" / f"{args.stage}_metrics_summary.json"
    output_csv = OUTPUTS_ROOT / "tables" / f"{args.stage}_metrics_summary.csv"
    dump_json(rows, output_json)
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        dump_csv_rows(rows, output_csv, fieldnames=fieldnames)
    print({"rows": len(rows), "json": str(output_json), "csv": str(output_csv)})


if __name__ == "__main__":
    main()
