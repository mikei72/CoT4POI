import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.config import load_analysis_config
from ablation_study.common.io import dump_json, load_json
from ablation_study.common.paths import OUTPUTS_ROOT


def _match_cot_substage(row: dict, table_name: str) -> bool:
    if not table_name.startswith("cot_"):
        return True
    norm_path = str(row.get("path", "")).replace("\\", "/").lower()
    if table_name == "cot_macro_table":
        return "/cot/" in norm_path and "/macro/" in norm_path
    if table_name == "cot_fine_table":
        return "/cot/" in norm_path and "/fine/" in norm_path
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready tables from collected metrics.")
    parser.add_argument("--table_name", required=True)
    args = parser.parse_args()

    analysis_cfg = load_analysis_config("default")
    table_cfg = analysis_cfg["tables"][args.table_name]
    if args.table_name.startswith("cot_"):
        summary = load_json(OUTPUTS_ROOT / "tables" / "cot_metrics_summary.json")
    else:
        summary = load_json(OUTPUTS_ROOT / "tables" / "final_llm_metrics_summary.json")

    rows = []
    for row in summary:
        if row.get("dataset") not in table_cfg["datasets"]:
            continue
        if row.get("variant") not in table_cfg["variants"]:
            continue
        if row.get("split") != "test":
            continue
        if not _match_cot_substage(row, args.table_name):
            continue
        rows.append({key: row.get(key) for key in ["dataset", "variant", *table_cfg["metrics"]]})

    out_path = OUTPUTS_ROOT / "tables" / f"{args.table_name}.json"
    dump_json(rows, out_path)
    print({"table_name": args.table_name, "rows": len(rows), "output": str(out_path)})


if __name__ == "__main__":
    main()
