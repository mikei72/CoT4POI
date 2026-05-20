import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.io import dump_json, load_json
from ablation_study.common.paths import PROJECT_ROOT, output_root

def main() -> None:
    parser = argparse.ArgumentParser(description="Run input-masking analysis for the final LLM.")
    parser.add_argument("--dataset_name", required=True, choices=["nyc"])
    parser.add_argument(
        "--masked_field",
        required=True,
        choices=["preference", "macro", "fine"],
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lora_output_dir", required=True)
    parser.add_argument("--source_variant", default="full")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print({"dataset": args.dataset_name, "masked_field": args.masked_field})
        return

    source_dir = output_root("final_llm", args.dataset_name, args.source_variant, "data")
    output_dir = output_root("final_llm", args.dataset_name, "input_masking", args.masked_field)

    masked_files = {}
    for split in ["train", "val", "test"]:
        rows = load_json(source_dir / f"final_sft_{split}.json")
        for row in rows:
            if args.masked_field == "preference":
                row["system_preference"] = "No specific preference."
            elif args.masked_field == "macro":
                row["aux_macro"] = "None"
            elif args.masked_field == "fine":
                row["aux_fine"] = "None"
        out_path = output_dir / f"final_sft_{split}.json"
        dump_json(rows, out_path)
        masked_files[split] = str(out_path.relative_to(PROJECT_ROOT))

    from ablation_study.final_llm_ablation.evaluate import evaluate_sft_json

    metrics = evaluate_sft_json(
        data_path=output_dir / "final_sft_test.json",
        model_path=args.model_name_or_path,
        lora_output_dir=args.lora_output_dir,
        output_path=output_dir / "evaluation_results.json",
    )
    dump_json({"masked_field": args.masked_field, "files": masked_files, "metrics": metrics}, output_dir / "summary.json")
    print({"masked_field": args.masked_field, "metrics": metrics})


if __name__ == "__main__":
    main()
