import argparse
import asyncio
from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.config import load_stage_config
from ablation_study.common.cot_helpers import load_category_mapping, normalize_category
from ablation_study.common.external_results import resolve_cot_macro_full_files
from ablation_study.common.io import dump_json, load_json
from ablation_study.common.metrics import format_metrics, ranking_metrics
from ablation_study.common.paths import PROJECT_ROOT, ensure_dir, output_root
from ablation_study.common.registry import COT_MACRO_VARIANTS
from ablation_study.cot_ablation.pipeline.data_processing import build_processed_splits
from ablation_study.cot_ablation.pipeline.stages import flags_for_macro_variant


async def _augment_preferences_for_splits(
    split_paths: dict[str, Path],
    preference_dir: Path,
    model_name: str,
    max_concurrent_requests: int,
) -> dict[str, Path]:
    from ablation_study.cot_ablation.pipeline.preference import add_preferences_to_file

    updated_paths = dict(split_paths)
    for split, src_path in split_paths.items():
        add_path = preference_dir / f"{split}.json"
        print(f"[macro] preference stage: split={split}, input={src_path}, output={add_path}")
        stats = await add_preferences_to_file(
            input_path=src_path,
            output_path=add_path,
            model_name=model_name,
            max_concurrent_requests=max_concurrent_requests,
        )
        print(f"[macro] preference done: split={split}, stats={stats}\n")
        updated_paths[split] = add_path
    return updated_paths


def evaluate_existing_macro_file(path: Path) -> dict:
    data = load_json(path)
    ranks: list[int | None] = []
    for item in data:
        macro_dict = item.get("macro", {}) or {}
        labels = list(macro_dict.keys())
        true_macro = item.get("result", {}).get("macro_category")
        rank = None
        if true_macro:
            try:
                rank = labels.index(str(true_macro).lower())
            except ValueError:
                rank = None
        ranks.append(rank)
    return format_metrics(ranking_metrics(ranks, ks=(1, 3, 5)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run macro-category ablation.")
    parser.add_argument("--dataset_name", required=True, choices=["nyc", "tky", "ca"])
    parser.add_argument("--variant", required=True, choices=sorted(COT_MACRO_VARIANTS))
    parser.add_argument("--model_name", default="cross-encoder/nli-roberta-base")
    parser.add_argument("--preference_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--max_concurrent_requests", type=int, default=10)
    parser.add_argument("--prefer_external_full", action="store_true", default=True)
    parser.add_argument("--no_prefer_external_full", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_stage_config("cot", args.dataset_name)
    flags = flags_for_macro_variant(args.variant)
    print(f"[macro] start dataset={args.dataset_name}, variant={args.variant}, flags={flags.__dict__}")
    if args.dry_run:
        print(
            {
                "dataset": args.dataset_name,
                "variant": args.variant,
                "description": COT_MACRO_VARIANTS[args.variant].description,
                "flags": flags.__dict__,
            }
        )
        return

    dataset_root = PROJECT_ROOT / config["dataset"]["root_dir"]
    variant_root = output_root("cot", args.dataset_name, "macro", args.variant)
    processed_dir = ensure_dir(variant_root / "processed")
    macro_dir = ensure_dir(variant_root / "predictions")
    metrics_path = variant_root / "metrics.json"
    prefer_external_full = args.prefer_external_full and (not args.no_prefer_external_full)
    map_file = dataset_root / "dataIntegration" / "corrected_categories_with_macro.csv"

    if args.variant == "full" and prefer_external_full:
        try:
            external_files = resolve_cot_macro_full_files(args.dataset_name)
            fine_to_macro = {}
            if map_file.exists():
                for macro_name, fine_list in load_category_mapping(map_file).items():
                    for fine in fine_list:
                        fine_to_macro[normalize_category(fine)] = macro_name

            split_metrics = {}
            artifacts = {}
            for split in ["train", "val", "test"]:
                dst_path = macro_dir / f"{split}.json"
                shutil.copy2(external_files[split], dst_path)
                if fine_to_macro:
                    data = load_json(dst_path)
                    for item in data:
                        item["result"]["macro_category"] = fine_to_macro.get(normalize_category(item["result"]["category_name"]))
                    dump_json(data, dst_path)
                split_metrics[split] = evaluate_existing_macro_file(dst_path)
                artifacts[split] = str(dst_path.relative_to(PROJECT_ROOT))

            dump_json(
                {
                    "dataset": args.dataset_name,
                    "variant": args.variant,
                    "description": COT_MACRO_VARIANTS[args.variant].description,
                    "metrics": split_metrics,
                    "artifacts": artifacts,
                    "source_mode": "external_full_macro",
                },
                metrics_path,
            )
            print(f"Reused external full macro outputs for {args.dataset_name}: {artifacts}")
            return
        except FileNotFoundError:
            pass

    split_paths = build_processed_splits(
        dataset_root=dataset_root,
        output_dir=processed_dir,
        use_time_discretization=flags.use_time_discretization,
    )
    print(f"[macro] processed splits ready: {split_paths}")

    if flags.use_preference:
        preference_dir = ensure_dir(variant_root / "preference")
        split_paths = asyncio.run(
            _augment_preferences_for_splits(
                split_paths=split_paths,
                preference_dir=preference_dir,
                model_name=args.preference_model,
                max_concurrent_requests=args.max_concurrent_requests,
            )
        )

    if map_file.exists():
        fine_to_macro = {}
        for macro_name, fine_list in load_category_mapping(map_file).items():
            for fine in fine_list:
                fine_to_macro[normalize_category(fine)] = macro_name

        for split, path in split_paths.items():
            print(f"[macro] mapping fine->macro for split={split}, path={path}\n")
            data = load_json(path)
            for item in data:
                item["result"]["macro_category"] = fine_to_macro.get(normalize_category(item["result"]["category_name"]))
            dump_json(data, path)

    from ablation_study.cot_ablation.pipeline.macro_stage import initialize_classifier, run_macro_file

    classifier = initialize_classifier(args.model_name)
    split_metrics = {}
    for split, src_path in split_paths.items():
        output_path = macro_dir / f"{split}.json"
        print(f"[macro] infer split={split}, input={src_path}, output={output_path}, use_preference={flags.use_preference}")
        split_metrics[split] = run_macro_file(classifier, src_path, output_path, use_preference=flags.use_preference)
        print(f"[macro] infer done split={split}, metrics={split_metrics[split]}\n")

    dump_json(
        {
            "dataset": args.dataset_name,
            "variant": args.variant,
            "description": COT_MACRO_VARIANTS[args.variant].description,
            "metrics": split_metrics,
            "artifacts": {split: str((macro_dir / f"{split}.json").relative_to(PROJECT_ROOT)) for split in split_paths},
        },
        metrics_path,
    )
    print(f"Saved macro ablation outputs to {variant_root}")


if __name__ == "__main__":
    main()
