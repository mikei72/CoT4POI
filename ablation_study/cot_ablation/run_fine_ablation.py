import argparse
import asyncio
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.config import load_stage_config
from ablation_study.common.artifacts import resolve_macro_variant_files
from ablation_study.common.external_results import resolve_cot_macro_full_files
from ablation_study.common.io import dump_json, load_json
from ablation_study.common.paths import PROJECT_ROOT, ensure_dir, output_root
from ablation_study.common.registry import COT_FINE_VARIANTS
from ablation_study.cot_ablation.pipeline.data_processing import build_processed_splits
from ablation_study.cot_ablation.pipeline.stages import flags_for_fine_variant


async def _augment_preferences_for_splits(
    split_paths: dict[str, Path],
    preference_dir: Path,
    model_name: str,
    max_concurrent_requests: int,
) -> dict[str, Path]:
    from ablation_study.cot_ablation.pipeline.preference import add_preferences_to_file

    updated_paths = dict(split_paths)
    for split, src_path in split_paths.items():
        out_path = preference_dir / f"{split}.json"
        print(f"[fine] preference stage: split={split}, input={src_path}, output={out_path}")
        stats = await add_preferences_to_file(
            input_path=src_path,
            output_path=out_path,
            model_name=model_name,
            max_concurrent_requests=max_concurrent_requests,
        )
        print(f"[fine] preference done: split={split}, stats={stats}")
        updated_paths[split] = out_path
    return updated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine-category ablation.")
    parser.add_argument("--dataset_name", required=True, choices=["nyc", "tky", "ca"])
    parser.add_argument("--variant", required=True, choices=sorted(COT_FINE_VARIANTS))
    parser.add_argument("--macro_model", default="cross-encoder/nli-roberta-base")
    parser.add_argument("--preference_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--reranker_model", default="BAAI/bge-reranker-base")
    parser.add_argument("--max_concurrent_requests", type=int, default=10)
    parser.add_argument("--prefer_external_full", action="store_true", default=True)
    parser.add_argument("--no_prefer_external_full", action="store_true")
    parser.add_argument("--skip_train", action="store_true", help="Skip fine-model training and reuse an existing model_dir for inference.")
    parser.add_argument("--model_dir", type=str, default=None, help="Existing fine model directory (e.g., .../checkpoints/best_model).")
    parser.add_argument("--force_train", action="store_true", help="Force training even for full variant.")
    parser.add_argument("--infer_seed", type=int, default=42, help="Random seed for fine inference candidate sampling.")
    parser.add_argument("--split_paths_file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--force_infer_same_process",
        action="store_true",
        help="Run inference in current process (default behavior is subprocess after training for stable GPU state).",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_stage_config("cot", args.dataset_name)
    flags = flags_for_fine_variant(args.variant)
    print(f"[fine] start dataset={args.dataset_name}, variant={args.variant}, flags={flags.__dict__}")
    if args.dry_run:
        print(
            {
                "dataset": args.dataset_name,
                "variant": args.variant,
                "description": COT_FINE_VARIANTS[args.variant].description,
                "flags": flags.__dict__,
            }
        )
        return

    dataset_root = PROJECT_ROOT / config["dataset"]["root_dir"]
    variant_root = output_root("cot", args.dataset_name, "fine", args.variant)
    processed_dir = ensure_dir(variant_root / "processed")
    preference_dir = ensure_dir(variant_root / "preference")
    fine_dir = ensure_dir(variant_root / "fine")
    metrics_path = variant_root / "metrics.json"
    prefer_external_full = args.prefer_external_full and (not args.no_prefer_external_full)
    skip_train = args.skip_train or (args.variant == "full" and not args.force_train)
    if args.split_paths_file:
        loaded = load_json(Path(args.split_paths_file))
        split_paths = {k: Path(v) for k, v in loaded.items()}
        print(f"[fine] reuse split paths from file: {args.split_paths_file} -> {split_paths}")
    else:
        # Chain-style reuse:
        # w_o_td      -> use macro w_o_td outputs
        # w_o_preference -> use macro w_o_preference outputs
        # w_o_macro   -> use macro full outputs (macro field will be ignored by fine model)
        # history_only -> build local no-TD/no-pref/no-macro data
        if args.variant == "w_o_td":
            split_paths = resolve_macro_variant_files(args.dataset_name, "w_o_td", prefer_external_full=prefer_external_full)
        elif args.variant == "w_o_preference":
            split_paths = resolve_macro_variant_files(args.dataset_name, "w_o_preference", prefer_external_full=prefer_external_full)
        elif args.variant == "w_o_macro":
            split_paths = resolve_macro_variant_files(args.dataset_name, "full", prefer_external_full=prefer_external_full)
        elif args.variant == "history_only":
            split_paths = build_processed_splits(dataset_root, processed_dir, use_time_discretization=False)
        elif args.variant == "full":
            try:
                split_paths = resolve_cot_macro_full_files(args.dataset_name)
                print(f"[fine] full variant reuse macro full files: {split_paths}")
            except FileNotFoundError:
                split_paths = build_processed_splits(dataset_root, processed_dir, use_time_discretization=flags.use_time_discretization)
                if flags.use_preference:
                    split_paths = asyncio.run(
                        _augment_preferences_for_splits(
                            split_paths=split_paths,
                            preference_dir=preference_dir,
                            model_name=args.preference_model,
                            max_concurrent_requests=args.max_concurrent_requests,
                        )
                    )
        else:
            split_paths = build_processed_splits(dataset_root, processed_dir, use_time_discretization=flags.use_time_discretization)
            if flags.use_preference:
                split_paths = asyncio.run(
                    _augment_preferences_for_splits(
                        split_paths=split_paths,
                        preference_dir=preference_dir,
                        model_name=args.preference_model,
                        max_concurrent_requests=args.max_concurrent_requests,
                    )
                )
    print(f"[fine] input splits ready: {split_paths}")

    from ablation_study.cot_ablation.pipeline.fine_model import FineTrainConfig, infer_fine_file, train_fine_model

    map_file = dataset_root / "dataIntegration" / "corrected_categories_with_macro.csv"
    train_config = FineTrainConfig(
        base_model=args.reranker_model,
        map_file=map_file,
        output_dir=ensure_dir(fine_dir / "checkpoints"),
    )
    if skip_train:
        if args.model_dir:
            model_dir = Path(args.model_dir)
        else:
            if args.variant == "full":
                # Full defaults to legacy checkpoint produced by main pipeline.
                model_dir = dataset_root / "experiment" / "checkpoints" / "best_model"
                if not model_dir.exists():
                    model_dir = train_config.output_dir / "best_model"
            else:
                # Other variants default to their own ablation checkpoint.
                model_dir = train_config.output_dir / "best_model"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Requested --skip_train but model_dir does not exist: {model_dir}. "
                "Pass --model_dir or run training once first."
            )
        print(f"[fine] skip training and reuse model_dir={model_dir}")
    else:
        model_dir = train_fine_model(
            train_file=split_paths["train"],
            val_file=split_paths["val"],
            use_preference=flags.use_preference,
            use_macro=flags.use_macro,
            config=train_config,
        )
        print(f"[fine] train done, model_dir={model_dir}")

        if not args.force_infer_same_process:
            split_paths_file = variant_root / "inference_split_paths.json"
            dump_json({k: str(v) for k, v in split_paths.items()}, split_paths_file)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset_name",
                args.dataset_name,
                "--variant",
                args.variant,
                "--skip_train",
                "--model_dir",
                str(model_dir),
                "--split_paths_file",
                str(split_paths_file),
                "--force_infer_same_process",
                "--infer_seed",
                str(args.infer_seed),
            ]
            if args.no_prefer_external_full:
                cmd.append("--no_prefer_external_full")
            elif args.prefer_external_full:
                cmd.append("--prefer_external_full")
            print("[fine] launch inference in fresh subprocess to avoid post-training GPU slowdown:")
            print("[fine] " + " ".join(cmd))
            subprocess.run(cmd, check=True)
            return

    split_metrics = {}
    artifacts = {}
    for split in ["train", "val", "test"]:
        output_path = fine_dir / f"{split}.json"
        print(
            f"[fine] infer split={split}, input={split_paths[split]}, output={output_path}, "
            f"use_preference={flags.use_preference}, use_macro={flags.use_macro}"
        )
        split_metrics[split] = infer_fine_file(
            model_dir=model_dir,
            input_file=split_paths[split],
            output_file=output_path,
            map_file=map_file,
            use_preference=flags.use_preference,
            use_macro=flags.use_macro,
            infer_seed=args.infer_seed,
        )
        print(f"[fine] infer done split={split}, metrics={split_metrics[split]}")
        artifacts[split] = str(output_path.relative_to(PROJECT_ROOT))

    dump_json(
        {
            "dataset": args.dataset_name,
            "variant": args.variant,
            "description": COT_FINE_VARIANTS[args.variant].description,
            "metrics": split_metrics,
            "artifacts": artifacts,
            "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
        },
        metrics_path,
    )
    print(f"Saved fine ablation outputs to {variant_root}")


if __name__ == "__main__":
    main()
