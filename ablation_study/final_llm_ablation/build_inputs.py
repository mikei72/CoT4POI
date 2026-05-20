import argparse
from pathlib import Path
import sys
import shutil

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.config import load_stage_config
from ablation_study.common.cot_helpers import build_sft_examples, load_raw_split
from ablation_study.common.external_results import resolve_cot_fine_full_files, resolve_final_sft_full_files
from ablation_study.common.io import dump_json, load_json
from ablation_study.common.paths import PROJECT_ROOT, ensure_dir, output_root

def main() -> None:
    parser = argparse.ArgumentParser(description="Build final LLM inputs for ablation variants.")
    parser.add_argument("--dataset_name", required=True, choices=["nyc", "tky", "ca"])
    parser.add_argument("--variant", required=True, choices=["full", "w_o_fine", "w_o_macro", "w_o_preference", "w_o_td"])
    parser.add_argument("--source_root", default="ablation_study/outputs/cot")
    parser.add_argument("--prefer_external_full", action="store_true", default=False)
    parser.add_argument("--no_prefer_external_full", action="store_true")
    args = parser.parse_args()

    load_stage_config("final_llm", args.dataset_name)
    dataset_root = PROJECT_ROOT / "datasets" / args.dataset_name
    source_root = PROJECT_ROOT / args.source_root / args.dataset_name
    output_dir = output_root("final_llm", args.dataset_name, args.variant, "data")
    prefer_external_full = args.prefer_external_full and (not args.no_prefer_external_full)

    variant_sources = {
        "full": {
            "sem_variant": source_root / "fine" / "full" / "fine",
            "include_preference": True,
            "include_macro": True,
            "include_fine": True,
        },
        "w_o_fine": {
            "sem_variant": source_root / "fine" / "full" / "fine",
            "include_preference": True,
            "include_macro": True,
            "include_fine": False,
        },
        "w_o_macro": {
            "sem_variant": source_root / "fine" / "w_o_macro" / "fine",
            "include_preference": True,
            "include_macro": False,
            "include_fine": True,
        },
        "w_o_preference": {
            "sem_variant": source_root / "fine" / "w_o_preference" / "fine",
            "include_preference": False,
            "include_macro": True,
            "include_fine": True,
        },
        "w_o_td": {
            "sem_variant": source_root / "fine" / "w_o_td" / "fine",
            "include_preference": True,
            "include_macro": True,
            "include_fine": True,
        },
    }
    recipe = variant_sources[args.variant]
    ensure_dir(output_dir)

    external_sft_files = {}
    if args.variant in {"full", "w_o_fine"} and prefer_external_full:
        try:
            external_sft_files = resolve_final_sft_full_files(args.dataset_name)
        except FileNotFoundError:
            external_sft_files = {}

    if args.variant in {"full", "w_o_fine"} and prefer_external_full:
        try:
            external_files = resolve_cot_fine_full_files(args.dataset_name)
        except FileNotFoundError:
            external_files = {}
    elif args.variant in {"full", "w_o_fine"}:
        external_files = {}
    else:
        external_files = {}

    if args.variant == "full" and external_sft_files:
        written = {}
        for split in ["train", "val", "test"]:
            out_path = output_dir / f"final_sft_{split}.json"
            shutil.copy2(external_sft_files[split], out_path)
            written[split] = str(out_path.relative_to(PROJECT_ROOT))
        manifest = {
            "dataset": args.dataset_name,
            "variant": args.variant,
            "files": written,
            "source_mode": "external_full_sft",
        }
        dump_json(manifest, output_dir.parent / "manifest.json")
        print(manifest)
        return

    if args.variant == "w_o_fine" and external_sft_files:
        written = {}
        for split in ["train", "val", "test"]:
            rows = load_json(external_sft_files[split])
            for row in rows:
                row["aux_fine"] = "None"
            out_path = output_dir / f"final_sft_{split}.json"
            dump_json(rows, out_path)
            written[split] = str(out_path.relative_to(PROJECT_ROOT))
        manifest = {
            "dataset": args.dataset_name,
            "variant": args.variant,
            "files": written,
            "source_mode": "external_full_sft_mask_fine",
        }
        dump_json(manifest, output_dir.parent / "manifest.json")
        print(manifest)
        return

    written = {}
    for split in ["train", "val", "test"]:
        orig_data = load_raw_split(dataset_root, split)
        if external_files:
            sem_path = external_files[split]
        else:
            sem_path = recipe["sem_variant"] / f"{split}.json"
        sem_data = load_json(sem_path)
        merged = build_sft_examples(
            orig_data=orig_data,
            sem_data=sem_data,
            include_preference=recipe["include_preference"],
            include_macro=recipe["include_macro"],
            include_fine=recipe["include_fine"],
        )
        output_path = output_dir / f"final_sft_{split}.json"
        dump_json(merged, output_path)
        written[split] = str(output_path.relative_to(PROJECT_ROOT))

    manifest = {
        "dataset": args.dataset_name,
        "variant": args.variant,
        "files": written,
        "source_mode": "external_full" if external_files else "ablation_outputs",
    }
    dump_json(manifest, output_dir.parent / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()
