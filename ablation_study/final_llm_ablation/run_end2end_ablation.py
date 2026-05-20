import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablation_study.common.config import load_stage_config
from ablation_study.common.io import dump_json
from ablation_study.common.paths import PROJECT_ROOT, ensure_dir, output_root
from ablation_study.common.registry import FINAL_LLM_VARIANTS
from ablation_study.common.runtime import run_command


def _fmt_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end final LLM ablation.")
    parser.add_argument("--dataset_name", required=True, choices=["nyc"])
    parser.add_argument(
        "--variant",
        required=True,
        choices=["full", "w_o_fine", "w_o_macro", "w_o_preference", "w_o_td"],
    )
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--training_script", default="supervised-fine-tune-qlora.py")
    parser.add_argument("--torchrun_bin", default="torchrun")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--lora_output_dir", default=None, help="Reuse existing LoRA checkpoint directory for evaluation.")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_stage_config("final_llm", args.dataset_name)
    training_cfg = config["training"]
    model_name_or_path = args.model_name_or_path or training_cfg["model_name_or_path"]

    data_dir = output_root("final_llm", args.dataset_name, args.variant, "data")
    run_dir = output_root("final_llm", args.dataset_name, args.variant, "run")
    eval_dir = ensure_dir(run_dir / "eval")
    metrics_path = run_dir / "metrics.json"
    train_output_root = Path(str(training_cfg.get("output_dir", run_dir))).resolve()
    train_output_dir = train_output_root / "ablation" / args.variant
    ensure_dir(train_output_dir)
    lora_output_dir = Path(args.lora_output_dir) if args.lora_output_dir else train_output_dir

    summary = {
        "dataset": args.dataset_name,
        "variant": args.variant,
        "description": FINAL_LLM_VARIANTS[args.variant].description,
        "data_dir": _fmt_path(data_dir),
        "train_output_dir": _fmt_path(train_output_dir),
        "lora_output_dir": _fmt_path(lora_output_dir.resolve()),
    }

    if not args.skip_train:
        command = [
            args.torchrun_bin,
            "--nproc_per_node",
            str(args.nproc_per_node),
            args.training_script,
            "--model_name_or_path",
            model_name_or_path,
            "--bf16",
            "True",
            "--output_dir",
            str(train_output_dir),
            "--model_max_length",
            str(training_cfg["model_max_length"]),
            "--use_flash_attn",
            "True" if training_cfg["use_flash_attn"] else "False",
            "--data_path",
            str(data_dir / "final_sft_train.json"),
            "--eval_data_path",
            str(data_dir / "final_sft_val.json"),
            "--low_rank_training",
            "True" if training_cfg["low_rank_training"] else "False",
            "--num_train_epochs",
            str(training_cfg["num_train_epochs"]),
            "--per_device_train_batch_size",
            str(training_cfg["per_device_train_batch_size"]),
            "--per_device_eval_batch_size",
            str(training_cfg["per_device_eval_batch_size"]),
            "--gradient_accumulation_steps",
            str(training_cfg["gradient_accumulation_steps"]),
            "--evaluation_strategy",
            "steps",
            "--eval_steps",
            str(training_cfg["eval_steps"]),
            "--save_strategy",
            "steps",
            "--save_steps",
            str(training_cfg["save_steps"]),
            "--save_total_limit",
            str(training_cfg["save_total_limit"]),
            "--load_best_model_at_end",
            "True",
            "--metric_for_best_model",
            "eval_loss",
            "--greater_is_better",
            "False",
            "--learning_rate",
            str(training_cfg["learning_rate"]),
            "--weight_decay",
            "0.0",
            "--warmup_steps",
            str(training_cfg["warmup_steps"]),
            "--lr_scheduler_type",
            str(training_cfg["lr_scheduler_type"]),
            "--logging_steps",
            str(training_cfg["logging_steps"]),
            "--deepspeed",
            str(PROJECT_ROOT / training_cfg["deepspeed"]),
            "--tf32",
            "True",
        ]
        summary["train_return_code"] = run_command(command, cwd=PROJECT_ROOT, dry_run=args.dry_run)

    if not args.skip_eval:
        from ablation_study.final_llm_ablation.evaluate import evaluate_sft_json

        eval_metrics = evaluate_sft_json(
            data_path=data_dir / "final_sft_test.json",
            model_path=model_name_or_path,
            lora_output_dir=str(lora_output_dir),
            output_path=eval_dir / "evaluation_results.json",
            flash_attn=args.flash_attn or training_cfg["use_flash_attn"],
        )
        summary["eval_metrics"] = eval_metrics

    dump_json(summary, metrics_path)
    print(summary)


if __name__ == "__main__":
    main()
