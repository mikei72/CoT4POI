from pathlib import Path


ABLAT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ABLAT_ROOT.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
OUTPUTS_ROOT = ABLAT_ROOT / "outputs"
CONFIGS_ROOT = ABLAT_ROOT / "configs"


def dataset_root(dataset_name: str) -> Path:
    return DATASETS_ROOT / dataset_name


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_root(*parts: str) -> Path:
    return ensure_dir(OUTPUTS_ROOT.joinpath(*parts))


def split_output_dir(dataset_name: str, stage: str, variant: str, split: str | None = None) -> Path:
    root = output_root(stage, dataset_name, variant)
    if split:
        return ensure_dir(root / split)
    return root
