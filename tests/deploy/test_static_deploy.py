from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.static_deploy_check import run_checks

REPOSITORY_ROOT = Path(__file__).parents[2]
DEPLOYMENT_FILES = (
    "Dockerfile.api",
    "Dockerfile.demo",
    ".dockerignore",
    "docker-compose.yml",
    ".env.example",
    ".github/workflows/ci.yml",
    "requirements.lock",
    "pyproject.toml",
    "README.md",
)


def _copy_deployment_files(tmp_path: Path) -> Path:
    for relative_path in DEPLOYMENT_FILES:
        source = REPOSITORY_ROOT / relative_path
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for relative_path in (
        "src/next_poi/serving/app.py",
        "src/next_poi/serving/smoke_test.py",
        "src/next_poi/demo/app.py",
        "src/next_poi/evaluation/__main__.py",
        "tests/fixtures/synthetic/train.csv",
        "tests/fixtures/synthetic/validation.csv",
        "tests/fixtures/synthetic/test.csv",
    ):
        source = REPOSITORY_ROOT / relative_path
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return tmp_path


def _failed_names(root: Path) -> set[str]:
    return {result.name for result in run_checks(root) if not result.passed}


def test_live_deployment_configuration_passes_all_static_checks() -> None:
    results = run_checks(REPOSITORY_ROOT)

    assert results
    assert all(result.passed for result in results), results


def test_yaml_parse_failure_is_reported(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    (root / "docker-compose.yml").write_text("services: [\n", encoding="utf-8")

    failures = _failed_names(root)

    assert "compose_services" in failures
    assert "compose_volumes" in failures


def test_broad_docker_copy_is_rejected(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    dockerfile = root / "Dockerfile.api"
    dockerfile.write_text(
        dockerfile.read_text(encoding="utf-8") + "\nCOPY . /leaked-context\n",
        encoding="utf-8",
    )

    assert "dockerfiles" in _failed_names(root)


def test_missing_protected_context_pattern_is_rejected(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    dockerignore = root / ".dockerignore"
    dockerignore.write_text(
        dockerignore.read_text(encoding="utf-8").replace("/datasets\n", ""),
        encoding="utf-8",
    )

    assert "docker_context" in _failed_names(root)


def test_writable_bundle_volume_is_rejected(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    compose = root / "docker-compose.yml"
    compose.write_text(
        compose.read_text(encoding="utf-8").replace("        read_only: true\n", ""),
        encoding="utf-8",
    )

    assert "compose_volumes" in _failed_names(root)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ('      NEXT_POI_BUNDLE: /opt/next-poi/bundle\n', '      API_KEY: baked\n'),
        ('      - "${NEXT_POI_API_PORT:-8000}:8000"', '      - "9000:9000"'),
    ),
)
def test_secret_or_wrong_port_is_rejected(tmp_path: Path, old: str, new: str) -> None:
    root = _copy_deployment_files(tmp_path)
    compose = root / "docker-compose.yml"
    compose.write_text(
        compose.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    assert "compose_ports_secrets_cpu" in _failed_names(root)


def test_demo_must_depend_only_on_healthy_api(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    compose = root / "docker-compose.yml"
    compose.write_text(
        compose.read_text(encoding="utf-8").replace(
            "        condition: service_healthy", "        condition: service_started"
        ),
        encoding="utf-8",
    )

    assert "compose_dependencies_health" in _failed_names(root)


def test_api_healthcheck_must_validate_bundle_readiness(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    compose = root / "docker-compose.yml"
    compose.write_text(
        compose.read_text(encoding="utf-8").replace(
            "http://127.0.0.1:8000/ready", "http://127.0.0.1:8000/health"
        ),
        encoding="utf-8",
    )

    assert "compose_dependencies_health" in _failed_names(root)


def test_api_image_healthcheck_must_validate_bundle_readiness(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    dockerfile = root / "Dockerfile.api"
    dockerfile.write_text(
        dockerfile.read_text(encoding="utf-8").replace(
            "http://127.0.0.1:8000/ready", "http://127.0.0.1:8000/health"
        ),
        encoding="utf-8",
    )

    assert "dockerfiles" in _failed_names(root)


def test_api_monitoring_path_must_use_writable_volume(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    compose = root / "docker-compose.yml"
    compose.write_text(
        compose.read_text(encoding="utf-8").replace(
            "/var/lib/next-poi/monitoring/events.jsonl", "/tmp/events.jsonl"
        ),
        encoding="utf-8",
    )

    assert "compose_volumes" in _failed_names(root)


def test_missing_command_target_is_rejected(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    (root / "src/next_poi/demo/app.py").unlink()

    assert "command_references" in _failed_names(root)


def test_legacy_environment_variables_must_remain_empty(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    environment = root / ".env.example"
    environment.write_text(
        environment.read_text(encoding="utf-8").replace("OPENAI_API_KEY=", "OPENAI_API_KEY=x"),
        encoding="utf-8",
    )

    assert "environment_example" in _failed_names(root)


def test_ci_must_keep_fixture_api_closure(tmp_path: Path) -> None:
    root = _copy_deployment_files(tmp_path)
    workflow = root / ".github/workflows/ci.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "python -m next_poi.serving.smoke_test", "python -m next_poi.serving.not_a_gate"
        ),
        encoding="utf-8",
    )

    assert "ci_cpu_gate" in _failed_names(root)
