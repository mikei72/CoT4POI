"""Static deployment checks for environments without a Docker CLI."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_DOCKERFILES = {
    "Dockerfile.api": {"requirements.lock", "pyproject.toml", "README.md", "src"},
    "Dockerfile.demo": {"requirements.lock", "pyproject.toml", "README.md", "src"},
}
_REQUIRED_IGNORES = {
    "/.git",
    "/.venv",
    "/.env",
    "/.env.*",
    "/datasets",
    "/model",
    "/experiment",
    "/artifacts",
    "/mlruns",
    "/monitoring/events",
    "/monitoring_events",
}
_SECRET_NAME = re.compile(r"(?:^|_)(?:API_KEY|KEY|PASSWORD|SECRET|TOKEN)(?:$|_)", re.I)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


class StaticDeployError(ValueError):
    """Raised when a deployment invariant is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticDeployError(message)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StaticDeployError(f"cannot parse YAML {path.name}: {exc}") from exc
    _require(isinstance(document, Mapping), f"{path.name} must contain a YAML mapping")
    return document


def _logical_dockerfile_lines(text: str) -> tuple[str, ...]:
    logical: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current = f"{current} {stripped}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical.append(current)
        current = ""
    _require(not current, "Dockerfile ends with an incomplete continuation")
    return tuple(logical)


def _copy_sources(lines: Sequence[str]) -> set[str]:
    sources: set[str] = set()
    for line in lines:
        instruction, _, value = line.partition(" ")
        if instruction.upper() == "ADD":
            raise StaticDeployError("ADD is forbidden; use narrow local COPY instructions")
        if instruction.upper() != "COPY":
            continue
        try:
            tokens = shlex.split(value)
        except ValueError as exc:
            raise StaticDeployError("COPY instruction cannot be parsed") from exc
        _require(len(tokens) >= 2, "COPY must include a source and destination")
        _require(not tokens[0].startswith("--from="), "multi-stage COPY is not expected here")
        for source in tokens[:-1]:
            normalized = source.rstrip("/")
            _require(normalized not in {"", ".", ".."}, "COPY must not include the full context")
            _require(not normalized.startswith("/"), "COPY source must be context-relative")
            sources.add(normalized)
    return sources


def _check_dockerfiles(root: Path) -> str:
    for filename, allowed_sources in _DOCKERFILES.items():
        path = root / filename
        _require(path.is_file(), f"missing {filename}")
        text = path.read_text(encoding="utf-8")
        lines = _logical_dockerfile_lines(text)
        sources = _copy_sources(lines)
        _require(sources == allowed_sources, f"{filename} has unexpected COPY sources: {sources}")
        _require(
            "python -m pip install -r requirements.lock" in text,
            f"{filename} must install requirements.lock",
        )
        _require(
            "python -m pip install -e . --no-deps" in text,
            f"{filename} must install the project without dependency re-resolution",
        )
        user_lines = [line for line in lines if line.partition(" ")[0].upper() == "USER"]
        _require(user_lines, f"{filename} must set a non-root USER")
        _require(user_lines[-1].split(maxsplit=1)[1] != "root", f"{filename} ends as root")
        lower = text.lower()
        _require("cuda" not in lower and "nvidia" not in lower, f"{filename} must be CPU-only")
        _require(not _SECRET_NAME.search(text), f"{filename} must not bake secret variables")
        _require("HEALTHCHECK" in text, f"{filename} must define an image healthcheck")
        if filename == "Dockerfile.api":
            _require(
                "127.0.0.1:8000/ready" in text,
                "Dockerfile.api must validate bundle readiness",
            )
        else:
            _require(
                "127.0.0.1:8501/_stcore/health" in text,
                "Dockerfile.demo must validate Streamlit health",
            )
    return "Dockerfiles use locked installs, narrow COPY, CPU images, and non-root users"


def _check_context(root: Path) -> str:
    path = root / ".dockerignore"
    _require(path.is_file(), "missing .dockerignore")
    patterns = {
        line.strip().rstrip("/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(_REQUIRED_IGNORES - patterns)
    _require(not missing, f".dockerignore is missing protected paths: {missing}")
    return "Docker context excludes VCS, environments, data, weights, and runtime artifacts"


def _service(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    services = document.get("services")
    _require(isinstance(services, Mapping), "Compose services must be a mapping")
    service = services.get(name)
    _require(isinstance(service, Mapping), f"Compose service {name!r} is missing")
    return service


def _command_text(service: Mapping[str, Any]) -> str:
    command = service.get("command")
    if isinstance(command, str):
        return command
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        return " ".join(command)
    raise StaticDeployError("service command must be a string or string list")


def _volume(service: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    volumes = service.get("volumes")
    _require(isinstance(volumes, list), "service volumes must use long-form list syntax")
    matches = [
        item
        for item in volumes
        if isinstance(item, Mapping) and item.get("target") == target
    ]
    _require(len(matches) == 1, f"expected exactly one volume targeting {target}")
    return matches[0]


def _healthcheck_text(service: Mapping[str, Any]) -> str:
    healthcheck = service.get("healthcheck")
    _require(isinstance(healthcheck, Mapping), "service healthcheck is missing")
    test = healthcheck.get("test")
    _require(isinstance(test, list), "healthcheck test must use list syntax")
    _require(all(isinstance(item, str) for item in test), "healthcheck test must be strings")
    return " ".join(test)


def _check_compose_services(root: Path) -> str:
    document = _load_yaml(root / "docker-compose.yml")
    services = document.get("services")
    _require(isinstance(services, Mapping), "Compose services must be a mapping")
    _require(
        set(services) == {"api", "demo", "mlflow"},
        "Compose service set must be api/demo/mlflow",
    )
    for name, dockerfile in {
        "api": "Dockerfile.api",
        "demo": "Dockerfile.demo",
        "mlflow": "Dockerfile.api",
    }.items():
        service = _service(document, name)
        build = service.get("build")
        _require(isinstance(build, Mapping), f"{name} build must be a mapping")
        _require(build.get("context") == ".", f"{name} build context must be repository root")
        _require(build.get("dockerfile") == dockerfile, f"{name} uses the wrong Dockerfile")
        _require("args" not in build, f"{name} must not pass image build arguments")
        _require((root / dockerfile).is_file(), f"referenced {dockerfile} does not exist")

    api_command = _command_text(_service(document, "api"))
    demo_command = _command_text(_service(document, "demo"))
    mlflow_command = _command_text(_service(document, "mlflow"))
    _require("next_poi.serving.app:app" in api_command, "api must run the canonical app")
    _require("uvicorn" in api_command, "api command must run Uvicorn")
    _require(
        "streamlit run src/next_poi/demo/app.py" in demo_command,
        "demo must run the HTTP-only Streamlit app",
    )
    _require("mlflow server" in mlflow_command, "mlflow must run its local UI server")
    _require("file:///mlruns" in mlflow_command, "mlflow must use the local file store")
    return "Compose defines the three CPU services and canonical commands"


def _check_compose_volumes(root: Path) -> str:
    document = _load_yaml(root / "docker-compose.yml")
    api = _service(document, "api")
    demo = _service(document, "demo")
    mlflow = _service(document, "mlflow")

    bundle = _volume(api, "/opt/next-poi/bundle")
    _require(bundle.get("type") == "bind", "bundle must be an explicit local bind")
    _require(bundle.get("read_only") is True, "bundle volume must be read-only")
    _require(
        bundle.get("source")
        == "${NEXT_POI_BUNDLE_HOST_PATH:-./artifacts/smoke/b3/bundle}",
        "bundle source must match the evaluation output layout",
    )
    monitoring = _volume(api, "/var/lib/next-poi/monitoring")
    _require(monitoring.get("type") == "volume", "monitoring must use a named volume")
    _require(monitoring.get("read_only") is not True, "monitoring volume must be writable")
    api_environment = api.get("environment")
    _require(isinstance(api_environment, Mapping), "api environment must be a mapping")
    _require(
        api_environment.get("NEXT_POI_MONITORING_PATH")
        == "/var/lib/next-poi/monitoring/events.jsonl",
        "api monitoring path must point inside its writable volume",
    )
    mlruns = _volume(mlflow, "/mlruns")
    _require(mlruns.get("type") == "volume", "MLflow must use a named volume")
    _require(mlruns.get("read_only") is not True, "MLflow volume must be writable")
    _require(not demo.get("volumes"), "demo must not mount data, bundles, or artifacts")

    serialized = json.dumps(document, sort_keys=True).lower()
    _require("datasets/nyc" not in serialized, "Compose must not mount repository NYC data")
    top_level = document.get("volumes")
    _require(isinstance(top_level, Mapping), "Compose named volumes must be declared")
    _require({"monitoring", "mlruns"}.issubset(top_level), "named volumes are incomplete")
    return "Bundle is read-only; monitoring and MLflow stores are isolated writable volumes"


def _check_compose_network_and_health(root: Path) -> str:
    document = _load_yaml(root / "docker-compose.yml")
    api = _service(document, "api")
    demo = _service(document, "demo")
    mlflow = _service(document, "mlflow")
    depends_on = demo.get("depends_on")
    _require(isinstance(depends_on, Mapping), "demo must depend on api readiness")
    api_dependency = depends_on.get("api")
    _require(isinstance(api_dependency, Mapping), "demo api dependency must be explicit")
    _require(
        api_dependency.get("condition") == "service_healthy",
        "demo must wait for a healthy api",
    )
    _require(set(depends_on) == {"api"}, "demo may depend only on api")
    environment = demo.get("environment")
    _require(isinstance(environment, Mapping), "demo environment must be a mapping")
    _require(
        environment.get("NEXT_POI_API_BASE_URL") == "http://api:8000",
        "demo must connect to the Compose api service",
    )
    _require("127.0.0.1:8000/ready" in _healthcheck_text(api), "api ready check is invalid")
    _require(
        "127.0.0.1:8501/_stcore/health" in _healthcheck_text(demo),
        "demo healthcheck is invalid",
    )
    _require(
        "127.0.0.1:5000/health" in _healthcheck_text(mlflow),
        "mlflow healthcheck is invalid",
    )
    return "Demo depends only on healthy API; every service has an in-container healthcheck"


def _check_compose_ports_and_secrets(root: Path) -> str:
    document = _load_yaml(root / "docker-compose.yml")
    expected = {"api": "8000", "demo": "8501", "mlflow": "5000"}
    for name, container_port in expected.items():
        service = _service(document, name)
        ports = service.get("ports")
        _require(isinstance(ports, list) and len(ports) == 1, f"{name} must publish one port")
        _require(
            isinstance(ports[0], str) and ports[0].endswith(f":{container_port}"),
            f"{name} publishes the wrong container port",
        )
        _require("secrets" not in service, f"{name} must not bake or mount repository secrets")
        _require("env_file" not in service, f"{name} must not load a repository .env file")
        environment = service.get("environment", {})
        _require(isinstance(environment, Mapping), f"{name} environment must be a mapping")
        secret_names = sorted(str(key) for key in environment if _SECRET_NAME.search(str(key)))
        _require(not secret_names, f"{name} includes secret environment names: {secret_names}")

    serialized = json.dumps(document, sort_keys=True).lower()
    _require("nvidia" not in serialized and "cuda" not in serialized, "Compose must be CPU-only")
    _require('"gpus"' not in serialized, "Compose must not request GPUs")
    return "Ports are explicit and no image args, env files, secret mounts, or GPU runtime are used"


def _check_command_references(root: Path) -> str:
    required_paths = {
        "src/next_poi/serving/app.py",
        "src/next_poi/demo/app.py",
        "requirements.lock",
        "pyproject.toml",
        "README.md",
        "src/next_poi/evaluation/__main__.py",
        "src/next_poi/serving/smoke_test.py",
        "tests/fixtures/synthetic/train.csv",
        "tests/fixtures/synthetic/validation.csv",
        "tests/fixtures/synthetic/test.csv",
    }
    missing = sorted(path for path in required_paths if not (root / path).is_file())
    _require(not missing, f"deployment commands reference missing paths: {missing}")
    return "Docker and Compose command references resolve to repository files"


def _check_environment_example(root: Path) -> str:
    path = root / ".env.example"
    _require(path.is_file(), "missing .env.example")
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        _require(separator == "=", f".env.example line {line_number} is not an assignment")
        _require(name and name not in entries, f"duplicate .env.example variable: {name}")
        entries[name] = value

    legacy = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "SILICONFLOW_API_KEY",
        "SILICONFLOW_BASE_URL",
    }
    _require(legacy.issubset(entries), ".env.example must preserve all four legacy API variables")
    _require(
        all(entries[name] == "" for name in legacy),
        "legacy API variables must remain empty in .env.example",
    )
    expected_production = {
        "NEXT_POI_BUNDLE_HOST_PATH": "./artifacts/smoke/b3/bundle",
        "NEXT_POI_API_PORT": "8000",
        "NEXT_POI_DEMO_PORT": "8501",
        "NEXT_POI_MLFLOW_PORT": "5000",
    }
    _require(
        all(entries.get(name) == value for name, value in expected_production.items()),
        "production environment examples do not match Compose defaults",
    )
    unexpected_secrets = sorted(
        name for name, value in entries.items() if _SECRET_NAME.search(name) and value
    )
    _require(not unexpected_secrets, f".env.example contains secret values: {unexpected_secrets}")
    return "Legacy API variables are preserved empty and production defaults contain no secrets"


def _workflow_run_text(workflow: Mapping[str, Any]) -> str:
    jobs = workflow.get("jobs")
    _require(isinstance(jobs, Mapping), "CI jobs must be a mapping")
    _require(set(jobs) == {"cpu-synthetic"}, "CI must contain only the CPU synthetic job")
    job = jobs["cpu-synthetic"]
    _require(isinstance(job, Mapping), "CPU synthetic job must be a mapping")
    _require(job.get("runs-on") == "ubuntu-latest", "CI must run on a CPU GitHub runner")
    steps = job.get("steps")
    _require(isinstance(steps, list), "CI steps must be a list")
    setup_steps = [
        step
        for step in steps
        if isinstance(step, Mapping)
        and str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    _require(len(setup_steps) == 1, "CI must have one Python setup step")
    setup = setup_steps[0].get("with")
    _require(isinstance(setup, Mapping), "Python setup options must be a mapping")
    _require(str(setup.get("python-version")) == "3.10", "CI Python must be 3.10")
    run_commands = [
        step.get("run") for step in steps if isinstance(step, Mapping) and "run" in step
    ]
    _require(
        all(isinstance(command, str) for command in run_commands),
        "every CI run command must be a string",
    )
    return "\n".join(run_commands)


def _check_ci(root: Path) -> str:
    workflow = _load_yaml(root / ".github/workflows/ci.yml")
    commands = _workflow_run_text(workflow)
    required_fragments = {
        "python -m pip install -r requirements.lock",
        "python -m pip install -e . --no-deps",
        "python -m ruff check src tests scripts",
        "python -m compileall -q src tests scripts",
        "python -m pytest -q",
        "python scripts/static_deploy_check.py",
        "python -m next_poi.evaluation",
        "--dataset synthetic",
        "--train tests/fixtures/synthetic/train.csv",
        "--validation tests/fixtures/synthetic/validation.csv",
        "--test tests/fixtures/synthetic/test.csv",
        "--output artifacts/smoke",
        "--variant b3",
        "python -m next_poi.serving.smoke_test",
        "--bundle artifacts/smoke/b3/bundle",
    }
    missing = sorted(fragment for fragment in required_fragments if fragment not in commands)
    _require(not missing, f"CI is missing local gate commands: {missing}")
    lower_commands = commands.lower()
    forbidden = sorted(
        token for token in ("docker ", "nvidia", "cuda", "full-gpu") if token in lower_commands
    )
    _require(not forbidden, f"CI includes forbidden dynamic/GPU commands: {forbidden}")
    return "CI matches local lint/compile/test/static and fixture-to-API CPU gates"


def _run_check(name: str, operation: Callable[[], str]) -> CheckResult:
    try:
        detail = operation()
    except (OSError, StaticDeployError) as exc:
        return CheckResult(name=name, passed=False, detail=str(exc))
    return CheckResult(name=name, passed=True, detail=detail)


def run_checks(root: str | Path) -> tuple[CheckResult, ...]:
    """Run deterministic static checks without invoking Docker."""

    repository_root = Path(root).resolve()
    checks: tuple[tuple[str, Callable[[Path], str]], ...] = (
        ("dockerfiles", _check_dockerfiles),
        ("docker_context", _check_context),
        ("compose_services", _check_compose_services),
        ("compose_volumes", _check_compose_volumes),
        ("compose_dependencies_health", _check_compose_network_and_health),
        ("compose_ports_secrets_cpu", _check_compose_ports_and_secrets),
        ("command_references", _check_command_references),
        ("environment_example", _check_environment_example),
        ("ci_cpu_gate", _check_ci),
    )
    return tuple(
        _run_check(name, lambda operation=operation: operation(repository_root))
        for name, operation in checks
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args(argv)
    results = run_checks(args.root)
    passed = sum(result.passed for result in results)
    payload = {
        "status": "passed" if passed == len(results) else "failed",
        "passed": passed,
        "failed": len(results) - passed,
        "checks": [asdict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
