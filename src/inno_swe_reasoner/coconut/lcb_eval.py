from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from inno_swe_reasoner.utils.logger import get_logger


def _find_first_available_module(module_names: list[str]) -> str | None:
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is not None:
            return module_name
    return None


def _find_repo_root_from_spec(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    module_path = Path(spec.origin).resolve()
    # Walk up to find the lcb_runner package, then use its parent as repo root.
    for parent in module_path.parents:
        if (parent / "lcb_runner").is_dir():
            return parent
    return module_path.parent


def run_lcb_custom_evaluator(
    custom_output_file: Path,
    evaluator_module: str | None = None,
    extra_args: list[str] | None = None,
) -> None:
    logger = get_logger()
    modules_to_try = (
        [evaluator_module]
        if evaluator_module
        else [
            "lcb_runner.runner.custom_evaluator",
            "livecodebench.runner.custom_evaluator",
        ]
    )

    module_name = _find_first_available_module(modules_to_try)
    if module_name is None:
        raise RuntimeError(
            "LiveCodeBench custom evaluator module not found. "
            "Install LiveCodeBench and/or set lcb_custom_evaluator_module in config."
        )
    repo_root = _find_repo_root_from_spec(module_name)
    custom_output_file = custom_output_file.resolve()
    cmd = [
        sys.executable,
        "-m",
        module_name,
        "--custom_output_file",
        str(custom_output_file),
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running LiveCodeBench custom evaluator...")
    logger.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root)
