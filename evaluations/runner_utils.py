"""Small batch-runner utilities; native solvers are imported by the runners."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re


def selection(value: str, name: str) -> tuple[str, ...]:
    items = tuple(value.replace(",", " ").split())
    if not items or len(set(items)) != len(items):
        raise ValueError(f"{name} must be a nonempty selection without duplicates")
    return items


def scene_selection(value: str) -> tuple[str, ...]:
    scenes = tuple(s if "." in s else f"faro.{s}"
                   for s in selection(value, "SCENES_TO_RUN"))
    if len(set(scenes)) != len(scenes) or any(
        re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", s) is None for s in scenes
    ):
        raise ValueError("SCENES_TO_RUN must contain distinct dotted scene IDs or short Faro IDs")
    return scenes


def run_selection(value: str) -> tuple[int, ...]:
    try:
        runs = tuple(int(r) for r in selection(value, "RUNS_TO_RUN"))
    except ValueError as exc:
        raise ValueError("RUNS_TO_RUN must contain distinct positive integer run IDs") from exc
    if any(r < 1 for r in runs) or len(set(runs)) != len(runs):
        raise ValueError("RUNS_TO_RUN must contain distinct positive integer run IDs")
    return runs


def duration_seconds(value: str) -> float:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh])\s*", value.lower())
    if match is None:
        raise ValueError("TIME_PER_RUN must be a duration such as '30s', '10m', or '2h'")
    seconds = float(match[1]) * {"s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("TIME_PER_RUN must be finite and positive")
    return seconds


def resolve_directory(value: str | Path, directory: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


@dataclass(frozen=True)
class BatchSettings:
    scenes: tuple[str, ...]
    runs: tuple[int, ...]
    output_dir: Path

    @property
    def solver_cache_dir(self) -> Path:
        return self.output_dir / "solver_cache"


def batch_settings(directory: Path, *, scenes: str, runs: str,
                   output_dir: str | Path) -> BatchSettings:
    return BatchSettings(scene_selection(scenes), run_selection(runs),
                         resolve_directory(output_dir, directory))


@contextmanager
def working_directory(path: Path):
    """Create and enter the directory used for native solver artifacts."""
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
