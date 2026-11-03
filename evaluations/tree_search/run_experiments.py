"""Tree-search preset for the shared CEM/tree search batch runner."""
from dataclasses import asdict
from pathlib import Path
import sys

# Edit these settings, then run this file.
SCENES_TO_RUN = "tree_easy tree_hard"
VARIANTS_TO_RUN = "KSO MEKSO MEKSOTO TO"
RUNS_TO_RUN = "1 2 3 4 5"
TIME_PER_RUN = "2h"
OUTPUT_DIR = "results"         # Relative to this script, or an absolute path
BASE_SEED = 0

TREE_SEARCH_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(TREE_SEARCH_DIR.parents[1]))
    sys.path.insert(0, str(TREE_SEARCH_DIR.parents[1] / "src"))

from evaluations.search_batch import BatchConfig, run_batch
from evaluations.runner_utils import batch_settings, duration_seconds, selection


def settings_from_file() -> BatchConfig:
    settings = batch_settings(TREE_SEARCH_DIR, scenes=SCENES_TO_RUN, runs=RUNS_TO_RUN,
                              output_dir=OUTPUT_DIR)
    return BatchConfig(**asdict(settings),
        methods=tuple(f"TREE_SEARCH_{v}" for v in selection(VARIANTS_TO_RUN, "VARIANTS_TO_RUN")),
        timeout_s=duration_seconds(TIME_PER_RUN), base_seed=BASE_SEED)


def main() -> None:
    run_batch(settings_from_file())


if __name__ == "__main__":
    main()
