"""Summarize tree batches using the settings in run_experiments.py."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

from evaluations.search_stats import main as summarize
from evaluations.tree_search.run_experiments import settings_from_file


def main() -> None:
    summarize(settings_from_file())


if __name__ == "__main__":
    main()
