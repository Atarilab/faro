# FARO evaluations

Contact-planning experiments. See [installation](../install.md); use `faro_env`.

Edit settings at the top of the runner and run the file.
Use the adjacent `get_stats.py` to summarize results.

| Runner | Default workload | Sequential runtime |
| --- | --- | --- |
| [LLM](llm/run_experiments.py) | 4 scenes × 5 runs × 100 plans | ~50 hours |
| [Tree search](tree_search/run_experiments.py) | 2 scenes × 4 variants × 5 runs | ~80 hours |
| [AHMP CEM](ahmp/run_experiments.py) | 2 scenes × 2 methods × 5 runs | ~40 hours |

Search totals use `TIME_PER_RUN = "2h"`; setup and solver overruns add time.
LLM timing comes from saved runs and depends on hardware.

For parallel execution, split scenes, methods or run IDs and give each process
its own `OUTPUT_DIR`. Four tree-search processes, one variant each, have
~20 hours of search each, assuming sufficient CPU and memory.

[Human-input examples](human_input/) solve and visualize individual contact sequences.
