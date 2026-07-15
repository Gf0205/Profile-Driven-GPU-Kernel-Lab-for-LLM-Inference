# Results

Raw benchmark CSV files and profiler traces are ignored and should not be
committed. The normal workflow is:

1. Run with `--no-write` on AutoDL RTX 4090.
2. Return the complete terminal output to Codex.
3. Check correctness and timing stability before interpreting performance.
4. Commit only the curated table and analysis in the study README.
