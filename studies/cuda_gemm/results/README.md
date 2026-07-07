# Results

Do not commit raw AutoDL result files by default. The normal workflow is:

1. Run `benchmark.py --no-write` on AutoDL RTX 3090.
2. Copy the terminal output back to Codex.
3. Update local summary docs from the returned output.
4. Commit only curated summary/analysis files.

