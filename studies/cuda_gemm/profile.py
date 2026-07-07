from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

# Avoid shadowing the Python standard-library `profile` module when torch
# imports cProfile internally. Prefer running `profiler.py` directly.
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.cuda_gemm.profiler import main  # noqa: E402


if __name__ == "__main__":
    main()

