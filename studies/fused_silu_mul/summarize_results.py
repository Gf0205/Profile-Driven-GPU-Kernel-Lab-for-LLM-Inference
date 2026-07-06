from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Fused SiLU-Mul benchmark CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def fmt(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["shape_name"]][row["provider"]] = row

    lines = [
        "| Shape | Regime | PyTorch ms | compile ms | Triton ms | Triton speedup vs PyTorch | Triton gap vs compile | Triton max_diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for shape_name, providers in grouped.items():
        pytorch = providers.get("pytorch_unfused")
        compiled = providers.get("torch_compile")
        triton = providers.get("triton")
        if not (pytorch and compiled and triton):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    shape_name,
                    triton["regime"],
                    fmt(pytorch["latency_ms"], 4),
                    fmt(compiled["latency_ms"], 4),
                    fmt(triton["latency_ms"], 4),
                    fmt(triton["speedup_vs_pytorch"], 2) + "x",
                    fmt(triton["gap_vs_torch_compile"], 2) + "x",
                    f"{float(triton['max_diff']):.3e}",
                ]
            )
            + " |"
        )

    summary = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(summary)


if __name__ == "__main__":
    main()

