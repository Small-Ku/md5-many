#!/usr/bin/env python3
"""Turn Criterion paired-baseline estimates into a conservative CI gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def annotation(kind: str, title: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        safe = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::{kind} title={title}::{safe}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("criterion_root", type=Path)
    parser.add_argument("--warn-threshold", type=float, default=0.03)
    parser.add_argument("--fail-point", type=float, default=0.07)
    parser.add_argument("--fail-lower", type=float, default=0.05)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.criterion_root.glob("**/change/estimates.json")):
        benchmark = path.relative_to(args.criterion_root).as_posix().removesuffix(
            "/change/estimates.json"
        )
        # Reference implementations are useful context but are not md5-many
        # regression gates.
        if "rustcrypto" in benchmark.lower():
            continue

        data = json.loads(path.read_text())
        estimate = data["mean"]
        ci = estimate["confidence_interval"]
        point = float(estimate["point_estimate"])
        lower = float(ci["lower_bound"])
        upper = float(ci["upper_bound"])

        if point >= args.fail_point and lower >= args.fail_lower:
            status = "FAIL"
        elif point >= args.warn_threshold:
            status = "WARN"
        else:
            status = "PASS"

        rows.append(
            {
                "benchmark": benchmark,
                "point": point,
                "lower": lower,
                "upper": upper,
                "status": status,
            }
        )

    if not rows:
        print("No comparable md5-many Criterion results were found.", file=sys.stderr)
        annotation("error", "Performance guard", "No comparable Criterion results were found")
        return 1

    rows.sort(key=lambda row: float(row["point"]), reverse=True)
    failures = [row for row in rows if row["status"] == "FAIL"]
    warnings = [row for row in rows if row["status"] == "WARN"]

    lines = [
        "### Paired performance comparison",
        "",
        f"Compared {len(rows)} md5-many benchmarks on the same runner.",
        "Positive values mean slower execution time at HEAD.",
        "",
        "| Status | Benchmark | Mean change | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['status']} | `{row['benchmark']}` | {pct(float(row['point']))} | "
            f"{pct(float(row['lower']))} … {pct(float(row['upper']))} |"
        )

    lines.extend(
        [
            "",
            f"Guard: warn at >= {args.warn_threshold * 100:.1f}% mean slowdown; fail only when "
            f"mean slowdown is >= {args.fail_point * 100:.1f}% **and** the 95% CI lower bound is "
            f">= {args.fail_lower * 100:.1f}%.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    print(summary)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(summary)

    for row in warnings:
        annotation(
            "warning",
            "Possible performance regression",
            f"{row['benchmark']}: mean {pct(float(row['point']))}, "
            f"95% CI {pct(float(row['lower']))}..{pct(float(row['upper']))}",
        )
    for row in failures:
        annotation(
            "error",
            "Performance regression",
            f"{row['benchmark']}: mean {pct(float(row['point']))}, "
            f"95% CI {pct(float(row['lower']))}..{pct(float(row['upper']))}",
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")

    if failures and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
