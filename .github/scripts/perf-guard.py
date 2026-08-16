#!/usr/bin/env python3
"""Turn ABBA Criterion measurements into a conservative CI performance gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


def pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def annotation(kind: str, title: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        safe = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::{kind} title={title}::{safe}")


def load_direction(root: Path, *, reverse: bool) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for path in sorted(root.glob("**/change/estimates.json")):
        benchmark = path.relative_to(root).as_posix().removesuffix("/change/estimates.json")
        if "rustcrypto" in benchmark.lower():
            continue

        data = json.loads(path.read_text())
        estimate = data["mean"]
        ci = estimate["confidence_interval"]
        point = float(estimate["point_estimate"])
        lower = float(ci["lower_bound"])
        upper = float(ci["upper_bound"])

        if reverse:
            # Criterion measured base/head - 1. Convert it back to
            # candidate/base - 1. Inversion reverses the confidence bounds.
            if min(point, lower, upper) <= -1.0:
                raise ValueError(f"invalid reverse ratio for {benchmark}")
            point = 1.0 / (1.0 + point) - 1.0
            lower, upper = (
                1.0 / (1.0 + upper) - 1.0,
                1.0 / (1.0 + lower) - 1.0,
            )

        rows[benchmark] = {"point": point, "lower": lower, "upper": upper}
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--warn-threshold", type=float, default=0.03)
    parser.add_argument("--fail-point", type=float, default=0.07)
    parser.add_argument("--fail-lower", type=float, default=0.05)
    parser.add_argument("--order-noise", type=float, default=0.05)
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    forward = load_direction(args.report_root / "forward", reverse=False)
    reverse = load_direction(args.report_root / "reverse", reverse=True)
    benchmarks = sorted(set(forward) & set(reverse))
    if not benchmarks:
        print("No benchmarks had both forward and reverse Criterion results.", file=sys.stderr)
        annotation("error", "Performance guard", "No bidirectional Criterion results were found")
        return 1

    same_sha = bool(args.base_sha and args.head_sha and args.base_sha == args.head_sha)
    rows: list[dict[str, object]] = []
    for benchmark in benchmarks:
        fwd = forward[benchmark]
        rev = reverse[benchmark]
        # The geometric mean is symmetric in the two measurement orders and
        # cancels smooth multiplicative clock/frequency drift better than an
        # arithmetic average of percentage changes.
        abba = math.sqrt((1.0 + fwd["point"]) * (1.0 + rev["point"])) - 1.0
        disagreement = abs(fwd["point"] - rev["point"])

        confirmed_fail = (
            fwd["point"] >= args.fail_point
            and fwd["lower"] >= args.fail_lower
            and rev["point"] >= args.fail_point
            and rev["lower"] >= args.fail_lower
        )
        confirmed_warn = (
            fwd["point"] >= args.warn_threshold and rev["point"] >= args.warn_threshold
        )

        if same_sha and (abs(abba) >= args.warn_threshold or disagreement >= args.order_noise):
            status = "NOISY"
        elif confirmed_fail:
            status = "FAIL"
        elif confirmed_warn:
            status = "WARN"
        elif disagreement >= args.order_noise:
            status = "NOISY"
        else:
            status = "PASS"

        rows.append(
            {
                "benchmark": benchmark,
                "abba": abba,
                "disagreement": disagreement,
                "forward": fwd,
                "reverse": rev,
                "status": status,
            }
        )

    rows.sort(key=lambda row: float(row["abba"]), reverse=True)
    failures = [row for row in rows if row["status"] == "FAIL"]
    warnings = [row for row in rows if row["status"] == "WARN"]
    noisy = [row for row in rows if row["status"] == "NOISY"]

    lines = ["### ABBA paired performance comparison", ""]
    if args.base_sha and args.head_sha:
        lines.append(f"Base `{args.base_sha[:12]}` → candidate `{args.head_sha[:12]}`.")
        lines.append("")
    if same_sha:
        lines.extend(
            [
                "**Calibration mode:** baseline and candidate are the same commit. Any apparent change is measurement noise.",
                "",
            ]
        )
    lines.extend(
        [
            f"Compared {len(rows)} md5-many benchmarks in both measurement orders on the same runner.",
            "Positive values mean slower execution time at HEAD. `ABBA` is the geometric mean of the two order-normalized ratios.",
            "",
            "| Status | Benchmark | ABBA | Forward 95% CI | Reverse 95% CI | Order spread |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        fwd = row["forward"]
        rev = row["reverse"]
        lines.append(
            f"| {row['status']} | `{row['benchmark']}` | {pct(float(row['abba']))} | "
            f"{pct(float(fwd['lower']))} … {pct(float(fwd['upper']))} | "
            f"{pct(float(rev['lower']))} … {pct(float(rev['upper']))} | "
            f"{pct(float(row['disagreement']))} |"
        )

    lines.extend(
        [
            "",
            f"Guard: warn only when **both orders** have >= {args.warn_threshold * 100:.1f}% mean slowdown; fail only when both orders have >= {args.fail_point * 100:.1f}% mean slowdown and both 95% CI lower bounds are >= {args.fail_lower * 100:.1f}%.",
            f"An order-to-order point-estimate spread >= {args.order_noise * 100:.1f}% is reported as `NOISY` instead of being treated as a regression by itself.",
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
            f"{row['benchmark']}: ABBA {pct(float(row['abba']))}; slowdown appears in both orders",
        )
    for row in noisy:
        annotation(
            "warning",
            "Noisy performance comparison",
            f"{row['benchmark']}: ABBA {pct(float(row['abba']))}; order spread {pct(float(row['disagreement']))}",
        )
    for row in failures:
        annotation(
            "error",
            "Performance regression",
            f"{row['benchmark']}: ABBA {pct(float(row['abba']))}; both measurement orders confirm the regression",
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")

    if failures and not args.no_fail and not same_sha:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
