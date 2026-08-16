#!/usr/bin/env bash
set -euo pipefail

out=${1:-perf-runner.txt}

{
  echo "# Runner fingerprint"
  echo
  echo "## uname"
  uname -a
  echo
  echo "## lscpu"
  lscpu
  echo
  echo "## rustc"
  rustc -Vv
  echo
  echo "## cargo"
  cargo -V
} | tee "$out"

perf_cpu=$(python3 - <<'PY'
import os
cpus = sorted(os.sched_getaffinity(0))
if not cpus:
    raise SystemExit("runner exposes no schedulable CPUs")
print(cpus[0])
PY
)

echo "PERF_CPU=$perf_cpu" | tee -a "$out"
if [[ -n ${GITHUB_ENV:-} ]]; then
  echo "PERF_CPU=$perf_cpu" >> "$GITHUB_ENV"
fi

if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
  {
    echo "### Runner fingerprint"
    echo
    echo '```text'
    grep -E '^(Architecture:|Vendor ID:|Model name:|CPU family:|Model:|Flags:|PERF_CPU=)' "$out" || true
    echo '```'
  } >> "$GITHUB_STEP_SUMMARY"
fi
