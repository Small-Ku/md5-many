#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <base-manifest> <head-manifest> <filter1|filter2|...|ALL>" >&2
  exit 2
fi

base_manifest=$1
head_manifest=$2
filter_spec=$3
base_target=${PERF_BASE_TARGET:?PERF_BASE_TARGET is required}
head_target=${PERF_HEAD_TARGET:?PERF_HEAD_TARGET is required}
perf_cpu=${PERF_CPU:?PERF_CPU is required}
baseline=${PERF_BASELINE_NAME:-ci-base}
sample_size=${PERF_SAMPLE_SIZE:-20}
warmup=${PERF_WARMUP_SECONDS:-1}
measurement=${PERF_MEASUREMENT_SECONDS:-2}

if [[ $filter_spec == ALL ]]; then
  filters=("")
else
  IFS='|' read -r -a filters <<< "$filter_spec"
fi

criterion_args=(
  --noplot
  --color never
  --sample-size "$sample_size"
  --warm-up-time "$warmup"
  --measurement-time "$measurement"
  --confidence-level 0.95
  --noise-threshold 0.02
)

run_bench() {
  local manifest=$1
  local target=$2
  local mode=$3
  local filter=$4
  local -a args=()

  if [[ -n $filter ]]; then
    args+=("$filter")
  fi
  args+=("${criterion_args[@]}" "$mode" "$baseline")

  echo "::group::$(basename "$(dirname "$manifest")") filter=${filter:-ALL} ${mode} ${baseline}"
  CARGO_TARGET_DIR="$target" \
    taskset -c "$perf_cpu" \
    cargo bench --locked --manifest-path "$manifest" --bench throughput -- "${args[@]}"
  echo "::endgroup::"
}

# Compile both revisions before taking measurements so build activity does not
# occur between the paired benchmark runs.
CARGO_TARGET_DIR="$base_target" cargo bench --locked --manifest-path "$base_manifest" --bench throughput --no-run
CARGO_TARGET_DIR="$head_target" cargo bench --locked --manifest-path "$head_manifest" --bench throughput --no-run

rm -rf "$base_target/criterion" "$head_target/criterion"

for filter in "${filters[@]}"; do
  run_bench "$base_manifest" "$base_target" --save-baseline "$filter"
done

# Criterion stores named baselines under target/criterion. Keep build outputs
# isolated between revisions, but copy only the sampled baseline data so the
# head run can compare against measurements from the same VM.
mkdir -p "$head_target/criterion"
cp -a "$base_target/criterion/." "$head_target/criterion/"

for filter in "${filters[@]}"; do
  # Lenient comparison lets a PR introduce a new benchmark while still
  # comparing every benchmark that exists in both revisions.
  run_bench "$head_manifest" "$head_target" --baseline-lenient "$filter"
done
