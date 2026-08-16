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
report_root=${PERF_REPORT_ROOT:?PERF_REPORT_ROOT is required}
perf_cpu=${PERF_CPU:?PERF_CPU is required}
baseline_prefix=${PERF_BASELINE_NAME:-ci-pair}
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
  local baseline=$4
  local filter=$5
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

clear_changes() {
  local target=$1
  if [[ -d $target/criterion ]]; then
    find "$target/criterion" -type d -name change -prune -exec rm -rf {} +
  fi
}

copy_named_baseline() {
  local source=$1
  local destination=$2
  local baseline=$3
  local count=0

  mkdir -p "$destination/criterion"
  while IFS= read -r -d '' path; do
    local rel=${path#"$source/criterion/"}
    local dest="$destination/criterion/$rel"
    mkdir -p "$(dirname "$dest")"
    rm -rf "$dest"
    cp -a "$path" "$dest"
    count=$((count + 1))
  done < <(find "$source/criterion" -type d -name "$baseline" -print0)

  if (( count == 0 )); then
    echo "no Criterion baseline named $baseline was produced" >&2
    return 1
  fi
}

collect_changes() {
  local source=$1
  local destination=$2
  local count=0

  mkdir -p "$destination"
  while IFS= read -r -d '' path; do
    local rel=${path#"$source/criterion/"}
    local dest="$destination/$rel"
    mkdir -p "$(dirname "$dest")"
    cp "$path" "$dest"
    count=$((count + 1))
  done < <(find "$source/criterion" -path '*/change/estimates.json' -print0)

  if (( count == 0 )); then
    echo "no comparable Criterion change estimates were produced" >&2
    exit 1
  fi
}

# Compile both revisions before taking measurements so build activity does not
# occur between the paired benchmark runs.
CARGO_TARGET_DIR="$base_target" cargo bench --locked --manifest-path "$base_manifest" --bench throughput --no-run
CARGO_TARGET_DIR="$head_target" cargo bench --locked --manifest-path "$head_manifest" --bench throughput --no-run

rm -rf "$base_target/criterion" "$head_target/criterion" "$report_root"
mkdir -p "$report_root/forward" "$report_root/reverse"

# Measure every filter in ABBA order. A cloud VM can change effective CPU
# frequency or steal time during a long job; a single base-then-head pass can
# therefore report a large false regression even when both revisions are the
# same commit. The reverse pass lets the guard require the slowdown to survive
# both measurement orders.
for index in "${!filters[@]}"; do
  filter=${filters[$index]}
  optional=false
  if [[ $filter == \?* ]]; then
    optional=true
    filter=${filter#\?}
  fi
  forward_baseline="${baseline_prefix}-f${index}"
  reverse_baseline="${baseline_prefix}-r${index}"

  # A: baseline revision. Hardware-specific sentinels may be explicitly marked
  # optional with a leading '?'. If the baseline produces no matching Criterion
  # data (for example an AVX-512-only benchmark on an AVX2 runner), skip that
  # filter. Required filters still fail loudly so typos or removed benchmarks
  # cannot silently weaken the regression guard.
  run_bench "$base_manifest" "$base_target" --save-baseline "$forward_baseline" "$filter"
  if ! copy_named_baseline "$base_target" "$head_target" "$forward_baseline"; then
    if [[ $optional == true ]]; then
      echo "::notice::Skipping optional performance filter '$filter': baseline produced no benchmark on this runner"
      if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
        echo "- SKIP \`$filter\`: baseline produced no benchmark on this runner." >> "$GITHUB_STEP_SUMMARY"
      fi
      continue
    fi
    echo "required performance filter '$filter' produced no Criterion baseline" >&2
    exit 1
  fi

  # B: candidate revision compared with A.
  clear_changes "$head_target"
  run_bench "$head_manifest" "$head_target" --baseline-lenient "$forward_baseline" "$filter"
  collect_changes "$head_target" "$report_root/forward"

  # B again, now as the baseline for the reverse-order measurement.
  run_bench "$head_manifest" "$head_target" --save-baseline "$reverse_baseline" "$filter"
  copy_named_baseline "$head_target" "$base_target" "$reverse_baseline"

  # A again. The guard mathematically inverts this base/head comparison back
  # into candidate/base orientation.
  clear_changes "$base_target"
  run_bench "$base_manifest" "$base_target" --baseline-lenient "$reverse_baseline" "$filter"
  collect_changes "$base_target" "$report_root/reverse"
done
