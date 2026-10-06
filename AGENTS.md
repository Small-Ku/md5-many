# AGENTS.md

Developer guidance for `md5-many`.

## Architecture

`md5-many` has two performance domains:

1. Single stream: MD5 blocks are sequentially dependent.
   - x86-64 uses the optimized backend in `src/scalar_x86_64.rs`.
   - `src/scalar.rs` remains the portable implementation and correctness oracle.
   - On preferred Intel AVX-512F/VL CPUs, the AVX-512VL one-shot loop keeps A/B/C/D in XMM state across consecutive blocks.
   - The RustCrypto block adapter converts scalar state only once per `compress_blocks` batch.
   - Keep the always-inlined vector-state core inside an AVX-512 target-feature context.
   - Do not call that core from a generic non-target-feature closure. Rust otherwise outlines target-feature intrinsic thunks and destroys performance.
2. Independent messages: `src/simd.rs` maps independent MD5 states to SIMD lanes through `fearless_simd`.

Specialized x86 kernels use 8 lanes for AVX2 and 16 lanes for AVX-512.
Two or three native groups can be interleaved round-by-round to hide one MD5 chain's dependency latency.
Four-way interleaving was benchmarked and rejected due to register/issue pressure. Do not reintroduce it without new evidence.

### Data layout

Inputs arrive message-major (AoS).
Native x86 kernels load one 64-byte block per message and transpose the 16 MD5 `u32` words into lane-major vectors (SoA).
Keep the whole message loop inside one `fearless_simd::kernel!` boundary. Per-block kernel transitions were measured to be extremely expensive.

### Equal-length scheduling

- AVX2: 8-way native, 16-way dual, 24-way triple.
- AVX-512: 16-way native, 32-way dual, 48-way triple.
- AArch64 on measured Neoverse-N2 hardware:
  - Production single-stream uses the portable Rust compressor rather than the hand-scheduled GPR path.
  - Equal-length batches use native NEON 4/8/12-way kernels. The 8/12-way groups are round-interleaved for ILP.
  - Prefer 12-way groups.
  - For a final 16-message region, use 8+8 rather than 12+4.
  - Measured under-filled groups may duplicate the final real lane:
  - 6/7/10/11/15 lanes do so from 55 B upward.
  - 5 lanes use the measured padding/alignment crossover.
  - 9 lanes use a more conservative alignment/padding/long-message crossover.
  - 13/14 lanes stay on the ordinary composition because padding to 16 regressed every measured point.
- Equal-length padding uses `build_padded_block` rather than byte-at-a-time synthesis.
- A pure padding block shared by every lane is parsed once and broadcast instead of loaded/transposed N times.
- Under-filled AVX2 dual/triple candidates duplicate a real lane rather than falling into a small tail:
  - 9-15 messages use padded dual.
  - 17-23 padded triple.
  - 26-31 equal/near-mixed batches use two dual kernels.
- Low-occupancy x86 dual-GPR scheduling is CPU-specific and always requires an explicit BMI1 CPUID guard:
  - AMD family 19h uses the measured tiny/1:16 two-message skew policy.
    - For three-message batches, pair the two longest only when the second-longest padded workload is <=1/4 of the longest.
  - Intel family 6/model `0xCF` uses dual-GPR for separately measured two-message equal/tiny/AVX-512-tail regions.
  - The same model also uses dual-GPR for exactly-two block-aligned incremental updates/finalization.
    - For three messages, it may pair the two longest when either the quarter-gap condition holds or the shortest padded workload is <=1/6 of the second-longest.
  - Other x86 CPUs retain the previous scalar/AVX2 crossover.
- For mixed 4–8-message AVX2 tails, allow the dynamic 2x-skew partitioner only when the long partition has at most two messages.
  - This crossover lets the recursive tail collapse to scalar/dual-scalar work.
  - Do not broaden it to one-short/many-long shapes. The extra partition can cost 17–25% at moderate skew.
- On measured AMD Family 19h AVX-512 hosts, short equal 9-16-message batches use two AVX2 chains up to 17 padded blocks.
  - Do not broaden this heuristic to other x86 families without measurements.
- AVX-512 hosts normally keep 2-8-message batches on AVX2. The measured x86 family 6/model `0xCF` crossover is an explicit exception:
  - Equal batches use padded ZMM from 512 B, or from 128 B when the length is a multiple of 64.
  - Mixed batches still require the shortest message to be at least 512 B.
  - Keep this model-specific unless another CPU is benchmarked directly.

### Mixed-length scheduling

- Process the common full-block prefix with the same native transpose/compression machinery.
- Build only divergent padded tails separately.
- On measured Neoverse-N2 AArch64, use the native NEON mixed kernel for consecutive 4/8/12-message chunks whose messages need the same total number of padded MD5 blocks.
  - Check the first four lanes once.
  - Extend the same-block-count run only to 8/12 lanes.
  - Do not repeatedly rescan rejected heterogeneous prefixes.
- Dual/triple mixed kernels interleave independent SIMD state chains just like equal-length kernels.
- The no-allocation skew planner also protects under-filled dual/triple and partial-tail fast paths.
  - If padded block counts differ by at least 2x, recursively partition short and long lanes.
  - Hash the sub-batches, then scatter digests back to the caller's order.
- AVX-512 17-31 and 33-47-message mixed/equal batches can use padded dual/triple kernels.
- Selected 50-63-message shapes stay in two AVX-512 dual kernels when this avoids a pathological tiny or large AVX2 tail.

## Module layout

```text
src/lib.rs            public API and tests
src/scalar.rs         portable scalar MD5 and dispatch wrapper
src/scalar_x86_64.rs  optimized x86-64 single-stream compression
src/simd.rs           SIMD dispatch, native x86 kernels, schedulers and fallback
src/block_api.rs      RustCrypto digest block adapter
src/consts.rs         MD5 IV, round constants and shifts
benches/throughput.rs user-facing Criterion performance suite
docs/performance.md   measured backend/scheduler evidence and rejected experiments
examples/probe.rs     detected native lane count
```

## Verification

Use the repository toolchain or normal Rust installation:

```bash
cargo test --locked
cargo test --locked --no-default-features --features libm
cargo test --locked --no-default-features --features libm,digest
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo package --locked
```

`fearless_simd` 0.7 requires either `std` or `libm`. Plain `--no-default-features` is not a valid dependency configuration.

For performance work:
- Compare before/after with the same Criterion benchmark name.
- Inspect release machine code when an optimization depends on a particular ISA instruction.
- Do not keep a change only because its algebraic rewrite looks plausible.

[`docs/performance.md`](docs/performance.md) owns the measurements and rejected experiments that justify current dispatch/scheduler choices.
Update it when new CPU evidence changes or extends a crossover.

Criterion groups protect specific crossovers:
- `x86-small-batch-*` protects the <=8-message AVX2/AVX-512 crossover.
- `x86-two-message-*` covers the dual-scalar pair path and skew guard.
- `x86-small-skew-*` covers the three-message quarter-gap, 4–8-message clustered-tail wins, and the one-short/many-long guard shape.

## GitHub Actions performance guard

### Performance candidate delivery (Scientific-Spiral)

Keep performance exploration local or on a branch while feasibility or value is unknown.
Open the first correct authority PR when all of these conditions hold:
- same-host measurements show a useful win;
- correctness is established;
- ISA feasibility is established;
- no known blocker is likely to overturn the direction.
Continue noise checks, portability, performance sentinels, cleanup, and review on that PR. PR-ready does not mean implementation-complete or accepted.
Treat PR commits as review and evidence carriers. Prefer additive corrections during review.
The maintainer accepts the final tree and squashes it into canonical history.

Performance sentinels live in `.github/workflows/ci.yml`.
They run downstream of the normal test matrix, quality/package checks, and MSRV checks through `needs`.
A broken or uncompilable change therefore never consumes benchmark runners.

Performance jobs compare baseline and candidate on the same GitHub-hosted VM. Do not compare absolute numbers across workflow runs.
Keep build outputs separate. Pin both revisions to the same schedulable CPU.

Each filter is measured in ABBA order (`base -> head -> head -> base`):
- Criterion produces a normal candidate/base comparison and a reverse base/head comparison.
- The guard inverts the reverse comparison back into candidate/base orientation.

A regression can fail CI only under these conditions:
- Both measurement orders independently show >=7% mean slowdown.
- Both 95% confidence-interval lower bounds are still >=5%.

Additional sentinel rules apply:
- Both orders at >=3% mean slowdown produce a warning.
- A >=5% disagreement between the two order-normalized point estimates is reported as `NOISY`, not as a regression by itself.
- The summary also reports the geometric mean of the two ratios, which is useful for cancelling smooth multiplicative frequency drift.
- RustCrypto reference benchmarks are reported by Criterion but excluded from the md5-many gate.

Run the sentinel suite on both `ubuntu-24.04` x86-64 and `ubuntu-24.04-arm` AArch64 only after correctness/MSRV/quality jobs succeed.
Pull requests compare the PR head against its base SHA.
Pushes to `master` and manual CI runs compare the candidate against the latest previous reachable release tag automatically.

A sentinel prefixed with `?` is explicitly hardware-optional:
- If the baseline produces no matching Criterion benchmark on that runner, the pair runner records a skip and continues.
- Keep this marker limited to benchmarks whose implementation intentionally returns early when the ISA is unavailable (currently `?x86-small-batch`, which requires AVX-512).
- Ordinary filters remain required so renamed, removed, or misspelled benchmarks fail CI instead of silently weakening coverage.

The workflow file `.github/workflows/performance.yml` is reserved for manual full-suite investigation:
- The `workflow_dispatch` trigger automatically selects the highest version-like release tag reachable from the candidate while excluding tags that point at the candidate itself.
- This tag selection naturally advances after each release.
- The `compare_ref` input remains an optional override.
- Manual full runs are non-blocking by default unless `enforce` is selected.

If baseline and candidate resolve to the same SHA, enter calibration mode and treat all apparent changes as runner noise.
Do not turn small cross-run throughput differences into hard gates.
Add or adjust a CI sentinel when a scheduler/backend crossover needs protection.

## Performance invariants

- Do not infer backend preference from ISA availability alone.
  - On measured AMD family 19h/Zen 4-class hardware, XMM-width AVX-512VL is substantially slower for a sequential MD5 stream.
  - On the same hardware, AVX-512 is substantially faster for sufficiently occupied `Md5Many` workloads.
  - Keep single-stream and multi-buffer dispatch policies independent.
- Keep single-stream AVX-512VL Intel-preferred unless direct same-host measurements establish a new CPU-specific exception.
- Keep the AMD family 19h low-occupancy overlap, quarter-gap, and `long_count <= 2` guards unless replacement measurements cover their known counterexamples.
- Treat hosted-runner CPU models as observations, not guarantees. Add CPU-specific dispatch only from direct measurements, and record the evidence in `docs/performance.md`.

## Invariants

- Edition 2024; declared MSRV Rust 1.89.
- `#![no_std]` core; `std` is feature-gated.
- `#![deny(unsafe_op_in_unsafe_fn)]` and documented safety assumptions around intrinsics.
- Check every padding, transpose, scheduler, or round change against the reference `md-5` implementation and randomized batch tests.
- Keep `vendor/` and `.cargo/config*` out of the repository and release bundle.
