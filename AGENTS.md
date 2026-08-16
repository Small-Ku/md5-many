# AGENTS.md

Developer guidance for `md5-many`.

## Architecture

`md5-many` has two performance domains:

1. **Single stream**: MD5 blocks are sequentially dependent. x86-64 uses the optimized backend in `src/scalar_x86_64.rs`; `src/scalar.rs` remains the portable implementation and correctness oracle.
2. **Independent messages**: `src/simd.rs` maps independent MD5 states to SIMD lanes through `fearless_simd`.

Specialized x86 kernels use 8 lanes for AVX2 and 16 lanes for AVX-512. Two or three native groups can be interleaved round-by-round to hide the dependency latency of one MD5 chain. Four-way interleaving was benchmarked and rejected due to register/issue pressure; do not reintroduce it without new evidence.

### Data layout

Inputs arrive message-major (AoS). Native x86 kernels load one 64-byte block per message and transpose the 16 MD5 `u32` words into lane-major vectors (SoA). Keep the whole message loop inside a single `fearless_simd::kernel!` boundary; per-block kernel transitions were measured to be extremely expensive.

### Equal-length scheduling

- AVX2: 8-way native, 16-way dual, 24-way triple.
- AVX-512: 16-way native, 32-way dual, 48-way triple.
- Equal-length padding uses `build_padded_block` rather than byte-at-a-time synthesis.
- A pure padding block shared by every lane is parsed once and broadcast instead of loaded/transposed N times.
- Under-filled AVX2 dual/triple candidates duplicate a real lane rather than falling into a small tail: 9-15 messages use padded dual, 17-23 padded triple, and 26-31 equal/near-mixed batches use two dual kernels.
- On measured AMD family 19h x86-64 CPUs with BMI1, a two-message batch can use the interleaved dual-scalar GPR backend; the G/I rounds use `ANDN`, so keep the explicit CPUID BMI1 guard rather than inferring support from family/model. Tiny pairs (<=32 padded blocks on the longer side) always qualify; larger pairs require the shorter side to have at least 1/16 as many padded blocks. This avoids paying dual setup for extreme skew. Three-message tails normally prefer sparse AVX2, except when the second-longest padded workload is <=1/4 of the longest; then pair the two longest messages in the BMI1 dual backend and hash the remaining lane scalar. Other x86 CPUs retain the previous scalar/AVX2 crossover.
- On measured AMD Family 19h AVX-512 hosts, short equal 9-16-message batches (up to 17 padded blocks) use two AVX2 chains; do not broaden this heuristic to other x86 families without measurements.
- AVX-512 hosts normally keep 2-8-message batches on AVX2. The measured x86 family 6/model `0xCF` crossover is an explicit exception: equal batches at >=512 B and mixed batches whose shortest message is >=512 B use a padded ZMM kernel. Keep this model-specific unless another CPU is benchmarked directly.

### Mixed-length scheduling

- Process the common full-block prefix with the same native transpose/compression machinery.
- Build only divergent padded tails separately.
- Dual/triple mixed kernels interleave independent SIMD state chains just like equal-length kernels.
- The no-allocation skew planner also protects under-filled dual/triple and partial-tail fast paths. If padded block counts differ by at least 2x, it recursively partitions short and long lanes, hashes the sub-batches, then scatters digests back to the caller's order.
- AVX-512 17-31 and 33-47-message mixed/equal batches can use padded dual/triple kernels. Selected 50-63-message shapes stay in two AVX-512 dual kernels when that avoids a pathological tiny or large AVX2 tail.

## Module layout

```text
src/lib.rs            public API and tests
src/scalar.rs         portable scalar MD5 and dispatch wrapper
src/scalar_x86_64.rs  optimized x86-64 single-stream compression
src/simd.rs           SIMD dispatch, native x86 kernels, schedulers and fallback
src/block_api.rs      RustCrypto digest block adapter
src/consts.rs         MD5 IV, round constants and shifts
benches/throughput.rs user-facing Criterion performance suite
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

`fearless_simd` 0.7 requires either `std` or `libm`; plain `--no-default-features` is not a valid dependency configuration.

For performance work, compare before/after with the same Criterion benchmark name and inspect release machine code when an optimization depends on a particular ISA instruction. A plausible algebraic rewrite is not sufficient reason to keep a change. On AVX-512 x86 hosts, the `x86-small-batch-*` Criterion groups compare runtime `auto` dispatch against a forced AVX2 level and should be used before changing the <=8-message crossover.

## Invariants

- Edition 2024; declared MSRV Rust 1.89.
- `#![no_std]` core; `std` is feature-gated.
- `#![deny(unsafe_op_in_unsafe_fn)]` and documented safety assumptions around intrinsics.
- Every padding, transpose, scheduler or round change must be checked against the reference `md-5` implementation and randomized batch tests.
- Keep `vendor/` and `.cargo/config*` out of the repository and release bundle.
