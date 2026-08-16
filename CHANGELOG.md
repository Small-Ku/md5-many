# Changelog

All notable changes to `md5-many` will be documented in this file.

## Unreleased

- Centralize measured backend/scheduler crossover evidence, rejected optimization experiments, and current AArch64 observations in `docs/performance.md`; keep README and developer guidance focused on policy rather than benchmark tables.
- Keep Intel AVX-512VL single-stream state in XMM registers across consecutive blocks, removing the per-block scalar/XMM bridge. On the measured Xeon Platinum 8573C this improves 4 KiB–1 MiB one-shot and streaming throughput by about 4.6–4.7% over `0.1.0-alpha.2`.
- Make the GitHub Actions performance guard bidirectional (ABBA), requiring regressions to reproduce in both base/head measurement orders before they can block a PR; same-SHA manual runs are reported explicitly as runner-noise calibration.
- Resolve manual performance baselines from the latest previous reachable release tag automatically, so new release tags do not require editing the workflow default.
- Chain performance sentinels behind correctness, MSRV, and quality/package CI jobs for pull requests, `master` pushes, and manual CI runs; non-PR runs automatically compare against the latest previous reachable release tag. Keep the separate Performance workflow for manual full-suite comparisons only.
- Let explicitly hardware-gated performance sentinels skip cleanly when the runner lacks their ISA while keeping all ordinary sentinel filters strict; this prevents AVX2-only x86 runners from failing solely because the AVX-512 small-batch benchmark intentionally does not exist there.
- Avoid compiling the x86-only padded-block helper on AArch64, removing a benchmark-build dead-code warning.

## 0.1.0-alpha.2 - 2026-08-16

Performance-focused prerelease with new single-stream architecture backends and more selective low-occupancy batch scheduling.

- Add an Intel-only XMM-width AVX-512VL single-stream compressor using `VPTERNLOGD` and `VPROLD`, used by both one-shot `md5()` and streaming `Md5`; AVX-512-capable AMD CPUs retain the faster scalar path observed on the measured host.
- Add a hand-scheduled little-endian AArch64 single-stream integer compressor using paired loads, `BIC`/`ORN`, and immediate rotates, while retaining the portable implementation on other AArch64 configurations.
- Add an AMD family 19h x86-64 dual-scalar two-message backend. Two independent NoLEA/G-shortcut state chains are interleaved in GPRs, avoiding both sparse AVX2 lanes and sequential-scalar dependency stalls; BMI1 `ANDN` shortens the throughput-bound G/I rounds, while explicit CPUID gating and a measured overlap guard keep unsupported or extremely unbalanced pairs on the existing path.
- Split strongly skewed three-message tails on measured AMD family 19h CPUs into a BMI1 dual-scalar pair plus one scalar hash when the second-longest padded workload is at most one quarter of the longest; equal and near-equal triples remain on AVX2.
- Reuse the AVX2 skew partitioner for 4–8-message tails when partitioning leaves at most two long messages, avoiding sparse-vector long tails without regressing one-short/many-long batches.
- Extend Criterion coverage for two-message overlap guards and 3–8-message skew crossovers, including regression cases that must remain on SIMD.
- Keep BMI1-specific tests behind the `std` feature so the `no_std + libm` test configurations continue to build cleanly.

## 0.1.0-alpha.1 - 2026-08-16

First public prerelease.

- High-throughput single-stream MD5 with an optimized x86-64 scalar backend.
- Runtime-dispatched multi-buffer SIMD through `fearless_simd`.
- AVX2 8-way kernels with dual- and triple-chain instruction-level parallelism.
- AVX-512 16-way kernels using `VPTERNLOGD` and `VPROLD`, with dual- and triple-chain scheduling.
- Equal-length, mixed-length, partial-batch, and skew-aware batch scheduling without heap allocation in the hashing path.
- RustCrypto `digest` compatibility and `no_std` support through the `libm` feature.
- Correctness tests against reference implementations plus Criterion throughput benchmarks.

MD5 is cryptographically broken; this crate is intended for legacy interoperability and non-adversarial checksumming only.
