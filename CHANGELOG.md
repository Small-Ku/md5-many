# Changelog

All notable changes to `md5-many` will be documented in this file.

## Unreleased

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
