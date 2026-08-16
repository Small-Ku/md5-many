# Changelog

All notable changes to `md5-many` will be documented in this file.

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
