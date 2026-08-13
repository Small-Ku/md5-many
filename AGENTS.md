# AGENTS.md

Developer and AI agent guidance for working in `md5-many`.

---

## 1. Project Overview & Architecture

`md5-many` is a high-throughput Rust implementation of MD5 focusing on **multi-buffer SIMD parallelism** via [`fearless_simd`](https://crates.io/crates/fearless_simd).

### Core Concepts

1. **Sequential vs Multi-Buffer Dependency**:
   - Single-stream MD5 has an inherent block-to-block data dependency; vectorizing a single message is limited.
   - Independent messages can execute simultaneously in separate SIMD lanes.
   - Target native widths: 16 lanes on AVX-512, 8 lanes on AVX2, and 4 lanes on SSE4.2 / AArch64 NEON / WASM SIMD128.

2. **AoS to SoA Transpose**:
   - Equal-length batch inputs are loaded in message-major order (Array of Structures / AoS).
   - Blocks are transposed into 16 lane-vectors of `u32` words (Structure of Arrays / SoA) before running the unrolled 64-round compression function.

3. **Kernel Boundary Minimization**:
   - Crossing `fearless_simd::kernel!` boundaries per-block incurs high overhead.
   - The entire equal-length streaming loop stays inside a single kernel dispatch region.

4. **Batch Processing Strategy**:
   - **Equal-length batches**: Fast path with unrolled multi-block loops.
   - **Mixed-length batches**: Lanes are advanced lockstep per 64-byte block; each lane's digest is finalized immediately upon reaching its padded final block.
   - **Tail / Small batches**: one-message tails use scalar MD5; equal-length 2-7 message batches pad to the AVX2 8-way kernel, and AVX-512 uses its 16-way kernel for 9-16 messages.

---

## 2. Module Layout

```text
md5_many/
├── Cargo.toml          # Features: std (default), digest (default), libm
├── src/
│   ├── lib.rs          # Public API: md5, md5_many, Md5Many, RustCrypto exports
│   ├── simd.rs         # SIMD dispatch, AVX2 8-way + AVX-512 16-way kernels, SoA transposes
│   ├── scalar.rs       # Portable scalar MD5 implementation & reference logic
│   ├── block_api.rs    # RustCrypto `digest` trait adapter (Md5Core)
│   └── consts.rs       # Round constants (K), shifts (S), initial state (IV)
├── benches/
│   └── throughput.rs   # Criterion benchmarks comparing scalar, RustCrypto `md-5`, md-5, and SIMD
└── examples/
    └── probe.rs        # Runtime SIMD probe and sanity verification utility
```

---

## 3. Development & Verification Workflows

### Standard Test Suite
```bash
cargo test
```
Runs unit tests, RFC test vectors, mixed/equal length property tests (`proptest`), and padding boundary checks.

### Feature Matrix Verification
Ensure all feature combinations build cleanly:
```bash
# Default (std + digest)
cargo check

# no_std with libm and digest
cargo check --no-default-features --features libm,digest

# no_std without Digest integration
cargo check --no-default-features --features libm
```

### Benchmarks
```bash
cargo bench --bench throughput
```

### Probing SIMD Level
```bash
cargo run --example probe
```

---

## 4. Coding Standards & Invariants

- **Rust Edition & MSRV**: Edition 2024, Rust 1.89+.
- **`#![no_std]` core**: Core crate logic must work without `std`. `fearless_simd` 0.7 requires either `std` or `libm`, so no-std verification enables this crate's `libm` feature. Use conditional `extern crate std;` under `#[cfg(feature = "std")]`.
- **Unsafe Code Guidelines**:
  - `#![deny(unsafe_op_in_unsafe_fn)]` is enforced.
  - Keep `unsafe` blocks minimal and well-documented with safety rationale comments (especially around SIMD intrinsics and transposes).
- **Public API Documentation**:
  - `#![warn(missing_docs)]` is active. All public items, traits, and structs must have doc comments and usage examples.
- **Precision & Correctness**:
  - Any modifications to padding, length encoding, or round operations must be validated against the `proptest` and RFC suites.
