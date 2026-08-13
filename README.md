# md5-many

High-throughput, multi-buffer MD5 implementation for Rust, accelerated by [`fearless_simd`](https://crates.io/crates/fearless_simd).

[![Crates.io](https://img.shields.io/crates/v/md5-many.svg)](https://crates.io/crates/md5-many)
[![Documentation](https://docs.rs/md5-many/badge.svg)](https://docs.rs/md5-many)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](Cargo.toml)

---

## Overview

A standard MD5 digest has a sequential dependency between consecutive 64-byte blocks, making single-stream hashing fundamentally scalar-bound. However, multiple independent messages can be processed simultaneously across parallel SIMD vector lanes.

`md5-many` provides:
- **Single-message hashing**: standard portable scalar MD5.
- **Multi-buffer SIMD hashing (`Md5Many`)**: parallel computation across SIMD lanes (8-way on AVX2, 4-way on SSE4.2 / NEON / WASM SIMD128).
- **RustCrypto trait compatibility**: optional `digest` crate integration.
- **`no_std` support**: lightweight embedded and WebAssembly readiness.

---

## Usage

Add `md5-many` to your `Cargo.toml`:

```toml
[dependencies]
md5-many = "0.1"
```

### 1. Single-Message Hashing

```rust
use md5_many::md5;

let digest = md5(b"The quick brown fox jumps over the lazy dog");
assert_eq!(
    digest,
    [
        0x9e, 0x10, 0x7d, 0x9d, 0x37, 0x2b, 0xb6, 0x82,
        0x6b, 0xd8, 0x1d, 0x35, 0x42, 0xa4, 0x19, 0xd6,
    ]
);
```

### 2. Multi-Buffer Batch Hashing (`Md5Many`)

When you have multiple independent inputs to hash (such as batch deduplication, file chunk verification, or asset processing), `Md5Many` computes digests concurrently:

```rust
use md5_many::Md5Many;

let inputs: [&[u8]; 4] = [b"alpha", b"beta", b"gamma", b"delta"];
let mut outputs = [[0u8; 16]; 4];

let hasher = Md5Many::new();
hasher.hash_many(&inputs, &mut outputs);
```

`Md5Many` automatically handles both equal-length and mixed-length batches, utilizing specialized SIMD kernels and falling back cleanly on under-filled tail batches.

### 3. RustCrypto `digest` Compatibility

When the `digest` feature is enabled (default), `md5_many::Md5` implements the standard RustCrypto traits:

```rust
use md5_many::{Digest, Md5};

let mut hasher = Md5::new();
hasher.update(b"hello ");
hasher.update(b"world");
let result = hasher.finalize();
```

---

## Feature Flags

| Feature | Default | Description |
| :--- | :---: | :--- |
| `std` | **Yes** | Enables runtime SIMD detection via standard library facilities. |
| `digest` | **Yes** | Implements traits from the [`digest`](https://crates.io/crates/digest) crate (`Digest`, `FixedOutput`, etc.). |
| `libm` | No | Enables floating-point helpers for `no_std` environments where required. |

To use in a `#![no_std]` environment:

```toml
[dependencies]
md5-many = { version = "0.1", default-features = false, features = ["libm", "digest"] }
```

---

## Benchmarks

Run the benchmark suite with:

```bash
cargo bench --bench throughput
```

The benchmarks compare single-stream scalar MD5, `RustCrypto `md-5``, RustCrypto `md-5`, multi-buffer SIMD throughput, and SIMD lane-fill efficiencies.

---

## Security Advisory

> [!CAUTION]
> **MD5 is cryptographically broken.**
>
> Do **not** use this crate for digital signatures, certificates, password hashing, HMACs, collision-resistant identifiers, or adversary-controlled data integrity. This crate is intended strictly for legacy protocol interoperability, non-cryptographic checksumming, and cache indexing where MD5 is mandated.

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
