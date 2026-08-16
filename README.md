# md5-many

High-throughput MD5 for Rust, with an optimized single-stream path and runtime-dispatched multi-buffer SIMD powered by [`fearless_simd`](https://crates.io/crates/fearless_simd).

> **MD5 is cryptographically broken.** Use this crate only where MD5 is required for legacy interoperability or non-adversarial checksumming. Do not use it for signatures, passwords, certificates, collision-resistant identifiers, or attacker-controlled integrity checks.

## What it does

A single MD5 stream has a dependency between consecutive 64-byte blocks, so wide SIMD is most useful when several independent messages are available at once. `md5-many` therefore exposes two complementary paths:

- `md5()` / `Md5`: single-message hashing. x86-64 uses an optimized NoLEA-style scalar compressor; `md5()` and the streaming `Md5` path can additionally select an XMM-width AVX-512VL compressor on supported Intel CPUs. Little-endian AArch64 uses a hand-scheduled integer compressor. Other targets retain the portable Rust compressor.
- `Md5Many`: batches independent messages into SIMD lanes and chooses a scheduler appropriate to the detected CPU and workload.

On x86 the specialized backend currently includes:

- Single-stream AVX-512VL: XMM registers with `VPTERNLOGD`/`VPROLD`, selected only on supported Intel CPUs for one-shot and streaming hashing; AMD AVX-512 stays on the faster NoLEA scalar path.
- AVX2: 8-message native kernels plus interleaved 16-message dual-chain and 24-message triple-chain kernels.
- AVX-512: 16-message native kernels plus interleaved 32-message dual-chain and 48-message triple-chain kernels.
- AVX-512 rounds use `VPTERNLOGD` for the MD5 Boolean functions and `VPROLD` for rotates.
- Equal-length inputs use whole-stream kernels, AoS-to-SoA transposes, and broadcast pure-padding blocks.
- Mixed-length inputs process the common full-block prefix at full SIMD speed, then handle only divergent tails separately.
- Partial batches can duplicate the last real input into unused lanes so they still benefit from dual/triple instruction-level parallelism instead of falling into a small tail.
- Highly skewed mixed batches, including under-filled dual/triple candidates, are repartitioned without allocation so one short message does not force many long messages through a slow divergent tail.
- On measured AMD family 19h x86-64 CPUs with BMI1, two-message batches can use a dual-scalar GPR kernel that interleaves two independent NoLEA/G-shortcut chains and uses three-operand `ANDN` in the throughput-bound G/I rounds. A CPUID check and overlap-aware skew guard keep unsupported or extremely unbalanced long pairs on the existing path. Three-message tails normally stay on sparse AVX2, but a strongly skewed triple (second-longest padded work at most one quarter of the longest) pairs the two longest messages and hashes the third scalar; this avoids a measured 9–31% divergence penalty. Other x86 CPUs retain the conservative scalar/AVX2 crossover.
- AVX-512 small batches remain on AVX2 by default. A narrowly measured x86 family 6/model `0xCF` tuning uses padded AVX-512 for 2-8-message equal or mixed batches once every message is at least 512 B; other AVX-512 CPUs keep the conservative AVX2 choice until measured.

Little-endian AArch64 uses a separate hand-scheduled single-stream integer kernel with paired message/constant loads, `BIC`/`ORN` Boolean forms, and immediate `ROR`. Multi-buffer AArch64 hashing remains on the `fearless_simd` NEON path.

Other `fearless_simd` targets retain the portable multi-buffer implementation, including SSE-class x86, AArch64 NEON, WASM SIMD, and scalar fallback as supported by the selected `fearless_simd` release.

## Usage

```toml
[dependencies]
md5-many = "0.1.0-alpha.1"
```

### Single message

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

### Batch hashing

```rust
use md5_many::Md5Many;

let inputs: [&[u8]; 4] = [b"alpha", b"beta", b"gamma", b"delta"];
let mut outputs = [[0u8; 16]; 4];

let hasher = Md5Many::new();
hasher.hash_many(&inputs, &mut outputs);
```

Construct `Md5Many` once and reuse it when possible; this avoids repeating runtime CPU-feature detection.

### RustCrypto `digest` compatibility

The default `digest` feature exposes `md5_many::Md5`:

```rust
use md5_many::{Digest, Md5};

let mut hasher = Md5::new();
hasher.update(b"hello ");
hasher.update(b"world");
let result = hasher.finalize();
```

## Features

| Feature | Default | Purpose |
| --- | :---: | --- |
| `std` | yes | Runtime SIMD detection through `fearless_simd`. |
| `digest` | yes | RustCrypto `digest` trait compatibility. |
| `libm` | no | Required `fearless_simd` support for `no_std` builds. |

`fearless_simd` 0.7 requires at least one of its `std` or `libm` modes. A `no_std` dependency therefore looks like:

```toml
[dependencies]
md5-many = { version = "0.1.0-alpha.1", default-features = false, features = ["libm", "digest"] }
```

Or omit `digest` if the block-trait adapter is not needed.

## Testing and benchmarks

```bash
cargo test --locked
cargo bench --locked --bench throughput
```

The Criterion suite contains:

- single-stream comparison against RustCrypto `md-5`;
- native-width equal-length batches at 64 B, 1 KiB, 64 KiB, and 1 MiB;
- lane-fill crossover measurements;
- partial-batch scaling around native, dual, and triple scheduler boundaries;
- short mixed-length padding-boundary workloads;
- deliberately skewed under-filled mixed batches, including a two-message long partition, to catch divergent-tail and low-occupancy regressions;
- one-, two-, and three-native-batch mixed workloads around 64 KiB;
- batch scaling through eight native SIMD groups;
- on AVX-512 x86 hosts, forced `auto` versus AVX2 comparisons for 8-message equal/mixed 1 KiB and 64 KiB workloads, so small-batch crossover changes can be measured without relying on Criterion history.

To inspect the runtime native lane width:

```bash
cargo run --release --example probe
```

A Zen 3 CPU such as a Ryzen 7 5800X3D reports an 8-lane AVX2 engine. An AVX-512-capable host reports 16 native lanes, while the scheduler may internally interleave two or three native groups to expose more instruction-level parallelism.

## `no_std` verification

```bash
cargo test --locked --no-default-features --features libm
cargo test --locked --no-default-features --features libm,digest
```

## Implementation provenance

The optimized x86-64 scalar compressor is a direct Rust port of the
`md5_block_noleag` scheduling from `animetosho/md5-optimisation` commit
`7cd4ad511f8cddbeed584c4087fb9506d94e8b87`. The XMM-width single-stream
AVX-512VL compressor ports the same repository's `md5_block_avx512` packed
message/constant schedule. The little-endian AArch64 scalar kernel is based on
the same repository's `md5-arm64-asm.h` scheduling ideas, rewritten as Rust
inline assembly. Its author releases that source into the Public Domain, or
under CC0-1.0 where a public-domain dedication is not recognized.

## License

`md5-many` is dual-licensed under MIT OR Apache-2.0. See `LICENSE-MIT` and
`LICENSE-APACHE`.
