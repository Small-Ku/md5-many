#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
//! High-throughput MD5 with portable multi-buffer SIMD.
//!
//! `md5-many` keeps the standard single-stream MD5 API separate from its
//! SIMD batch engine. A single MD5 stream has a dependency from one 64-byte
//! block to the next, while independent messages can occupy independent SIMD
//! lanes. On x86-64 the batch scheduler combines 8-way AVX2 or 16-way AVX-512
//! native kernels with interleaved dual- and triple-batch kernels to expose
//! additional instruction-level parallelism.
//!
//! MD5 is cryptographically broken and must not be used for signatures,
//! password hashing, or adversarial integrity. This crate targets legacy
//! interoperability and non-adversarial checksumming workloads.

#[cfg(feature = "std")]
extern crate std;

mod consts;
mod incremental;
mod scalar;
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    any(test, feature = "bench-internals")
))]
mod scalar_aarch64;
#[cfg(target_arch = "x86_64")]
mod scalar_x86_64;
#[cfg(target_arch = "x86_64")]
mod scalar_x86_64_avx512;
#[cfg(target_arch = "x86_64")]
mod scalar_x86_64_dual;
mod simd;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod simd_aarch64;

#[cfg(feature = "digest")]
/// RustCrypto block-level compatibility API.
pub mod block_api;

#[cfg(feature = "digest")]
pub use digest::{self, Digest};

#[cfg(feature = "digest")]
digest::buffer_fixed!(
    /// Streaming MD5 hasher compatible with RustCrypto's `digest` traits.
    pub struct Md5(block_api::Md5Core);
    impl: BaseFixedTraits AlgorithmName Default Clone HashMarker Reset FixedOutputReset;
);

/// A raw 128-bit MD5 digest.
pub type Md5Digest = [u8; 16];

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench_internals {
    //! Unstable backend hooks for this crate's own microbenchmarks.

    use super::Md5Digest;

    /// Hash through the pre-specialization generic one-shot framing path.
    #[must_use]
    pub fn md5_generic(input: &[u8]) -> Md5Digest {
        crate::scalar::hash_generic(input)
    }

    /// Hash a <=55-byte message through the one-block framing candidate.
    #[must_use]
    pub fn md5_short_one_block(input: &[u8]) -> Md5Digest {
        crate::scalar::hash_short_one_block(input)
    }

    /// Force the portable Rust compressor with generic one-shot framing.
    #[must_use]
    pub fn md5_portable(input: &[u8]) -> Md5Digest {
        crate::scalar::hash_portable(input)
    }

    /// Force the portable compressor with the <=55-byte one-block framing.
    #[must_use]
    pub fn md5_portable_short(input: &[u8]) -> Md5Digest {
        crate::scalar::hash_portable_short(input)
    }

    /// Force the little-endian AArch64 GPR compressor with generic one-shot framing.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[must_use]
    pub fn md5_aarch64_gpr(input: &[u8]) -> Md5Digest {
        crate::scalar::hash_aarch64_gpr(input)
    }

    /// Force the AArch64 GPR compressor with the <=55-byte one-block framing.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[must_use]
    pub fn md5_aarch64_gpr_short(input: &[u8]) -> Md5Digest {
        crate::scalar::hash_aarch64_gpr_short(input)
    }

    /// Force the native four-way AArch64 NEON equal-length candidate.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[must_use]
    pub fn md5_aarch64_neon4(inputs: [&[u8]; 4]) -> [Md5Digest; 4] {
        crate::simd_aarch64::hash_equal_len4(inputs)
    }

    /// Force the native eight-way AArch64 NEON round-interleaved candidate.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[must_use]
    pub fn md5_aarch64_neon8(inputs: [&[u8]; 8]) -> [Md5Digest; 8] {
        crate::simd_aarch64::hash_equal_len8(inputs)
    }

    /// Force the native twelve-way AArch64 NEON round-interleaved candidate.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[must_use]
    pub fn md5_aarch64_neon12(inputs: [&[u8]; 12]) -> [Md5Digest; 12] {
        crate::simd_aarch64::hash_equal_len12(inputs)
    }

    /// Hash a non-empty block-aligned message through the compact final-block candidate.
    #[must_use]
    pub fn md5_aligned(input: &[u8]) -> Md5Digest {
        crate::scalar::hash_aligned(input)
    }

    /// Whether the x86-64 AVX-512VL single-stream backend can execute.
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    pub fn x86_avx512_supported() -> bool {
        crate::scalar_x86_64_avx512::is_supported()
    }

    /// Force the x86-64 NoLEA scalar single-stream backend.
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    pub fn md5_x86_nolea(input: &[u8]) -> Md5Digest {
        crate::scalar::hash_x86_nolea(input)
    }

    /// Force the generic-framing x86-64 AVX-512VL single-stream backend.
    ///
    /// # Panics
    ///
    /// Panics when AVX-512F or AVX-512VL is unavailable.
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    pub fn md5_x86_avx512_generic(input: &[u8]) -> Md5Digest {
        assert!(x86_avx512_supported());
        // SAFETY: the assertion above checks both required target features.
        unsafe { crate::scalar_x86_64_avx512::hash_generic(input) }
    }

    /// Force the x86-64 AVX-512VL single-stream backend.
    ///
    /// # Panics
    ///
    /// Panics when AVX-512F or AVX-512VL is unavailable.
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    pub fn md5_x86_avx512(input: &[u8]) -> Md5Digest {
        assert!(x86_avx512_supported());
        // SAFETY: the assertion above checks both required target features.
        unsafe { crate::scalar_x86_64_avx512::hash(input) }
    }

    /// Force the benchmark-only packed-digest AVX-512VL candidate.
    ///
    /// # Panics
    ///
    /// Panics when AVX-512F or AVX-512VL is unavailable.
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    pub fn md5_x86_avx512_packed_digest(input: &[u8]) -> Md5Digest {
        assert!(x86_avx512_supported());
        // SAFETY: the assertion above checks both required target features.
        unsafe { crate::scalar_x86_64_avx512::hash_packed_digest(input) }
    }
}

pub use incremental::Md5State;

/// Compute the MD5 digest of a single byte slice.
///
/// On x86-64 this uses the optimized scalar compressor and may select an
/// XMM-width AVX-512VL single-stream backend on supported Intel CPUs.
/// AArch64 and other non-x86 targets use the portable Rust compressor;
/// the AArch64 integer assembly backend is retained only for benchmarking. For independent-message SIMD
/// batching, use [`Md5Many`] or [`md5_many`].
#[must_use]
pub fn md5(input: &[u8]) -> Md5Digest {
    scalar::hash(input)
}

/// Runtime-dispatched batch MD5 engine.
///
/// Construct this once and reuse it so CPU feature detection is not repeated.
#[derive(Clone, Copy, Debug)]
pub struct Md5Many {
    level: fearless_simd::Level,
}

impl Md5Many {
    /// Detect the best SIMD level available on the current process.
    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[must_use]
    pub fn new() -> Self {
        Self::from_level(fearless_simd::Level::new())
    }

    /// Construct a batch engine from an already-detected Fearless SIMD level.
    #[must_use]
    pub const fn from_level(level: fearless_simd::Level) -> Self {
        Self { level }
    }

    /// Return the native number of `u32` SIMD lanes used by this engine.
    ///
    /// AVX-512 is 16 lanes, AVX2 is 8 lanes, and SSE4.2/NEON/WASM use 4 lanes.
    #[must_use]
    pub const fn lanes(self) -> usize {
        simd::lanes_with_level(self.level)
    }

    /// Hash many independent messages into `outputs`.
    ///
    /// Equal-length groups use specialized whole-stream SIMD kernels. Mixed
    /// groups process their common full-block prefix at SIMD speed and only
    /// diverge for the tail; skewed groups may be repartitioned to avoid
    /// wasting lanes on already-finished messages.
    ///
    /// # Panics
    ///
    /// Panics if `outputs` is shorter than `inputs`.
    pub fn hash_many(self, inputs: &[&[u8]], outputs: &mut [Md5Digest]) {
        simd::hash_many_with_level(self.level, inputs, outputs);
    }

    /// Increment several independent MD5 streams in parallel.
    ///
    /// Each `inputs[i]` is appended to `states[i]`. Complete 64-byte blocks
    /// are compacted into SIMD lanes even when chunk lengths differ, while
    /// partial blocks remain buffered in the corresponding [`Md5State`].
    ///
    /// # Panics
    ///
    /// Panics if `states` and `inputs` have different lengths.
    pub fn update_many(self, states: &mut [Md5State], inputs: &[&[u8]]) {
        incremental::update_many_with_level(self.level, states, inputs);
    }

    /// Finalize several incremental MD5 streams in parallel.
    ///
    /// Finalization is non-destructive: the states are not modified and may
    /// receive more data afterwards. Padding blocks are batched through the
    /// multi-buffer compressor where profitable.
    ///
    /// # Panics
    ///
    /// Panics if `outputs` is shorter than `states`.
    pub fn finalize_many(self, states: &[Md5State], outputs: &mut [Md5Digest]) {
        incremental::finalize_many_with_level(self.level, states, outputs);
    }
}

#[cfg(any(feature = "std", target_arch = "wasm32"))]
impl Default for Md5Many {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash many independent messages using runtime SIMD detection.
///
/// For repeated calls, prefer constructing [`Md5Many`] once and reusing it.
///
/// # Panics
///
/// Panics if `outputs` is shorter than `inputs`.
#[cfg(any(feature = "std", target_arch = "wasm32"))]
pub fn md5_many(inputs: &[&[u8]], outputs: &mut [Md5Digest]) {
    Md5Many::new().hash_many(inputs, outputs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use md5::Md5 as ReferenceMd5;

    fn reference(input: &[u8]) -> Md5Digest {
        let out = <ReferenceMd5 as md5::Digest>::digest(input);
        out.as_slice().try_into().expect("MD5 is 16 bytes")
    }

    #[test]
    fn rfc_vectors() {
        let vectors: &[(&[u8], [u8; 16])] = &[
            (b"", hex_literal::hex!("d41d8cd98f00b204e9800998ecf8427e")),
            (b"a", hex_literal::hex!("0cc175b9c0f1b6a831c399e269772661")),
            (
                b"abc",
                hex_literal::hex!("900150983cd24fb0d6963f7d28e17f72"),
            ),
            (
                b"message digest",
                hex_literal::hex!("f96b697d7cb7938d525a2f31aaf161d0"),
            ),
            (
                b"abcdefghijklmnopqrstuvwxyz",
                hex_literal::hex!("c3fcd3d76192e4007dfb496cca67e13b"),
            ),
        ];
        for (input, expected) in vectors {
            assert_eq!(md5(input), *expected);
        }
    }

    #[test]
    fn boundary_lengths_match_reference() {
        let mut data = [0u8; 257];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = index as u8;
        }
        for len in 0..=data.len() {
            assert_eq!(md5(&data[..len]), reference(&data[..len]), "len={len}");
        }
    }

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[test]
    fn many_equal_length_matches_reference() {
        let messages: [[u8; 192]; 13] = core::array::from_fn(|message| {
            core::array::from_fn(|byte| (message * 31 + byte * 17) as u8)
        });
        let inputs: [&[u8]; 13] = core::array::from_fn(|i| messages[i].as_slice());
        let mut outputs = [[0u8; 16]; 13];

        let engine = Md5Many::new();
        engine.hash_many(&inputs, &mut outputs);
        for (input, output) in inputs.iter().zip(outputs) {
            assert_eq!(output, reference(input));
        }
    }

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[test]
    fn many_equal_length_lane_counts_match_reference() {
        let storage: [[u8; 193]; 16] =
            core::array::from_fn(|lane| core::array::from_fn(|i| (lane * 37 + i * 13) as u8));
        let engine = Md5Many::new();

        for active in 2..=16 {
            let inputs = &storage[..active];
            let inputs: std::vec::Vec<&[u8]> = inputs.iter().map(<[u8; 193]>::as_slice).collect();
            let mut outputs = std::vec![[0u8; 16]; active];
            engine.hash_many(&inputs, &mut outputs);
            for lane in 0..active {
                assert_eq!(
                    outputs[lane],
                    reference(inputs[lane]),
                    "active={active}, lane={lane}"
                );
            }
        }
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx512_padding_boundaries_match_reference() {
        let engine = Md5Many::new();
        if engine.lanes() < 16 {
            return;
        }

        let storage: [[u8; 129]; 16] =
            core::array::from_fn(|lane| core::array::from_fn(|i| (lane * 23 + i * 5) as u8));
        for len in [0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129] {
            let inputs: [&[u8]; 16] = core::array::from_fn(|lane| &storage[lane][..len]);
            let mut outputs = [[0u8; 16]; 16];
            engine.hash_many(&inputs, &mut outputs);
            for lane in 0..16 {
                assert_eq!(
                    outputs[lane],
                    reference(inputs[lane]),
                    "lane={lane}, len={len}"
                );
            }
        }
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx512_small_batch_crossover_matches_reference() {
        let engine = Md5Many::new();
        if engine.lanes() < 16 {
            return;
        }

        let storage: [[u8; 1216]; 8] =
            core::array::from_fn(|lane| core::array::from_fn(|i| (lane * 41 + i * 7) as u8));

        for active in [2usize, 4, 8] {
            for len in [512usize, 513, 1024] {
                let inputs: std::vec::Vec<&[u8]> = storage[..active]
                    .iter()
                    .map(|input| &input[..len])
                    .collect();
                let mut outputs = std::vec![[0u8; 16]; active];
                engine.hash_many(&inputs, &mut outputs);
                for lane in 0..active {
                    assert_eq!(
                        outputs[lane],
                        reference(inputs[lane]),
                        "equal active={active}, lane={lane}, len={len}"
                    );
                }
            }

            let inputs: std::vec::Vec<&[u8]> = storage[..active]
                .iter()
                .enumerate()
                .map(|(lane, input)| &input[..512 + lane * 73])
                .collect();
            let mut outputs = std::vec![[0u8; 16]; active];
            engine.hash_many(&inputs, &mut outputs);
            for lane in 0..active {
                assert_eq!(
                    outputs[lane],
                    reference(inputs[lane]),
                    "mixed active={active}, lane={lane}"
                );
            }
        }
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn x86_triple_batches_match_reference() {
        let detected = fearless_simd::Level::new();

        if let Some(avx2) = detected.as_avx2() {
            let engine = Md5Many::from_level(fearless_simd::Level::Avx2(avx2));
            for mixed in [false, true] {
                let storage: std::vec::Vec<std::vec::Vec<u8>> = (0..24)
                    .map(|lane| {
                        let len = if mixed { 4096 - lane * 8 } else { 4096 };
                        (0..len).map(|i| (lane * 31 + i * 11) as u8).collect()
                    })
                    .collect();
                let inputs: std::vec::Vec<&[u8]> =
                    storage.iter().map(std::vec::Vec::as_slice).collect();
                let mut outputs = std::vec![[0u8; 16]; inputs.len()];
                engine.hash_many(&inputs, &mut outputs);
                for (lane, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
                    assert_eq!(*output, reference(input), "avx2 mixed={mixed} lane={lane}");
                }
            }
        }

        if let Some(avx512) = detected.as_avx512() {
            let engine = Md5Many::from_level(fearless_simd::Level::Avx512(avx512));
            for mixed in [false, true] {
                let storage: std::vec::Vec<std::vec::Vec<u8>> = (0..48)
                    .map(|lane| {
                        let len = if mixed { 4096 - lane * 4 } else { 4096 };
                        (0..len).map(|i| (lane * 17 + i * 13) as u8).collect()
                    })
                    .collect();
                let inputs: std::vec::Vec<&[u8]> =
                    storage.iter().map(std::vec::Vec::as_slice).collect();
                let mut outputs = std::vec![[0u8; 16]; inputs.len()];
                engine.hash_many(&inputs, &mut outputs);
                for (lane, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
                    assert_eq!(
                        *output,
                        reference(input),
                        "avx512 mixed={mixed} lane={lane}"
                    );
                }
            }
        }
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn x86_partial_batch_boundaries_match_reference() {
        fn check_counts(engine: Md5Many, counts: &[usize], base_len: usize) {
            let equal_storage: std::vec::Vec<std::vec::Vec<u8>> = (0..64)
                .map(|lane| (0..base_len).map(|i| (lane * 41 + i * 17) as u8).collect())
                .collect();
            let mixed_storage: std::vec::Vec<std::vec::Vec<u8>> = (0..64)
                .map(|lane| {
                    let len = base_len - (lane % 13) * 7;
                    (0..len).map(|i| (lane * 23 + i * 29) as u8).collect()
                })
                .collect();

            for &count in counts {
                for (kind, storage) in [("equal", &equal_storage), ("mixed", &mixed_storage)] {
                    let inputs: std::vec::Vec<&[u8]> = storage[..count]
                        .iter()
                        .map(std::vec::Vec::as_slice)
                        .collect();
                    let mut outputs = std::vec![[0u8; 16]; count];
                    engine.hash_many(&inputs, &mut outputs);
                    for (lane, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
                        assert_eq!(
                            *output,
                            reference(input),
                            "{kind} count={count} lane={lane}"
                        );
                    }
                }
            }
        }

        let detected = fearless_simd::Level::new();
        if let Some(avx2) = detected.as_avx2() {
            let engine = Md5Many::from_level(fearless_simd::Level::Avx2(avx2));
            check_counts(
                engine,
                &[
                    2, 3, 8, 9, 15, 16, 17, 23, 24, 25, 26, 31, 32, 49, 50, 55, 56, 63, 64,
                ],
                257,
            );

            // Exercise both sides of the two-message one-block crossover.
            for len in [55usize, 56] {
                let storage = [std::vec![0x35u8; len], std::vec![0xa7u8; len]];
                let inputs = [storage[0].as_slice(), storage[1].as_slice()];
                let mut outputs = [[0u8; 16]; 2];
                engine.hash_many(&inputs, &mut outputs);
                for lane in 0..2 {
                    assert_eq!(
                        outputs[lane],
                        reference(inputs[lane]),
                        "len={len} lane={lane}"
                    );
                }
            }
        }

        if let Some(avx512) = detected.as_avx512() {
            let engine = Md5Many::from_level(fearless_simd::Level::Avx512(avx512));
            check_counts(
                engine,
                &[8, 9, 16, 17, 31, 32, 33, 47, 48, 49, 50, 56, 63, 64],
                2048,
            );
        }
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn x86_underfilled_skew_batches_match_reference() {
        fn check(engine: Md5Many, count: usize, short_count: usize) {
            let storage: std::vec::Vec<std::vec::Vec<u8>> = (0..count)
                .map(|lane| {
                    let len = if lane < short_count { 129 } else { 4097 };
                    (0..len).map(|i| (lane * 47 + i * 19) as u8).collect()
                })
                .collect();
            let inputs: std::vec::Vec<&[u8]> =
                storage.iter().map(std::vec::Vec::as_slice).collect();
            let mut outputs = std::vec![[0u8; 16]; count];
            engine.hash_many(&inputs, &mut outputs);
            for (lane, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
                assert_eq!(
                    *output,
                    reference(input),
                    "count={count} short_count={short_count} lane={lane}"
                );
            }
        }

        let detected = fearless_simd::Level::new();
        if let Some(avx2) = detected.as_avx2() {
            let engine = Md5Many::from_level(fearless_simd::Level::Avx2(avx2));
            for (count, short_count) in
                [(3, 2), (9, 4), (15, 7), (17, 8), (23, 8), (26, 8), (31, 15)]
            {
                check(engine, count, short_count);
            }
        }
        if let Some(avx512) = detected.as_avx512() {
            let engine = Md5Many::from_level(fearless_simd::Level::Avx512(avx512));
            for (count, short_count) in [
                (3, 2),
                (4, 2),
                (4, 3),
                (5, 3),
                (6, 4),
                (7, 5),
                (8, 6),
                (17, 8),
                (31, 15),
                (33, 16),
                (47, 16),
                (50, 16),
                (57, 16),
                (63, 31),
            ] {
                check(engine, count, short_count);
            }
        }
    }

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[test]
    fn many_padding_boundaries_match_reference() {
        let storage: [[u8; 129]; 8] =
            core::array::from_fn(|lane| core::array::from_fn(|i| (lane * 19 + i * 7) as u8));

        for len in [0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129] {
            let inputs: [&[u8]; 8] = core::array::from_fn(|lane| &storage[lane][..len]);
            let mut outputs = [[0u8; 16]; 8];
            Md5Many::new().hash_many(&inputs, &mut outputs);
            for lane in 0..8 {
                assert_eq!(
                    outputs[lane],
                    reference(inputs[lane]),
                    "lane={lane}, len={len}"
                );
            }
        }
    }

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[test]
    fn many_mixed_lengths_are_correct() {
        let a = b"".as_slice();
        let b = b"abc".as_slice();
        let c = [0x5au8; 65];
        let d = [0xa5u8; 130];
        let inputs: [&[u8]; 4] = [a, b, &c, &d];
        let mut outputs = [[0u8; 16]; 4];
        Md5Many::new().hash_many(&inputs, &mut outputs);
        for i in 0..inputs.len() {
            assert_eq!(outputs[i], reference(inputs[i]));
        }
    }

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[test]
    fn many_mixed_lengths_cross_multiple_block_counts() {
        let storage: [[u8; 321]; 8] =
            core::array::from_fn(|lane| core::array::from_fn(|i| (lane * 43 + i * 11) as u8));
        let lengths = [0usize, 3, 55, 64, 119, 128, 193, 321];
        let inputs: [&[u8]; 8] = core::array::from_fn(|lane| &storage[lane][..lengths[lane]]);
        let mut outputs = [[0u8; 16]; 8];

        Md5Many::new().hash_many(&inputs, &mut outputs);
        for lane in 0..inputs.len() {
            assert_eq!(outputs[lane], reference(inputs[lane]), "lane={lane}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn large_dual_batches_and_skew_partition_match_reference() {
        let engine = Md5Many::new();

        // Covers repeated dual-batch scheduling beyond one native pair.
        let equal_storage: std::vec::Vec<std::vec::Vec<u8>> = (0..64)
            .map(|lane| (0..1025).map(|i| (lane * 29 + i * 7) as u8).collect())
            .collect();
        let equal_inputs: std::vec::Vec<&[u8]> =
            equal_storage.iter().map(std::vec::Vec::as_slice).collect();
        let mut outputs = std::vec![[0u8; 16]; equal_inputs.len()];
        engine.hash_many(&equal_inputs, &mut outputs);
        for (lane, (input, output)) in equal_inputs.iter().zip(&outputs).enumerate() {
            assert_eq!(*output, reference(input), "equal lane={lane}");
        }

        // Alternating short/long messages forces the no-alloc skew planner to
        // repartition each dual candidate, then scatter digests back to the
        // caller's original order.
        let mixed_storage: std::vec::Vec<std::vec::Vec<u8>> = (0..64)
            .map(|lane| {
                let len = if lane % 2 == 0 { 1024 } else { 16 * 1024 };
                (0..len).map(|i| (lane * 17 + i * 11) as u8).collect()
            })
            .collect();
        let mixed_inputs: std::vec::Vec<&[u8]> =
            mixed_storage.iter().map(std::vec::Vec::as_slice).collect();
        outputs.fill([0u8; 16]);
        engine.hash_many(&mixed_inputs, &mut outputs);
        for (lane, (input, output)) in mixed_inputs.iter().zip(&outputs).enumerate() {
            assert_eq!(*output, reference(input), "mixed lane={lane}");
        }
    }

    #[test]
    fn incremental_single_state_boundaries_match_reference() {
        let data: [u8; 321] = core::array::from_fn(|i| (i * 37 + 11) as u8);
        for len in 0..=data.len() {
            let mut state = Md5State::new();
            let split1 = core::cmp::min(len, 31);
            let split2 = core::cmp::min(len, 97);
            state.update(&data[..split1]);
            state.update(&data[split1..split2]);
            state.update(&data[split2..len]);
            assert_eq!(state.finalize(), reference(&data[..len]), "len={len}");
            assert_eq!(state.bytes_hashed(), len as u64);
        }
    }

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[test]
    fn incremental_many_mixed_chunks_match_reference() {
        let engine = Md5Many::new();
        let storage: std::vec::Vec<std::vec::Vec<u8>> = (0..37)
            .map(|lane| {
                let len = (lane * 431 + 57) % 5001;
                (0..len).map(|i| (lane * 53 + i * 19) as u8).collect()
            })
            .collect();
        let mut states = std::vec![Md5State::new(); storage.len()];
        let mut offsets = std::vec![0usize; storage.len()];
        let mut round = 0usize;

        while offsets
            .iter()
            .zip(&storage)
            .any(|(&offset, message)| offset < message.len())
        {
            let chunks: std::vec::Vec<&[u8]> = storage
                .iter()
                .enumerate()
                .map(|(lane, message)| {
                    let start = offsets[lane];
                    let remaining = message.len() - start;
                    let proposed = 1 + ((round * 17 + lane * 29) % 172);
                    let take = core::cmp::min(remaining, proposed);
                    &message[start..start + take]
                })
                .collect();

            engine.update_many(&mut states, &chunks);
            for lane in 0..storage.len() {
                offsets[lane] += chunks[lane].len();
            }
            round += 1;
        }

        let mut outputs = std::vec![[0u8; 16]; states.len()];
        engine.finalize_many(&states, &mut outputs);
        for lane in 0..storage.len() {
            assert_eq!(
                outputs[lane],
                reference(&storage[lane]),
                "mixed incremental lane={lane}"
            );
        }

        // Finalization is a snapshot rather than a consuming operation.
        let suffixes: std::vec::Vec<std::vec::Vec<u8>> = (0..states.len())
            .map(|lane| std::vec![(lane as u8).wrapping_mul(7); lane % 91])
            .collect();
        let suffix_refs: std::vec::Vec<&[u8]> =
            suffixes.iter().map(std::vec::Vec::as_slice).collect();
        engine.update_many(&mut states, &suffix_refs);
        engine.finalize_many(&states, &mut outputs);
        for lane in 0..storage.len() {
            let mut expected = storage[lane].clone();
            expected.extend_from_slice(&suffixes[lane]);
            assert_eq!(
                outputs[lane],
                reference(&expected),
                "post-finalize incremental lane={lane}"
            );
        }
    }

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[test]
    fn incremental_many_partial_block_completion_matches_reference() {
        let engine = Md5Many::new();
        let storage: [[u8; 129]; 16] =
            core::array::from_fn(|lane| core::array::from_fn(|i| (lane * 23 + i * 41) as u8));
        let mut states = [Md5State::new(); 16];

        let first: [&[u8]; 16] = core::array::from_fn(|lane| &storage[lane][..31]);
        engine.update_many(&mut states, &first);
        let second: [&[u8]; 16] = core::array::from_fn(|lane| &storage[lane][31..64]);
        engine.update_many(&mut states, &second);
        let third: [&[u8]; 16] = core::array::from_fn(|lane| &storage[lane][64..129]);
        engine.update_many(&mut states, &third);

        let mut outputs = [[0u8; 16]; 16];
        engine.finalize_many(&states, &mut outputs);
        for lane in 0..16 {
            assert_eq!(outputs[lane], reference(&storage[lane]), "lane={lane}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn incremental_many_lockstep_then_mixed_matches_reference() {
        let engine = Md5Many::new();
        let lanes = engine.lanes();
        let storage: std::vec::Vec<std::vec::Vec<u8>> = (0..lanes)
            .map(|lane| (0..257).map(|i| (lane * 31 + i * 43 + 7) as u8).collect())
            .collect();
        let mut states = std::vec![Md5State::new(); lanes];

        // The first update is deliberately lockstep and leaves a partial
        // block in every stream. The second update diverges lane-by-lane and
        // must fall back to the general compaction scheduler without losing
        // the buffered prefix.
        let first: std::vec::Vec<&[u8]> = storage.iter().map(|message| &message[..47]).collect();
        engine.update_many(&mut states, &first);

        let second_lengths: std::vec::Vec<usize> =
            (0..lanes).map(|lane| 1 + ((lane * 37) % 173)).collect();
        let second: std::vec::Vec<&[u8]> = storage
            .iter()
            .zip(&second_lengths)
            .map(|(message, &len)| &message[47..47 + len])
            .collect();
        engine.update_many(&mut states, &second);

        let mut outputs = std::vec![[0u8; 16]; lanes];
        engine.finalize_many(&states, &mut outputs);
        for lane in 0..lanes {
            let len = 47 + second_lengths[lane];
            assert_eq!(
                outputs[lane],
                reference(&storage[lane][..len]),
                "lane={lane}, len={len}"
            );
        }
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn incremental_avx512_stateful_kernel_matches_reference() {
        let detected = fearless_simd::Level::new();
        let Some(avx512) = detected.as_avx512() else {
            return;
        };

        let engine = Md5Many::from_level(fearless_simd::Level::Avx512(avx512));
        let storage: [[u8; 521]; 16] =
            core::array::from_fn(|lane| core::array::from_fn(|i| (lane * 59 + i * 17) as u8));
        let mut states = [Md5State::new(); 16];

        for (start, end) in [(0, 17), (17, 64), (64, 129), (129, 257), (257, 521)] {
            let inputs: [&[u8]; 16] = core::array::from_fn(|lane| &storage[lane][start..end]);
            engine.update_many(&mut states, &inputs);
        }

        let mut outputs = [[0u8; 16]; 16];
        engine.finalize_many(&states, &mut outputs);
        for lane in 0..16 {
            assert_eq!(outputs[lane], reference(&storage[lane]), "lane={lane}");
        }
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn incremental_x86_interleaved_groups_match_reference() {
        fn check(engine: Md5Many, count: usize) {
            let storage: std::vec::Vec<std::vec::Vec<u8>> = (0..count)
                .map(|lane| (0..521).map(|i| (lane * 71 + i * 23 + 5) as u8).collect())
                .collect();
            let mut states = std::vec![Md5State::new(); count];

            // Three exactly block-aligned updates exercise the native/dual/
            // triple stateful kernels directly. The remaining uneven chunks
            // verify that the same states transition back through buffered
            // incremental scheduling before interleaved finalization.
            for (start, end) in [
                (0, 64),
                (64, 128),
                (128, 192),
                (192, 239),
                (239, 320),
                (320, 521),
            ] {
                let inputs: std::vec::Vec<&[u8]> =
                    storage.iter().map(|message| &message[start..end]).collect();
                engine.update_many(&mut states, &inputs);
            }

            let mut outputs = std::vec![[0u8; 16]; count];
            engine.finalize_many(&states, &mut outputs);
            for lane in 0..count {
                assert_eq!(
                    outputs[lane],
                    reference(&storage[lane]),
                    "count={count} lane={lane}"
                );
            }
        }

        let detected = fearless_simd::Level::new();
        if let Some(avx2) = detected.as_avx2() {
            let engine = Md5Many::from_level(fearless_simd::Level::Avx2(avx2));
            check(engine, 16);
            check(engine, 24);
        }
        if let Some(avx512) = detected.as_avx512() {
            let engine = Md5Many::from_level(fearless_simd::Level::Avx512(avx512));
            check(engine, 32);
            check(engine, 48);
        }
    }

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(64))]

        #[test]
        fn random_batches_match_reference(
            messages in proptest::collection::vec(
                proptest::collection::vec(proptest::prelude::any::<u8>(), 0..513),
                0..65,
            )
        ) {
            let inputs: std::vec::Vec<&[u8]> = messages.iter().map(std::vec::Vec::as_slice).collect();
            let mut outputs = std::vec![[0u8; 16]; inputs.len()];
            Md5Many::new().hash_many(&inputs, &mut outputs);
            for (input, output) in inputs.iter().zip(outputs) {
                proptest::prop_assert_eq!(output, reference(input));
            }
        }
    }

    #[cfg(feature = "digest")]
    #[test]
    fn digest_trait_compatibility() {
        let mut hasher = Md5::new();
        hasher.update(b"fearless");
        hasher.update(b" md5");
        let got = hasher.finalize();
        assert_eq!(got.as_slice(), reference(b"fearless md5"));
    }

    #[cfg(feature = "digest")]
    #[test]
    fn digest_streaming_multiblock_matches_reference() {
        let data: [u8; 4097] = core::array::from_fn(|i| (i * 29 + 7) as u8);
        let mut hasher = Md5::new();
        hasher.update(&data[..13]);
        hasher.update(&data[13..4013]);
        hasher.update(&data[4013..]);
        let got = hasher.finalize();
        assert_eq!(got.as_slice(), reference(&data));
    }
}
