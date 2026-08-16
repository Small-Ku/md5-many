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
mod scalar;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod scalar_aarch64;
#[cfg(target_arch = "x86_64")]
mod scalar_x86_64;
#[cfg(target_arch = "x86_64")]
mod scalar_x86_64_avx512;
#[cfg(target_arch = "x86_64")]
mod scalar_x86_64_dual;
mod simd;

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

/// Compute the MD5 digest of a single byte slice.
///
/// On x86-64 this uses the optimized scalar compressor and may select an
/// XMM-width AVX-512VL single-stream backend on supported Intel CPUs.
/// Little-endian AArch64 uses a hand-scheduled integer compressor; other
/// targets use the portable Rust compressor. For independent-message SIMD
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
        Self {
            level: fearless_simd::Level::new(),
        }
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
    pub fn lanes(self) -> usize {
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
}
