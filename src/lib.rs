#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
//! High-throughput MD5 with portable multi-buffer SIMD.
//!
//! `fearless-md5` keeps the standard single-stream MD5 API separate from its
//! SIMD batch engine. A single MD5 stream has a dependency from one 64-byte
//! block to the next, while independent messages can occupy independent SIMD
//! lanes. On x86-64, `fearless_simd` 0.5 therefore maps naturally to 8-way
//! AVX2 MD5; SSE4.2, AArch64 NEON and WASM SIMD use four lanes.
//!
//! MD5 is cryptographically broken and must not be used for signatures,
//! password hashing, or adversarial integrity. This crate targets legacy
//! interoperability and non-adversarial checksumming workloads.

#[cfg(feature = "std")]
extern crate std;

mod consts;
mod scalar;
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
/// This is the portable scalar path. The SIMD acceleration in this crate is
/// designed for independent-message batches; use [`Md5Many`] or [`md5_many`]
/// when multiple messages are available at once.
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
    /// AVX2 is 8 lanes; the current SSE4.2/NEON/WASM paths are 4 lanes.
    #[must_use]
    pub fn lanes(self) -> usize {
        simd::lanes_with_level(self.level)
    }

    /// Hash many independent messages into `outputs`.
    ///
    /// Contiguous groups whose inputs have the same length use a streamlined
    /// SIMD path. Mixed-length lanes advance together; each lane's digest is
    /// materialized as soon as that message reaches its final padded block.
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

    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(64))]

        #[test]
        fn random_batches_match_reference(
            messages in proptest::collection::vec(
                proptest::collection::vec(proptest::prelude::any::<u8>(), 0..513),
                0..17,
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
