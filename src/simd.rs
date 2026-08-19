use fearless_simd::{Level, Simd, SimdBase, dispatch};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __cpuid, __cpuid_count, __m128i, __m256i, __m512i, _mm_loadu_si128, _mm_storeu_si128,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
    _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi128_si256,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256,
    _mm512_add_epi32, _mm512_castsi128_si512, _mm512_castsi512_si128, _mm512_extracti32x4_epi32,
    _mm512_inserti32x4, _mm512_loadu_si512, _mm512_permutex2var_epi32, _mm512_rol_epi32,
    _mm512_set1_epi32, _mm512_setr_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
    _mm512_ternarylogic_epi32,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __cpuid, __cpuid_count, __m128i, __m256i, __m512i, _mm_loadu_si128, _mm_storeu_si128,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
    _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi128_si256,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256,
    _mm512_add_epi32, _mm512_castsi128_si512, _mm512_castsi512_si128, _mm512_extracti32x4_epi32,
    _mm512_inserti32x4, _mm512_loadu_si512, _mm512_permutex2var_epi32, _mm512_rol_epi32,
    _mm512_set1_epi32, _mm512_setr_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
    _mm512_ternarylogic_epi32,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::{Avx2, Avx512};

use crate::consts::{K, S, STATE_INIT};
use crate::incremental::Md5State;
use crate::scalar;

/// Maximum lane count supported by the current implementation.
///
/// `fearless_simd` 0.7 uses up to 16 native `u32` lanes with AVX-512.
const MAX_LANES: usize = 16;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum X86TuningClass {
    Generic = 1,
    AmdFamily19h = 2,
    IntelFamily06ModelCf = 3,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(unused_unsafe)]
fn x86_tuning_class() -> X86TuningClass {
    use core::sync::atomic::{AtomicU8, Ordering};

    // 0 = unknown. CPUID is serializing enough to matter for short hashes, so
    // classify once and keep model-specific crossover decisions out of the hot path.
    static CACHED: AtomicU8 = AtomicU8::new(0);
    match CACHED.load(Ordering::Relaxed) {
        1 => return X86TuningClass::Generic,
        2 => return X86TuningClass::AmdFamily19h,
        3 => return X86TuningClass::IntelFamily06ModelCf,
        _ => {}
    }

    // The crate's supported x86 targets provide CPUID. Rust 1.89 still
    // declares these intrinsics unsafe, so keep the compatibility boundary
    // explicit even though newer compilers no longer require it.
    // SAFETY: leaf 0 and leaf 1 are basic CPUID leaves on supported x86 CPUs.
    let (leaf0, leaf1) = unsafe { (__cpuid(0), __cpuid(1)) };
    let base_family = (leaf1.eax >> 8) & 0x0f;
    let ext_family = (leaf1.eax >> 20) & 0xff;
    let family = if base_family == 0x0f {
        base_family + ext_family
    } else {
        base_family
    };
    let base_model = (leaf1.eax >> 4) & 0x0f;
    let ext_model = (leaf1.eax >> 16) & 0x0f;
    let model = if base_family == 0x06 || base_family == 0x0f {
        base_model | (ext_model << 4)
    } else {
        base_model
    };

    let authentic_amd =
        leaf0.ebx == 0x6874_7541 && leaf0.edx == 0x6974_6e65 && leaf0.ecx == 0x444d_4163;
    let genuine_intel =
        leaf0.ebx == 0x756e_6547 && leaf0.edx == 0x4965_6e69 && leaf0.ecx == 0x6c65_746e;

    let class = if authentic_amd && family == 0x19 {
        X86TuningClass::AmdFamily19h
    } else if genuine_intel && family == 0x06 && model == 0xcf {
        // Deliberately narrow: the <=8-message ZMM crossover was measured on
        // this exact model. Other AVX-512 CPUs keep the conservative AVX2
        // small-batch path until we have measurements for them.
        X86TuningClass::IntelFamily06ModelCf
    } else {
        X86TuningClass::Generic
    };

    CACHED.store(class as u8, Ordering::Relaxed);
    class
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn amd_family_19h() -> bool {
    x86_tuning_class() == X86TuningClass::AmdFamily19h
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn intel_family_06_model_cf() -> bool {
    x86_tuning_class() == X86TuningClass::IntelFamily06ModelCf
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(unused_unsafe)]
fn x86_has_bmi1() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};

    // 0 = unknown, 1 = unavailable, 2 = available. Hypervisors may mask
    // BMI1 independently of family/model, so check the feature bit rather
    // than assuming every reported Family 19h guest exposes ANDN.
    static CACHED: AtomicU8 = AtomicU8::new(0);
    match CACHED.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }

    // SAFETY: leaf 0 is always available on supported x86 targets; leaf 7 is
    // queried only when leaf 0 reports that structured extended features are
    // available. Rust 1.89 declares both intrinsics unsafe.
    let max_leaf = unsafe { __cpuid(0) }.eax;
    let available = max_leaf >= 7 && (unsafe { __cpuid_count(7, 0) }.ebx & (1 << 3)) != 0;
    CACHED.store(if available { 2 } else { 1 }, Ordering::Relaxed);
    available
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn prefer_dual_scalar_pair(inputs: &[&[u8]]) -> bool {
    if !amd_family_19h() || !x86_has_bmi1() || inputs.len() != 2 {
        return false;
    }

    let blocks0 = padded_blocks_for_len(inputs[0].len());
    let blocks1 = padded_blocks_for_len(inputs[1].len());
    let min_blocks = core::cmp::min(blocks0, blocks1);
    let max_blocks = core::cmp::max(blocks0, blocks1);

    // Tiny pairs amortize the dual-GPR setup even when their lengths differ.
    // For larger messages, require at least a 1:16 common-block opportunity;
    // more extreme skew leaves too little work to overlap before the long lane
    // falls back to the ordinary scalar compressor.
    max_blocks <= 32 || min_blocks.saturating_mul(16) >= max_blocks
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn hash_three_dual_scalar(inputs: &[&[u8]], outputs: &mut [[u8; 16]]) -> bool {
    if !amd_family_19h() || !x86_has_bmi1() || inputs.len() != 3 {
        return false;
    }

    // Pair the two longest messages to maximize the common dual-GPR prefix.
    let mut order = [0usize, 1, 2];
    let blocks = [
        padded_blocks_for_len(inputs[0].len()),
        padded_blocks_for_len(inputs[1].len()),
        padded_blocks_for_len(inputs[2].len()),
    ];
    if blocks[order[0]] > blocks[order[1]] {
        order.swap(0, 1);
    }
    if blocks[order[1]] > blocks[order[2]] {
        order.swap(1, 2);
    }
    if blocks[order[0]] > blocks[order[1]] {
        order.swap(0, 1);
    }

    // Sparse AVX2 remains decisively better for equal and near-equal triples.
    // Split only when at most one lane carries more than a quarter of the
    // longest lane's padded-block work; this crossover is stable from 4 KiB
    // through 64 KiB on the measured AMD Family 19h host.
    if blocks[order[1]].saturating_mul(4) > blocks[order[2]] {
        return false;
    }

    // SAFETY: x86_has_bmi1() checked CPUID above.
    let pair =
        unsafe { crate::scalar_x86_64_dual::hash_pair_bmi1([inputs[order[1]], inputs[order[2]]]) };
    outputs[order[1]] = pair[0];
    outputs[order[2]] = pair[1];
    outputs[order[0]] = scalar::hash(inputs[order[0]]);
    true
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_equal_len_avx2_kernel(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
        debug_assert_eq!(inputs.len(), 8);
        debug_assert_eq!(outputs.len(), 8);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);

                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    // SAFETY: every entry is a 64-byte block, and unaligned AVX2
                    // loads read exactly bytes 0..32 and 32..64 respectively.
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    // SAFETY: bytes 32..64 are inside the same 64-byte block.
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (g, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $z), _mm256_andnot_si256($z, $y))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }

        macro_rules! step {
            (g, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t = _mm256_add_epi32($a, _mm256_andnot_si256($d, $c));
                t = _mm256_add_epi32(t, _mm256_set1_epi32(K[$round] as i32));
                t = _mm256_add_epi32(t, $words[$word]);
                t = _mm256_add_epi32(t, _mm256_and_si256($d, $b));
                let rotated = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t),
                );
                $a = _mm256_add_epi32($b, rotated);
            }};
            ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t = _mm256_add_epi32($a, mix!($which, $b, $c, $d, $ones));
                t = _mm256_add_epi32(t, _mm256_set1_epi32(K[$round] as i32));
                t = _mm256_add_epi32(t, $words[$word]);
                let rotated = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t),
                );
                $a = _mm256_add_epi32($b, rotated);
            }};
        }

        macro_rules! compress {
            ($word_expr:expr, $a:ident, $b:ident, $c:ident, $d:ident, $ones:ident) => {{
                let words = $word_expr;
                let initial = [$a, $b, $c, $d];
                step!(f, $a, $b, $c, $d, words, $ones, 0, 0, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 1, 1, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 2, 2, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 3, 3, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 4, 4, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 5, 5, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 6, 6, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 7, 7, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 8, 8, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 9, 9, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 10, 10, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 11, 11, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 12, 12, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 13, 13, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 14, 14, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 15, 15, 22);
                step!(g, $a, $b, $c, $d, words, $ones, 1, 16, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 6, 17, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 11, 18, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 0, 19, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 5, 20, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 10, 21, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 15, 22, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 4, 23, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 9, 24, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 14, 25, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 3, 26, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 8, 27, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 13, 28, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 2, 29, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 7, 30, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 12, 31, 20);
                step!(h, $a, $b, $c, $d, words, $ones, 5, 32, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 8, 33, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 11, 34, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 14, 35, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 1, 36, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 4, 37, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 7, 38, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 10, 39, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 13, 40, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 0, 41, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 3, 42, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 6, 43, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 9, 44, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 12, 45, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 15, 46, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 2, 47, 23);
                step!(i, $a, $b, $c, $d, words, $ones, 0, 48, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 7, 49, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 14, 50, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 5, 51, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 12, 52, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 3, 53, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 10, 54, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 1, 55, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 8, 56, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 15, 57, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 6, 58, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 13, 59, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 4, 60, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 11, 61, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 2, 62, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 9, 63, 21);
                $a = _mm256_add_epi32(initial[0], $a);
                $b = _mm256_add_epi32(initial[1], $b);
                $c = _mm256_add_epi32(initial[2], $c);
                $d = _mm256_add_epi32(initial[3], $d);
            }};
        }

        let mut a = _mm256_set1_epi32(STATE_INIT[0] as i32);
        let mut b = _mm256_set1_epi32(STATE_INIT[1] as i32);
        let mut c = _mm256_set1_epi32(STATE_INIT[2] as i32);
        let mut d = _mm256_set1_epi32(STATE_INIT[3] as i32);
        let all_ones = _mm256_set1_epi32(-1);

        let len = inputs[0].len();
        let full_blocks = len / 64;
        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d, all_ones);
        }

        let padded_blocks = padded_blocks_for_len(len);
        for block_index in full_blocks..padded_blocks {
            if block_index * 64 >= len {
                let padded = build_padded_block(inputs[0], padded_blocks, block_index);
                let words: [__m256i; 16] = core::array::from_fn(|word| {
                    let offset = word * 4;
                    let value = u32::from_le_bytes(
                        padded[offset..offset + 4]
                            .try_into()
                            .expect("four-byte word"),
                    );
                    _mm256_set1_epi32(value as i32)
                });
                compress!(words, a, b, c, d, all_ones);
                continue;
            }

            let padded: [[u8; 64]; 8] = core::array::from_fn(|lane| {
                build_padded_block(inputs[lane], padded_blocks, block_index)
            });
            let blocks: [&[u8; 64]; 8] = [
                &padded[0], &padded[1], &padded[2], &padded[3], &padded[4], &padded[5], &padded[6],
                &padded[7],
            ];
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d, all_ones);
        }

        let states = [a, b, c, d];
        let mut lanes = [[0u32; 8]; 4];
        for word in 0..4 {
            // SAFETY: each destination is exactly eight u32 values (32 bytes).
            unsafe {
                _mm256_storeu_si256(lanes[word].as_mut_ptr().cast::<__m256i>(), states[word]);
            }
        }
        for lane in 0..8 {
            for word in 0..4 {
                outputs[lane][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes[word][lane].to_le_bytes());
            }
        }
    }
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx2_state_kernel(avx2: Avx2, states: &mut [[u32; 4]], inputs: &[&[u8]]) {
        debug_assert_eq!(states.len(), 8);
        debug_assert_eq!(inputs.len(), 8);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        debug_assert!(inputs[0].len().is_multiple_of(64));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);

                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    // SAFETY: every entry is a 64-byte block, and unaligned AVX2
                    // loads read exactly bytes 0..32 and 32..64 respectively.
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    // SAFETY: bytes 32..64 are inside the same 64-byte block.
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (g, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $z), _mm256_andnot_si256($z, $y))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }

        macro_rules! step {
            (g, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t = _mm256_add_epi32($a, _mm256_andnot_si256($d, $c));
                t = _mm256_add_epi32(t, _mm256_set1_epi32(K[$round] as i32));
                t = _mm256_add_epi32(t, $words[$word]);
                t = _mm256_add_epi32(t, _mm256_and_si256($d, $b));
                let rotated = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t),
                );
                $a = _mm256_add_epi32($b, rotated);
            }};
            ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t = _mm256_add_epi32($a, mix!($which, $b, $c, $d, $ones));
                t = _mm256_add_epi32(t, _mm256_set1_epi32(K[$round] as i32));
                t = _mm256_add_epi32(t, $words[$word]);
                let rotated = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t),
                );
                $a = _mm256_add_epi32($b, rotated);
            }};
        }

        macro_rules! compress {
            ($word_expr:expr, $a:ident, $b:ident, $c:ident, $d:ident, $ones:ident) => {{
                let words = $word_expr;
                let initial = [$a, $b, $c, $d];
                step!(f, $a, $b, $c, $d, words, $ones, 0, 0, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 1, 1, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 2, 2, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 3, 3, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 4, 4, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 5, 5, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 6, 6, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 7, 7, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 8, 8, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 9, 9, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 10, 10, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 11, 11, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 12, 12, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 13, 13, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 14, 14, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 15, 15, 22);
                step!(g, $a, $b, $c, $d, words, $ones, 1, 16, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 6, 17, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 11, 18, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 0, 19, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 5, 20, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 10, 21, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 15, 22, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 4, 23, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 9, 24, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 14, 25, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 3, 26, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 8, 27, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 13, 28, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 2, 29, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 7, 30, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 12, 31, 20);
                step!(h, $a, $b, $c, $d, words, $ones, 5, 32, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 8, 33, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 11, 34, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 14, 35, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 1, 36, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 4, 37, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 7, 38, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 10, 39, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 13, 40, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 0, 41, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 3, 42, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 6, 43, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 9, 44, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 12, 45, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 15, 46, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 2, 47, 23);
                step!(i, $a, $b, $c, $d, words, $ones, 0, 48, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 7, 49, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 14, 50, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 5, 51, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 12, 52, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 3, 53, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 10, 54, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 1, 55, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 8, 56, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 15, 57, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 6, 58, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 13, 59, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 4, 60, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 11, 61, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 2, 62, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 9, 63, 21);
                $a = _mm256_add_epi32(initial[0], $a);
                $b = _mm256_add_epi32(initial[1], $b);
                $c = _mm256_add_epi32(initial[2], $c);
                $d = _mm256_add_epi32(initial[3], $d);
            }};
        }

        let mut lane_words = [[0u32; 8]; 4];
        for word in 0..4 {
            for lane in 0..8 {
                lane_words[word][lane] = states[lane][word];
            }
        }
        // SAFETY: each lane_words row is exactly eight u32 values (32 bytes).
        let mut a = unsafe { _mm256_loadu_si256(lane_words[0].as_ptr().cast::<__m256i>()) };
        let mut b = unsafe { _mm256_loadu_si256(lane_words[1].as_ptr().cast::<__m256i>()) };
        let mut c = unsafe { _mm256_loadu_si256(lane_words[2].as_ptr().cast::<__m256i>()) };
        let mut d = unsafe { _mm256_loadu_si256(lane_words[3].as_ptr().cast::<__m256i>()) };
        let all_ones = _mm256_set1_epi32(-1);

        let block_count = inputs[0].len() / 64;
        for block_index in 0..block_count {
            let offset = block_index * 64;
            let blocks: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d, all_ones);
        }

        let vector_state = [a, b, c, d];
        for word in 0..4 {
            // SAFETY: each destination is exactly eight u32 values (32 bytes).
            unsafe {
                _mm256_storeu_si256(
                    lane_words[word].as_mut_ptr().cast::<__m256i>(),
                    vector_state[word],
                );
            }
        }
        for lane in 0..8 {
            for word in 0..4 {
                states[lane][word] = lane_words[word][lane];
            }
        }
    }
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_mixed_len_avx2_kernel(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
        debug_assert_eq!(inputs.len(), 8);
        debug_assert_eq!(outputs.len(), 8);
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);

                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    // SAFETY: every entry is a 64-byte block, and unaligned AVX2
                    // loads read exactly bytes 0..32 and 32..64 respectively.
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    // SAFETY: bytes 32..64 are inside the same 64-byte block.
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (g, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $z), _mm256_andnot_si256($z, $y))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }

        macro_rules! step {
            (g, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t = _mm256_add_epi32($a, _mm256_andnot_si256($d, $c));
                t = _mm256_add_epi32(t, _mm256_set1_epi32(K[$round] as i32));
                t = _mm256_add_epi32(t, $words[$word]);
                t = _mm256_add_epi32(t, _mm256_and_si256($d, $b));
                let rotated = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t),
                );
                $a = _mm256_add_epi32($b, rotated);
            }};
            ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t = _mm256_add_epi32($a, mix!($which, $b, $c, $d, $ones));
                t = _mm256_add_epi32(t, _mm256_set1_epi32(K[$round] as i32));
                t = _mm256_add_epi32(t, $words[$word]);
                let rotated = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t),
                );
                $a = _mm256_add_epi32($b, rotated);
            }};
        }

        macro_rules! compress {
            ($word_expr:expr, $a:ident, $b:ident, $c:ident, $d:ident, $ones:ident) => {{
                let words = $word_expr;
                let initial = [$a, $b, $c, $d];
                step!(f, $a, $b, $c, $d, words, $ones, 0, 0, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 1, 1, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 2, 2, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 3, 3, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 4, 4, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 5, 5, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 6, 6, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 7, 7, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 8, 8, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 9, 9, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 10, 10, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 11, 11, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 12, 12, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 13, 13, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 14, 14, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 15, 15, 22);
                step!(g, $a, $b, $c, $d, words, $ones, 1, 16, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 6, 17, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 11, 18, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 0, 19, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 5, 20, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 10, 21, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 15, 22, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 4, 23, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 9, 24, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 14, 25, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 3, 26, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 8, 27, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 13, 28, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 2, 29, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 7, 30, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 12, 31, 20);
                step!(h, $a, $b, $c, $d, words, $ones, 5, 32, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 8, 33, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 11, 34, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 14, 35, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 1, 36, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 4, 37, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 7, 38, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 10, 39, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 13, 40, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 0, 41, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 3, 42, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 6, 43, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 9, 44, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 12, 45, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 15, 46, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 2, 47, 23);
                step!(i, $a, $b, $c, $d, words, $ones, 0, 48, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 7, 49, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 14, 50, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 5, 51, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 12, 52, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 3, 53, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 10, 54, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 1, 55, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 8, 56, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 15, 57, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 6, 58, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 13, 59, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 4, 60, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 11, 61, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 2, 62, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 9, 63, 21);
                $a = _mm256_add_epi32(initial[0], $a);
                $b = _mm256_add_epi32(initial[1], $b);
                $c = _mm256_add_epi32(initial[2], $c);
                $d = _mm256_add_epi32(initial[3], $d);
            }};
        }
        let mut full_counts = [0usize; 8];
        let mut block_counts = [0usize; 8];
        let mut common_full = usize::MAX;
        let mut max_blocks = 0usize;
        for lane in 0..8 {
            full_counts[lane] = inputs[lane].len() / 64;
            block_counts[lane] = padded_blocks_for_len(inputs[lane].len());
            common_full = core::cmp::min(common_full, full_counts[lane]);
            max_blocks = core::cmp::max(max_blocks, block_counts[lane]);
        }
        let mut a = _mm256_set1_epi32(STATE_INIT[0] as i32);
        let mut b = _mm256_set1_epi32(STATE_INIT[1] as i32);
        let mut c = _mm256_set1_epi32(STATE_INIT[2] as i32);
        let mut d = _mm256_set1_epi32(STATE_INIT[3] as i32);
        let all_ones = _mm256_set1_epi32(-1);

        // All lanes have real full blocks throughout this prefix, so use the
        // same direct transpose/compression path as equal-length hashing.
        for block_index in 0..common_full {
            let offset = block_index * 64;
            let blocks: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d, all_ones);
        }

        for block_index in common_full..max_blocks {
            let base = block_index * 64;
            let mut scratch = [[0u8; 64]; 8];
            for lane in 0..8 {
                if block_index >= full_counts[lane] && block_index < block_counts[lane] {
                    scratch[lane] =
                        build_padded_block(inputs[lane], block_counts[lane], block_index);
                }
            }
            let blocks: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                if block_index < full_counts[lane] {
                    inputs[lane][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[lane]
                }
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d, all_ones);

            if block_counts.iter().any(|&count| count == block_index + 1) {
                let states = [a, b, c, d];
                let mut lanes_out = [[0u32; 8]; 4];
                for word in 0..4 {
                    // SAFETY: each destination exactly matches one SIMD vector.
                    unsafe {
                        _mm256_storeu_si256(
                            lanes_out[word].as_mut_ptr().cast::<__m256i>(),
                            states[word],
                        );
                    }
                }
                for lane in 0..8 {
                    if block_counts[lane] == block_index + 1 {
                        for word in 0..4 {
                            outputs[lane][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes_out[word][lane].to_le_bytes());
                        }
                    }
                }
            }
        }
    }
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_equal_len_avx2_dual_kernel(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
        debug_assert_eq!(inputs.len(), 16);
        debug_assert_eq!(outputs.len(), 16);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);

                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    // SAFETY: every entry is a 64-byte block, and unaligned AVX2
                    // loads read exactly bytes 0..32 and 32..64 respectively.
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    // SAFETY: bytes 32..64 are inside the same 64-byte block.
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (g, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $z), _mm256_andnot_si256($z, $y))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }

        macro_rules! step2 {
            (g, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t0 = _mm256_add_epi32($a0, _mm256_andnot_si256($d0, $c0));
                let mut t1 = _mm256_add_epi32($a1, _mm256_andnot_si256($d1, $c1));
                let key = _mm256_set1_epi32(K[$round] as i32);
                t0 = _mm256_add_epi32(t0, key);
                t1 = _mm256_add_epi32(t1, key);
                t0 = _mm256_add_epi32(t0, $words0[$word]);
                t1 = _mm256_add_epi32(t1, $words1[$word]);
                t0 = _mm256_add_epi32(t0, _mm256_and_si256($d0, $b0));
                t1 = _mm256_add_epi32(t1, _mm256_and_si256($d1, $b1));
                let rotated0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let rotated1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                $a0 = _mm256_add_epi32($b0, rotated0);
                $a1 = _mm256_add_epi32($b1, rotated1);
            }};
            ($which:ident, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed0 = mix!($which, $b0, $c0, $d0, $ones);
                let mixed1 = mix!($which, $b1, $c1, $d1, $ones);
                let mut t0 = _mm256_add_epi32($a0, mixed0);
                let mut t1 = _mm256_add_epi32($a1, mixed1);
                let key = _mm256_set1_epi32(K[$round] as i32);
                t0 = _mm256_add_epi32(t0, key);
                t1 = _mm256_add_epi32(t1, key);
                t0 = _mm256_add_epi32(t0, $words0[$word]);
                t1 = _mm256_add_epi32(t1, $words1[$word]);
                let rotated0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let rotated1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                $a0 = _mm256_add_epi32($b0, rotated0);
                $a1 = _mm256_add_epi32($b1, rotated1);
            }};
        }

        macro_rules! compress2 {
            ($words0:expr, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $words1:expr, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $ones:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 0, 0, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 1, 1, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 2, 2, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 3, 3, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 4, 4, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 5, 5, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 6, 6, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 7, 7, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 8, 8, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 9, 9, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 10, 10, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 11, 11, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 12, 12, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 13, 13, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 14, 14, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 15, 15, 22
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 1, 16, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 6, 17, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 11, 18, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 0, 19, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 5, 20, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 10, 21, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 15, 22, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 4, 23, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 9, 24, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 14, 25, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 3, 26, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 8, 27, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 13, 28, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 2, 29, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 7, 30, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 12, 31, 20
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 5, 32, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 8, 33, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 11, 34, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 14, 35, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 1, 36, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 4, 37, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 7, 38, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 10, 39, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 13, 40, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 0, 41, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 3, 42, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 6, 43, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 9, 44, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 12, 45, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 15, 46, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 2, 47, 23
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 0, 48, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 7, 49, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 14, 50, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 5, 51, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 12, 52, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 3, 53, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 10, 54, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 1, 55, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 8, 56, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 15, 57, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 6, 58, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 13, 59, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 4, 60, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 11, 61, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 2, 62, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 9, 63, 21
                );
                $a0 = _mm256_add_epi32(initial0[0], $a0);
                $b0 = _mm256_add_epi32(initial0[1], $b0);
                $c0 = _mm256_add_epi32(initial0[2], $c0);
                $d0 = _mm256_add_epi32(initial0[3], $d0);
                $a1 = _mm256_add_epi32(initial1[0], $a1);
                $b1 = _mm256_add_epi32(initial1[1], $b1);
                $c1 = _mm256_add_epi32(initial1[2], $c1);
                $d1 = _mm256_add_epi32(initial1[3], $d1);
            }};
        }

        let len = inputs[0].len();
        let mut a0 = _mm256_set1_epi32(STATE_INIT[0] as i32);
        let mut b0 = _mm256_set1_epi32(STATE_INIT[1] as i32);
        let mut c0 = _mm256_set1_epi32(STATE_INIT[2] as i32);
        let mut d0 = _mm256_set1_epi32(STATE_INIT[3] as i32);
        let mut a1 = a0;
        let mut b1 = b0;
        let mut c1 = c0;
        let mut d1 = d0;
        let all_ones = _mm256_set1_epi32(-1);
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 8][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, all_ones);
        }

        let padded_blocks = padded_blocks_for_len(len);
        for block_index in full_blocks..padded_blocks {
            if block_index * 64 >= len {
                let padded = build_padded_block(inputs[0], padded_blocks, block_index);
                let words: [__m256i; 16] = core::array::from_fn(|word| {
                    let offset = word * 4;
                    let value = u32::from_le_bytes(
                        padded[offset..offset + 4]
                            .try_into()
                            .expect("four-byte word"),
                    );
                    _mm256_set1_epi32(value as i32)
                });
                compress2!(words, a0, b0, c0, d0, words, a1, b1, c1, d1, all_ones);
                continue;
            }

            let padded: [[u8; 64]; 16] = core::array::from_fn(|lane| {
                build_padded_block(inputs[lane], padded_blocks, block_index)
            });
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| &padded[lane]);
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| &padded[lane + 8]);
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, all_ones);
        }

        let states0 = [a0, b0, c0, d0];
        let states1 = [a1, b1, c1, d1];
        let mut lanes0 = [[0u32; 8]; 4];
        let mut lanes1 = [[0u32; 8]; 4];
        for word in 0..4 {
            // SAFETY: each destination is exactly eight u32 values (32 bytes).
            unsafe {
                _mm256_storeu_si256(lanes0[word].as_mut_ptr().cast::<__m256i>(), states0[word]);
                _mm256_storeu_si256(lanes1[word].as_mut_ptr().cast::<__m256i>(), states1[word]);
            }
        }
        for lane in 0..8 {
            for word in 0..4 {
                outputs[lane][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes0[word][lane].to_le_bytes());
                outputs[lane + 8][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes1[word][lane].to_le_bytes());
            }
        }
    }
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_equal_len_avx2_triple_kernel(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
        debug_assert_eq!(inputs.len(), 24);
        debug_assert_eq!(outputs.len(), 24);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);
                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);
                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }
        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }
        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }
        macro_rules! step3 {
            (g, $a0:ident,$b0:ident,$c0:ident,$d0:ident, $a1:ident,$b1:ident,$c1:ident,$d1:ident, $a2:ident,$b2:ident,$c2:ident,$d2:ident, $w0:ident,$w1:ident,$w2:ident,$ones:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm256_set1_epi32(K[$round] as i32);
                let mut t0 = _mm256_add_epi32($a0, _mm256_andnot_si256($d0, $c0));
                let mut t1 = _mm256_add_epi32($a1, _mm256_andnot_si256($d1, $c1));
                let mut t2 = _mm256_add_epi32($a2, _mm256_andnot_si256($d2, $c2));
                t0 = _mm256_add_epi32(_mm256_add_epi32(t0, key), $w0[$word]);
                t1 = _mm256_add_epi32(_mm256_add_epi32(t1, key), $w1[$word]);
                t2 = _mm256_add_epi32(_mm256_add_epi32(t2, key), $w2[$word]);
                t0 = _mm256_add_epi32(t0, _mm256_and_si256($d0, $b0));
                t1 = _mm256_add_epi32(t1, _mm256_and_si256($d1, $b1));
                t2 = _mm256_add_epi32(t2, _mm256_and_si256($d2, $b2));
                let r0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let r1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                let r2 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t2),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t2),
                );
                $a0 = _mm256_add_epi32($b0, r0);
                $a1 = _mm256_add_epi32($b1, r1);
                $a2 = _mm256_add_epi32($b2, r2);
            }};
            ($which:ident, $a0:ident,$b0:ident,$c0:ident,$d0:ident, $a1:ident,$b1:ident,$c1:ident,$d1:ident, $a2:ident,$b2:ident,$c2:ident,$d2:ident, $w0:ident,$w1:ident,$w2:ident,$ones:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm256_set1_epi32(K[$round] as i32);
                let mut t0 = _mm256_add_epi32($a0, mix!($which, $b0, $c0, $d0, $ones));
                let mut t1 = _mm256_add_epi32($a1, mix!($which, $b1, $c1, $d1, $ones));
                let mut t2 = _mm256_add_epi32($a2, mix!($which, $b2, $c2, $d2, $ones));
                t0 = _mm256_add_epi32(_mm256_add_epi32(t0, key), $w0[$word]);
                t1 = _mm256_add_epi32(_mm256_add_epi32(t1, key), $w1[$word]);
                t2 = _mm256_add_epi32(_mm256_add_epi32(t2, key), $w2[$word]);
                let r0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let r1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                let r2 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t2),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t2),
                );
                $a0 = _mm256_add_epi32($b0, r0);
                $a1 = _mm256_add_epi32($b1, r1);
                $a2 = _mm256_add_epi32($b2, r2);
            }};
        }
        macro_rules! compress3 {
            ($words0:expr,$a0:ident,$b0:ident,$c0:ident,$d0:ident, $words1:expr,$a1:ident,$b1:ident,$c1:ident,$d1:ident, $words2:expr,$a2:ident,$b2:ident,$c2:ident,$d2:ident, $ones:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let words2 = $words2;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                let initial2 = [$a2, $b2, $c2, $d2];
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 0, 0, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 1, 1, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 2, 2, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 3, 3, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 4, 4, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 5, 5, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 6, 6, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 7, 7, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 8, 8, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 9, 9, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 10, 10, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 11, 11, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 12, 12, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 13, 13, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 14, 14, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 15, 15, 22
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 1, 16, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 6, 17, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 11, 18, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 0, 19, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 5, 20, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 10, 21, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 15, 22, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 4, 23, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 9, 24, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 14, 25, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 3, 26, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 8, 27, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 13, 28, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 2, 29, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 7, 30, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 12, 31, 20
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 5, 32, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 8, 33, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 11, 34, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 14, 35, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 1, 36, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 4, 37, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 7, 38, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 10, 39, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 13, 40, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 0, 41, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 3, 42, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 6, 43, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 9, 44, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 12, 45, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 15, 46, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 2, 47, 23
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 0, 48, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 7, 49, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 14, 50, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 5, 51, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 12, 52, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 3, 53, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 10, 54, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 1, 55, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 8, 56, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 15, 57, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 6, 58, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 13, 59, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 4, 60, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 11, 61, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 2, 62, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 9, 63, 21
                );
                $a0 = _mm256_add_epi32(initial0[0], $a0);
                $b0 = _mm256_add_epi32(initial0[1], $b0);
                $c0 = _mm256_add_epi32(initial0[2], $c0);
                $d0 = _mm256_add_epi32(initial0[3], $d0);
                $a1 = _mm256_add_epi32(initial1[0], $a1);
                $b1 = _mm256_add_epi32(initial1[1], $b1);
                $c1 = _mm256_add_epi32(initial1[2], $c1);
                $d1 = _mm256_add_epi32(initial1[3], $d1);
                $a2 = _mm256_add_epi32(initial2[0], $a2);
                $b2 = _mm256_add_epi32(initial2[1], $b2);
                $c2 = _mm256_add_epi32(initial2[2], $c2);
                $d2 = _mm256_add_epi32(initial2[3], $d2);
            }};
        }

        let len = inputs[0].len();
        let mut a0 = _mm256_set1_epi32(STATE_INIT[0] as i32);
        let mut b0 = _mm256_set1_epi32(STATE_INIT[1] as i32);
        let mut c0 = _mm256_set1_epi32(STATE_INIT[2] as i32);
        let mut d0 = _mm256_set1_epi32(STATE_INIT[3] as i32);
        let mut a1 = a0;
        let mut b1 = b0;
        let mut c1 = c0;
        let mut d1 = d0;
        let mut a2 = a0;
        let mut b2 = b0;
        let mut c2 = c0;
        let mut d2 = d0;
        let all_ones = _mm256_set1_epi32(-1);
        let full_blocks = len / 64;
        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 8][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks2: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2, all_ones
            );
        }
        let padded_blocks = padded_blocks_for_len(len);
        for block_index in full_blocks..padded_blocks {
            if block_index * 64 >= len {
                let padded = build_padded_block(inputs[0], padded_blocks, block_index);
                let words: [__m256i; 16] = core::array::from_fn(|word| {
                    let offset = word * 4;
                    let value = u32::from_le_bytes(
                        padded[offset..offset + 4]
                            .try_into()
                            .expect("four-byte word"),
                    );
                    _mm256_set1_epi32(value as i32)
                });
                compress3!(
                    words, a0, b0, c0, d0, words, a1, b1, c1, d1, words, a2, b2, c2, d2, all_ones
                );
                continue;
            }

            let padded: [[u8; 64]; 24] = core::array::from_fn(|lane| {
                build_padded_block(inputs[lane], padded_blocks, block_index)
            });
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| &padded[lane]);
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| &padded[lane + 8]);
            let blocks2: [&[u8; 64]; 8] = core::array::from_fn(|lane| &padded[lane + 16]);
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2, all_ones
            );
        }
        let states0 = [a0, b0, c0, d0];
        let states1 = [a1, b1, c1, d1];
        let states2 = [a2, b2, c2, d2];
        let mut lanes0 = [[0u32; 8]; 4];
        let mut lanes1 = [[0u32; 8]; 4];
        let mut lanes2 = [[0u32; 8]; 4];
        for word in 0..4 {
            unsafe {
                _mm256_storeu_si256(lanes0[word].as_mut_ptr().cast::<__m256i>(), states0[word]);
                _mm256_storeu_si256(lanes1[word].as_mut_ptr().cast::<__m256i>(), states1[word]);
                _mm256_storeu_si256(lanes2[word].as_mut_ptr().cast::<__m256i>(), states2[word]);
            }
        }
        for lane in 0..8 {
            for word in 0..4 {
                outputs[lane][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes0[word][lane].to_le_bytes());
                outputs[lane + 8][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes1[word][lane].to_le_bytes());
                outputs[lane + 16][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes2[word][lane].to_le_bytes());
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_mixed_len_avx2_triple_kernel(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
        debug_assert_eq!(inputs.len(), 24);
        debug_assert_eq!(outputs.len(), 24);
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);
                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);
                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }
        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }
        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }
        macro_rules! step3 {
            (g, $a0:ident,$b0:ident,$c0:ident,$d0:ident, $a1:ident,$b1:ident,$c1:ident,$d1:ident, $a2:ident,$b2:ident,$c2:ident,$d2:ident, $w0:ident,$w1:ident,$w2:ident,$ones:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm256_set1_epi32(K[$round] as i32);
                let mut t0 = _mm256_add_epi32($a0, _mm256_andnot_si256($d0, $c0));
                let mut t1 = _mm256_add_epi32($a1, _mm256_andnot_si256($d1, $c1));
                let mut t2 = _mm256_add_epi32($a2, _mm256_andnot_si256($d2, $c2));
                t0 = _mm256_add_epi32(_mm256_add_epi32(t0, key), $w0[$word]);
                t1 = _mm256_add_epi32(_mm256_add_epi32(t1, key), $w1[$word]);
                t2 = _mm256_add_epi32(_mm256_add_epi32(t2, key), $w2[$word]);
                t0 = _mm256_add_epi32(t0, _mm256_and_si256($d0, $b0));
                t1 = _mm256_add_epi32(t1, _mm256_and_si256($d1, $b1));
                t2 = _mm256_add_epi32(t2, _mm256_and_si256($d2, $b2));
                let r0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let r1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                let r2 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t2),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t2),
                );
                $a0 = _mm256_add_epi32($b0, r0);
                $a1 = _mm256_add_epi32($b1, r1);
                $a2 = _mm256_add_epi32($b2, r2);
            }};
            ($which:ident, $a0:ident,$b0:ident,$c0:ident,$d0:ident, $a1:ident,$b1:ident,$c1:ident,$d1:ident, $a2:ident,$b2:ident,$c2:ident,$d2:ident, $w0:ident,$w1:ident,$w2:ident,$ones:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm256_set1_epi32(K[$round] as i32);
                let mut t0 = _mm256_add_epi32($a0, mix!($which, $b0, $c0, $d0, $ones));
                let mut t1 = _mm256_add_epi32($a1, mix!($which, $b1, $c1, $d1, $ones));
                let mut t2 = _mm256_add_epi32($a2, mix!($which, $b2, $c2, $d2, $ones));
                t0 = _mm256_add_epi32(_mm256_add_epi32(t0, key), $w0[$word]);
                t1 = _mm256_add_epi32(_mm256_add_epi32(t1, key), $w1[$word]);
                t2 = _mm256_add_epi32(_mm256_add_epi32(t2, key), $w2[$word]);
                let r0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let r1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                let r2 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t2),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t2),
                );
                $a0 = _mm256_add_epi32($b0, r0);
                $a1 = _mm256_add_epi32($b1, r1);
                $a2 = _mm256_add_epi32($b2, r2);
            }};
        }
        macro_rules! compress3 {
            ($words0:expr,$a0:ident,$b0:ident,$c0:ident,$d0:ident, $words1:expr,$a1:ident,$b1:ident,$c1:ident,$d1:ident, $words2:expr,$a2:ident,$b2:ident,$c2:ident,$d2:ident, $ones:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let words2 = $words2;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                let initial2 = [$a2, $b2, $c2, $d2];
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 0, 0, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 1, 1, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 2, 2, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 3, 3, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 4, 4, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 5, 5, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 6, 6, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 7, 7, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 8, 8, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 9, 9, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 10, 10, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 11, 11, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 12, 12, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 13, 13, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 14, 14, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 15, 15, 22
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 1, 16, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 6, 17, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 11, 18, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 0, 19, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 5, 20, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 10, 21, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 15, 22, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 4, 23, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 9, 24, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 14, 25, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 3, 26, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 8, 27, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 13, 28, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 2, 29, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 7, 30, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 12, 31, 20
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 5, 32, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 8, 33, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 11, 34, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 14, 35, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 1, 36, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 4, 37, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 7, 38, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 10, 39, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 13, 40, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 0, 41, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 3, 42, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 6, 43, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 9, 44, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 12, 45, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 15, 46, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 2, 47, 23
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 0, 48, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 7, 49, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 14, 50, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 5, 51, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 12, 52, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 3, 53, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 10, 54, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 1, 55, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 8, 56, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 15, 57, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 6, 58, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 13, 59, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 4, 60, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 11, 61, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 2, 62, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 9, 63, 21
                );
                $a0 = _mm256_add_epi32(initial0[0], $a0);
                $b0 = _mm256_add_epi32(initial0[1], $b0);
                $c0 = _mm256_add_epi32(initial0[2], $c0);
                $d0 = _mm256_add_epi32(initial0[3], $d0);
                $a1 = _mm256_add_epi32(initial1[0], $a1);
                $b1 = _mm256_add_epi32(initial1[1], $b1);
                $c1 = _mm256_add_epi32(initial1[2], $c1);
                $d1 = _mm256_add_epi32(initial1[3], $d1);
                $a2 = _mm256_add_epi32(initial2[0], $a2);
                $b2 = _mm256_add_epi32(initial2[1], $b2);
                $c2 = _mm256_add_epi32(initial2[2], $c2);
                $d2 = _mm256_add_epi32(initial2[3], $d2);
            }};
        }

        let mut full_counts = [0usize; 24];
        let mut block_counts = [0usize; 24];
        let mut common_full = usize::MAX;
        let mut max_blocks = 0usize;
        for lane in 0..24 {
            full_counts[lane] = inputs[lane].len() / 64;
            block_counts[lane] = padded_blocks_for_len(inputs[lane].len());
            common_full = core::cmp::min(common_full, full_counts[lane]);
            max_blocks = core::cmp::max(max_blocks, block_counts[lane]);
        }
        let mut a0 = _mm256_set1_epi32(STATE_INIT[0] as i32);
        let mut b0 = _mm256_set1_epi32(STATE_INIT[1] as i32);
        let mut c0 = _mm256_set1_epi32(STATE_INIT[2] as i32);
        let mut d0 = _mm256_set1_epi32(STATE_INIT[3] as i32);
        let mut a1 = a0;
        let mut b1 = b0;
        let mut c1 = c0;
        let mut d1 = d0;
        let mut a2 = a0;
        let mut b2 = b0;
        let mut c2 = c0;
        let mut d2 = d0;
        let all_ones = _mm256_set1_epi32(-1);

        for block_index in 0..common_full {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 8][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks2: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2, all_ones
            );
        }

        for block_index in common_full..max_blocks {
            let base = block_index * 64;
            let mut scratch = [[0u8; 64]; 24];
            for lane in 0..24 {
                if block_index >= full_counts[lane] && block_index < block_counts[lane] {
                    scratch[lane] =
                        build_padded_block(inputs[lane], block_counts[lane], block_index);
                }
            }
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                if block_index < full_counts[lane] {
                    inputs[lane][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[lane]
                }
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                let i = lane + 8;
                if block_index < full_counts[i] {
                    inputs[i][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[i]
                }
            });
            let blocks2: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                let i = lane + 16;
                if block_index < full_counts[i] {
                    inputs[i][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[i]
                }
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2, all_ones
            );

            if block_counts.iter().any(|&count| count == block_index + 1) {
                let states0 = [a0, b0, c0, d0];
                let states1 = [a1, b1, c1, d1];
                let states2 = [a2, b2, c2, d2];
                let mut lanes0 = [[0u32; 8]; 4];
                let mut lanes1 = [[0u32; 8]; 4];
                let mut lanes2 = [[0u32; 8]; 4];
                for word in 0..4 {
                    unsafe {
                        _mm256_storeu_si256(
                            lanes0[word].as_mut_ptr().cast::<__m256i>(),
                            states0[word],
                        );
                        _mm256_storeu_si256(
                            lanes1[word].as_mut_ptr().cast::<__m256i>(),
                            states1[word],
                        );
                        _mm256_storeu_si256(
                            lanes2[word].as_mut_ptr().cast::<__m256i>(),
                            states2[word],
                        );
                    }
                }
                for lane in 0..8 {
                    if block_counts[lane] == block_index + 1 {
                        for word in 0..4 {
                            outputs[lane][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes0[word][lane].to_le_bytes());
                        }
                    }
                    let i1 = lane + 8;
                    if block_counts[i1] == block_index + 1 {
                        for word in 0..4 {
                            outputs[i1][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes1[word][lane].to_le_bytes());
                        }
                    }
                    let i2 = lane + 16;
                    if block_counts[i2] == block_index + 1 {
                        for word in 0..4 {
                            outputs[i2][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes2[word][lane].to_le_bytes());
                        }
                    }
                }
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_mixed_len_avx2_dual_kernel(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
        debug_assert_eq!(inputs.len(), 16);
        debug_assert_eq!(outputs.len(), 16);
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);

                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    // SAFETY: every entry is a 64-byte block, and unaligned AVX2
                    // loads read exactly bytes 0..32 and 32..64 respectively.
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    // SAFETY: bytes 32..64 are inside the same 64-byte block.
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (g, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $z), _mm256_andnot_si256($z, $y))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }

        macro_rules! step2 {
            (g, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t0 = _mm256_add_epi32($a0, _mm256_andnot_si256($d0, $c0));
                let mut t1 = _mm256_add_epi32($a1, _mm256_andnot_si256($d1, $c1));
                let key = _mm256_set1_epi32(K[$round] as i32);
                t0 = _mm256_add_epi32(t0, key);
                t1 = _mm256_add_epi32(t1, key);
                t0 = _mm256_add_epi32(t0, $words0[$word]);
                t1 = _mm256_add_epi32(t1, $words1[$word]);
                t0 = _mm256_add_epi32(t0, _mm256_and_si256($d0, $b0));
                t1 = _mm256_add_epi32(t1, _mm256_and_si256($d1, $b1));
                let rotated0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let rotated1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                $a0 = _mm256_add_epi32($b0, rotated0);
                $a1 = _mm256_add_epi32($b1, rotated1);
            }};
            ($which:ident, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed0 = mix!($which, $b0, $c0, $d0, $ones);
                let mixed1 = mix!($which, $b1, $c1, $d1, $ones);
                let mut t0 = _mm256_add_epi32($a0, mixed0);
                let mut t1 = _mm256_add_epi32($a1, mixed1);
                let key = _mm256_set1_epi32(K[$round] as i32);
                t0 = _mm256_add_epi32(t0, key);
                t1 = _mm256_add_epi32(t1, key);
                t0 = _mm256_add_epi32(t0, $words0[$word]);
                t1 = _mm256_add_epi32(t1, $words1[$word]);
                let rotated0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let rotated1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                $a0 = _mm256_add_epi32($b0, rotated0);
                $a1 = _mm256_add_epi32($b1, rotated1);
            }};
        }

        macro_rules! compress2 {
            ($words0:expr, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $words1:expr, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $ones:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 0, 0, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 1, 1, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 2, 2, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 3, 3, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 4, 4, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 5, 5, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 6, 6, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 7, 7, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 8, 8, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 9, 9, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 10, 10, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 11, 11, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 12, 12, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 13, 13, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 14, 14, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 15, 15, 22
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 1, 16, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 6, 17, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 11, 18, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 0, 19, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 5, 20, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 10, 21, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 15, 22, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 4, 23, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 9, 24, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 14, 25, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 3, 26, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 8, 27, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 13, 28, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 2, 29, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 7, 30, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 12, 31, 20
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 5, 32, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 8, 33, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 11, 34, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 14, 35, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 1, 36, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 4, 37, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 7, 38, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 10, 39, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 13, 40, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 0, 41, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 3, 42, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 6, 43, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 9, 44, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 12, 45, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 15, 46, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 2, 47, 23
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 0, 48, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 7, 49, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 14, 50, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 5, 51, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 12, 52, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 3, 53, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 10, 54, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 1, 55, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 8, 56, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 15, 57, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 6, 58, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 13, 59, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 4, 60, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 11, 61, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 2, 62, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 9, 63, 21
                );
                $a0 = _mm256_add_epi32(initial0[0], $a0);
                $b0 = _mm256_add_epi32(initial0[1], $b0);
                $c0 = _mm256_add_epi32(initial0[2], $c0);
                $d0 = _mm256_add_epi32(initial0[3], $d0);
                $a1 = _mm256_add_epi32(initial1[0], $a1);
                $b1 = _mm256_add_epi32(initial1[1], $b1);
                $c1 = _mm256_add_epi32(initial1[2], $c1);
                $d1 = _mm256_add_epi32(initial1[3], $d1);
            }};
        }

        let mut full_counts = [0usize; 16];
        let mut block_counts = [0usize; 16];
        let mut common_full = usize::MAX;
        let mut max_blocks = 0usize;
        for lane in 0..16 {
            full_counts[lane] = inputs[lane].len() / 64;
            block_counts[lane] = padded_blocks_for_len(inputs[lane].len());
            common_full = core::cmp::min(common_full, full_counts[lane]);
            max_blocks = core::cmp::max(max_blocks, block_counts[lane]);
        }
        let mut a0 = _mm256_set1_epi32(STATE_INIT[0] as i32);
        let mut b0 = _mm256_set1_epi32(STATE_INIT[1] as i32);
        let mut c0 = _mm256_set1_epi32(STATE_INIT[2] as i32);
        let mut d0 = _mm256_set1_epi32(STATE_INIT[3] as i32);
        let mut a1 = a0;
        let mut b1 = b0;
        let mut c1 = c0;
        let mut d1 = d0;
        let all_ones = _mm256_set1_epi32(-1);

        for block_index in 0..common_full {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 8][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, all_ones);
        }

        for block_index in common_full..max_blocks {
            let base = block_index * 64;
            let mut scratch = [[0u8; 64]; 16];
            for lane in 0..16 {
                if block_index >= full_counts[lane] && block_index < block_counts[lane] {
                    scratch[lane] =
                        build_padded_block(inputs[lane], block_counts[lane], block_index);
                }
            }
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                if block_index < full_counts[lane] {
                    inputs[lane][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[lane]
                }
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                let index = lane + 8;
                if block_index < full_counts[index] {
                    inputs[index][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[index]
                }
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, all_ones);

            if block_counts.iter().any(|&count| count == block_index + 1) {
                let states0 = [a0, b0, c0, d0];
                let states1 = [a1, b1, c1, d1];
                let mut lanes0 = [[0u32; 8]; 4];
                let mut lanes1 = [[0u32; 8]; 4];
                for word in 0..4 {
                    unsafe {
                        _mm256_storeu_si256(
                            lanes0[word].as_mut_ptr().cast::<__m256i>(),
                            states0[word],
                        );
                        _mm256_storeu_si256(
                            lanes1[word].as_mut_ptr().cast::<__m256i>(),
                            states1[word],
                        );
                    }
                }
                for lane in 0..8 {
                    if block_counts[lane] == block_index + 1 {
                        for word in 0..4 {
                            outputs[lane][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes0[word][lane].to_le_bytes());
                        }
                    }
                    let index = lane + 8;
                    if block_counts[index] == block_index + 1 {
                        for word in 0..4 {
                            outputs[index][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes1[word][lane].to_le_bytes());
                        }
                    }
                }
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx512_state_kernel(
        avx512: Avx512,
        states: &mut [[u32; 4]],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(states.len(), 16);
        debug_assert_eq!(inputs.len(), 16);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        debug_assert!(inputs[0].len().is_multiple_of(64));
        let _ = avx512;

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }

                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }

                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }

                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    // SAFETY: each entry is a full 64-byte MD5 block and the
                    // unaligned AVX-512 load reads exactly those 64 bytes.
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }

        macro_rules! step {
            ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed = mix!($which, $b, $c, $d);
                let mut t = _mm512_add_epi32($a, mixed);
                t = _mm512_add_epi32(t, _mm512_set1_epi32(K[$round] as i32));
                t = _mm512_add_epi32(t, $words[$word]);
                $a = _mm512_add_epi32($b, _mm512_rol_epi32::<$shift>(t));
            }};
        }

        macro_rules! compress {
            ($words:expr, $a:ident, $b:ident, $c:ident, $d:ident) => {{
                let words = $words;
                let initial = [$a, $b, $c, $d];
                step!(f, $a, $b, $c, $d, words, 0, 0, 7);
                step!(f, $d, $a, $b, $c, words, 1, 1, 12);
                step!(f, $c, $d, $a, $b, words, 2, 2, 17);
                step!(f, $b, $c, $d, $a, words, 3, 3, 22);
                step!(f, $a, $b, $c, $d, words, 4, 4, 7);
                step!(f, $d, $a, $b, $c, words, 5, 5, 12);
                step!(f, $c, $d, $a, $b, words, 6, 6, 17);
                step!(f, $b, $c, $d, $a, words, 7, 7, 22);
                step!(f, $a, $b, $c, $d, words, 8, 8, 7);
                step!(f, $d, $a, $b, $c, words, 9, 9, 12);
                step!(f, $c, $d, $a, $b, words, 10, 10, 17);
                step!(f, $b, $c, $d, $a, words, 11, 11, 22);
                step!(f, $a, $b, $c, $d, words, 12, 12, 7);
                step!(f, $d, $a, $b, $c, words, 13, 13, 12);
                step!(f, $c, $d, $a, $b, words, 14, 14, 17);
                step!(f, $b, $c, $d, $a, words, 15, 15, 22);
                step!(g, $a, $b, $c, $d, words, 1, 16, 5);
                step!(g, $d, $a, $b, $c, words, 6, 17, 9);
                step!(g, $c, $d, $a, $b, words, 11, 18, 14);
                step!(g, $b, $c, $d, $a, words, 0, 19, 20);
                step!(g, $a, $b, $c, $d, words, 5, 20, 5);
                step!(g, $d, $a, $b, $c, words, 10, 21, 9);
                step!(g, $c, $d, $a, $b, words, 15, 22, 14);
                step!(g, $b, $c, $d, $a, words, 4, 23, 20);
                step!(g, $a, $b, $c, $d, words, 9, 24, 5);
                step!(g, $d, $a, $b, $c, words, 14, 25, 9);
                step!(g, $c, $d, $a, $b, words, 3, 26, 14);
                step!(g, $b, $c, $d, $a, words, 8, 27, 20);
                step!(g, $a, $b, $c, $d, words, 13, 28, 5);
                step!(g, $d, $a, $b, $c, words, 2, 29, 9);
                step!(g, $c, $d, $a, $b, words, 7, 30, 14);
                step!(g, $b, $c, $d, $a, words, 12, 31, 20);
                step!(h, $a, $b, $c, $d, words, 5, 32, 4);
                step!(h, $d, $a, $b, $c, words, 8, 33, 11);
                step!(h, $c, $d, $a, $b, words, 11, 34, 16);
                step!(h, $b, $c, $d, $a, words, 14, 35, 23);
                step!(h, $a, $b, $c, $d, words, 1, 36, 4);
                step!(h, $d, $a, $b, $c, words, 4, 37, 11);
                step!(h, $c, $d, $a, $b, words, 7, 38, 16);
                step!(h, $b, $c, $d, $a, words, 10, 39, 23);
                step!(h, $a, $b, $c, $d, words, 13, 40, 4);
                step!(h, $d, $a, $b, $c, words, 0, 41, 11);
                step!(h, $c, $d, $a, $b, words, 3, 42, 16);
                step!(h, $b, $c, $d, $a, words, 6, 43, 23);
                step!(h, $a, $b, $c, $d, words, 9, 44, 4);
                step!(h, $d, $a, $b, $c, words, 12, 45, 11);
                step!(h, $c, $d, $a, $b, words, 15, 46, 16);
                step!(h, $b, $c, $d, $a, words, 2, 47, 23);
                step!(i, $a, $b, $c, $d, words, 0, 48, 6);
                step!(i, $d, $a, $b, $c, words, 7, 49, 10);
                step!(i, $c, $d, $a, $b, words, 14, 50, 15);
                step!(i, $b, $c, $d, $a, words, 5, 51, 21);
                step!(i, $a, $b, $c, $d, words, 12, 52, 6);
                step!(i, $d, $a, $b, $c, words, 3, 53, 10);
                step!(i, $c, $d, $a, $b, words, 10, 54, 15);
                step!(i, $b, $c, $d, $a, words, 1, 55, 21);
                step!(i, $a, $b, $c, $d, words, 8, 56, 6);
                step!(i, $d, $a, $b, $c, words, 15, 57, 10);
                step!(i, $c, $d, $a, $b, words, 6, 58, 15);
                step!(i, $b, $c, $d, $a, words, 13, 59, 21);
                step!(i, $a, $b, $c, $d, words, 4, 60, 6);
                step!(i, $d, $a, $b, $c, words, 11, 61, 10);
                step!(i, $c, $d, $a, $b, words, 2, 62, 15);
                step!(i, $b, $c, $d, $a, words, 9, 63, 21);
                $a = _mm512_add_epi32(initial[0], $a);
                $b = _mm512_add_epi32(initial[1], $b);
                $c = _mm512_add_epi32(initial[2], $c);
                $d = _mm512_add_epi32(initial[3], $d);
            }};
        }

        let len = inputs[0].len();
        let mut lane_words = [[0u32; 16]; 4];
        for word in 0..4 {
            for lane in 0..16 {
                lane_words[word][lane] = states[lane][word];
            }
        }
        // SAFETY: each lane_words row is exactly sixteen u32 values (64 bytes).
        let mut a = unsafe { _mm512_loadu_si512(lane_words[0].as_ptr().cast::<__m512i>()) };
        let mut b = unsafe { _mm512_loadu_si512(lane_words[1].as_ptr().cast::<__m512i>()) };
        let mut c = unsafe { _mm512_loadu_si512(lane_words[2].as_ptr().cast::<__m512i>()) };
        let mut d = unsafe { _mm512_loadu_si512(lane_words[3].as_ptr().cast::<__m512i>()) };
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d);
        }

        let vector_state = [a, b, c, d];
        for word in 0..4 {
            // SAFETY: each destination is exactly sixteen u32 values (64 bytes).
            unsafe {
                _mm512_storeu_si512(
                    lane_words[word].as_mut_ptr().cast::<__m512i>(),
                    vector_state[word],
                );
            }
        }
        for lane in 0..16 {
            for word in 0..4 {
                states[lane][word] = lane_words[word][lane];
            }
        }
    }
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_equal_len_avx512_kernel(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
        debug_assert_eq!(inputs.len(), 16);
        debug_assert_eq!(outputs.len(), 16);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }

                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }

                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }

                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    // SAFETY: each entry is a full 64-byte MD5 block and the
                    // unaligned AVX-512 load reads exactly those 64 bytes.
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }

        macro_rules! step {
            ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed = mix!($which, $b, $c, $d);
                let mut t = _mm512_add_epi32($a, mixed);
                t = _mm512_add_epi32(t, _mm512_set1_epi32(K[$round] as i32));
                t = _mm512_add_epi32(t, $words[$word]);
                $a = _mm512_add_epi32($b, _mm512_rol_epi32::<$shift>(t));
            }};
        }

        macro_rules! compress {
            ($words:expr, $a:ident, $b:ident, $c:ident, $d:ident) => {{
                let words = $words;
                let initial = [$a, $b, $c, $d];
                step!(f, $a, $b, $c, $d, words, 0, 0, 7);
                step!(f, $d, $a, $b, $c, words, 1, 1, 12);
                step!(f, $c, $d, $a, $b, words, 2, 2, 17);
                step!(f, $b, $c, $d, $a, words, 3, 3, 22);
                step!(f, $a, $b, $c, $d, words, 4, 4, 7);
                step!(f, $d, $a, $b, $c, words, 5, 5, 12);
                step!(f, $c, $d, $a, $b, words, 6, 6, 17);
                step!(f, $b, $c, $d, $a, words, 7, 7, 22);
                step!(f, $a, $b, $c, $d, words, 8, 8, 7);
                step!(f, $d, $a, $b, $c, words, 9, 9, 12);
                step!(f, $c, $d, $a, $b, words, 10, 10, 17);
                step!(f, $b, $c, $d, $a, words, 11, 11, 22);
                step!(f, $a, $b, $c, $d, words, 12, 12, 7);
                step!(f, $d, $a, $b, $c, words, 13, 13, 12);
                step!(f, $c, $d, $a, $b, words, 14, 14, 17);
                step!(f, $b, $c, $d, $a, words, 15, 15, 22);
                step!(g, $a, $b, $c, $d, words, 1, 16, 5);
                step!(g, $d, $a, $b, $c, words, 6, 17, 9);
                step!(g, $c, $d, $a, $b, words, 11, 18, 14);
                step!(g, $b, $c, $d, $a, words, 0, 19, 20);
                step!(g, $a, $b, $c, $d, words, 5, 20, 5);
                step!(g, $d, $a, $b, $c, words, 10, 21, 9);
                step!(g, $c, $d, $a, $b, words, 15, 22, 14);
                step!(g, $b, $c, $d, $a, words, 4, 23, 20);
                step!(g, $a, $b, $c, $d, words, 9, 24, 5);
                step!(g, $d, $a, $b, $c, words, 14, 25, 9);
                step!(g, $c, $d, $a, $b, words, 3, 26, 14);
                step!(g, $b, $c, $d, $a, words, 8, 27, 20);
                step!(g, $a, $b, $c, $d, words, 13, 28, 5);
                step!(g, $d, $a, $b, $c, words, 2, 29, 9);
                step!(g, $c, $d, $a, $b, words, 7, 30, 14);
                step!(g, $b, $c, $d, $a, words, 12, 31, 20);
                step!(h, $a, $b, $c, $d, words, 5, 32, 4);
                step!(h, $d, $a, $b, $c, words, 8, 33, 11);
                step!(h, $c, $d, $a, $b, words, 11, 34, 16);
                step!(h, $b, $c, $d, $a, words, 14, 35, 23);
                step!(h, $a, $b, $c, $d, words, 1, 36, 4);
                step!(h, $d, $a, $b, $c, words, 4, 37, 11);
                step!(h, $c, $d, $a, $b, words, 7, 38, 16);
                step!(h, $b, $c, $d, $a, words, 10, 39, 23);
                step!(h, $a, $b, $c, $d, words, 13, 40, 4);
                step!(h, $d, $a, $b, $c, words, 0, 41, 11);
                step!(h, $c, $d, $a, $b, words, 3, 42, 16);
                step!(h, $b, $c, $d, $a, words, 6, 43, 23);
                step!(h, $a, $b, $c, $d, words, 9, 44, 4);
                step!(h, $d, $a, $b, $c, words, 12, 45, 11);
                step!(h, $c, $d, $a, $b, words, 15, 46, 16);
                step!(h, $b, $c, $d, $a, words, 2, 47, 23);
                step!(i, $a, $b, $c, $d, words, 0, 48, 6);
                step!(i, $d, $a, $b, $c, words, 7, 49, 10);
                step!(i, $c, $d, $a, $b, words, 14, 50, 15);
                step!(i, $b, $c, $d, $a, words, 5, 51, 21);
                step!(i, $a, $b, $c, $d, words, 12, 52, 6);
                step!(i, $d, $a, $b, $c, words, 3, 53, 10);
                step!(i, $c, $d, $a, $b, words, 10, 54, 15);
                step!(i, $b, $c, $d, $a, words, 1, 55, 21);
                step!(i, $a, $b, $c, $d, words, 8, 56, 6);
                step!(i, $d, $a, $b, $c, words, 15, 57, 10);
                step!(i, $c, $d, $a, $b, words, 6, 58, 15);
                step!(i, $b, $c, $d, $a, words, 13, 59, 21);
                step!(i, $a, $b, $c, $d, words, 4, 60, 6);
                step!(i, $d, $a, $b, $c, words, 11, 61, 10);
                step!(i, $c, $d, $a, $b, words, 2, 62, 15);
                step!(i, $b, $c, $d, $a, words, 9, 63, 21);
                $a = _mm512_add_epi32(initial[0], $a);
                $b = _mm512_add_epi32(initial[1], $b);
                $c = _mm512_add_epi32(initial[2], $c);
                $d = _mm512_add_epi32(initial[3], $d);
            }};
        }

        let len = inputs[0].len();
        let mut a = _mm512_set1_epi32(STATE_INIT[0] as i32);
        let mut b = _mm512_set1_epi32(STATE_INIT[1] as i32);
        let mut c = _mm512_set1_epi32(STATE_INIT[2] as i32);
        let mut d = _mm512_set1_epi32(STATE_INIT[3] as i32);
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d);
        }

        let padded_blocks = padded_blocks_for_len(len);
        for block_index in full_blocks..padded_blocks {
            if block_index * 64 >= len {
                let padded = build_padded_block(inputs[0], padded_blocks, block_index);
                let words: [__m512i; 16] = core::array::from_fn(|word| {
                    let offset = word * 4;
                    let value = u32::from_le_bytes(
                        padded[offset..offset + 4]
                            .try_into()
                            .expect("four-byte word"),
                    );
                    _mm512_set1_epi32(value as i32)
                });
                compress!(words, a, b, c, d);
                continue;
            }

            let padded: [[u8; 64]; 16] = core::array::from_fn(|lane| {
                build_padded_block(inputs[lane], padded_blocks, block_index)
            });
            let blocks: [&[u8; 64]; 16] = core::array::from_fn(|lane| &padded[lane]);
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d);
        }

        let states = [a, b, c, d];
        let mut lanes = [[0u32; 16]; 4];
        for word in 0..4 {
            // SAFETY: each destination is exactly sixteen u32 values (64 bytes).
            unsafe {
                _mm512_storeu_si512(lanes[word].as_mut_ptr().cast::<__m512i>(), states[word]);
            }
        }
        for lane in 0..16 {
            for word in 0..4 {
                outputs[lane][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes[word][lane].to_le_bytes());
            }
        }
    }
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_mixed_len_avx512_kernel(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
        debug_assert_eq!(inputs.len(), 16);
        debug_assert_eq!(outputs.len(), 16);

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }

                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }

                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }

                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    // SAFETY: each entry is a full 64-byte MD5 block and the
                    // unaligned AVX-512 load reads exactly those 64 bytes.
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }

        macro_rules! step {
            ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed = mix!($which, $b, $c, $d);
                let mut t = _mm512_add_epi32($a, mixed);
                t = _mm512_add_epi32(t, _mm512_set1_epi32(K[$round] as i32));
                t = _mm512_add_epi32(t, $words[$word]);
                $a = _mm512_add_epi32($b, _mm512_rol_epi32::<$shift>(t));
            }};
        }

        macro_rules! compress {
            ($words:expr, $a:ident, $b:ident, $c:ident, $d:ident) => {{
                let words = $words;
                let initial = [$a, $b, $c, $d];
                step!(f, $a, $b, $c, $d, words, 0, 0, 7);
                step!(f, $d, $a, $b, $c, words, 1, 1, 12);
                step!(f, $c, $d, $a, $b, words, 2, 2, 17);
                step!(f, $b, $c, $d, $a, words, 3, 3, 22);
                step!(f, $a, $b, $c, $d, words, 4, 4, 7);
                step!(f, $d, $a, $b, $c, words, 5, 5, 12);
                step!(f, $c, $d, $a, $b, words, 6, 6, 17);
                step!(f, $b, $c, $d, $a, words, 7, 7, 22);
                step!(f, $a, $b, $c, $d, words, 8, 8, 7);
                step!(f, $d, $a, $b, $c, words, 9, 9, 12);
                step!(f, $c, $d, $a, $b, words, 10, 10, 17);
                step!(f, $b, $c, $d, $a, words, 11, 11, 22);
                step!(f, $a, $b, $c, $d, words, 12, 12, 7);
                step!(f, $d, $a, $b, $c, words, 13, 13, 12);
                step!(f, $c, $d, $a, $b, words, 14, 14, 17);
                step!(f, $b, $c, $d, $a, words, 15, 15, 22);
                step!(g, $a, $b, $c, $d, words, 1, 16, 5);
                step!(g, $d, $a, $b, $c, words, 6, 17, 9);
                step!(g, $c, $d, $a, $b, words, 11, 18, 14);
                step!(g, $b, $c, $d, $a, words, 0, 19, 20);
                step!(g, $a, $b, $c, $d, words, 5, 20, 5);
                step!(g, $d, $a, $b, $c, words, 10, 21, 9);
                step!(g, $c, $d, $a, $b, words, 15, 22, 14);
                step!(g, $b, $c, $d, $a, words, 4, 23, 20);
                step!(g, $a, $b, $c, $d, words, 9, 24, 5);
                step!(g, $d, $a, $b, $c, words, 14, 25, 9);
                step!(g, $c, $d, $a, $b, words, 3, 26, 14);
                step!(g, $b, $c, $d, $a, words, 8, 27, 20);
                step!(g, $a, $b, $c, $d, words, 13, 28, 5);
                step!(g, $d, $a, $b, $c, words, 2, 29, 9);
                step!(g, $c, $d, $a, $b, words, 7, 30, 14);
                step!(g, $b, $c, $d, $a, words, 12, 31, 20);
                step!(h, $a, $b, $c, $d, words, 5, 32, 4);
                step!(h, $d, $a, $b, $c, words, 8, 33, 11);
                step!(h, $c, $d, $a, $b, words, 11, 34, 16);
                step!(h, $b, $c, $d, $a, words, 14, 35, 23);
                step!(h, $a, $b, $c, $d, words, 1, 36, 4);
                step!(h, $d, $a, $b, $c, words, 4, 37, 11);
                step!(h, $c, $d, $a, $b, words, 7, 38, 16);
                step!(h, $b, $c, $d, $a, words, 10, 39, 23);
                step!(h, $a, $b, $c, $d, words, 13, 40, 4);
                step!(h, $d, $a, $b, $c, words, 0, 41, 11);
                step!(h, $c, $d, $a, $b, words, 3, 42, 16);
                step!(h, $b, $c, $d, $a, words, 6, 43, 23);
                step!(h, $a, $b, $c, $d, words, 9, 44, 4);
                step!(h, $d, $a, $b, $c, words, 12, 45, 11);
                step!(h, $c, $d, $a, $b, words, 15, 46, 16);
                step!(h, $b, $c, $d, $a, words, 2, 47, 23);
                step!(i, $a, $b, $c, $d, words, 0, 48, 6);
                step!(i, $d, $a, $b, $c, words, 7, 49, 10);
                step!(i, $c, $d, $a, $b, words, 14, 50, 15);
                step!(i, $b, $c, $d, $a, words, 5, 51, 21);
                step!(i, $a, $b, $c, $d, words, 12, 52, 6);
                step!(i, $d, $a, $b, $c, words, 3, 53, 10);
                step!(i, $c, $d, $a, $b, words, 10, 54, 15);
                step!(i, $b, $c, $d, $a, words, 1, 55, 21);
                step!(i, $a, $b, $c, $d, words, 8, 56, 6);
                step!(i, $d, $a, $b, $c, words, 15, 57, 10);
                step!(i, $c, $d, $a, $b, words, 6, 58, 15);
                step!(i, $b, $c, $d, $a, words, 13, 59, 21);
                step!(i, $a, $b, $c, $d, words, 4, 60, 6);
                step!(i, $d, $a, $b, $c, words, 11, 61, 10);
                step!(i, $c, $d, $a, $b, words, 2, 62, 15);
                step!(i, $b, $c, $d, $a, words, 9, 63, 21);
                $a = _mm512_add_epi32(initial[0], $a);
                $b = _mm512_add_epi32(initial[1], $b);
                $c = _mm512_add_epi32(initial[2], $c);
                $d = _mm512_add_epi32(initial[3], $d);
            }};
        }

        let mut full_counts = [0usize; 16];
        let mut block_counts = [0usize; 16];
        let mut common_full = usize::MAX;
        let mut max_blocks = 0usize;
        for lane in 0..16 {
            full_counts[lane] = inputs[lane].len() / 64;
            block_counts[lane] = padded_blocks_for_len(inputs[lane].len());
            common_full = core::cmp::min(common_full, full_counts[lane]);
            max_blocks = core::cmp::max(max_blocks, block_counts[lane]);
        }
        let mut a = _mm512_set1_epi32(STATE_INIT[0] as i32);
        let mut b = _mm512_set1_epi32(STATE_INIT[1] as i32);
        let mut c = _mm512_set1_epi32(STATE_INIT[2] as i32);
        let mut d = _mm512_set1_epi32(STATE_INIT[3] as i32);

        // All lanes have real full blocks throughout this prefix, so use the
        // same direct transpose/compression path as equal-length hashing.
        for block_index in 0..common_full {
            let offset = block_index * 64;
            let blocks: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d);
        }

        for block_index in common_full..max_blocks {
            let base = block_index * 64;
            let mut scratch = [[0u8; 64]; 16];
            for lane in 0..16 {
                if block_index >= full_counts[lane] && block_index < block_counts[lane] {
                    scratch[lane] =
                        build_padded_block(inputs[lane], block_counts[lane], block_index);
                }
            }
            let blocks: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                if block_index < full_counts[lane] {
                    inputs[lane][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[lane]
                }
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d);

            if block_counts.iter().any(|&count| count == block_index + 1) {
                let states = [a, b, c, d];
                let mut lanes_out = [[0u32; 16]; 4];
                for word in 0..4 {
                    // SAFETY: each destination exactly matches one SIMD vector.
                    unsafe {
                        _mm512_storeu_si512(
                            lanes_out[word].as_mut_ptr().cast::<__m512i>(),
                            states[word],
                        );
                    }
                }
                for lane in 0..16 {
                    if block_counts[lane] == block_index + 1 {
                        for word in 0..4 {
                            outputs[lane][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes_out[word][lane].to_le_bytes());
                        }
                    }
                }
            }
        }
    }
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_equal_len_avx512_dual_kernel(
        avx512: Avx512,
        inputs: &[&[u8]],
        outputs: &mut [[u8; 16]],
    ) {
        debug_assert_eq!(inputs.len(), 32);
        debug_assert_eq!(outputs.len(), 32);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }

                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }

                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }

                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    // SAFETY: each entry is a full 64-byte MD5 block and the
                    // unaligned AVX-512 load reads exactly those 64 bytes.
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }

        macro_rules! step2 {
            ($which:ident, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed0 = mix!($which, $b0, $c0, $d0);
                let mixed1 = mix!($which, $b1, $c1, $d1);
                let mut t0 = _mm512_add_epi32($a0, mixed0);
                let mut t1 = _mm512_add_epi32($a1, mixed1);
                let key = _mm512_set1_epi32(K[$round] as i32);
                t0 = _mm512_add_epi32(t0, key);
                t1 = _mm512_add_epi32(t1, key);
                t0 = _mm512_add_epi32(t0, $words0[$word]);
                t1 = _mm512_add_epi32(t1, $words1[$word]);
                $a0 = _mm512_add_epi32($b0, _mm512_rol_epi32::<$shift>(t0));
                $a1 = _mm512_add_epi32($b1, _mm512_rol_epi32::<$shift>(t1));
            }};
        }

        macro_rules! compress2 {
            ($words0:expr, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $words1:expr, $a1:ident, $b1:ident, $c1:ident, $d1:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 0, 0, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 1, 1, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 2, 2, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 3, 3, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 4, 4, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 5, 5, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 6, 6, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 7, 7, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 8, 8, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 9, 9, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 10, 10, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 11, 11, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 12, 12, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 13, 13, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 14, 14, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 15, 15, 22
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 1, 16, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 6, 17, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 11, 18, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 0, 19, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 5, 20, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 10, 21, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 15, 22, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 4, 23, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 9, 24, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 14, 25, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 3, 26, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 8, 27, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 13, 28, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 2, 29, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 7, 30, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 12, 31, 20
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 5, 32, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 8, 33, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 11, 34, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 14, 35, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 1, 36, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 4, 37, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 7, 38, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 10, 39, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 13, 40, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 0, 41, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 3, 42, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 6, 43, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 9, 44, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 12, 45, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 15, 46, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 2, 47, 23
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 0, 48, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 7, 49, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 14, 50, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 5, 51, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 12, 52, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 3, 53, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 10, 54, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 1, 55, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 8, 56, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 15, 57, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 6, 58, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 13, 59, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 4, 60, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 11, 61, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 2, 62, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 9, 63, 21
                );
                $a0 = _mm512_add_epi32(initial0[0], $a0);
                $b0 = _mm512_add_epi32(initial0[1], $b0);
                $c0 = _mm512_add_epi32(initial0[2], $c0);
                $d0 = _mm512_add_epi32(initial0[3], $d0);
                $a1 = _mm512_add_epi32(initial1[0], $a1);
                $b1 = _mm512_add_epi32(initial1[1], $b1);
                $c1 = _mm512_add_epi32(initial1[2], $c1);
                $d1 = _mm512_add_epi32(initial1[3], $d1);
            }};
        }

        let len = inputs[0].len();
        let mut a0 = _mm512_set1_epi32(STATE_INIT[0] as i32);
        let mut b0 = _mm512_set1_epi32(STATE_INIT[1] as i32);
        let mut c0 = _mm512_set1_epi32(STATE_INIT[2] as i32);
        let mut d0 = _mm512_set1_epi32(STATE_INIT[3] as i32);
        let mut a1 = a0;
        let mut b1 = b0;
        let mut c1 = c0;
        let mut d1 = d0;
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1);
        }

        let padded_blocks = padded_blocks_for_len(len);
        for block_index in full_blocks..padded_blocks {
            if block_index * 64 >= len {
                let padded = build_padded_block(inputs[0], padded_blocks, block_index);
                let words: [__m512i; 16] = core::array::from_fn(|word| {
                    let offset = word * 4;
                    let value = u32::from_le_bytes(
                        padded[offset..offset + 4]
                            .try_into()
                            .expect("four-byte word"),
                    );
                    _mm512_set1_epi32(value as i32)
                });
                compress2!(words, a0, b0, c0, d0, words, a1, b1, c1, d1);
                continue;
            }

            let padded: [[u8; 64]; 32] = core::array::from_fn(|lane| {
                build_padded_block(inputs[lane], padded_blocks, block_index)
            });
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| &padded[lane]);
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| &padded[lane + 16]);
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1);
        }

        let states0 = [a0, b0, c0, d0];
        let states1 = [a1, b1, c1, d1];
        let mut lanes0 = [[0u32; 16]; 4];
        let mut lanes1 = [[0u32; 16]; 4];
        for word in 0..4 {
            // SAFETY: destinations are each sixteen u32 values (64 bytes).
            unsafe {
                _mm512_storeu_si512(lanes0[word].as_mut_ptr().cast::<__m512i>(), states0[word]);
                _mm512_storeu_si512(lanes1[word].as_mut_ptr().cast::<__m512i>(), states1[word]);
            }
        }
        for lane in 0..16 {
            for word in 0..4 {
                outputs[lane][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes0[word][lane].to_le_bytes());
                outputs[lane + 16][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes1[word][lane].to_le_bytes());
            }
        }
    }
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_equal_len_avx512_triple_kernel(
        avx512: Avx512,
        inputs: &[&[u8]],
        outputs: &mut [[u8; 16]],
    ) {
        debug_assert_eq!(inputs.len(), 48);
        debug_assert_eq!(outputs.len(), 48);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx512;

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }
                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }
                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }
                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }
        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }
        macro_rules! mix {
            (f,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }
        macro_rules! step3 {
            ($which:ident,$a0:ident,$b0:ident,$c0:ident,$d0:ident,$a1:ident,$b1:ident,$c1:ident,$d1:ident,$a2:ident,$b2:ident,$c2:ident,$d2:ident,$w0:ident,$w1:ident,$w2:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm512_set1_epi32(K[$round] as i32);
                let mut t0 = _mm512_add_epi32($a0, mix!($which, $b0, $c0, $d0));
                let mut t1 = _mm512_add_epi32($a1, mix!($which, $b1, $c1, $d1));
                let mut t2 = _mm512_add_epi32($a2, mix!($which, $b2, $c2, $d2));
                t0 = _mm512_add_epi32(_mm512_add_epi32(t0, key), $w0[$word]);
                t1 = _mm512_add_epi32(_mm512_add_epi32(t1, key), $w1[$word]);
                t2 = _mm512_add_epi32(_mm512_add_epi32(t2, key), $w2[$word]);
                $a0 = _mm512_add_epi32($b0, _mm512_rol_epi32::<$shift>(t0));
                $a1 = _mm512_add_epi32($b1, _mm512_rol_epi32::<$shift>(t1));
                $a2 = _mm512_add_epi32($b2, _mm512_rol_epi32::<$shift>(t2));
            }};
        }
        macro_rules! compress3 {
            ($words0:expr,$a0:ident,$b0:ident,$c0:ident,$d0:ident,$words1:expr,$a1:ident,$b1:ident,$c1:ident,$d1:ident,$words2:expr,$a2:ident,$b2:ident,$c2:ident,$d2:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let words2 = $words2;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                let initial2 = [$a2, $b2, $c2, $d2];
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 0, 0, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 1, 1, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 2, 2, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 3, 3, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 4, 4, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 5, 5, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 6, 6, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 7, 7, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 8, 8, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 9, 9, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 10, 10, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 11, 11, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 12, 12, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 13, 13, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 14, 14, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 15, 15, 22
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 1, 16, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 6, 17, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 11, 18, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 0, 19, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 5, 20, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 10, 21, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 15, 22, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 4, 23, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 9, 24, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 14, 25, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 3, 26, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 8, 27, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 13, 28, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 2, 29, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 7, 30, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 12, 31, 20
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 5, 32, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 8, 33, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 11, 34, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 14, 35, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 1, 36, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 4, 37, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 7, 38, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 10, 39, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 13, 40, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 0, 41, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 3, 42, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 6, 43, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 9, 44, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 12, 45, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 15, 46, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 2, 47, 23
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 0, 48, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 7, 49, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 14, 50, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 5, 51, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 12, 52, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 3, 53, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 10, 54, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 1, 55, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 8, 56, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 15, 57, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 6, 58, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 13, 59, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 4, 60, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 11, 61, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 2, 62, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 9, 63, 21
                );
                $a0 = _mm512_add_epi32(initial0[0], $a0);
                $b0 = _mm512_add_epi32(initial0[1], $b0);
                $c0 = _mm512_add_epi32(initial0[2], $c0);
                $d0 = _mm512_add_epi32(initial0[3], $d0);
                $a1 = _mm512_add_epi32(initial1[0], $a1);
                $b1 = _mm512_add_epi32(initial1[1], $b1);
                $c1 = _mm512_add_epi32(initial1[2], $c1);
                $d1 = _mm512_add_epi32(initial1[3], $d1);
                $a2 = _mm512_add_epi32(initial2[0], $a2);
                $b2 = _mm512_add_epi32(initial2[1], $b2);
                $c2 = _mm512_add_epi32(initial2[2], $c2);
                $d2 = _mm512_add_epi32(initial2[3], $d2);
            }};
        }
        let len = inputs[0].len();
        let mut a0 = _mm512_set1_epi32(STATE_INIT[0] as i32);
        let mut b0 = _mm512_set1_epi32(STATE_INIT[1] as i32);
        let mut c0 = _mm512_set1_epi32(STATE_INIT[2] as i32);
        let mut d0 = _mm512_set1_epi32(STATE_INIT[3] as i32);
        let mut a1 = a0;
        let mut b1 = b0;
        let mut c1 = c0;
        let mut d1 = d0;
        let mut a2 = a0;
        let mut b2 = b0;
        let mut c2 = c0;
        let mut d2 = d0;
        let full_blocks = len / 64;
        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks2: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 32][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2
            );
        }
        let padded_blocks = padded_blocks_for_len(len);
        for block_index in full_blocks..padded_blocks {
            if block_index * 64 >= len {
                let padded = build_padded_block(inputs[0], padded_blocks, block_index);
                let words: [__m512i; 16] = core::array::from_fn(|word| {
                    let offset = word * 4;
                    let value = u32::from_le_bytes(
                        padded[offset..offset + 4]
                            .try_into()
                            .expect("four-byte word"),
                    );
                    _mm512_set1_epi32(value as i32)
                });
                compress3!(
                    words, a0, b0, c0, d0, words, a1, b1, c1, d1, words, a2, b2, c2, d2
                );
                continue;
            }

            let padded: [[u8; 64]; 48] = core::array::from_fn(|lane| {
                build_padded_block(inputs[lane], padded_blocks, block_index)
            });
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| &padded[lane]);
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| &padded[lane + 16]);
            let blocks2: [&[u8; 64]; 16] = core::array::from_fn(|lane| &padded[lane + 32]);
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2
            );
        }
        let states0 = [a0, b0, c0, d0];
        let states1 = [a1, b1, c1, d1];
        let states2 = [a2, b2, c2, d2];
        let mut lanes0 = [[0u32; 16]; 4];
        let mut lanes1 = [[0u32; 16]; 4];
        let mut lanes2 = [[0u32; 16]; 4];
        for word in 0..4 {
            unsafe {
                _mm512_storeu_si512(lanes0[word].as_mut_ptr().cast::<__m512i>(), states0[word]);
                _mm512_storeu_si512(lanes1[word].as_mut_ptr().cast::<__m512i>(), states1[word]);
                _mm512_storeu_si512(lanes2[word].as_mut_ptr().cast::<__m512i>(), states2[word]);
            }
        }
        for lane in 0..16 {
            for word in 0..4 {
                outputs[lane][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes0[word][lane].to_le_bytes());
                outputs[lane + 16][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes1[word][lane].to_le_bytes());
                outputs[lane + 32][word * 4..word * 4 + 4]
                    .copy_from_slice(&lanes2[word][lane].to_le_bytes());
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_mixed_len_avx512_triple_kernel(
        avx512: Avx512,
        inputs: &[&[u8]],
        outputs: &mut [[u8; 16]],
    ) {
        debug_assert_eq!(inputs.len(), 48);
        debug_assert_eq!(outputs.len(), 48);
        let _ = avx512;

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }
                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }
                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }
                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }
        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }
        macro_rules! mix {
            (f,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }
        macro_rules! step3 {
            ($which:ident,$a0:ident,$b0:ident,$c0:ident,$d0:ident,$a1:ident,$b1:ident,$c1:ident,$d1:ident,$a2:ident,$b2:ident,$c2:ident,$d2:ident,$w0:ident,$w1:ident,$w2:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm512_set1_epi32(K[$round] as i32);
                let mut t0 = _mm512_add_epi32($a0, mix!($which, $b0, $c0, $d0));
                let mut t1 = _mm512_add_epi32($a1, mix!($which, $b1, $c1, $d1));
                let mut t2 = _mm512_add_epi32($a2, mix!($which, $b2, $c2, $d2));
                t0 = _mm512_add_epi32(_mm512_add_epi32(t0, key), $w0[$word]);
                t1 = _mm512_add_epi32(_mm512_add_epi32(t1, key), $w1[$word]);
                t2 = _mm512_add_epi32(_mm512_add_epi32(t2, key), $w2[$word]);
                $a0 = _mm512_add_epi32($b0, _mm512_rol_epi32::<$shift>(t0));
                $a1 = _mm512_add_epi32($b1, _mm512_rol_epi32::<$shift>(t1));
                $a2 = _mm512_add_epi32($b2, _mm512_rol_epi32::<$shift>(t2));
            }};
        }
        macro_rules! compress3 {
            ($words0:expr,$a0:ident,$b0:ident,$c0:ident,$d0:ident,$words1:expr,$a1:ident,$b1:ident,$c1:ident,$d1:ident,$words2:expr,$a2:ident,$b2:ident,$c2:ident,$d2:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let words2 = $words2;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                let initial2 = [$a2, $b2, $c2, $d2];
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 0, 0, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 1, 1, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 2, 2, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 3, 3, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 4, 4, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 5, 5, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 6, 6, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 7, 7, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 8, 8, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 9, 9, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 10, 10, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 11, 11, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 12, 12, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 13, 13, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 14, 14, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 15, 15, 22
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 1, 16, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 6, 17, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 11, 18, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 0, 19, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 5, 20, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 10, 21, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 15, 22, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 4, 23, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 9, 24, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 14, 25, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 3, 26, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 8, 27, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 13, 28, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 2, 29, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 7, 30, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 12, 31, 20
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 5, 32, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 8, 33, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 11, 34, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 14, 35, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 1, 36, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 4, 37, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 7, 38, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 10, 39, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 13, 40, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 0, 41, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 3, 42, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 6, 43, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 9, 44, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 12, 45, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 15, 46, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 2, 47, 23
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 0, 48, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 7, 49, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 14, 50, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 5, 51, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 12, 52, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 3, 53, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 10, 54, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 1, 55, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 8, 56, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 15, 57, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 6, 58, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 13, 59, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 4, 60, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 11, 61, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 2, 62, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 9, 63, 21
                );
                $a0 = _mm512_add_epi32(initial0[0], $a0);
                $b0 = _mm512_add_epi32(initial0[1], $b0);
                $c0 = _mm512_add_epi32(initial0[2], $c0);
                $d0 = _mm512_add_epi32(initial0[3], $d0);
                $a1 = _mm512_add_epi32(initial1[0], $a1);
                $b1 = _mm512_add_epi32(initial1[1], $b1);
                $c1 = _mm512_add_epi32(initial1[2], $c1);
                $d1 = _mm512_add_epi32(initial1[3], $d1);
                $a2 = _mm512_add_epi32(initial2[0], $a2);
                $b2 = _mm512_add_epi32(initial2[1], $b2);
                $c2 = _mm512_add_epi32(initial2[2], $c2);
                $d2 = _mm512_add_epi32(initial2[3], $d2);
            }};
        }
        let mut full_counts = [0usize; 48];
        let mut block_counts = [0usize; 48];
        let mut common_full = usize::MAX;
        let mut max_blocks = 0usize;
        for lane in 0..48 {
            full_counts[lane] = inputs[lane].len() / 64;
            block_counts[lane] = padded_blocks_for_len(inputs[lane].len());
            common_full = core::cmp::min(common_full, full_counts[lane]);
            max_blocks = core::cmp::max(max_blocks, block_counts[lane]);
        }
        let mut a0 = _mm512_set1_epi32(STATE_INIT[0] as i32);
        let mut b0 = _mm512_set1_epi32(STATE_INIT[1] as i32);
        let mut c0 = _mm512_set1_epi32(STATE_INIT[2] as i32);
        let mut d0 = _mm512_set1_epi32(STATE_INIT[3] as i32);
        let mut a1 = a0;
        let mut b1 = b0;
        let mut c1 = c0;
        let mut d1 = d0;
        let mut a2 = a0;
        let mut b2 = b0;
        let mut c2 = c0;
        let mut d2 = d0;
        for block_index in 0..common_full {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks2: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 32][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2
            );
        }
        for block_index in common_full..max_blocks {
            let base = block_index * 64;
            let mut scratch = [[0u8; 64]; 48];
            for lane in 0..48 {
                if block_index >= full_counts[lane] && block_index < block_counts[lane] {
                    scratch[lane] =
                        build_padded_block(inputs[lane], block_counts[lane], block_index);
                }
            }
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                if block_index < full_counts[lane] {
                    inputs[lane][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[lane]
                }
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                let i = lane + 16;
                if block_index < full_counts[i] {
                    inputs[i][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[i]
                }
            });
            let blocks2: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                let i = lane + 32;
                if block_index < full_counts[i] {
                    inputs[i][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[i]
                }
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2
            );
            if block_counts.iter().any(|&count| count == block_index + 1) {
                let states0 = [a0, b0, c0, d0];
                let states1 = [a1, b1, c1, d1];
                let states2 = [a2, b2, c2, d2];
                let mut lanes0 = [[0u32; 16]; 4];
                let mut lanes1 = [[0u32; 16]; 4];
                let mut lanes2 = [[0u32; 16]; 4];
                for word in 0..4 {
                    unsafe {
                        _mm512_storeu_si512(
                            lanes0[word].as_mut_ptr().cast::<__m512i>(),
                            states0[word],
                        );
                        _mm512_storeu_si512(
                            lanes1[word].as_mut_ptr().cast::<__m512i>(),
                            states1[word],
                        );
                        _mm512_storeu_si512(
                            lanes2[word].as_mut_ptr().cast::<__m512i>(),
                            states2[word],
                        );
                    }
                }
                for lane in 0..16 {
                    if block_counts[lane] == block_index + 1 {
                        for word in 0..4 {
                            outputs[lane][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes0[word][lane].to_le_bytes());
                        }
                    }
                    let i1 = lane + 16;
                    if block_counts[i1] == block_index + 1 {
                        for word in 0..4 {
                            outputs[i1][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes1[word][lane].to_le_bytes());
                        }
                    }
                    let i2 = lane + 32;
                    if block_counts[i2] == block_index + 1 {
                        for word in 0..4 {
                            outputs[i2][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes2[word][lane].to_le_bytes());
                        }
                    }
                }
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn hash_mixed_len_avx512_dual_kernel(
        avx512: Avx512,
        inputs: &[&[u8]],
        outputs: &mut [[u8; 16]],
    ) {
        debug_assert_eq!(inputs.len(), 32);
        debug_assert_eq!(outputs.len(), 32);

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }

                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }

                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }

                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    // SAFETY: each entry is a full 64-byte MD5 block and the
                    // unaligned AVX-512 load reads exactly those 64 bytes.
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }

        macro_rules! step2 {
            ($which:ident, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed0 = mix!($which, $b0, $c0, $d0);
                let mixed1 = mix!($which, $b1, $c1, $d1);
                let mut t0 = _mm512_add_epi32($a0, mixed0);
                let mut t1 = _mm512_add_epi32($a1, mixed1);
                let key = _mm512_set1_epi32(K[$round] as i32);
                t0 = _mm512_add_epi32(t0, key);
                t1 = _mm512_add_epi32(t1, key);
                t0 = _mm512_add_epi32(t0, $words0[$word]);
                t1 = _mm512_add_epi32(t1, $words1[$word]);
                $a0 = _mm512_add_epi32($b0, _mm512_rol_epi32::<$shift>(t0));
                $a1 = _mm512_add_epi32($b1, _mm512_rol_epi32::<$shift>(t1));
            }};
        }

        macro_rules! compress2 {
            ($words0:expr, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $words1:expr, $a1:ident, $b1:ident, $c1:ident, $d1:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 0, 0, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 1, 1, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 2, 2, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 3, 3, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 4, 4, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 5, 5, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 6, 6, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 7, 7, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 8, 8, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 9, 9, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 10, 10, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 11, 11, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 12, 12, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 13, 13, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 14, 14, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 15, 15, 22
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 1, 16, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 6, 17, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 11, 18, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 0, 19, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 5, 20, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 10, 21, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 15, 22, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 4, 23, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 9, 24, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 14, 25, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 3, 26, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 8, 27, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 13, 28, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 2, 29, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 7, 30, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 12, 31, 20
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 5, 32, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 8, 33, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 11, 34, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 14, 35, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 1, 36, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 4, 37, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 7, 38, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 10, 39, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 13, 40, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 0, 41, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 3, 42, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 6, 43, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 9, 44, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 12, 45, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 15, 46, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 2, 47, 23
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 0, 48, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 7, 49, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 14, 50, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 5, 51, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 12, 52, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 3, 53, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 10, 54, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 1, 55, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 8, 56, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 15, 57, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 6, 58, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 13, 59, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 4, 60, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 11, 61, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 2, 62, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 9, 63, 21
                );
                $a0 = _mm512_add_epi32(initial0[0], $a0);
                $b0 = _mm512_add_epi32(initial0[1], $b0);
                $c0 = _mm512_add_epi32(initial0[2], $c0);
                $d0 = _mm512_add_epi32(initial0[3], $d0);
                $a1 = _mm512_add_epi32(initial1[0], $a1);
                $b1 = _mm512_add_epi32(initial1[1], $b1);
                $c1 = _mm512_add_epi32(initial1[2], $c1);
                $d1 = _mm512_add_epi32(initial1[3], $d1);
            }};
        }

        let mut full_counts = [0usize; 32];
        let mut block_counts = [0usize; 32];
        let mut common_full = usize::MAX;
        let mut max_blocks = 0usize;
        for lane in 0..32 {
            full_counts[lane] = inputs[lane].len() / 64;
            block_counts[lane] = padded_blocks_for_len(inputs[lane].len());
            common_full = core::cmp::min(common_full, full_counts[lane]);
            max_blocks = core::cmp::max(max_blocks, block_counts[lane]);
        }
        let mut a0 = _mm512_set1_epi32(STATE_INIT[0] as i32);
        let mut b0 = _mm512_set1_epi32(STATE_INIT[1] as i32);
        let mut c0 = _mm512_set1_epi32(STATE_INIT[2] as i32);
        let mut d0 = _mm512_set1_epi32(STATE_INIT[3] as i32);
        let mut a1 = a0;
        let mut b1 = b0;
        let mut c1 = c0;
        let mut d1 = d0;

        for block_index in 0..common_full {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1);
        }

        for block_index in common_full..max_blocks {
            let base = block_index * 64;
            let mut scratch = [[0u8; 64]; 32];
            for lane in 0..32 {
                if block_index >= full_counts[lane] && block_index < block_counts[lane] {
                    scratch[lane] =
                        build_padded_block(inputs[lane], block_counts[lane], block_index);
                }
            }
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                if block_index < full_counts[lane] {
                    inputs[lane][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[lane]
                }
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                let index = lane + 16;
                if block_index < full_counts[index] {
                    inputs[index][base..base + 64]
                        .try_into()
                        .expect("full MD5 block")
                } else {
                    &scratch[index]
                }
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1);

            if block_counts.iter().any(|&count| count == block_index + 1) {
                let states0 = [a0, b0, c0, d0];
                let states1 = [a1, b1, c1, d1];
                let mut lanes0 = [[0u32; 16]; 4];
                let mut lanes1 = [[0u32; 16]; 4];
                for word in 0..4 {
                    unsafe {
                        _mm512_storeu_si512(
                            lanes0[word].as_mut_ptr().cast::<__m512i>(),
                            states0[word],
                        );
                        _mm512_storeu_si512(
                            lanes1[word].as_mut_ptr().cast::<__m512i>(),
                            states1[word],
                        );
                    }
                }
                for lane in 0..16 {
                    if block_counts[lane] == block_index + 1 {
                        for word in 0..4 {
                            outputs[lane][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes0[word][lane].to_le_bytes());
                        }
                    }
                    let index = lane + 16;
                    if block_counts[index] == block_index + 1 {
                        for word in 0..4 {
                            outputs[index][word * 4..word * 4 + 4]
                                .copy_from_slice(&lanes1[word][lane].to_le_bytes());
                        }
                    }
                }
            }
        }
    }
);

#[inline(always)]
fn rol<SIMD: Simd>(x: SIMD::u32s, shift: u32) -> SIMD::u32s {
    (x << shift) | (x >> (32 - shift))
}

#[inline(always)]
fn f<SIMD: Simd>(x: SIMD::u32s, y: SIMD::u32s, z: SIMD::u32s) -> SIMD::u32s {
    (x & y) | ((!x) & z)
}

#[inline(always)]
fn g<SIMD: Simd>(x: SIMD::u32s, y: SIMD::u32s, z: SIMD::u32s) -> SIMD::u32s {
    (x & z) | (y & !z)
}

#[inline(always)]
fn h<SIMD: Simd>(x: SIMD::u32s, y: SIMD::u32s, z: SIMD::u32s) -> SIMD::u32s {
    x ^ y ^ z
}

#[inline(always)]
fn i<SIMD: Simd>(x: SIMD::u32s, y: SIMD::u32s, z: SIMD::u32s) -> SIMD::u32s {
    y ^ (x | !z)
}

#[inline(always)]
fn compress_words<SIMD: Simd>(state: &mut [SIMD::u32s; 4], words: &[SIMD::u32s; 16]) {
    let [mut a, mut b, mut c, mut d] = *state;
    let initial = *state;

    macro_rules! step {
        ($mix:ident, $a:ident, $b:ident, $c:ident, $d:ident, $word:expr, $round:expr) => {{
            $a = $b
                + rol::<SIMD>(
                    $a + $mix::<SIMD>($b, $c, $d) + K[$round] + words[$word],
                    S[$round],
                );
        }};
    }

    step!(f, a, b, c, d, 0, 0);
    step!(f, d, a, b, c, 1, 1);
    step!(f, c, d, a, b, 2, 2);
    step!(f, b, c, d, a, 3, 3);
    step!(f, a, b, c, d, 4, 4);
    step!(f, d, a, b, c, 5, 5);
    step!(f, c, d, a, b, 6, 6);
    step!(f, b, c, d, a, 7, 7);
    step!(f, a, b, c, d, 8, 8);
    step!(f, d, a, b, c, 9, 9);
    step!(f, c, d, a, b, 10, 10);
    step!(f, b, c, d, a, 11, 11);
    step!(f, a, b, c, d, 12, 12);
    step!(f, d, a, b, c, 13, 13);
    step!(f, c, d, a, b, 14, 14);
    step!(f, b, c, d, a, 15, 15);

    step!(g, a, b, c, d, 1, 16);
    step!(g, d, a, b, c, 6, 17);
    step!(g, c, d, a, b, 11, 18);
    step!(g, b, c, d, a, 0, 19);
    step!(g, a, b, c, d, 5, 20);
    step!(g, d, a, b, c, 10, 21);
    step!(g, c, d, a, b, 15, 22);
    step!(g, b, c, d, a, 4, 23);
    step!(g, a, b, c, d, 9, 24);
    step!(g, d, a, b, c, 14, 25);
    step!(g, c, d, a, b, 3, 26);
    step!(g, b, c, d, a, 8, 27);
    step!(g, a, b, c, d, 13, 28);
    step!(g, d, a, b, c, 2, 29);
    step!(g, c, d, a, b, 7, 30);
    step!(g, b, c, d, a, 12, 31);

    step!(h, a, b, c, d, 5, 32);
    step!(h, d, a, b, c, 8, 33);
    step!(h, c, d, a, b, 11, 34);
    step!(h, b, c, d, a, 14, 35);
    step!(h, a, b, c, d, 1, 36);
    step!(h, d, a, b, c, 4, 37);
    step!(h, c, d, a, b, 7, 38);
    step!(h, b, c, d, a, 10, 39);
    step!(h, a, b, c, d, 13, 40);
    step!(h, d, a, b, c, 0, 41);
    step!(h, c, d, a, b, 3, 42);
    step!(h, b, c, d, a, 6, 43);
    step!(h, a, b, c, d, 9, 44);
    step!(h, d, a, b, c, 12, 45);
    step!(h, c, d, a, b, 15, 46);
    step!(h, b, c, d, a, 2, 47);

    step!(i, a, b, c, d, 0, 48);
    step!(i, d, a, b, c, 7, 49);
    step!(i, c, d, a, b, 14, 50);
    step!(i, b, c, d, a, 5, 51);
    step!(i, a, b, c, d, 12, 52);
    step!(i, d, a, b, c, 3, 53);
    step!(i, c, d, a, b, 10, 54);
    step!(i, b, c, d, a, 1, 55);
    step!(i, a, b, c, d, 8, 56);
    step!(i, d, a, b, c, 15, 57);
    step!(i, c, d, a, b, 6, 58);
    step!(i, b, c, d, a, 13, 59);
    step!(i, a, b, c, d, 4, 60);
    step!(i, d, a, b, c, 11, 61);
    step!(i, c, d, a, b, 2, 62);
    step!(i, b, c, d, a, 9, 63);

    state[0] = initial[0] + a;
    state[1] = initial[1] + b;
    state[2] = initial[2] + c;
    state[3] = initial[3] + d;
}
#[inline(always)]
fn full_block_words<SIMD: Simd>(
    simd: SIMD,
    inputs: &[&[u8]],
    active: usize,
    block_index: usize,
) -> [SIMD::u32s; 16] {
    core::array::from_fn(|word_index| {
        SIMD::u32s::from_fn(simd, |lane| {
            if lane >= active {
                return 0;
            }
            let offset = block_index * 64 + word_index * 4;
            u32::from_le_bytes(
                inputs[lane][offset..offset + 4]
                    .try_into()
                    .expect("four-byte word"),
            )
        })
    })
}

#[inline(always)]
fn compress_many_blocks_inner<SIMD: Simd>(simd: SIMD, states: &mut [[u32; 4]], inputs: &[&[u8]]) {
    debug_assert_eq!(states.len(), inputs.len());
    debug_assert!(
        inputs
            .first()
            .is_none_or(|first| inputs.iter().all(|input| input.len() == first.len()))
    );
    debug_assert!(inputs.first().is_none_or(|input| input.len() % 64 == 0));

    let lanes = SIMD::u32s::N;
    debug_assert!(lanes <= MAX_LANES);

    let mut start = 0;
    while start < inputs.len() {
        let end = core::cmp::min(start + lanes, inputs.len());
        let input_chunk = &inputs[start..end];
        let state_chunk = &mut states[start..end];
        let active = input_chunk.len();

        // Under-filled two-stream batches are cheaper on the optimized scalar
        // compressors than paying a whole SIMD round schedule.
        if active < 3 {
            for (state, input) in state_chunk.iter_mut().zip(input_chunk) {
                for block in input.chunks_exact(64) {
                    let block: &[u8; 64] = block.try_into().expect("64-byte chunk");
                    scalar::compress_block(state, block);
                }
            }
            start = end;
            continue;
        }

        let mut vector_state: [SIMD::u32s; 4] = core::array::from_fn(|word| {
            SIMD::u32s::from_fn(simd, |lane| {
                if lane < active {
                    state_chunk[lane][word]
                } else {
                    STATE_INIT[word]
                }
            })
        });

        let block_count = input_chunk[0].len() / 64;
        for block_index in 0..block_count {
            let words = full_block_words(simd, input_chunk, active, block_index);
            compress_words::<SIMD>(&mut vector_state, &words);
        }

        for lane in 0..active {
            for word in 0..4 {
                state_chunk[lane][word] = vector_state[word][lane];
            }
        }
        start = end;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx2_dual_state_kernel(
        avx2: Avx2,
        states: &mut [[u32; 4]],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(inputs.len(), 16);
        debug_assert_eq!(states.len(), 16);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);

                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    // SAFETY: every entry is a 64-byte block, and unaligned AVX2
                    // loads read exactly bytes 0..32 and 32..64 respectively.
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    // SAFETY: bytes 32..64 are inside the same 64-byte block.
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (g, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $z), _mm256_andnot_si256($z, $y))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }

        macro_rules! step2 {
            (g, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t0 = _mm256_add_epi32($a0, _mm256_andnot_si256($d0, $c0));
                let mut t1 = _mm256_add_epi32($a1, _mm256_andnot_si256($d1, $c1));
                let key = _mm256_set1_epi32(K[$round] as i32);
                t0 = _mm256_add_epi32(t0, key);
                t1 = _mm256_add_epi32(t1, key);
                t0 = _mm256_add_epi32(t0, $words0[$word]);
                t1 = _mm256_add_epi32(t1, $words1[$word]);
                t0 = _mm256_add_epi32(t0, _mm256_and_si256($d0, $b0));
                t1 = _mm256_add_epi32(t1, _mm256_and_si256($d1, $b1));
                let rotated0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let rotated1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                $a0 = _mm256_add_epi32($b0, rotated0);
                $a1 = _mm256_add_epi32($b1, rotated1);
            }};
            ($which:ident, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed0 = mix!($which, $b0, $c0, $d0, $ones);
                let mixed1 = mix!($which, $b1, $c1, $d1, $ones);
                let mut t0 = _mm256_add_epi32($a0, mixed0);
                let mut t1 = _mm256_add_epi32($a1, mixed1);
                let key = _mm256_set1_epi32(K[$round] as i32);
                t0 = _mm256_add_epi32(t0, key);
                t1 = _mm256_add_epi32(t1, key);
                t0 = _mm256_add_epi32(t0, $words0[$word]);
                t1 = _mm256_add_epi32(t1, $words1[$word]);
                let rotated0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let rotated1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                $a0 = _mm256_add_epi32($b0, rotated0);
                $a1 = _mm256_add_epi32($b1, rotated1);
            }};
        }

        macro_rules! compress2 {
            ($words0:expr, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $words1:expr, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $ones:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 0, 0, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 1, 1, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 2, 2, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 3, 3, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 4, 4, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 5, 5, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 6, 6, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 7, 7, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 8, 8, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 9, 9, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 10, 10, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 11, 11, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 12, 12, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 13, 13, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 14, 14, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 15, 15, 22
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 1, 16, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 6, 17, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 11, 18, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 0, 19, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 5, 20, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 10, 21, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 15, 22, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 4, 23, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 9, 24, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 14, 25, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 3, 26, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 8, 27, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 13, 28, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 2, 29, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 7, 30, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 12, 31, 20
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 5, 32, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 8, 33, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 11, 34, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 14, 35, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 1, 36, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 4, 37, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 7, 38, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 10, 39, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 13, 40, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 0, 41, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 3, 42, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 6, 43, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 9, 44, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 12, 45, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 15, 46, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 2, 47, 23
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 0, 48, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 7, 49, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 14, 50, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 5, 51, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 12, 52, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 3, 53, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 10, 54, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 1, 55, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 8, 56, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 15, 57, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 6, 58, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 13, 59, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 4, 60, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 11, 61, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 2, 62, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 9, 63, 21
                );
                $a0 = _mm256_add_epi32(initial0[0], $a0);
                $b0 = _mm256_add_epi32(initial0[1], $b0);
                $c0 = _mm256_add_epi32(initial0[2], $c0);
                $d0 = _mm256_add_epi32(initial0[3], $d0);
                $a1 = _mm256_add_epi32(initial1[0], $a1);
                $b1 = _mm256_add_epi32(initial1[1], $b1);
                $c1 = _mm256_add_epi32(initial1[2], $c1);
                $d1 = _mm256_add_epi32(initial1[3], $d1);
            }};
        }

        let len = inputs[0].len();
        let mut lane_words0 = [[0u32; 8]; 4];
        let mut lane_words1 = [[0u32; 8]; 4];
        for word in 0..4 {
            for lane in 0..8 {
                lane_words0[word][lane] = states[lane][word];
                lane_words1[word][lane] = states[lane + 8][word];
            }
        }
        // SAFETY: lane_words0[0] exactly matches one native vector.
        let mut a0 = unsafe { _mm256_loadu_si256(lane_words0[0].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words0[1] exactly matches one native vector.
        let mut b0 = unsafe { _mm256_loadu_si256(lane_words0[1].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words0[2] exactly matches one native vector.
        let mut c0 = unsafe { _mm256_loadu_si256(lane_words0[2].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words0[3] exactly matches one native vector.
        let mut d0 = unsafe { _mm256_loadu_si256(lane_words0[3].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words1[0] exactly matches one native vector.
        let mut a1 = unsafe { _mm256_loadu_si256(lane_words1[0].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words1[1] exactly matches one native vector.
        let mut b1 = unsafe { _mm256_loadu_si256(lane_words1[1].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words1[2] exactly matches one native vector.
        let mut c1 = unsafe { _mm256_loadu_si256(lane_words1[2].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words1[3] exactly matches one native vector.
        let mut d1 = unsafe { _mm256_loadu_si256(lane_words1[3].as_ptr().cast::<__m256i>()) };
        let all_ones = _mm256_set1_epi32(-1);
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 8][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, all_ones);
        }

        let states0 = [a0, b0, c0, d0];
        let states1 = [a1, b1, c1, d1];
        for word in 0..4 {
            unsafe {
                _mm256_storeu_si256(
                    lane_words0[word].as_mut_ptr().cast::<__m256i>(),
                    states0[word],
                )
            };
        }
        for word in 0..4 {
            unsafe {
                _mm256_storeu_si256(
                    lane_words1[word].as_mut_ptr().cast::<__m256i>(),
                    states1[word],
                )
            };
        }
        for lane in 0..8 {
            for word in 0..4 {
                states[lane][word] = lane_words0[word][lane];
                states[lane + 8][word] = lane_words1[word][lane];
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx2_triple_state_kernel(
        avx2: Avx2,
        states: &mut [[u32; 4]],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(inputs.len(), 24);
        debug_assert_eq!(states.len(), 24);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);
                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);
                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }
        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }
        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }
        macro_rules! step3 {
            (g, $a0:ident,$b0:ident,$c0:ident,$d0:ident, $a1:ident,$b1:ident,$c1:ident,$d1:ident, $a2:ident,$b2:ident,$c2:ident,$d2:ident, $w0:ident,$w1:ident,$w2:ident,$ones:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm256_set1_epi32(K[$round] as i32);
                let mut t0 = _mm256_add_epi32($a0, _mm256_andnot_si256($d0, $c0));
                let mut t1 = _mm256_add_epi32($a1, _mm256_andnot_si256($d1, $c1));
                let mut t2 = _mm256_add_epi32($a2, _mm256_andnot_si256($d2, $c2));
                t0 = _mm256_add_epi32(_mm256_add_epi32(t0, key), $w0[$word]);
                t1 = _mm256_add_epi32(_mm256_add_epi32(t1, key), $w1[$word]);
                t2 = _mm256_add_epi32(_mm256_add_epi32(t2, key), $w2[$word]);
                t0 = _mm256_add_epi32(t0, _mm256_and_si256($d0, $b0));
                t1 = _mm256_add_epi32(t1, _mm256_and_si256($d1, $b1));
                t2 = _mm256_add_epi32(t2, _mm256_and_si256($d2, $b2));
                let r0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let r1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                let r2 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t2),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t2),
                );
                $a0 = _mm256_add_epi32($b0, r0);
                $a1 = _mm256_add_epi32($b1, r1);
                $a2 = _mm256_add_epi32($b2, r2);
            }};
            ($which:ident, $a0:ident,$b0:ident,$c0:ident,$d0:ident, $a1:ident,$b1:ident,$c1:ident,$d1:ident, $a2:ident,$b2:ident,$c2:ident,$d2:ident, $w0:ident,$w1:ident,$w2:ident,$ones:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm256_set1_epi32(K[$round] as i32);
                let mut t0 = _mm256_add_epi32($a0, mix!($which, $b0, $c0, $d0, $ones));
                let mut t1 = _mm256_add_epi32($a1, mix!($which, $b1, $c1, $d1, $ones));
                let mut t2 = _mm256_add_epi32($a2, mix!($which, $b2, $c2, $d2, $ones));
                t0 = _mm256_add_epi32(_mm256_add_epi32(t0, key), $w0[$word]);
                t1 = _mm256_add_epi32(_mm256_add_epi32(t1, key), $w1[$word]);
                t2 = _mm256_add_epi32(_mm256_add_epi32(t2, key), $w2[$word]);
                let r0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let r1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                let r2 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t2),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t2),
                );
                $a0 = _mm256_add_epi32($b0, r0);
                $a1 = _mm256_add_epi32($b1, r1);
                $a2 = _mm256_add_epi32($b2, r2);
            }};
        }
        macro_rules! compress3 {
            ($words0:expr,$a0:ident,$b0:ident,$c0:ident,$d0:ident, $words1:expr,$a1:ident,$b1:ident,$c1:ident,$d1:ident, $words2:expr,$a2:ident,$b2:ident,$c2:ident,$d2:ident, $ones:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let words2 = $words2;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                let initial2 = [$a2, $b2, $c2, $d2];
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 0, 0, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 1, 1, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 2, 2, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 3, 3, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 4, 4, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 5, 5, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 6, 6, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 7, 7, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 8, 8, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 9, 9, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 10, 10, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 11, 11, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 12, 12, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 13, 13, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 14, 14, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 15, 15, 22
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 1, 16, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 6, 17, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 11, 18, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 0, 19, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 5, 20, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 10, 21, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 15, 22, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 4, 23, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 9, 24, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 14, 25, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 3, 26, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 8, 27, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 13, 28, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 2, 29, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 7, 30, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 12, 31, 20
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 5, 32, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 8, 33, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 11, 34, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 14, 35, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 1, 36, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 4, 37, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 7, 38, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 10, 39, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 13, 40, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 0, 41, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 3, 42, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 6, 43, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 9, 44, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 12, 45, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 15, 46, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 2, 47, 23
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 0, 48, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 7, 49, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 14, 50, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 5, 51, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 12, 52, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 3, 53, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 10, 54, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 1, 55, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 8, 56, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 15, 57, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 6, 58, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 13, 59, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 4, 60, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 11, 61, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 2, 62, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 9, 63, 21
                );
                $a0 = _mm256_add_epi32(initial0[0], $a0);
                $b0 = _mm256_add_epi32(initial0[1], $b0);
                $c0 = _mm256_add_epi32(initial0[2], $c0);
                $d0 = _mm256_add_epi32(initial0[3], $d0);
                $a1 = _mm256_add_epi32(initial1[0], $a1);
                $b1 = _mm256_add_epi32(initial1[1], $b1);
                $c1 = _mm256_add_epi32(initial1[2], $c1);
                $d1 = _mm256_add_epi32(initial1[3], $d1);
                $a2 = _mm256_add_epi32(initial2[0], $a2);
                $b2 = _mm256_add_epi32(initial2[1], $b2);
                $c2 = _mm256_add_epi32(initial2[2], $c2);
                $d2 = _mm256_add_epi32(initial2[3], $d2);
            }};
        }

        let len = inputs[0].len();
        let mut lane_words0 = [[0u32; 8]; 4];
        let mut lane_words1 = [[0u32; 8]; 4];
        let mut lane_words2 = [[0u32; 8]; 4];
        for word in 0..4 {
            for lane in 0..8 {
                lane_words0[word][lane] = states[lane][word];
                lane_words1[word][lane] = states[lane + 8][word];
                lane_words2[word][lane] = states[lane + 16][word];
            }
        }
        // SAFETY: lane_words0[0] exactly matches one native vector.
        let mut a0 = unsafe { _mm256_loadu_si256(lane_words0[0].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words0[1] exactly matches one native vector.
        let mut b0 = unsafe { _mm256_loadu_si256(lane_words0[1].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words0[2] exactly matches one native vector.
        let mut c0 = unsafe { _mm256_loadu_si256(lane_words0[2].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words0[3] exactly matches one native vector.
        let mut d0 = unsafe { _mm256_loadu_si256(lane_words0[3].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words1[0] exactly matches one native vector.
        let mut a1 = unsafe { _mm256_loadu_si256(lane_words1[0].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words1[1] exactly matches one native vector.
        let mut b1 = unsafe { _mm256_loadu_si256(lane_words1[1].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words1[2] exactly matches one native vector.
        let mut c1 = unsafe { _mm256_loadu_si256(lane_words1[2].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words1[3] exactly matches one native vector.
        let mut d1 = unsafe { _mm256_loadu_si256(lane_words1[3].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words2[0] exactly matches one native vector.
        let mut a2 = unsafe { _mm256_loadu_si256(lane_words2[0].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words2[1] exactly matches one native vector.
        let mut b2 = unsafe { _mm256_loadu_si256(lane_words2[1].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words2[2] exactly matches one native vector.
        let mut c2 = unsafe { _mm256_loadu_si256(lane_words2[2].as_ptr().cast::<__m256i>()) };
        // SAFETY: lane_words2[3] exactly matches one native vector.
        let mut d2 = unsafe { _mm256_loadu_si256(lane_words2[3].as_ptr().cast::<__m256i>()) };
        let all_ones = _mm256_set1_epi32(-1);
        let full_blocks = len / 64;
        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 8][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks2: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2, all_ones
            );
        }
        let states0 = [a0, b0, c0, d0];
        let states1 = [a1, b1, c1, d1];
        let states2 = [a2, b2, c2, d2];
        for word in 0..4 {
            unsafe {
                _mm256_storeu_si256(
                    lane_words0[word].as_mut_ptr().cast::<__m256i>(),
                    states0[word],
                )
            };
        }
        for word in 0..4 {
            unsafe {
                _mm256_storeu_si256(
                    lane_words1[word].as_mut_ptr().cast::<__m256i>(),
                    states1[word],
                )
            };
        }
        for word in 0..4 {
            unsafe {
                _mm256_storeu_si256(
                    lane_words2[word].as_mut_ptr().cast::<__m256i>(),
                    states2[word],
                )
            };
        }
        for lane in 0..8 {
            for word in 0..4 {
                states[lane][word] = lane_words0[word][lane];
                states[lane + 8][word] = lane_words1[word][lane];
                states[lane + 16][word] = lane_words2[word][lane];
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx512_dual_state_kernel(
        avx512: Avx512,
        states: &mut [[u32; 4]],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(inputs.len(), 32);
        debug_assert_eq!(states.len(), 32);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }

                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }

                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }

                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    // SAFETY: each entry is a full 64-byte MD5 block and the
                    // unaligned AVX-512 load reads exactly those 64 bytes.
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }

        macro_rules! step2 {
            ($which:ident, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed0 = mix!($which, $b0, $c0, $d0);
                let mixed1 = mix!($which, $b1, $c1, $d1);
                let mut t0 = _mm512_add_epi32($a0, mixed0);
                let mut t1 = _mm512_add_epi32($a1, mixed1);
                let key = _mm512_set1_epi32(K[$round] as i32);
                t0 = _mm512_add_epi32(t0, key);
                t1 = _mm512_add_epi32(t1, key);
                t0 = _mm512_add_epi32(t0, $words0[$word]);
                t1 = _mm512_add_epi32(t1, $words1[$word]);
                $a0 = _mm512_add_epi32($b0, _mm512_rol_epi32::<$shift>(t0));
                $a1 = _mm512_add_epi32($b1, _mm512_rol_epi32::<$shift>(t1));
            }};
        }

        macro_rules! compress2 {
            ($words0:expr, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $words1:expr, $a1:ident, $b1:ident, $c1:ident, $d1:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 0, 0, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 1, 1, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 2, 2, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 3, 3, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 4, 4, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 5, 5, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 6, 6, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 7, 7, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 8, 8, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 9, 9, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 10, 10, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 11, 11, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 12, 12, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 13, 13, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 14, 14, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 15, 15, 22
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 1, 16, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 6, 17, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 11, 18, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 0, 19, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 5, 20, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 10, 21, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 15, 22, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 4, 23, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 9, 24, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 14, 25, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 3, 26, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 8, 27, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 13, 28, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 2, 29, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 7, 30, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 12, 31, 20
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 5, 32, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 8, 33, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 11, 34, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 14, 35, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 1, 36, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 4, 37, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 7, 38, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 10, 39, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 13, 40, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 0, 41, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 3, 42, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 6, 43, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 9, 44, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 12, 45, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 15, 46, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 2, 47, 23
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 0, 48, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 7, 49, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 14, 50, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 5, 51, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 12, 52, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 3, 53, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 10, 54, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 1, 55, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 8, 56, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 15, 57, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 6, 58, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 13, 59, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 4, 60, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 11, 61, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 2, 62, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 9, 63, 21
                );
                $a0 = _mm512_add_epi32(initial0[0], $a0);
                $b0 = _mm512_add_epi32(initial0[1], $b0);
                $c0 = _mm512_add_epi32(initial0[2], $c0);
                $d0 = _mm512_add_epi32(initial0[3], $d0);
                $a1 = _mm512_add_epi32(initial1[0], $a1);
                $b1 = _mm512_add_epi32(initial1[1], $b1);
                $c1 = _mm512_add_epi32(initial1[2], $c1);
                $d1 = _mm512_add_epi32(initial1[3], $d1);
            }};
        }

        let len = inputs[0].len();
        let mut lane_words0 = [[0u32; 16]; 4];
        let mut lane_words1 = [[0u32; 16]; 4];
        for word in 0..4 {
            for lane in 0..16 {
                lane_words0[word][lane] = states[lane][word];
                lane_words1[word][lane] = states[lane + 16][word];
            }
        }
        // SAFETY: lane_words0[0] exactly matches one native vector.
        let mut a0 = unsafe { _mm512_loadu_si512(lane_words0[0].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words0[1] exactly matches one native vector.
        let mut b0 = unsafe { _mm512_loadu_si512(lane_words0[1].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words0[2] exactly matches one native vector.
        let mut c0 = unsafe { _mm512_loadu_si512(lane_words0[2].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words0[3] exactly matches one native vector.
        let mut d0 = unsafe { _mm512_loadu_si512(lane_words0[3].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words1[0] exactly matches one native vector.
        let mut a1 = unsafe { _mm512_loadu_si512(lane_words1[0].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words1[1] exactly matches one native vector.
        let mut b1 = unsafe { _mm512_loadu_si512(lane_words1[1].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words1[2] exactly matches one native vector.
        let mut c1 = unsafe { _mm512_loadu_si512(lane_words1[2].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words1[3] exactly matches one native vector.
        let mut d1 = unsafe { _mm512_loadu_si512(lane_words1[3].as_ptr().cast::<__m512i>()) };
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1);
        }

        let states0 = [a0, b0, c0, d0];
        let states1 = [a1, b1, c1, d1];
        for word in 0..4 {
            unsafe {
                _mm512_storeu_si512(
                    lane_words0[word].as_mut_ptr().cast::<__m512i>(),
                    states0[word],
                )
            };
        }
        for word in 0..4 {
            unsafe {
                _mm512_storeu_si512(
                    lane_words1[word].as_mut_ptr().cast::<__m512i>(),
                    states1[word],
                )
            };
        }
        for lane in 0..16 {
            for word in 0..4 {
                states[lane][word] = lane_words0[word][lane];
                states[lane + 16][word] = lane_words1[word][lane];
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx512_triple_state_kernel(
        avx512: Avx512,
        states: &mut [[u32; 4]],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(inputs.len(), 48);
        debug_assert_eq!(states.len(), 48);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx512;

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }
                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }
                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }
                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }
        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }
        macro_rules! mix {
            (f,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }
        macro_rules! step3 {
            ($which:ident,$a0:ident,$b0:ident,$c0:ident,$d0:ident,$a1:ident,$b1:ident,$c1:ident,$d1:ident,$a2:ident,$b2:ident,$c2:ident,$d2:ident,$w0:ident,$w1:ident,$w2:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm512_set1_epi32(K[$round] as i32);
                let mut t0 = _mm512_add_epi32($a0, mix!($which, $b0, $c0, $d0));
                let mut t1 = _mm512_add_epi32($a1, mix!($which, $b1, $c1, $d1));
                let mut t2 = _mm512_add_epi32($a2, mix!($which, $b2, $c2, $d2));
                t0 = _mm512_add_epi32(_mm512_add_epi32(t0, key), $w0[$word]);
                t1 = _mm512_add_epi32(_mm512_add_epi32(t1, key), $w1[$word]);
                t2 = _mm512_add_epi32(_mm512_add_epi32(t2, key), $w2[$word]);
                $a0 = _mm512_add_epi32($b0, _mm512_rol_epi32::<$shift>(t0));
                $a1 = _mm512_add_epi32($b1, _mm512_rol_epi32::<$shift>(t1));
                $a2 = _mm512_add_epi32($b2, _mm512_rol_epi32::<$shift>(t2));
            }};
        }
        macro_rules! compress3 {
            ($words0:expr,$a0:ident,$b0:ident,$c0:ident,$d0:ident,$words1:expr,$a1:ident,$b1:ident,$c1:ident,$d1:ident,$words2:expr,$a2:ident,$b2:ident,$c2:ident,$d2:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let words2 = $words2;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                let initial2 = [$a2, $b2, $c2, $d2];
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 0, 0, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 1, 1, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 2, 2, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 3, 3, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 4, 4, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 5, 5, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 6, 6, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 7, 7, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 8, 8, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 9, 9, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 10, 10, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 11, 11, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 12, 12, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 13, 13, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 14, 14, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 15, 15, 22
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 1, 16, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 6, 17, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 11, 18, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 0, 19, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 5, 20, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 10, 21, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 15, 22, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 4, 23, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 9, 24, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 14, 25, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 3, 26, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 8, 27, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 13, 28, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 2, 29, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 7, 30, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 12, 31, 20
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 5, 32, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 8, 33, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 11, 34, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 14, 35, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 1, 36, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 4, 37, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 7, 38, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 10, 39, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 13, 40, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 0, 41, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 3, 42, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 6, 43, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 9, 44, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 12, 45, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 15, 46, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 2, 47, 23
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 0, 48, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 7, 49, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 14, 50, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 5, 51, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 12, 52, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 3, 53, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 10, 54, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 1, 55, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 8, 56, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 15, 57, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 6, 58, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 13, 59, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 4, 60, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 11, 61, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 2, 62, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 9, 63, 21
                );
                $a0 = _mm512_add_epi32(initial0[0], $a0);
                $b0 = _mm512_add_epi32(initial0[1], $b0);
                $c0 = _mm512_add_epi32(initial0[2], $c0);
                $d0 = _mm512_add_epi32(initial0[3], $d0);
                $a1 = _mm512_add_epi32(initial1[0], $a1);
                $b1 = _mm512_add_epi32(initial1[1], $b1);
                $c1 = _mm512_add_epi32(initial1[2], $c1);
                $d1 = _mm512_add_epi32(initial1[3], $d1);
                $a2 = _mm512_add_epi32(initial2[0], $a2);
                $b2 = _mm512_add_epi32(initial2[1], $b2);
                $c2 = _mm512_add_epi32(initial2[2], $c2);
                $d2 = _mm512_add_epi32(initial2[3], $d2);
            }};
        }
        let len = inputs[0].len();
        let mut lane_words0 = [[0u32; 16]; 4];
        let mut lane_words1 = [[0u32; 16]; 4];
        let mut lane_words2 = [[0u32; 16]; 4];
        for word in 0..4 {
            for lane in 0..16 {
                lane_words0[word][lane] = states[lane][word];
                lane_words1[word][lane] = states[lane + 16][word];
                lane_words2[word][lane] = states[lane + 32][word];
            }
        }
        // SAFETY: lane_words0[0] exactly matches one native vector.
        let mut a0 = unsafe { _mm512_loadu_si512(lane_words0[0].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words0[1] exactly matches one native vector.
        let mut b0 = unsafe { _mm512_loadu_si512(lane_words0[1].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words0[2] exactly matches one native vector.
        let mut c0 = unsafe { _mm512_loadu_si512(lane_words0[2].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words0[3] exactly matches one native vector.
        let mut d0 = unsafe { _mm512_loadu_si512(lane_words0[3].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words1[0] exactly matches one native vector.
        let mut a1 = unsafe { _mm512_loadu_si512(lane_words1[0].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words1[1] exactly matches one native vector.
        let mut b1 = unsafe { _mm512_loadu_si512(lane_words1[1].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words1[2] exactly matches one native vector.
        let mut c1 = unsafe { _mm512_loadu_si512(lane_words1[2].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words1[3] exactly matches one native vector.
        let mut d1 = unsafe { _mm512_loadu_si512(lane_words1[3].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words2[0] exactly matches one native vector.
        let mut a2 = unsafe { _mm512_loadu_si512(lane_words2[0].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words2[1] exactly matches one native vector.
        let mut b2 = unsafe { _mm512_loadu_si512(lane_words2[1].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words2[2] exactly matches one native vector.
        let mut c2 = unsafe { _mm512_loadu_si512(lane_words2[2].as_ptr().cast::<__m512i>()) };
        // SAFETY: lane_words2[3] exactly matches one native vector.
        let mut d2 = unsafe { _mm512_loadu_si512(lane_words2[3].as_ptr().cast::<__m512i>()) };
        let full_blocks = len / 64;
        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks2: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 32][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2
            );
        }
        let states0 = [a0, b0, c0, d0];
        let states1 = [a1, b1, c1, d1];
        let states2 = [a2, b2, c2, d2];
        for word in 0..4 {
            unsafe {
                _mm512_storeu_si512(
                    lane_words0[word].as_mut_ptr().cast::<__m512i>(),
                    states0[word],
                )
            };
        }
        for word in 0..4 {
            unsafe {
                _mm512_storeu_si512(
                    lane_words1[word].as_mut_ptr().cast::<__m512i>(),
                    states1[word],
                )
            };
        }
        for word in 0..4 {
            unsafe {
                _mm512_storeu_si512(
                    lane_words2[word].as_mut_ptr().cast::<__m512i>(),
                    states2[word],
                )
            };
        }
        for lane in 0..16 {
            for word in 0..4 {
                states[lane][word] = lane_words0[word][lane];
                states[lane + 16][word] = lane_words1[word][lane];
                states[lane + 32][word] = lane_words2[word][lane];
            }
        }
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! transpose4x4_epi32 {
    ($r0:expr, $r1:expr, $r2:expr, $r3:expr) => {{
        let t0 = _mm_unpacklo_epi32($r0, $r1);
        let t1 = _mm_unpackhi_epi32($r0, $r1);
        let t2 = _mm_unpacklo_epi32($r2, $r3);
        let t3 = _mm_unpackhi_epi32($r2, $r3);
        [
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        ]
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! load_md5_state8 {
    ($streams:expr, $base:expr) => {{
        let streams = &*$streams;
        let base = $base;
        macro_rules! load_lane {
            ($lane:expr) => {{
                // SAFETY: `state` is a contiguous `[u32; 4]`, exactly 16 bytes.
                unsafe { _mm_loadu_si128(streams[base + $lane].state.as_ptr().cast::<__m128i>()) }
            }};
        }
        let r0 = load_lane!(0);
        let r1 = load_lane!(1);
        let r2 = load_lane!(2);
        let r3 = load_lane!(3);
        let r4 = load_lane!(4);
        let r5 = load_lane!(5);
        let r6 = load_lane!(6);
        let r7 = load_lane!(7);
        let lo = transpose4x4_epi32!(r0, r1, r2, r3);
        let hi = transpose4x4_epi32!(r4, r5, r6, r7);
        core::array::from_fn(|word| {
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo[word]), hi[word])
        })
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! store_md5_state8 {
    ($streams:expr, $base:expr, $a:expr, $b:expr, $c:expr, $d:expr) => {{
        let streams = &mut *$streams;
        let base = $base;
        let v = [$a, $b, $c, $d];
        let lo: [__m128i; 4] = core::array::from_fn(|word| _mm256_castsi256_si128(v[word]));
        let hi: [__m128i; 4] = core::array::from_fn(|word| _mm256_extracti128_si256::<1>(v[word]));
        let rows0 = transpose4x4_epi32!(lo[0], lo[1], lo[2], lo[3]);
        let rows1 = transpose4x4_epi32!(hi[0], hi[1], hi[2], hi[3]);
        macro_rules! store_lane {
            ($lane:expr, $row:expr) => {{
                // SAFETY: `state` is a contiguous `[u32; 4]`, exactly 16 bytes.
                unsafe {
                    _mm_storeu_si128(
                        streams[base + $lane].state.as_mut_ptr().cast::<__m128i>(),
                        $row,
                    )
                };
            }};
        }
        store_lane!(0, rows0[0]);
        store_lane!(1, rows0[1]);
        store_lane!(2, rows0[2]);
        store_lane!(3, rows0[3]);
        store_lane!(4, rows1[0]);
        store_lane!(5, rows1[1]);
        store_lane!(6, rows1[2]);
        store_lane!(7, rows1[3]);
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! load_md5_state16 {
    ($streams:expr, $base:expr) => {{
        let streams = &*$streams;
        let base = $base;
        macro_rules! load_lane {
            ($lane:expr) => {{
                // SAFETY: `state` is a contiguous `[u32; 4]`, exactly 16 bytes.
                unsafe { _mm_loadu_si128(streams[base + $lane].state.as_ptr().cast::<__m128i>()) }
            }};
        }
        let g0 = transpose4x4_epi32!(load_lane!(0), load_lane!(1), load_lane!(2), load_lane!(3));
        let g1 = transpose4x4_epi32!(load_lane!(4), load_lane!(5), load_lane!(6), load_lane!(7));
        let g2 = transpose4x4_epi32!(load_lane!(8), load_lane!(9), load_lane!(10), load_lane!(11));
        let g3 = transpose4x4_epi32!(
            load_lane!(12),
            load_lane!(13),
            load_lane!(14),
            load_lane!(15)
        );
        core::array::from_fn(|word| {
            let v = _mm512_castsi128_si512(g0[word]);
            let v = _mm512_inserti32x4::<1>(v, g1[word]);
            let v = _mm512_inserti32x4::<2>(v, g2[word]);
            _mm512_inserti32x4::<3>(v, g3[word])
        })
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! store_md5_state16 {
    ($streams:expr, $base:expr, $a:expr, $b:expr, $c:expr, $d:expr) => {{
        let streams = &mut *$streams;
        let base = $base;
        let v = [$a, $b, $c, $d];
        let q0: [__m128i; 4] = core::array::from_fn(|word| _mm512_castsi512_si128(v[word]));
        let q1: [__m128i; 4] = core::array::from_fn(|word| _mm512_extracti32x4_epi32::<1>(v[word]));
        let q2: [__m128i; 4] = core::array::from_fn(|word| _mm512_extracti32x4_epi32::<2>(v[word]));
        let q3: [__m128i; 4] = core::array::from_fn(|word| _mm512_extracti32x4_epi32::<3>(v[word]));
        let rows0 = transpose4x4_epi32!(q0[0], q0[1], q0[2], q0[3]);
        let rows1 = transpose4x4_epi32!(q1[0], q1[1], q1[2], q1[3]);
        let rows2 = transpose4x4_epi32!(q2[0], q2[1], q2[2], q2[3]);
        let rows3 = transpose4x4_epi32!(q3[0], q3[1], q3[2], q3[3]);
        macro_rules! store_lane {
            ($lane:expr, $row:expr) => {{
                // SAFETY: `state` is a contiguous `[u32; 4]`, exactly 16 bytes.
                unsafe {
                    _mm_storeu_si128(
                        streams[base + $lane].state.as_mut_ptr().cast::<__m128i>(),
                        $row,
                    )
                };
            }};
        }
        store_lane!(0, rows0[0]);
        store_lane!(1, rows0[1]);
        store_lane!(2, rows0[2]);
        store_lane!(3, rows0[3]);
        store_lane!(4, rows1[0]);
        store_lane!(5, rows1[1]);
        store_lane!(6, rows1[2]);
        store_lane!(7, rows1[3]);
        store_lane!(8, rows2[0]);
        store_lane!(9, rows2[1]);
        store_lane!(10, rows2[2]);
        store_lane!(11, rows2[3]);
        store_lane!(12, rows3[0]);
        store_lane!(13, rows3[1]);
        store_lane!(14, rows3[2]);
        store_lane!(15, rows3[3]);
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx2_md5_state_kernel(
        avx2: Avx2,
        streams: &mut [Md5State],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(streams.len(), 8);
        debug_assert_eq!(inputs.len(), 8);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        debug_assert!(inputs[0].len().is_multiple_of(64));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);

                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    // SAFETY: every entry is a 64-byte block, and unaligned AVX2
                    // loads read exactly bytes 0..32 and 32..64 respectively.
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    // SAFETY: bytes 32..64 are inside the same 64-byte block.
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (g, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $z), _mm256_andnot_si256($z, $y))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }

        macro_rules! step {
            (g, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t = _mm256_add_epi32($a, _mm256_andnot_si256($d, $c));
                t = _mm256_add_epi32(t, _mm256_set1_epi32(K[$round] as i32));
                t = _mm256_add_epi32(t, $words[$word]);
                t = _mm256_add_epi32(t, _mm256_and_si256($d, $b));
                let rotated = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t),
                );
                $a = _mm256_add_epi32($b, rotated);
            }};
            ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t = _mm256_add_epi32($a, mix!($which, $b, $c, $d, $ones));
                t = _mm256_add_epi32(t, _mm256_set1_epi32(K[$round] as i32));
                t = _mm256_add_epi32(t, $words[$word]);
                let rotated = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t),
                );
                $a = _mm256_add_epi32($b, rotated);
            }};
        }

        macro_rules! compress {
            ($word_expr:expr, $a:ident, $b:ident, $c:ident, $d:ident, $ones:ident) => {{
                let words = $word_expr;
                let initial = [$a, $b, $c, $d];
                step!(f, $a, $b, $c, $d, words, $ones, 0, 0, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 1, 1, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 2, 2, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 3, 3, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 4, 4, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 5, 5, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 6, 6, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 7, 7, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 8, 8, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 9, 9, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 10, 10, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 11, 11, 22);
                step!(f, $a, $b, $c, $d, words, $ones, 12, 12, 7);
                step!(f, $d, $a, $b, $c, words, $ones, 13, 13, 12);
                step!(f, $c, $d, $a, $b, words, $ones, 14, 14, 17);
                step!(f, $b, $c, $d, $a, words, $ones, 15, 15, 22);
                step!(g, $a, $b, $c, $d, words, $ones, 1, 16, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 6, 17, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 11, 18, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 0, 19, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 5, 20, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 10, 21, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 15, 22, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 4, 23, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 9, 24, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 14, 25, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 3, 26, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 8, 27, 20);
                step!(g, $a, $b, $c, $d, words, $ones, 13, 28, 5);
                step!(g, $d, $a, $b, $c, words, $ones, 2, 29, 9);
                step!(g, $c, $d, $a, $b, words, $ones, 7, 30, 14);
                step!(g, $b, $c, $d, $a, words, $ones, 12, 31, 20);
                step!(h, $a, $b, $c, $d, words, $ones, 5, 32, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 8, 33, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 11, 34, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 14, 35, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 1, 36, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 4, 37, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 7, 38, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 10, 39, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 13, 40, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 0, 41, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 3, 42, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 6, 43, 23);
                step!(h, $a, $b, $c, $d, words, $ones, 9, 44, 4);
                step!(h, $d, $a, $b, $c, words, $ones, 12, 45, 11);
                step!(h, $c, $d, $a, $b, words, $ones, 15, 46, 16);
                step!(h, $b, $c, $d, $a, words, $ones, 2, 47, 23);
                step!(i, $a, $b, $c, $d, words, $ones, 0, 48, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 7, 49, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 14, 50, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 5, 51, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 12, 52, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 3, 53, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 10, 54, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 1, 55, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 8, 56, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 15, 57, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 6, 58, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 13, 59, 21);
                step!(i, $a, $b, $c, $d, words, $ones, 4, 60, 6);
                step!(i, $d, $a, $b, $c, words, $ones, 11, 61, 10);
                step!(i, $c, $d, $a, $b, words, $ones, 2, 62, 15);
                step!(i, $b, $c, $d, $a, words, $ones, 9, 63, 21);
                $a = _mm256_add_epi32(initial[0], $a);
                $b = _mm256_add_epi32(initial[1], $b);
                $c = _mm256_add_epi32(initial[2], $c);
                $d = _mm256_add_epi32(initial[3], $d);
            }};
        }

        let [mut a, mut b, mut c, mut d] = load_md5_state8!(streams, 0);
        let all_ones = _mm256_set1_epi32(-1);

        let block_count = inputs[0].len() / 64;
        for block_index in 0..block_count {
            let offset = block_index * 64;
            let blocks: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d, all_ones);
        }

        store_md5_state8!(streams, 0, a, b, c, d);
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx2_dual_md5_state_kernel(
        avx2: Avx2,
        streams: &mut [Md5State],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(inputs.len(), 16);
        debug_assert_eq!(streams.len(), 16);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);

                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    // SAFETY: every entry is a 64-byte block, and unaligned AVX2
                    // loads read exactly bytes 0..32 and 32..64 respectively.
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    // SAFETY: bytes 32..64 are inside the same 64-byte block.
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (g, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $z), _mm256_andnot_si256($z, $y))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }

        macro_rules! step2 {
            (g, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mut t0 = _mm256_add_epi32($a0, _mm256_andnot_si256($d0, $c0));
                let mut t1 = _mm256_add_epi32($a1, _mm256_andnot_si256($d1, $c1));
                let key = _mm256_set1_epi32(K[$round] as i32);
                t0 = _mm256_add_epi32(t0, key);
                t1 = _mm256_add_epi32(t1, key);
                t0 = _mm256_add_epi32(t0, $words0[$word]);
                t1 = _mm256_add_epi32(t1, $words1[$word]);
                t0 = _mm256_add_epi32(t0, _mm256_and_si256($d0, $b0));
                t1 = _mm256_add_epi32(t1, _mm256_and_si256($d1, $b1));
                let rotated0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let rotated1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                $a0 = _mm256_add_epi32($b0, rotated0);
                $a1 = _mm256_add_epi32($b1, rotated1);
            }};
            ($which:ident, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $ones:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed0 = mix!($which, $b0, $c0, $d0, $ones);
                let mixed1 = mix!($which, $b1, $c1, $d1, $ones);
                let mut t0 = _mm256_add_epi32($a0, mixed0);
                let mut t1 = _mm256_add_epi32($a1, mixed1);
                let key = _mm256_set1_epi32(K[$round] as i32);
                t0 = _mm256_add_epi32(t0, key);
                t1 = _mm256_add_epi32(t1, key);
                t0 = _mm256_add_epi32(t0, $words0[$word]);
                t1 = _mm256_add_epi32(t1, $words1[$word]);
                let rotated0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let rotated1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                $a0 = _mm256_add_epi32($b0, rotated0);
                $a1 = _mm256_add_epi32($b1, rotated1);
            }};
        }

        macro_rules! compress2 {
            ($words0:expr, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $words1:expr, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $ones:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 0, 0, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 1, 1, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 2, 2, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 3, 3, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 4, 4, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 5, 5, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 6, 6, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 7, 7, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 8, 8, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 9, 9, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 10, 10, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 11, 11, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 12, 12, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 13, 13, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 14, 14, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 15, 15, 22
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 1, 16, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 6, 17, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 11, 18, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 0, 19, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 5, 20, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 10, 21, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 15, 22, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 4, 23, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 9, 24, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 14, 25, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 3, 26, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 8, 27, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 13, 28, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 2, 29, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 7, 30, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 12, 31, 20
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 5, 32, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 8, 33, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 11, 34, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 14, 35, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 1, 36, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 4, 37, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 7, 38, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 10, 39, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 13, 40, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 0, 41, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 3, 42, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 6, 43, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 9, 44, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 12, 45, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 15, 46, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 2, 47, 23
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 0, 48, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 7, 49, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 14, 50, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 5, 51, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 12, 52, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 3, 53, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 10, 54, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 1, 55, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 8, 56, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 15, 57, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 6, 58, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 13, 59, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, $ones, 4, 60, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, $ones, 11, 61, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, $ones, 2, 62, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, $ones, 9, 63, 21
                );
                $a0 = _mm256_add_epi32(initial0[0], $a0);
                $b0 = _mm256_add_epi32(initial0[1], $b0);
                $c0 = _mm256_add_epi32(initial0[2], $c0);
                $d0 = _mm256_add_epi32(initial0[3], $d0);
                $a1 = _mm256_add_epi32(initial1[0], $a1);
                $b1 = _mm256_add_epi32(initial1[1], $b1);
                $c1 = _mm256_add_epi32(initial1[2], $c1);
                $d1 = _mm256_add_epi32(initial1[3], $d1);
            }};
        }

        let len = inputs[0].len();
        let [mut a0, mut b0, mut c0, mut d0] = load_md5_state8!(streams, 0);
        let [mut a1, mut b1, mut c1, mut d1] = load_md5_state8!(streams, 8);
        let all_ones = _mm256_set1_epi32(-1);
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 8][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, all_ones);
        }

        store_md5_state8!(streams, 0, a0, b0, c0, d0);
        store_md5_state8!(streams, 8, a1, b1, c1, d1);
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx2_triple_md5_state_kernel(
        avx2: Avx2,
        streams: &mut [Md5State],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(inputs.len(), 24);
        debug_assert_eq!(streams.len(), 24);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx2;

        macro_rules! transpose8 {
            ($rows:expr) => {{
                let rows = $rows;
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);
                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);
                [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ]
            }};
        }
        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut lo = [_mm256_setzero_si256(); 8];
                let mut hi = lo;
                for lane in 0..8 {
                    let ptr = blocks[lane].as_ptr();
                    lo[lane] = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
                    hi[lane] = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };
                }
                let lo = transpose8!(lo);
                let hi = transpose8!(hi);
                [
                    lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2],
                    hi[3], hi[4], hi[5], hi[6], hi[7],
                ]
            }};
        }
        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_or_si256(_mm256_and_si256($x, $y), _mm256_andnot_si256($x, $z))
            };
            (h, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256(_mm256_xor_si256($x, $y), $z)
            };
            (i, $x:expr, $y:expr, $z:expr, $ones:expr) => {
                _mm256_xor_si256($y, _mm256_or_si256($x, _mm256_xor_si256($z, $ones)))
            };
        }
        macro_rules! step3 {
            (g, $a0:ident,$b0:ident,$c0:ident,$d0:ident, $a1:ident,$b1:ident,$c1:ident,$d1:ident, $a2:ident,$b2:ident,$c2:ident,$d2:ident, $w0:ident,$w1:ident,$w2:ident,$ones:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm256_set1_epi32(K[$round] as i32);
                let mut t0 = _mm256_add_epi32($a0, _mm256_andnot_si256($d0, $c0));
                let mut t1 = _mm256_add_epi32($a1, _mm256_andnot_si256($d1, $c1));
                let mut t2 = _mm256_add_epi32($a2, _mm256_andnot_si256($d2, $c2));
                t0 = _mm256_add_epi32(_mm256_add_epi32(t0, key), $w0[$word]);
                t1 = _mm256_add_epi32(_mm256_add_epi32(t1, key), $w1[$word]);
                t2 = _mm256_add_epi32(_mm256_add_epi32(t2, key), $w2[$word]);
                t0 = _mm256_add_epi32(t0, _mm256_and_si256($d0, $b0));
                t1 = _mm256_add_epi32(t1, _mm256_and_si256($d1, $b1));
                t2 = _mm256_add_epi32(t2, _mm256_and_si256($d2, $b2));
                let r0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let r1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                let r2 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t2),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t2),
                );
                $a0 = _mm256_add_epi32($b0, r0);
                $a1 = _mm256_add_epi32($b1, r1);
                $a2 = _mm256_add_epi32($b2, r2);
            }};
            ($which:ident, $a0:ident,$b0:ident,$c0:ident,$d0:ident, $a1:ident,$b1:ident,$c1:ident,$d1:ident, $a2:ident,$b2:ident,$c2:ident,$d2:ident, $w0:ident,$w1:ident,$w2:ident,$ones:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm256_set1_epi32(K[$round] as i32);
                let mut t0 = _mm256_add_epi32($a0, mix!($which, $b0, $c0, $d0, $ones));
                let mut t1 = _mm256_add_epi32($a1, mix!($which, $b1, $c1, $d1, $ones));
                let mut t2 = _mm256_add_epi32($a2, mix!($which, $b2, $c2, $d2, $ones));
                t0 = _mm256_add_epi32(_mm256_add_epi32(t0, key), $w0[$word]);
                t1 = _mm256_add_epi32(_mm256_add_epi32(t1, key), $w1[$word]);
                t2 = _mm256_add_epi32(_mm256_add_epi32(t2, key), $w2[$word]);
                let r0 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t0),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t0),
                );
                let r1 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t1),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t1),
                );
                let r2 = _mm256_or_si256(
                    _mm256_slli_epi32::<$shift>(t2),
                    _mm256_srli_epi32::<{ 32 - $shift }>(t2),
                );
                $a0 = _mm256_add_epi32($b0, r0);
                $a1 = _mm256_add_epi32($b1, r1);
                $a2 = _mm256_add_epi32($b2, r2);
            }};
        }
        macro_rules! compress3 {
            ($words0:expr,$a0:ident,$b0:ident,$c0:ident,$d0:ident, $words1:expr,$a1:ident,$b1:ident,$c1:ident,$d1:ident, $words2:expr,$a2:ident,$b2:ident,$c2:ident,$d2:ident, $ones:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let words2 = $words2;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                let initial2 = [$a2, $b2, $c2, $d2];
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 0, 0, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 1, 1, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 2, 2, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 3, 3, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 4, 4, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 5, 5, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 6, 6, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 7, 7, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 8, 8, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 9, 9, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 10, 10, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 11, 11, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 12, 12, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 13, 13, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 14, 14, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 15, 15, 22
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 1, 16, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 6, 17, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 11, 18, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 0, 19, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 5, 20, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 10, 21, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 15, 22, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 4, 23, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 9, 24, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 14, 25, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 3, 26, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 8, 27, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 13, 28, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 2, 29, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 7, 30, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 12, 31, 20
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 5, 32, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 8, 33, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 11, 34, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 14, 35, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 1, 36, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 4, 37, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 7, 38, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 10, 39, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 13, 40, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 0, 41, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 3, 42, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 6, 43, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 9, 44, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 12, 45, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 15, 46, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 2, 47, 23
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 0, 48, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 7, 49, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 14, 50, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 5, 51, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 12, 52, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 3, 53, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 10, 54, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 1, 55, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 8, 56, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 15, 57, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 6, 58, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 13, 59, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, $ones, 4, 60, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, $ones, 11, 61, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, $ones, 2, 62, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, $ones, 9, 63, 21
                );
                $a0 = _mm256_add_epi32(initial0[0], $a0);
                $b0 = _mm256_add_epi32(initial0[1], $b0);
                $c0 = _mm256_add_epi32(initial0[2], $c0);
                $d0 = _mm256_add_epi32(initial0[3], $d0);
                $a1 = _mm256_add_epi32(initial1[0], $a1);
                $b1 = _mm256_add_epi32(initial1[1], $b1);
                $c1 = _mm256_add_epi32(initial1[2], $c1);
                $d1 = _mm256_add_epi32(initial1[3], $d1);
                $a2 = _mm256_add_epi32(initial2[0], $a2);
                $b2 = _mm256_add_epi32(initial2[1], $b2);
                $c2 = _mm256_add_epi32(initial2[2], $c2);
                $d2 = _mm256_add_epi32(initial2[3], $d2);
            }};
        }

        let len = inputs[0].len();
        let [mut a0, mut b0, mut c0, mut d0] = load_md5_state8!(streams, 0);
        let [mut a1, mut b1, mut c1, mut d1] = load_md5_state8!(streams, 8);
        let [mut a2, mut b2, mut c2, mut d2] = load_md5_state8!(streams, 16);
        let all_ones = _mm256_set1_epi32(-1);
        let full_blocks = len / 64;
        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 8][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks2: [&[u8; 64]; 8] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2, all_ones
            );
        }
        store_md5_state8!(streams, 0, a0, b0, c0, d0);
        store_md5_state8!(streams, 8, a1, b1, c1, d1);
        store_md5_state8!(streams, 16, a2, b2, c2, d2);
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx512_md5_state_kernel(
        avx512: Avx512,
        streams: &mut [Md5State],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(streams.len(), 16);
        debug_assert_eq!(inputs.len(), 16);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        debug_assert!(inputs[0].len().is_multiple_of(64));
        let _ = avx512;

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }

                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }

                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }

                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    // SAFETY: each entry is a full 64-byte MD5 block and the
                    // unaligned AVX-512 load reads exactly those 64 bytes.
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }

        macro_rules! step {
            ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $words:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed = mix!($which, $b, $c, $d);
                let mut t = _mm512_add_epi32($a, mixed);
                t = _mm512_add_epi32(t, _mm512_set1_epi32(K[$round] as i32));
                t = _mm512_add_epi32(t, $words[$word]);
                $a = _mm512_add_epi32($b, _mm512_rol_epi32::<$shift>(t));
            }};
        }

        macro_rules! compress {
            ($words:expr, $a:ident, $b:ident, $c:ident, $d:ident) => {{
                let words = $words;
                let initial = [$a, $b, $c, $d];
                step!(f, $a, $b, $c, $d, words, 0, 0, 7);
                step!(f, $d, $a, $b, $c, words, 1, 1, 12);
                step!(f, $c, $d, $a, $b, words, 2, 2, 17);
                step!(f, $b, $c, $d, $a, words, 3, 3, 22);
                step!(f, $a, $b, $c, $d, words, 4, 4, 7);
                step!(f, $d, $a, $b, $c, words, 5, 5, 12);
                step!(f, $c, $d, $a, $b, words, 6, 6, 17);
                step!(f, $b, $c, $d, $a, words, 7, 7, 22);
                step!(f, $a, $b, $c, $d, words, 8, 8, 7);
                step!(f, $d, $a, $b, $c, words, 9, 9, 12);
                step!(f, $c, $d, $a, $b, words, 10, 10, 17);
                step!(f, $b, $c, $d, $a, words, 11, 11, 22);
                step!(f, $a, $b, $c, $d, words, 12, 12, 7);
                step!(f, $d, $a, $b, $c, words, 13, 13, 12);
                step!(f, $c, $d, $a, $b, words, 14, 14, 17);
                step!(f, $b, $c, $d, $a, words, 15, 15, 22);
                step!(g, $a, $b, $c, $d, words, 1, 16, 5);
                step!(g, $d, $a, $b, $c, words, 6, 17, 9);
                step!(g, $c, $d, $a, $b, words, 11, 18, 14);
                step!(g, $b, $c, $d, $a, words, 0, 19, 20);
                step!(g, $a, $b, $c, $d, words, 5, 20, 5);
                step!(g, $d, $a, $b, $c, words, 10, 21, 9);
                step!(g, $c, $d, $a, $b, words, 15, 22, 14);
                step!(g, $b, $c, $d, $a, words, 4, 23, 20);
                step!(g, $a, $b, $c, $d, words, 9, 24, 5);
                step!(g, $d, $a, $b, $c, words, 14, 25, 9);
                step!(g, $c, $d, $a, $b, words, 3, 26, 14);
                step!(g, $b, $c, $d, $a, words, 8, 27, 20);
                step!(g, $a, $b, $c, $d, words, 13, 28, 5);
                step!(g, $d, $a, $b, $c, words, 2, 29, 9);
                step!(g, $c, $d, $a, $b, words, 7, 30, 14);
                step!(g, $b, $c, $d, $a, words, 12, 31, 20);
                step!(h, $a, $b, $c, $d, words, 5, 32, 4);
                step!(h, $d, $a, $b, $c, words, 8, 33, 11);
                step!(h, $c, $d, $a, $b, words, 11, 34, 16);
                step!(h, $b, $c, $d, $a, words, 14, 35, 23);
                step!(h, $a, $b, $c, $d, words, 1, 36, 4);
                step!(h, $d, $a, $b, $c, words, 4, 37, 11);
                step!(h, $c, $d, $a, $b, words, 7, 38, 16);
                step!(h, $b, $c, $d, $a, words, 10, 39, 23);
                step!(h, $a, $b, $c, $d, words, 13, 40, 4);
                step!(h, $d, $a, $b, $c, words, 0, 41, 11);
                step!(h, $c, $d, $a, $b, words, 3, 42, 16);
                step!(h, $b, $c, $d, $a, words, 6, 43, 23);
                step!(h, $a, $b, $c, $d, words, 9, 44, 4);
                step!(h, $d, $a, $b, $c, words, 12, 45, 11);
                step!(h, $c, $d, $a, $b, words, 15, 46, 16);
                step!(h, $b, $c, $d, $a, words, 2, 47, 23);
                step!(i, $a, $b, $c, $d, words, 0, 48, 6);
                step!(i, $d, $a, $b, $c, words, 7, 49, 10);
                step!(i, $c, $d, $a, $b, words, 14, 50, 15);
                step!(i, $b, $c, $d, $a, words, 5, 51, 21);
                step!(i, $a, $b, $c, $d, words, 12, 52, 6);
                step!(i, $d, $a, $b, $c, words, 3, 53, 10);
                step!(i, $c, $d, $a, $b, words, 10, 54, 15);
                step!(i, $b, $c, $d, $a, words, 1, 55, 21);
                step!(i, $a, $b, $c, $d, words, 8, 56, 6);
                step!(i, $d, $a, $b, $c, words, 15, 57, 10);
                step!(i, $c, $d, $a, $b, words, 6, 58, 15);
                step!(i, $b, $c, $d, $a, words, 13, 59, 21);
                step!(i, $a, $b, $c, $d, words, 4, 60, 6);
                step!(i, $d, $a, $b, $c, words, 11, 61, 10);
                step!(i, $c, $d, $a, $b, words, 2, 62, 15);
                step!(i, $b, $c, $d, $a, words, 9, 63, 21);
                $a = _mm512_add_epi32(initial[0], $a);
                $b = _mm512_add_epi32(initial[1], $b);
                $c = _mm512_add_epi32(initial[2], $c);
                $d = _mm512_add_epi32(initial[3], $d);
            }};
        }

        let len = inputs[0].len();
        let [mut a, mut b, mut c, mut d] = load_md5_state16!(streams, 0);
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words = load_transposed!(blocks);
            compress!(words, a, b, c, d);
        }

        store_md5_state16!(streams, 0, a, b, c, d);
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx512_dual_md5_state_kernel(
        avx512: Avx512,
        streams: &mut [Md5State],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(inputs.len(), 32);
        debug_assert_eq!(streams.len(), 32);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }

                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }

                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }

                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }

        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    // SAFETY: each entry is a full 64-byte MD5 block and the
                    // unaligned AVX-512 load reads exactly those 64 bytes.
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }

        macro_rules! mix {
            (f, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i, $x:expr, $y:expr, $z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }

        macro_rules! step2 {
            ($which:ident, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $a1:ident, $b1:ident, $c1:ident, $d1:ident, $words0:ident, $words1:ident, $word:expr, $round:expr, $shift:literal) => {{
                let mixed0 = mix!($which, $b0, $c0, $d0);
                let mixed1 = mix!($which, $b1, $c1, $d1);
                let mut t0 = _mm512_add_epi32($a0, mixed0);
                let mut t1 = _mm512_add_epi32($a1, mixed1);
                let key = _mm512_set1_epi32(K[$round] as i32);
                t0 = _mm512_add_epi32(t0, key);
                t1 = _mm512_add_epi32(t1, key);
                t0 = _mm512_add_epi32(t0, $words0[$word]);
                t1 = _mm512_add_epi32(t1, $words1[$word]);
                $a0 = _mm512_add_epi32($b0, _mm512_rol_epi32::<$shift>(t0));
                $a1 = _mm512_add_epi32($b1, _mm512_rol_epi32::<$shift>(t1));
            }};
        }

        macro_rules! compress2 {
            ($words0:expr, $a0:ident, $b0:ident, $c0:ident, $d0:ident, $words1:expr, $a1:ident, $b1:ident, $c1:ident, $d1:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 0, 0, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 1, 1, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 2, 2, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 3, 3, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 4, 4, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 5, 5, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 6, 6, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 7, 7, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 8, 8, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 9, 9, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 10, 10, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 11, 11, 22
                );
                step2!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 12, 12, 7
                );
                step2!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 13, 13, 12
                );
                step2!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 14, 14, 17
                );
                step2!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 15, 15, 22
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 1, 16, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 6, 17, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 11, 18, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 0, 19, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 5, 20, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 10, 21, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 15, 22, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 4, 23, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 9, 24, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 14, 25, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 3, 26, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 8, 27, 20
                );
                step2!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 13, 28, 5
                );
                step2!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 2, 29, 9
                );
                step2!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 7, 30, 14
                );
                step2!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 12, 31, 20
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 5, 32, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 8, 33, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 11, 34, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 14, 35, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 1, 36, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 4, 37, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 7, 38, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 10, 39, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 13, 40, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 0, 41, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 3, 42, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 6, 43, 23
                );
                step2!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 9, 44, 4
                );
                step2!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 12, 45, 11
                );
                step2!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 15, 46, 16
                );
                step2!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 2, 47, 23
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 0, 48, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 7, 49, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 14, 50, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 5, 51, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 12, 52, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 3, 53, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 10, 54, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 1, 55, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 8, 56, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 15, 57, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 6, 58, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 13, 59, 21
                );
                step2!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, words0, words1, 4, 60, 6
                );
                step2!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, words0, words1, 11, 61, 10
                );
                step2!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, words0, words1, 2, 62, 15
                );
                step2!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, words0, words1, 9, 63, 21
                );
                $a0 = _mm512_add_epi32(initial0[0], $a0);
                $b0 = _mm512_add_epi32(initial0[1], $b0);
                $c0 = _mm512_add_epi32(initial0[2], $c0);
                $d0 = _mm512_add_epi32(initial0[3], $d0);
                $a1 = _mm512_add_epi32(initial1[0], $a1);
                $b1 = _mm512_add_epi32(initial1[1], $b1);
                $c1 = _mm512_add_epi32(initial1[2], $c1);
                $d1 = _mm512_add_epi32(initial1[3], $d1);
            }};
        }

        let len = inputs[0].len();
        let [mut a0, mut b0, mut c0, mut d0] = load_md5_state16!(streams, 0);
        let [mut a1, mut b1, mut c1, mut d1] = load_md5_state16!(streams, 16);
        let full_blocks = len / 64;

        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            compress2!(words0, a0, b0, c0, d0, words1, a1, b1, c1, d1);
        }

        store_md5_state16!(streams, 0, a0, b0, c0, d0);
        store_md5_state16!(streams, 16, a1, b1, c1, d1);
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline]
    fn compress_equal_len_avx512_triple_md5_state_kernel(
        avx512: Avx512,
        streams: &mut [Md5State],
        inputs: &[&[u8]],
    ) {
        debug_assert_eq!(inputs.len(), 48);
        debug_assert_eq!(streams.len(), 48);
        debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
        let _ = avx512;

        macro_rules! transpose16 {
            ($rows:expr) => {{
                let rows = $rows;
                let pair_lo =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let pair_hi =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                let quad_lo =
                    _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
                let quad_hi =
                    _mm512_setr_epi32(8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
                let oct_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                let oct_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                let half_lo =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                let half_hi =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
                let mut s1 = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    let a = rows[pair * 2];
                    let b = rows[pair * 2 + 1];
                    s1[pair * 2] = _mm512_permutex2var_epi32(a, pair_lo, b);
                    s1[pair * 2 + 1] = _mm512_permutex2var_epi32(a, pair_hi, b);
                }
                let mut s2 = [_mm512_setzero_si512(); 16];
                for group in 0..4 {
                    let base = group * 4;
                    s2[base] = _mm512_permutex2var_epi32(s1[base], quad_lo, s1[base + 2]);
                    s2[base + 1] = _mm512_permutex2var_epi32(s1[base], quad_hi, s1[base + 2]);
                    s2[base + 2] = _mm512_permutex2var_epi32(s1[base + 1], quad_lo, s1[base + 3]);
                    s2[base + 3] = _mm512_permutex2var_epi32(s1[base + 1], quad_hi, s1[base + 3]);
                }
                let mut s3 = [_mm512_setzero_si512(); 16];
                for half in 0..2 {
                    let left = half * 8;
                    let right = left + 4;
                    for quarter in 0..4 {
                        s3[left + quarter * 2] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_lo,
                            s2[right + quarter],
                        );
                        s3[left + quarter * 2 + 1] = _mm512_permutex2var_epi32(
                            s2[left + quarter],
                            oct_hi,
                            s2[right + quarter],
                        );
                    }
                }
                let mut out = [_mm512_setzero_si512(); 16];
                for pair in 0..8 {
                    out[pair * 2] = _mm512_permutex2var_epi32(s3[pair], half_lo, s3[8 + pair]);
                    out[pair * 2 + 1] = _mm512_permutex2var_epi32(s3[pair], half_hi, s3[8 + pair]);
                }
                out
            }};
        }
        macro_rules! load_transposed {
            ($blocks:expr) => {{
                let blocks = $blocks;
                let mut rows = [_mm512_setzero_si512(); 16];
                for lane in 0..16 {
                    rows[lane] = unsafe { _mm512_loadu_si512(blocks[lane].as_ptr().cast()) };
                }
                transpose16!(rows)
            }};
        }
        macro_rules! mix {
            (f,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0xca>($x, $y, $z)
            };
            (g,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0xe4>($x, $y, $z)
            };
            (h,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0x96>($x, $y, $z)
            };
            (i,$x:expr,$y:expr,$z:expr) => {
                _mm512_ternarylogic_epi32::<0x39>($x, $y, $z)
            };
        }
        macro_rules! step3 {
            ($which:ident,$a0:ident,$b0:ident,$c0:ident,$d0:ident,$a1:ident,$b1:ident,$c1:ident,$d1:ident,$a2:ident,$b2:ident,$c2:ident,$d2:ident,$w0:ident,$w1:ident,$w2:ident,$word:expr,$round:expr,$shift:literal) => {{
                let key = _mm512_set1_epi32(K[$round] as i32);
                let mut t0 = _mm512_add_epi32($a0, mix!($which, $b0, $c0, $d0));
                let mut t1 = _mm512_add_epi32($a1, mix!($which, $b1, $c1, $d1));
                let mut t2 = _mm512_add_epi32($a2, mix!($which, $b2, $c2, $d2));
                t0 = _mm512_add_epi32(_mm512_add_epi32(t0, key), $w0[$word]);
                t1 = _mm512_add_epi32(_mm512_add_epi32(t1, key), $w1[$word]);
                t2 = _mm512_add_epi32(_mm512_add_epi32(t2, key), $w2[$word]);
                $a0 = _mm512_add_epi32($b0, _mm512_rol_epi32::<$shift>(t0));
                $a1 = _mm512_add_epi32($b1, _mm512_rol_epi32::<$shift>(t1));
                $a2 = _mm512_add_epi32($b2, _mm512_rol_epi32::<$shift>(t2));
            }};
        }
        macro_rules! compress3 {
            ($words0:expr,$a0:ident,$b0:ident,$c0:ident,$d0:ident,$words1:expr,$a1:ident,$b1:ident,$c1:ident,$d1:ident,$words2:expr,$a2:ident,$b2:ident,$c2:ident,$d2:ident) => {{
                let words0 = $words0;
                let words1 = $words1;
                let words2 = $words2;
                let initial0 = [$a0, $b0, $c0, $d0];
                let initial1 = [$a1, $b1, $c1, $d1];
                let initial2 = [$a2, $b2, $c2, $d2];
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 0, 0, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 1, 1, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 2, 2, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 3, 3, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 4, 4, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 5, 5, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 6, 6, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 7, 7, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 8, 8, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 9, 9, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 10, 10, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 11, 11, 22
                );
                step3!(
                    f, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 12, 12, 7
                );
                step3!(
                    f, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 13, 13, 12
                );
                step3!(
                    f, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 14, 14, 17
                );
                step3!(
                    f, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 15, 15, 22
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 1, 16, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 6, 17, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 11, 18, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 0, 19, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 5, 20, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 10, 21, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 15, 22, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 4, 23, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 9, 24, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 14, 25, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 3, 26, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 8, 27, 20
                );
                step3!(
                    g, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 13, 28, 5
                );
                step3!(
                    g, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 2, 29, 9
                );
                step3!(
                    g, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 7, 30, 14
                );
                step3!(
                    g, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 12, 31, 20
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 5, 32, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 8, 33, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 11, 34, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 14, 35, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 1, 36, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 4, 37, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 7, 38, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 10, 39, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 13, 40, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 0, 41, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 3, 42, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 6, 43, 23
                );
                step3!(
                    h, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 9, 44, 4
                );
                step3!(
                    h, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 12, 45, 11
                );
                step3!(
                    h, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 15, 46, 16
                );
                step3!(
                    h, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 2, 47, 23
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 0, 48, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 7, 49, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 14, 50, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 5, 51, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 12, 52, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 3, 53, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 10, 54, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 1, 55, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 8, 56, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 15, 57, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 6, 58, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 13, 59, 21
                );
                step3!(
                    i, $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, words0, words1,
                    words2, 4, 60, 6
                );
                step3!(
                    i, $d0, $a0, $b0, $c0, $d1, $a1, $b1, $c1, $d2, $a2, $b2, $c2, words0, words1,
                    words2, 11, 61, 10
                );
                step3!(
                    i, $c0, $d0, $a0, $b0, $c1, $d1, $a1, $b1, $c2, $d2, $a2, $b2, words0, words1,
                    words2, 2, 62, 15
                );
                step3!(
                    i, $b0, $c0, $d0, $a0, $b1, $c1, $d1, $a1, $b2, $c2, $d2, $a2, words0, words1,
                    words2, 9, 63, 21
                );
                $a0 = _mm512_add_epi32(initial0[0], $a0);
                $b0 = _mm512_add_epi32(initial0[1], $b0);
                $c0 = _mm512_add_epi32(initial0[2], $c0);
                $d0 = _mm512_add_epi32(initial0[3], $d0);
                $a1 = _mm512_add_epi32(initial1[0], $a1);
                $b1 = _mm512_add_epi32(initial1[1], $b1);
                $c1 = _mm512_add_epi32(initial1[2], $c1);
                $d1 = _mm512_add_epi32(initial1[3], $d1);
                $a2 = _mm512_add_epi32(initial2[0], $a2);
                $b2 = _mm512_add_epi32(initial2[1], $b2);
                $c2 = _mm512_add_epi32(initial2[2], $c2);
                $d2 = _mm512_add_epi32(initial2[3], $d2);
            }};
        }
        let len = inputs[0].len();
        let [mut a0, mut b0, mut c0, mut d0] = load_md5_state16!(streams, 0);
        let [mut a1, mut b1, mut c1, mut d1] = load_md5_state16!(streams, 16);
        let [mut a2, mut b2, mut c2, mut d2] = load_md5_state16!(streams, 32);
        let full_blocks = len / 64;
        for block_index in 0..full_blocks {
            let offset = block_index * 64;
            let blocks0: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks1: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 16][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let blocks2: [&[u8; 64]; 16] = core::array::from_fn(|lane| {
                inputs[lane + 32][offset..offset + 64]
                    .try_into()
                    .expect("full MD5 block")
            });
            let words0 = load_transposed!(blocks0);
            let words1 = load_transposed!(blocks1);
            let words2 = load_transposed!(blocks2);
            compress3!(
                words0, a0, b0, c0, d0, words1, a1, b1, c1, d1, words2, a2, b2, c2, d2
            );
        }
        store_md5_state16!(streams, 0, a0, b0, c0, d0);
        store_md5_state16!(streams, 16, a1, b1, c1, d1);
        store_md5_state16!(streams, 32, a2, b2, c2, d2);
    }
);

/// Compress block-aligned lockstep fragments directly from `Md5State`.
///
/// This avoids copying chaining states through a temporary contiguous array
/// on the incremental equal-chunk hot path.
#[inline]
pub(crate) fn compress_md5_states_blocks_validated_with_level(
    level: Level,
    streams: &mut [Md5State],
    inputs: &[&[u8]],
) {
    debug_assert_eq!(streams.len(), inputs.len());
    debug_assert!(!inputs.is_empty());
    debug_assert!(inputs[0].len().is_multiple_of(64));
    debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if streams.len() == 48
            && let Some(avx512) = level.as_avx512()
        {
            compress_equal_len_avx512_triple_md5_state_kernel(avx512, streams, inputs);
            return;
        }
        if streams.len() == 32
            && let Some(avx512) = level.as_avx512()
        {
            compress_equal_len_avx512_dual_md5_state_kernel(avx512, streams, inputs);
            return;
        }
        if streams.len() == 24
            && let Some(avx2) = level.as_avx2()
        {
            compress_equal_len_avx2_triple_md5_state_kernel(avx2, streams, inputs);
            return;
        }
        if streams.len() == 16 {
            if let Some(avx512) = level.as_avx512() {
                compress_equal_len_avx512_md5_state_kernel(avx512, streams, inputs);
                return;
            }
            if let Some(avx2) = level.as_avx2() {
                compress_equal_len_avx2_dual_md5_state_kernel(avx2, streams, inputs);
                return;
            }
        }
        if streams.len() == 8
            && let Some(avx2) = level.as_avx2()
        {
            compress_equal_len_avx2_md5_state_kernel(avx2, streams, inputs);
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if streams.len() <= 4
        && let Some(sse2) = level.as_sse2()
    {
        compress_md5_states_inner(sse2, streams, inputs);
        return;
    }

    // Portable and under-filled groups keep the compact temporary-state path.
    let mut states = [[0u32; 4]; 48];
    for (lane, stream) in streams.iter().enumerate() {
        states[lane] = stream.state;
    }
    compress_many_blocks_with_level(level, &mut states[..streams.len()], inputs);
    for (lane, stream) in streams.iter_mut().enumerate() {
        stream.state = states[lane];
    }
}

/// Compress equal-length, block-aligned message fragments starting from
/// caller-supplied MD5 states. This is the primitive used by incremental
/// multi-stream hashing; unlike the one-shot kernels it performs no padding.
pub(crate) fn compress_many_blocks_with_level(
    level: Level,
    states: &mut [[u32; 4]],
    inputs: &[&[u8]],
) {
    assert_eq!(states.len(), inputs.len(), "state/input length mismatch");
    if inputs.is_empty() {
        return;
    }
    let len = inputs[0].len();
    assert!(
        len.is_multiple_of(64),
        "incremental SIMD input must be block aligned"
    );
    assert!(
        inputs.iter().all(|input| input.len() == len),
        "incremental SIMD inputs must have equal lengths"
    );

    // A stateful update can temporarily have fewer active streams than the
    // engine's widest level. Select the narrowest native x86 vector width
    // that can hold the compacted batch instead of doing 4 streams in 16
    // AVX-512 lanes, for example.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if states.len() == 48
            && let Some(avx512) = level.as_avx512()
        {
            compress_equal_len_avx512_triple_state_kernel(avx512, states, inputs);
            return;
        }
        if states.len() == 32
            && let Some(avx512) = level.as_avx512()
        {
            compress_equal_len_avx512_dual_state_kernel(avx512, states, inputs);
            return;
        }
        if states.len() == 24
            && let Some(avx2) = level.as_avx2()
        {
            compress_equal_len_avx2_triple_state_kernel(avx2, states, inputs);
            return;
        }
        if states.len() == 16 {
            if let Some(avx512) = level.as_avx512() {
                compress_equal_len_avx512_state_kernel(avx512, states, inputs);
                return;
            }
            if let Some(avx2) = level.as_avx2() {
                compress_equal_len_avx2_dual_state_kernel(avx2, states, inputs);
                return;
            }
        }
        if states.len() <= 4
            && let Some(sse2) = level.as_sse2()
        {
            compress_many_blocks_inner(sse2, states, inputs);
            return;
        }
        if states.len() == 8
            && let Some(avx2) = level.as_avx2()
        {
            compress_equal_len_avx2_state_kernel(avx2, states, inputs);
            return;
        }
        if states.len() <= 8
            && let Some(avx2) = level.as_avx2()
        {
            compress_many_blocks_inner(avx2, states, inputs);
            return;
        }
    }

    dispatch!(level, simd => compress_many_blocks_inner(simd, states, inputs));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn compress_md5_states_inner<SIMD: Simd>(simd: SIMD, streams: &mut [Md5State], inputs: &[&[u8]]) {
    debug_assert_eq!(streams.len(), inputs.len());
    debug_assert!(streams.len() <= SIMD::u32s::N);
    debug_assert!(
        inputs
            .first()
            .is_none_or(|first| inputs.iter().all(|input| input.len() == first.len()))
    );
    debug_assert!(
        inputs
            .first()
            .is_none_or(|input| input.len().is_multiple_of(64))
    );

    let active = streams.len();
    let mut vector_state: [SIMD::u32s; 4] = core::array::from_fn(|word| {
        SIMD::u32s::from_fn(simd, |lane| {
            if lane < active {
                streams[lane].state[word]
            } else {
                STATE_INIT[word]
            }
        })
    });

    let block_count = inputs[0].len() / 64;
    for block_index in 0..block_count {
        let words = full_block_words(simd, inputs, active, block_index);
        compress_words::<SIMD>(&mut vector_state, &words);
    }

    for lane in 0..active {
        for (word, vector) in vector_state.iter().enumerate() {
            streams[lane].state[word] = vector[lane];
        }
    }
}

#[inline(always)]
fn padded_blocks_for_len(len: usize) -> usize {
    let full_blocks = len / 64;
    let tail = len & 63;
    full_blocks + if tail <= 55 { 1 } else { 2 }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn build_padded_block(input: &[u8], padded_blocks: usize, block_index: usize) -> [u8; 64] {
    let mut block = [0u8; 64];
    let base = block_index * 64;
    if base < input.len() {
        let count = core::cmp::min(64, input.len() - base);
        block[..count].copy_from_slice(&input[base..base + count]);
    }
    if input.len() >= base && input.len() < base + 64 {
        block[input.len() - base] = 0x80;
    }
    if block_index + 1 == padded_blocks {
        block[56..64].copy_from_slice(&(input.len() as u64).wrapping_mul(8).to_le_bytes());
    }
    block
}

#[inline(always)]
fn padded_byte(input: &[u8], padded_blocks: usize, absolute: usize) -> u8 {
    if absolute < input.len() {
        return input[absolute];
    }
    if absolute == input.len() {
        return 0x80;
    }

    let total_len = padded_blocks * 64;
    if absolute >= total_len - 8 {
        let length_byte = absolute - (total_len - 8);
        return ((input.len() as u64).wrapping_mul(8).to_le_bytes())[length_byte];
    }
    0
}

#[inline(always)]
fn padded_block_words<SIMD: Simd>(
    simd: SIMD,
    inputs: &[&[u8]],
    active: usize,
    padded_blocks: usize,
    block_index: usize,
) -> [SIMD::u32s; 16] {
    core::array::from_fn(|word_index| {
        SIMD::u32s::from_fn(simd, |lane| {
            if lane >= active {
                return 0;
            }
            let base = block_index * 64 + word_index * 4;
            let bytes = [
                padded_byte(inputs[lane], padded_blocks, base),
                padded_byte(inputs[lane], padded_blocks, base + 1),
                padded_byte(inputs[lane], padded_blocks, base + 2),
                padded_byte(inputs[lane], padded_blocks, base + 3),
            ];
            u32::from_le_bytes(bytes)
        })
    })
}

#[inline(always)]
fn hash_equal_len_chunk<SIMD: Simd>(simd: SIMD, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!(!inputs.is_empty());
    debug_assert_eq!(inputs.len(), outputs.len());
    debug_assert!(inputs.len() <= SIMD::u32s::N);
    debug_assert!(SIMD::u32s::N <= MAX_LANES);

    let active = inputs.len();
    let len = inputs[0].len();
    debug_assert!(inputs.iter().all(|input| input.len() == len));

    let mut state = STATE_INIT.map(|word| SIMD::u32s::splat(simd, word));
    let full_blocks = len / 64;

    for block_index in 0..full_blocks {
        let words = full_block_words(simd, inputs, active, block_index);
        compress_words::<SIMD>(&mut state, &words);
    }

    let padded_blocks = padded_blocks_for_len(len);
    for block_index in full_blocks..padded_blocks {
        let words = padded_block_words(simd, inputs, active, padded_blocks, block_index);
        compress_words::<SIMD>(&mut state, &words);
    }

    for lane in 0..active {
        for (word_index, state_word) in state.iter().enumerate() {
            let bytes = state_word[lane].to_le_bytes();
            outputs[lane][word_index * 4..word_index * 4 + 4].copy_from_slice(&bytes);
        }
    }
}

#[inline(always)]
fn mixed_block_words<SIMD: Simd>(
    simd: SIMD,
    inputs: &[&[u8]],
    active: usize,
    padded_blocks: &[usize; MAX_LANES],
    block_index: usize,
) -> [SIMD::u32s; 16] {
    core::array::from_fn(|word_index| {
        SIMD::u32s::from_fn(simd, |lane| {
            if lane >= active || block_index >= padded_blocks[lane] {
                return 0;
            }
            let input = inputs[lane];
            let base = block_index * 64 + word_index * 4;
            if block_index < input.len() / 64 {
                return u32::from_le_bytes(
                    input[base..base + 4].try_into().expect("four-byte word"),
                );
            }
            u32::from_le_bytes([
                padded_byte(input, padded_blocks[lane], base),
                padded_byte(input, padded_blocks[lane], base + 1),
                padded_byte(input, padded_blocks[lane], base + 2),
                padded_byte(input, padded_blocks[lane], base + 3),
            ])
        })
    })
}

#[inline(always)]
fn hash_mixed_len_chunk<SIMD: Simd>(simd: SIMD, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!(!inputs.is_empty());
    debug_assert_eq!(inputs.len(), outputs.len());
    debug_assert!(inputs.len() <= SIMD::u32s::N);

    let active = inputs.len();
    let mut block_counts = [0usize; MAX_LANES];
    let mut max_blocks = 0;
    for lane in 0..active {
        let blocks = padded_blocks_for_len(inputs[lane].len());
        block_counts[lane] = blocks;
        max_blocks = core::cmp::max(max_blocks, blocks);
    }

    let mut state = STATE_INIT.map(|word| SIMD::u32s::splat(simd, word));
    for block_index in 0..max_blocks {
        let words = mixed_block_words(simd, inputs, active, &block_counts, block_index);
        compress_words::<SIMD>(&mut state, &words);

        // A lane's state is final immediately after its own last padded block.
        // Later SIMD rounds may overwrite that lane, so materialize the digest now.
        for lane in 0..active {
            if block_counts[lane] == block_index + 1 {
                for (word_index, state_word) in state.iter().enumerate() {
                    let bytes = state_word[lane].to_le_bytes();
                    outputs[lane][word_index * 4..word_index * 4 + 4].copy_from_slice(&bytes);
                }
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_equal_len_avx2_padded(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((2..=8).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());
    debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

    let active = inputs.len();
    let padded_inputs: [&[u8]; 8] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 8];
    hash_equal_len_avx2_kernel(avx2, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_equal_len_avx2_dual_padded(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((9..=16).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());
    debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

    let active = inputs.len();
    let padded_inputs: [&[u8]; 16] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 16];
    hash_equal_len_avx2_dual_kernel(avx2, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_equal_len_avx2_triple_padded(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((17..=24).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());
    debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

    let active = inputs.len();
    let padded_inputs: [&[u8]; 24] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 24];
    hash_equal_len_avx2_triple_kernel(avx2, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_mixed_len_avx2_dual_padded(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((9..=16).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());

    let active = inputs.len();
    let padded_inputs: [&[u8]; 16] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 16];
    hash_mixed_len_avx2_dual_kernel(avx2, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_mixed_len_avx2_triple_padded(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((17..=24).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());

    let active = inputs.len();
    let padded_inputs: [&[u8]; 24] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 24];
    hash_mixed_len_avx2_triple_kernel(avx2, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_mixed_len_avx2_padded(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((3..=8).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());
    let active = inputs.len();
    let padded_inputs: [&[u8]; 8] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 8];
    hash_mixed_len_avx2_kernel(avx2, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_mixed_len_avx512_padded(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((2..=16).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());
    let active = inputs.len();
    let padded_inputs: [&[u8]; 16] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 16];
    hash_mixed_len_avx512_kernel(avx512, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_equal_len_avx512_dual_padded(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((17..=32).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());
    debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

    let active = inputs.len();
    let padded_inputs: [&[u8]; 32] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 32];
    hash_equal_len_avx512_dual_kernel(avx512, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_equal_len_avx512_triple_padded(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((33..=48).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());
    debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

    let active = inputs.len();
    let padded_inputs: [&[u8]; 48] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 48];
    hash_equal_len_avx512_triple_kernel(avx512, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_mixed_len_avx512_dual_padded(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((17..=32).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());

    let active = inputs.len();
    let padded_inputs: [&[u8]; 32] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 32];
    hash_mixed_len_avx512_dual_kernel(avx512, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_mixed_len_avx512_triple_padded(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((33..=48).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());

    let active = inputs.len();
    let padded_inputs: [&[u8]; 48] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 48];
    hash_mixed_len_avx512_triple_kernel(avx512, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_equal_len_avx512_padded(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((2..=16).contains(&inputs.len()));
    debug_assert_eq!(inputs.len(), outputs.len());
    debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

    let active = inputs.len();
    let padded_inputs: [&[u8]; 16] =
        core::array::from_fn(|lane| inputs[core::cmp::min(lane, active - 1)]);
    let mut padded_outputs = [[0u8; 16]; 16];
    hash_equal_len_avx512_kernel(avx512, &padded_inputs, &mut padded_outputs);
    outputs.copy_from_slice(&padded_outputs[..active]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn skew_partition<const N: usize, const HALF: usize>(inputs: &[&[u8]]) -> Option<[usize; N]> {
    debug_assert_eq!(inputs.len(), N);
    debug_assert_eq!(N, HALF * 2);

    let mut max_blocks = 0usize;
    let mut block_counts = [0usize; N];
    for lane in 0..N {
        let blocks = padded_blocks_for_len(inputs[lane].len());
        block_counts[lane] = blocks;
        max_blocks = core::cmp::max(max_blocks, blocks);
    }

    let mut short_count = 0usize;
    for &blocks in &block_counts {
        short_count += usize::from(blocks.saturating_mul(2) <= max_blocks);
    }
    if short_count < HALF {
        return None;
    }

    let mut selected = [false; N];
    let mut order = [0usize; N];
    let mut out = 0usize;
    for lane in 0..N {
        if block_counts[lane].saturating_mul(2) <= max_blocks && out < HALF {
            order[out] = lane;
            selected[lane] = true;
            out += 1;
        }
    }
    debug_assert_eq!(out, HALF);
    for (lane, &is_selected) in selected.iter().enumerate() {
        if !is_selected {
            order[out] = lane;
            out += 1;
        }
    }
    debug_assert_eq!(out, N);
    Some(order)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn skew_partition_dynamic<const MAX: usize>(inputs: &[&[u8]]) -> Option<([usize; MAX], usize)> {
    debug_assert!(!inputs.is_empty());
    debug_assert!(inputs.len() <= MAX);

    let mut max_blocks = 0usize;
    let mut block_counts = [0usize; MAX];
    for (lane, input) in inputs.iter().enumerate() {
        let blocks = padded_blocks_for_len(input.len());
        block_counts[lane] = blocks;
        max_blocks = core::cmp::max(max_blocks, blocks);
    }

    let mut short_count = 0usize;
    for &blocks in &block_counts[..inputs.len()] {
        short_count += usize::from(blocks.saturating_mul(2) <= max_blocks);
    }
    if short_count == 0 || short_count == inputs.len() {
        return None;
    }

    let mut order = [0usize; MAX];
    let mut short_out = 0usize;
    let mut long_out = short_count;
    for (lane, &blocks) in block_counts[..inputs.len()].iter().enumerate() {
        if blocks.saturating_mul(2) <= max_blocks {
            order[short_out] = lane;
            short_out += 1;
        } else {
            order[long_out] = lane;
            long_out += 1;
        }
    }
    debug_assert_eq!(short_out, short_count);
    debug_assert_eq!(long_out, inputs.len());
    Some((order, short_count))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hash_skewed_partial_avx2_with_limit<const MAX: usize>(
    avx2: Avx2,
    inputs: &[&[u8]],
    outputs: &mut [[u8; 16]],
    max_long_count: usize,
) -> bool {
    debug_assert!(!inputs.is_empty());
    debug_assert!(inputs.len() <= MAX);
    debug_assert_eq!(inputs.len(), outputs.len());

    let Some((order, split)) = skew_partition_dynamic::<MAX>(inputs) else {
        return false;
    };
    let active = inputs.len();
    if active - split > max_long_count {
        return false;
    }

    let reordered_inputs: [&[u8]; MAX] = core::array::from_fn(|lane| {
        let source = if lane < active {
            order[lane]
        } else {
            order[active - 1]
        };
        inputs[source]
    });
    let mut reordered_outputs = [[0u8; 16]; MAX];
    hash_many_avx2(
        avx2,
        &reordered_inputs[..split],
        &mut reordered_outputs[..split],
    );
    hash_many_avx2(
        avx2,
        &reordered_inputs[split..active],
        &mut reordered_outputs[split..active],
    );
    for lane in 0..active {
        outputs[order[lane]] = reordered_outputs[lane];
    }
    true
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hash_skewed_partial_avx2(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) -> bool {
    hash_skewed_partial_avx2_with_limit::<32>(avx2, inputs, outputs, usize::MAX)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hash_skewed_small_avx2(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) -> bool {
    // For 4-8 lane tails, splitting only helps when the long partition
    // collapses to scalar/dual-scalar work. Leaving three or more long lanes
    // on a recursive SIMD call adds partition overhead without removing the
    // sparse-lane bottleneck.
    hash_skewed_partial_avx2_with_limit::<8>(avx2, inputs, outputs, 2)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hash_skewed_partial_avx512(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) -> bool {
    debug_assert!(!inputs.is_empty());
    debug_assert!(inputs.len() <= 64);
    debug_assert_eq!(inputs.len(), outputs.len());

    let Some((order, split)) = skew_partition_dynamic::<64>(inputs) else {
        return false;
    };
    let active = inputs.len();
    let reordered_inputs: [&[u8]; 64] = core::array::from_fn(|lane| {
        let source = if lane < active {
            order[lane]
        } else {
            order[active - 1]
        };
        inputs[source]
    });
    let mut reordered_outputs = [[0u8; 16]; 64];
    hash_many_avx512(
        avx512,
        &reordered_inputs[..split],
        &mut reordered_outputs[..split],
    );
    hash_many_avx512(
        avx512,
        &reordered_inputs[split..active],
        &mut reordered_outputs[split..active],
    );
    for lane in 0..active {
        outputs[order[lane]] = reordered_outputs[lane];
    }
    true
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hash_partitioned_split_avx2(
    avx2: Avx2,
    inputs: &[&[u8]],
    outputs: &mut [[u8; 16]],
    order: [usize; 16],
) {
    debug_assert_eq!(inputs.len(), 16);
    debug_assert_eq!(outputs.len(), 16);

    if order.iter().enumerate().all(|(lane, &index)| lane == index) {
        hash_many_avx2(avx2, &inputs[..8], &mut outputs[..8]);
        hash_many_avx2(avx2, &inputs[8..], &mut outputs[8..]);
        return;
    }

    let reordered_inputs: [&[u8]; 16] = core::array::from_fn(|lane| inputs[order[lane]]);
    let mut reordered_outputs = [[0u8; 16]; 16];
    hash_many_avx2(avx2, &reordered_inputs[..8], &mut reordered_outputs[..8]);
    hash_many_avx2(avx2, &reordered_inputs[8..], &mut reordered_outputs[8..]);
    for lane in 0..16 {
        outputs[order[lane]] = reordered_outputs[lane];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hash_partitioned_split_avx512(
    avx512: Avx512,
    inputs: &[&[u8]],
    outputs: &mut [[u8; 16]],
    order: [usize; 32],
) {
    debug_assert_eq!(inputs.len(), 32);
    debug_assert_eq!(outputs.len(), 32);

    if order.iter().enumerate().all(|(lane, &index)| lane == index) {
        hash_many_avx512(avx512, &inputs[..16], &mut outputs[..16]);
        hash_many_avx512(avx512, &inputs[16..], &mut outputs[16..]);
        return;
    }

    let reordered_inputs: [&[u8]; 32] = core::array::from_fn(|lane| inputs[order[lane]]);
    let mut reordered_outputs = [[0u8; 16]; 32];
    hash_many_avx512(
        avx512,
        &reordered_inputs[..16],
        &mut reordered_outputs[..16],
    );
    hash_many_avx512(
        avx512,
        &reordered_inputs[16..],
        &mut reordered_outputs[16..],
    );
    for lane in 0..32 {
        outputs[order[lane]] = reordered_outputs[lane];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_many_avx2(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    let mut start = 0;
    while start < inputs.len() {
        let remaining = inputs.len() - start;

        // Under-filled dual/triple kernels are often faster than hashing one
        // full native group and then paying for a small scalar/SIMD tail. The
        // unused lanes duplicate the last real input and their digests are
        // discarded.
        if (9..=15).contains(&remaining) || (17..=23).contains(&remaining) {
            let input_chunk = &inputs[start..];
            let output_chunk = &mut outputs[start..];
            let same_len = input_chunk
                .first()
                .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));
            if !same_len && hash_skewed_partial_avx2(avx2, input_chunk, output_chunk) {
                break;
            }
            if remaining <= 15 {
                if same_len {
                    hash_equal_len_avx2_dual_padded(avx2, input_chunk, output_chunk);
                } else {
                    hash_mixed_len_avx2_dual_padded(avx2, input_chunk, output_chunk);
                }
            } else if same_len {
                hash_equal_len_avx2_triple_padded(avx2, input_chunk, output_chunk);
            } else {
                hash_mixed_len_avx2_triple_padded(avx2, input_chunk, output_chunk);
            }
            break;
        }

        if (26..=31).contains(&remaining) {
            let input_chunk = &inputs[start..];
            let same_len = input_chunk
                .first()
                .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));
            if !same_len && hash_skewed_partial_avx2(avx2, input_chunk, &mut outputs[start..]) {
                break;
            }
            if same_len {
                hash_equal_len_avx2_dual_kernel(
                    avx2,
                    &input_chunk[..16],
                    &mut outputs[start..start + 16],
                );
                hash_equal_len_avx2_dual_padded(
                    avx2,
                    &input_chunk[16..],
                    &mut outputs[start + 16..],
                );
            } else {
                hash_mixed_len_avx2_dual_kernel(
                    avx2,
                    &input_chunk[..16],
                    &mut outputs[start..start + 16],
                );
                hash_mixed_len_avx2_dual_padded(
                    avx2,
                    &input_chunk[16..],
                    &mut outputs[start + 16..],
                );
            }
            break;
        }

        let complete_avx2_groups = remaining / 8;

        if remaining >= 24 && complete_avx2_groups != 4 {
            let input_chunk = &inputs[start..start + 24];
            let same_len = input_chunk
                .first()
                .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));
            if same_len {
                hash_equal_len_avx2_triple_kernel(
                    avx2,
                    input_chunk,
                    &mut outputs[start..start + 24],
                );
                start += 24;
                continue;
            }
            let min_blocks = input_chunk
                .iter()
                .map(|input| padded_blocks_for_len(input.len()))
                .min()
                .unwrap_or(0);
            let max_blocks = input_chunk
                .iter()
                .map(|input| padded_blocks_for_len(input.len()))
                .max()
                .unwrap_or(0);
            if input_chunk.iter().all(|input| input.len() >= 64)
                && min_blocks.saturating_mul(2) >= max_blocks
            {
                hash_mixed_len_avx2_triple_kernel(
                    avx2,
                    input_chunk,
                    &mut outputs[start..start + 24],
                );
                start += 24;
                continue;
            }
        }

        if inputs.len() - start >= 16 {
            let input_chunk = &inputs[start..start + 16];
            let same_len = input_chunk
                .first()
                .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));
            if same_len {
                hash_equal_len_avx2_dual_kernel(avx2, input_chunk, &mut outputs[start..start + 16]);
                start += 16;
                continue;
            }

            if let Some(order) = skew_partition::<16, 8>(input_chunk) {
                hash_partitioned_split_avx2(
                    avx2,
                    input_chunk,
                    &mut outputs[start..start + 16],
                    order,
                );
                start += 16;
                continue;
            }
            if input_chunk.iter().all(|input| input.len() >= 64) {
                hash_mixed_len_avx2_dual_kernel(avx2, input_chunk, &mut outputs[start..start + 16]);
                start += 16;
                continue;
            }
        }

        let end = core::cmp::min(start + 8, inputs.len());
        let input_chunk = &inputs[start..end];
        let output_chunk = &mut outputs[start..end];

        if input_chunk.len() == 1 {
            output_chunk[0] = scalar::hash(input_chunk[0]);
            start = end;
            continue;
        }

        #[cfg(target_arch = "x86_64")]
        if hash_three_dual_scalar(input_chunk, output_chunk) {
            start = end;
            continue;
        }

        let same_len = input_chunk
            .first()
            .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));

        if input_chunk.len() >= 4
            && !same_len
            && hash_skewed_small_avx2(avx2, input_chunk, output_chunk)
        {
            start = end;
            continue;
        }

        let max_padded_blocks = input_chunk
            .iter()
            .map(|input| padded_blocks_for_len(input.len()))
            .max()
            .unwrap_or(0);
        let prefer_scalar = input_chunk.len() == 2 && max_padded_blocks == 1;

        #[cfg(target_arch = "x86_64")]
        if prefer_dual_scalar_pair(input_chunk) {
            // SAFETY: prefer_dual_scalar_pair checked CPUID BMI1 immediately above.
            let digests = unsafe {
                crate::scalar_x86_64_dual::hash_pair_bmi1([input_chunk[0], input_chunk[1]])
            };
            output_chunk.copy_from_slice(&digests);
            start = end;
            continue;
        }

        if prefer_scalar {
            for (input, output) in input_chunk.iter().zip(output_chunk) {
                *output = scalar::hash(input);
            }
        } else if same_len {
            if input_chunk.len() == 8 {
                hash_equal_len_avx2_kernel(avx2, input_chunk, output_chunk);
            } else {
                hash_equal_len_avx2_padded(avx2, input_chunk, output_chunk);
            }
        } else if input_chunk.len() < 3 {
            for (input, output) in input_chunk.iter().zip(output_chunk) {
                *output = scalar::hash(input);
            }
        } else if input_chunk.len() == 8 {
            hash_mixed_len_avx2_kernel(avx2, input_chunk, output_chunk);
        } else {
            hash_mixed_len_avx2_padded(avx2, input_chunk, output_chunk);
        }
        start = end;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn hash_many_avx512(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    let avx2 = avx512
        .level()
        .as_avx2()
        .expect("AVX-512 feature level includes AVX2");

    let mut start = 0;
    while start < inputs.len() {
        let remaining = inputs.len() - start;

        // As with AVX2, an under-filled interleaved kernel is faster than a
        // full native batch followed by a small tail. Duplicate the last real
        // input into inactive lanes and discard those outputs.
        if (17..=31).contains(&remaining) || (33..=47).contains(&remaining) {
            let input_chunk = &inputs[start..];
            let output_chunk = &mut outputs[start..];
            let same_len = input_chunk
                .first()
                .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));
            if !same_len && hash_skewed_partial_avx512(avx512, input_chunk, output_chunk) {
                break;
            }
            if remaining <= 31 {
                if same_len {
                    hash_equal_len_avx512_dual_padded(avx512, input_chunk, output_chunk);
                } else {
                    hash_mixed_len_avx512_dual_padded(avx512, input_chunk, output_chunk);
                }
            } else if same_len {
                hash_equal_len_avx512_triple_padded(avx512, input_chunk, output_chunk);
            } else {
                hash_mixed_len_avx512_triple_padded(avx512, input_chunk, output_chunk);
            }
            break;
        }

        if (50..=63).contains(&remaining) {
            let input_chunk = &inputs[start..];
            let same_len = input_chunk
                .first()
                .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));
            if !same_len && hash_skewed_partial_avx512(avx512, input_chunk, &mut outputs[start..]) {
                break;
            }
            if same_len {
                hash_equal_len_avx512_dual_kernel(
                    avx512,
                    &input_chunk[..32],
                    &mut outputs[start..start + 32],
                );
                hash_equal_len_avx512_dual_padded(
                    avx512,
                    &input_chunk[32..],
                    &mut outputs[start + 32..],
                );
                break;
            }
            // With a mixed batch, triple-48 plus a 2-message scalar tail is a
            // pronounced cliff, while 9-15 remaining lanes switch back to a
            // large AVX2 tail kernel. On the measured x86 targets it is faster
            // to stay in two AVX-512 dual kernels for those tail shapes.
            if remaining == 50 || remaining >= 57 {
                hash_mixed_len_avx512_dual_kernel(
                    avx512,
                    &input_chunk[..32],
                    &mut outputs[start..start + 32],
                );
                hash_mixed_len_avx512_dual_padded(
                    avx512,
                    &input_chunk[32..],
                    &mut outputs[start + 32..],
                );
                break;
            }
        }

        let complete_avx512_groups = remaining / 16;
        if remaining >= 48 && complete_avx512_groups % 2 == 1 {
            let input_chunk = &inputs[start..start + 48];
            let same_len = input_chunk
                .first()
                .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));
            if same_len {
                hash_equal_len_avx512_triple_kernel(
                    avx512,
                    input_chunk,
                    &mut outputs[start..start + 48],
                );
                start += 48;
                continue;
            }
            let min_blocks = input_chunk
                .iter()
                .map(|input| padded_blocks_for_len(input.len()))
                .min()
                .unwrap_or(0);
            let max_blocks = input_chunk
                .iter()
                .map(|input| padded_blocks_for_len(input.len()))
                .max()
                .unwrap_or(0);
            if input_chunk.iter().all(|input| input.len() >= 64)
                && min_blocks.saturating_mul(4) >= max_blocks.saturating_mul(3)
            {
                hash_mixed_len_avx512_triple_kernel(
                    avx512,
                    input_chunk,
                    &mut outputs[start..start + 48],
                );
                start += 48;
                continue;
            }
        }

        if inputs.len() - start >= 32 {
            let input_chunk = &inputs[start..start + 32];
            let same_len = input_chunk
                .first()
                .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));
            if same_len {
                hash_equal_len_avx512_dual_kernel(
                    avx512,
                    input_chunk,
                    &mut outputs[start..start + 32],
                );
                start += 32;
                continue;
            }

            if let Some(order) = skew_partition::<32, 16>(input_chunk) {
                hash_partitioned_split_avx512(
                    avx512,
                    input_chunk,
                    &mut outputs[start..start + 32],
                    order,
                );
                start += 32;
                continue;
            }
            if input_chunk.iter().all(|input| input.len() >= 64) {
                hash_mixed_len_avx512_dual_kernel(
                    avx512,
                    input_chunk,
                    &mut outputs[start..start + 32],
                );
                start += 32;
                continue;
            }
        }

        let end = core::cmp::min(start + 16, inputs.len());
        let input_chunk = &inputs[start..end];
        let output_chunk = &mut outputs[start..end];

        if input_chunk.len() == 1 {
            output_chunk[0] = scalar::hash(input_chunk[0]);
            start = end;
            continue;
        }

        #[cfg(target_arch = "x86_64")]
        if hash_three_dual_scalar(input_chunk, output_chunk) {
            start = end;
            continue;
        }

        let same_len = input_chunk
            .first()
            .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));

        #[cfg(target_arch = "x86_64")]
        if prefer_dual_scalar_pair(input_chunk) {
            // SAFETY: prefer_dual_scalar_pair checked CPUID BMI1 immediately above.
            let digests = unsafe {
                crate::scalar_x86_64_dual::hash_pair_bmi1([input_chunk[0], input_chunk[1]])
            };
            output_chunk.copy_from_slice(&digests);
            start = end;
            continue;
        }

        if same_len {
            if input_chunk.len() <= 8 {
                let len = input_chunk[0].len();
                let prefer_zmm = intel_family_06_model_cf() && len >= 512;
                if prefer_zmm {
                    hash_equal_len_avx512_padded(avx512, input_chunk, output_chunk);
                } else if input_chunk.len() == 8 {
                    hash_equal_len_avx2_kernel(avx2, input_chunk, output_chunk);
                } else {
                    hash_equal_len_avx2_padded(avx2, input_chunk, output_chunk);
                }
            } else if amd_family_19h() && padded_blocks_for_len(input_chunk[0].len()) <= 17 {
                // On AMD Family 19h AVX-512 parts, two interleaved YMM chains
                // beat one ZMM chain for short equal-length batches. At larger
                // sizes the AVX-512 native kernel wins back the fixed overhead.
                if input_chunk.len() == 16 {
                    hash_equal_len_avx2_dual_kernel(avx2, input_chunk, output_chunk);
                } else {
                    hash_equal_len_avx2_dual_padded(avx2, input_chunk, output_chunk);
                }
            } else if input_chunk.len() == 16 {
                hash_equal_len_avx512_kernel(avx512, input_chunk, output_chunk);
            } else {
                hash_equal_len_avx512_padded(avx512, input_chunk, output_chunk);
            }
        } else if input_chunk.len() <= 8 {
            let min_len = input_chunk
                .iter()
                .map(|input| input.len())
                .min()
                .unwrap_or(0);
            if intel_family_06_model_cf() && min_len >= 512 {
                hash_mixed_len_avx512_padded(avx512, input_chunk, output_chunk);
            } else {
                hash_many_avx2(avx2, input_chunk, output_chunk);
            }
        } else if input_chunk.len() == 16 {
            hash_mixed_len_avx512_kernel(avx512, input_chunk, output_chunk);
        } else {
            hash_mixed_len_avx512_padded(avx512, input_chunk, output_chunk);
        }

        start = end;
    }
}

#[inline(always)]
fn hash_many_inner<SIMD: Simd>(simd: SIMD, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    let lanes = SIMD::u32s::N;
    debug_assert!(lanes <= MAX_LANES);

    let mut start = 0;
    while start < inputs.len() {
        let end = core::cmp::min(start + lanes, inputs.len());
        let input_chunk = &inputs[start..end];
        let output_chunk = &mut outputs[start..end];

        // SIMD has fixed per-round cost. Tiny under-filled batches are cheaper on
        // the scalar path, especially on AVX2 where a full vector has eight lanes.
        if input_chunk.len() < 3 {
            for (input, output) in input_chunk.iter().zip(output_chunk) {
                *output = scalar::hash(input);
            }
            start = end;
            continue;
        }

        let same_len = input_chunk
            .first()
            .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));

        if same_len {
            hash_equal_len_chunk(simd, input_chunk, output_chunk);
        } else {
            hash_mixed_len_chunk(simd, input_chunk, output_chunk);
        }

        start = end;
    }
}

pub(crate) fn hash_many_with_level(level: Level, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    assert!(
        outputs.len() >= inputs.len(),
        "output slice is shorter than input slice"
    );
    let outputs = &mut outputs[..inputs.len()];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx512) = level.as_avx512() {
        hash_many_avx512(avx512, inputs, outputs);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = level.as_avx2() {
        hash_many_avx2(avx2, inputs, outputs);
        return;
    }

    dispatch!(level, simd => hash_many_inner(simd, inputs, outputs));
}

pub(crate) const fn lanes_with_level(level: Level) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match level {
            Level::Avx512(_) => 16,
            Level::Avx2(_) => 8,
            Level::Sse4_2(_) | Level::Sse2(_) => 4,
            _ => 1,
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        match level {
            Level::Neon(_) => 4,
            _ => 1,
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        match level {
            Level::WasmSimd128(_) => 4,
            _ => 1,
        }
    }

    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        let _ = level;
        1
    }
}
