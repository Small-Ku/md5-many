use fearless_simd::{Level, Simd, SimdBase, dispatch};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, __m512i, _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256,
    _mm512_add_epi32, _mm512_loadu_si512, _mm512_permutex2var_epi32, _mm512_rol_epi32,
    _mm512_set1_epi32, _mm512_setr_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
    _mm512_ternarylogic_epi32,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256,
    _mm512_add_epi32, _mm512_loadu_si512, _mm512_permutex2var_epi32, _mm512_rol_epi32,
    _mm512_set1_epi32, _mm512_setr_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
    _mm512_ternarylogic_epi32,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::{Avx2, Avx512};

use crate::consts::{K, S, STATE_INIT};
use crate::scalar;

/// Maximum lane count supported by the current implementation.
///
/// `fearless_simd` 0.7 uses up to 16 native `u32` lanes with AVX-512.
const MAX_LANES: usize = 16;

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
            let mut padded = [[0u8; 64]; 8];
            let base = block_index * 64;
            for lane in 0..8 {
                for (byte, slot) in padded[lane].iter_mut().enumerate() {
                    *slot = padded_byte(inputs[lane], padded_blocks, base + byte);
                }
            }
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
            let mut padded = [[0u8; 64]; 16];
            let base = block_index * 64;
            for lane in 0..16 {
                for (byte, slot) in padded[lane].iter_mut().enumerate() {
                    *slot = padded_byte(inputs[lane], padded_blocks, base + byte);
                }
            }
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
fn padded_blocks_for_len(len: usize) -> usize {
    let full_blocks = len / 64;
    let tail = len & 63;
    full_blocks + if tail <= 55 { 1 } else { 2 }
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
fn hash_equal_len_avx512_padded(avx512: Avx512, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    debug_assert!((9..=16).contains(&inputs.len()));
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
#[inline(always)]
fn hash_many_avx2(avx2: Avx2, inputs: &[&[u8]], outputs: &mut [[u8; 16]]) {
    let mut start = 0;
    while start < inputs.len() {
        let end = core::cmp::min(start + 8, inputs.len());
        let input_chunk = &inputs[start..end];
        let output_chunk = &mut outputs[start..end];

        if input_chunk.len() == 1 {
            output_chunk[0] = scalar::hash(input_chunk[0]);
            start = end;
            continue;
        }

        let same_len = input_chunk
            .first()
            .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));

        if same_len {
            hash_equal_len_avx2_padded(avx2, input_chunk, output_chunk);
        } else if input_chunk.len() < 3 {
            for (input, output) in input_chunk.iter().zip(output_chunk) {
                *output = scalar::hash(input);
            }
        } else {
            dispatch!(avx2.level(), simd => hash_many_inner(simd, input_chunk, output_chunk));
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
        let end = core::cmp::min(start + 16, inputs.len());
        let input_chunk = &inputs[start..end];
        let output_chunk = &mut outputs[start..end];

        if input_chunk.len() == 1 {
            output_chunk[0] = scalar::hash(input_chunk[0]);
            start = end;
            continue;
        }

        let same_len = input_chunk
            .first()
            .is_none_or(|first| input_chunk.iter().all(|input| input.len() == first.len()));

        if same_len {
            if input_chunk.len() <= 8 {
                hash_equal_len_avx2_padded(avx2, input_chunk, output_chunk);
            } else {
                hash_equal_len_avx512_padded(avx512, input_chunk, output_chunk);
            }
        } else if input_chunk.len() <= 8 {
            hash_many_avx2(avx2, input_chunk, output_chunk);
        } else {
            dispatch!(avx512.level(), simd => hash_many_inner(simd, input_chunk, output_chunk));
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

#[inline(always)]
fn lanes_inner<SIMD: Simd>(_simd: SIMD) -> usize {
    <SIMD::u32s as SimdBase<SIMD>>::N
}

pub(crate) fn lanes_with_level(level: Level) -> usize {
    dispatch!(level, simd => lanes_inner(simd))
}
