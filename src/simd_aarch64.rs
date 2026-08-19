use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vbslq_u32, vcombine_u32, vdupq_n_u32, veorq_u32, vget_high_u32,
    vget_low_u32, vld1q_u8, vornq_u32, vorrq_u32, vreinterpretq_u8_u32, vreinterpretq_u32_u8,
    vshlq_n_u32, vshrq_n_u32, vst1q_u8, vtrn1q_u32, vtrn2q_u32,
};

use crate::consts::{K, STATE_INIT};

#[inline(always)]
fn transpose4(rows: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
    let t0 = vtrn1q_u32(rows[0], rows[1]);
    let t1 = vtrn2q_u32(rows[0], rows[1]);
    let t2 = vtrn1q_u32(rows[2], rows[3]);
    let t3 = vtrn2q_u32(rows[2], rows[3]);
    [
        vcombine_u32(vget_low_u32(t0), vget_low_u32(t2)),
        vcombine_u32(vget_low_u32(t1), vget_low_u32(t3)),
        vcombine_u32(vget_high_u32(t0), vget_high_u32(t2)),
        vcombine_u32(vget_high_u32(t1), vget_high_u32(t3)),
    ]
}

#[inline(always)]
fn load_words(blocks: [&[u8; 64]; 4]) -> [uint32x4_t; 16] {
    let mut words = [vdupq_n_u32(0); 16];
    for group in 0..4 {
        let offset = group * 16;
        let rows = core::array::from_fn(|lane| {
            // SAFETY: each load reads exactly 16 bytes starting at one of
            // offsets 0, 16, 32, or 48 of a 64-byte MD5 block.
            unsafe { vreinterpretq_u32_u8(vld1q_u8(blocks[lane].as_ptr().add(offset))) }
        });
        let transposed = transpose4(rows);
        words[group * 4..group * 4 + 4].copy_from_slice(&transposed);
    }
    words
}

#[inline(always)]
fn compress_words(state: &mut [uint32x4_t; 4], words: &[uint32x4_t; 16]) {
    let [mut a, mut b, mut c, mut d] = *state;
    let initial = *state;

    macro_rules! mix {
        (f, $x:expr, $y:expr, $z:expr) => {
            vbslq_u32($x, $y, $z)
        };
        (g, $x:expr, $y:expr, $z:expr) => {
            vbslq_u32($z, $x, $y)
        };
        (h, $x:expr, $y:expr, $z:expr) => {
            veorq_u32(veorq_u32($x, $y), $z)
        };
        (i, $x:expr, $y:expr, $z:expr) => {
            veorq_u32($y, vornq_u32($x, $z))
        };
    }
    macro_rules! step {
        ($which:ident, $a:ident, $b:ident, $c:ident, $d:ident, $word:expr, $round:expr, $shift:literal) => {{
            let mixed = mix!($which, $b, $c, $d);
            let mut value = vaddq_u32($a, mixed);
            value = vaddq_u32(value, vdupq_n_u32(K[$round]));
            value = vaddq_u32(value, words[$word]);
            let rotated = vorrq_u32(
                vshlq_n_u32::<$shift>(value),
                vshrq_n_u32::<{ 32 - $shift }>(value),
            );
            $a = vaddq_u32($b, rotated);
        }};
    }

    step!(f, a, b, c, d, 0, 0, 7);
    step!(f, d, a, b, c, 1, 1, 12);
    step!(f, c, d, a, b, 2, 2, 17);
    step!(f, b, c, d, a, 3, 3, 22);
    step!(f, a, b, c, d, 4, 4, 7);
    step!(f, d, a, b, c, 5, 5, 12);
    step!(f, c, d, a, b, 6, 6, 17);
    step!(f, b, c, d, a, 7, 7, 22);
    step!(f, a, b, c, d, 8, 8, 7);
    step!(f, d, a, b, c, 9, 9, 12);
    step!(f, c, d, a, b, 10, 10, 17);
    step!(f, b, c, d, a, 11, 11, 22);
    step!(f, a, b, c, d, 12, 12, 7);
    step!(f, d, a, b, c, 13, 13, 12);
    step!(f, c, d, a, b, 14, 14, 17);
    step!(f, b, c, d, a, 15, 15, 22);
    step!(g, a, b, c, d, 1, 16, 5);
    step!(g, d, a, b, c, 6, 17, 9);
    step!(g, c, d, a, b, 11, 18, 14);
    step!(g, b, c, d, a, 0, 19, 20);
    step!(g, a, b, c, d, 5, 20, 5);
    step!(g, d, a, b, c, 10, 21, 9);
    step!(g, c, d, a, b, 15, 22, 14);
    step!(g, b, c, d, a, 4, 23, 20);
    step!(g, a, b, c, d, 9, 24, 5);
    step!(g, d, a, b, c, 14, 25, 9);
    step!(g, c, d, a, b, 3, 26, 14);
    step!(g, b, c, d, a, 8, 27, 20);
    step!(g, a, b, c, d, 13, 28, 5);
    step!(g, d, a, b, c, 2, 29, 9);
    step!(g, c, d, a, b, 7, 30, 14);
    step!(g, b, c, d, a, 12, 31, 20);
    step!(h, a, b, c, d, 5, 32, 4);
    step!(h, d, a, b, c, 8, 33, 11);
    step!(h, c, d, a, b, 11, 34, 16);
    step!(h, b, c, d, a, 14, 35, 23);
    step!(h, a, b, c, d, 1, 36, 4);
    step!(h, d, a, b, c, 4, 37, 11);
    step!(h, c, d, a, b, 7, 38, 16);
    step!(h, b, c, d, a, 10, 39, 23);
    step!(h, a, b, c, d, 13, 40, 4);
    step!(h, d, a, b, c, 0, 41, 11);
    step!(h, c, d, a, b, 3, 42, 16);
    step!(h, b, c, d, a, 6, 43, 23);
    step!(h, a, b, c, d, 9, 44, 4);
    step!(h, d, a, b, c, 12, 45, 11);
    step!(h, c, d, a, b, 15, 46, 16);
    step!(h, b, c, d, a, 2, 47, 23);
    step!(i, a, b, c, d, 0, 48, 6);
    step!(i, d, a, b, c, 7, 49, 10);
    step!(i, c, d, a, b, 14, 50, 15);
    step!(i, b, c, d, a, 5, 51, 21);
    step!(i, a, b, c, d, 12, 52, 6);
    step!(i, d, a, b, c, 3, 53, 10);
    step!(i, c, d, a, b, 10, 54, 15);
    step!(i, b, c, d, a, 1, 55, 21);
    step!(i, a, b, c, d, 8, 56, 6);
    step!(i, d, a, b, c, 15, 57, 10);
    step!(i, c, d, a, b, 6, 58, 15);
    step!(i, b, c, d, a, 13, 59, 21);
    step!(i, a, b, c, d, 4, 60, 6);
    step!(i, d, a, b, c, 11, 61, 10);
    step!(i, c, d, a, b, 2, 62, 15);
    step!(i, b, c, d, a, 9, 63, 21);

    state[0] = vaddq_u32(initial[0], a);
    state[1] = vaddq_u32(initial[1], b);
    state[2] = vaddq_u32(initial[2], c);
    state[3] = vaddq_u32(initial[3], d);
}

#[inline(always)]
fn padded_blocks_for_len(len: usize) -> usize {
    len / 64 + if (len & 63) <= 55 { 1 } else { 2 }
}

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

/// Hash four equal-length messages with a native AArch64 NEON kernel.
///
/// This remains an experimental backend until hardware benchmarks establish
/// its crossover against the generic Fearless SIMD NEON path.
pub(crate) fn hash_equal_len4(inputs: [&[u8]; 4]) -> [[u8; 16]; 4] {
    let len = inputs[0].len();
    assert!(inputs.iter().all(|input| input.len() == len));

    let mut state = STATE_INIT.map(vdupq_n_u32);
    let full_blocks = len / 64;
    for block_index in 0..full_blocks {
        let offset = block_index * 64;
        let blocks: [&[u8; 64]; 4] = core::array::from_fn(|lane| {
            inputs[lane][offset..offset + 64]
                .try_into()
                .expect("full MD5 block")
        });
        let words = load_words(blocks);
        compress_words(&mut state, &words);
    }

    let padded_blocks = padded_blocks_for_len(len);
    for block_index in full_blocks..padded_blocks {
        let padded: [[u8; 64]; 4] = core::array::from_fn(|lane| {
            build_padded_block(inputs[lane], padded_blocks, block_index)
        });
        let blocks: [&[u8; 64]; 4] = core::array::from_fn(|lane| &padded[lane]);
        let words = load_words(blocks);
        compress_words(&mut state, &words);
    }

    // Transpose the SoA state back to one `[A, B, C, D]` vector per
    // message. This module only exists on little-endian AArch64, so storing
    // each u32 vector as bytes directly produces MD5's little-endian digest.
    let digest_rows = transpose4(state);
    let mut outputs = [[0u8; 16]; 4];
    for lane in 0..4 {
        // SAFETY: each output has exactly 16 writable bytes.
        unsafe {
            vst1q_u8(
                outputs[lane].as_mut_ptr(),
                vreinterpretq_u8_u32(digest_rows[lane]),
            )
        };
    }
    outputs
}

#[cfg(test)]
mod tests {
    use super::hash_equal_len4;

    #[test]
    fn native_neon_equal_len_matches_scalar() {
        for &len in &[0usize, 1, 55, 56, 63, 64, 65, 119, 120, 128, 1024, 4096] {
            let data: [std::vec::Vec<u8>; 4] = core::array::from_fn(|lane| {
                (0..len)
                    .map(|index| (index as u8).wrapping_mul(17).wrapping_add(lane as u8 * 29))
                    .collect()
            });
            let refs = data.each_ref().map(|input| crate::scalar::hash(input));
            let got = hash_equal_len4(data.each_ref().map(|input| input.as_slice()));
            assert_eq!(got, refs, "len={len}");
        }
    }
}
