use crate::consts::{K, S, STATE_INIT};

#[inline(always)]
fn round_f(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | (!x & z)
}

#[inline(always)]
fn round_g(x: u32, y: u32, z: u32) -> u32 {
    (x & z).wrapping_add(y & !z)
}

#[inline(always)]
fn round_h(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

#[inline(always)]
fn round_i(x: u32, y: u32, z: u32) -> u32 {
    y ^ (x | !z)
}

#[inline]
pub(crate) fn compress_block(state: &mut [u32; 4], block: &[u8; 64]) {
    let mut words = [0u32; 16];
    for (word, bytes) in words.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
    }

    let [mut a, mut b, mut c, mut d] = *state;
    let initial = *state;

    macro_rules! step {
        ($mix:ident, $a:ident, $b:ident, $c:ident, $d:ident, $word:expr, $round:expr) => {{
            $a = $b.wrapping_add(
                $a.wrapping_add($mix($b, $c, $d))
                    .wrapping_add(K[$round])
                    .wrapping_add(words[$word])
                    .rotate_left(S[$round]),
            );
        }};
    }

    step!(round_f, a, b, c, d, 0, 0);
    step!(round_f, d, a, b, c, 1, 1);
    step!(round_f, c, d, a, b, 2, 2);
    step!(round_f, b, c, d, a, 3, 3);
    step!(round_f, a, b, c, d, 4, 4);
    step!(round_f, d, a, b, c, 5, 5);
    step!(round_f, c, d, a, b, 6, 6);
    step!(round_f, b, c, d, a, 7, 7);
    step!(round_f, a, b, c, d, 8, 8);
    step!(round_f, d, a, b, c, 9, 9);
    step!(round_f, c, d, a, b, 10, 10);
    step!(round_f, b, c, d, a, 11, 11);
    step!(round_f, a, b, c, d, 12, 12);
    step!(round_f, d, a, b, c, 13, 13);
    step!(round_f, c, d, a, b, 14, 14);
    step!(round_f, b, c, d, a, 15, 15);

    step!(round_g, a, b, c, d, 1, 16);
    step!(round_g, d, a, b, c, 6, 17);
    step!(round_g, c, d, a, b, 11, 18);
    step!(round_g, b, c, d, a, 0, 19);
    step!(round_g, a, b, c, d, 5, 20);
    step!(round_g, d, a, b, c, 10, 21);
    step!(round_g, c, d, a, b, 15, 22);
    step!(round_g, b, c, d, a, 4, 23);
    step!(round_g, a, b, c, d, 9, 24);
    step!(round_g, d, a, b, c, 14, 25);
    step!(round_g, c, d, a, b, 3, 26);
    step!(round_g, b, c, d, a, 8, 27);
    step!(round_g, a, b, c, d, 13, 28);
    step!(round_g, d, a, b, c, 2, 29);
    step!(round_g, c, d, a, b, 7, 30);
    step!(round_g, b, c, d, a, 12, 31);

    step!(round_h, a, b, c, d, 5, 32);
    step!(round_h, d, a, b, c, 8, 33);
    step!(round_h, c, d, a, b, 11, 34);
    step!(round_h, b, c, d, a, 14, 35);
    step!(round_h, a, b, c, d, 1, 36);
    step!(round_h, d, a, b, c, 4, 37);
    step!(round_h, c, d, a, b, 7, 38);
    step!(round_h, b, c, d, a, 10, 39);
    step!(round_h, a, b, c, d, 13, 40);
    step!(round_h, d, a, b, c, 0, 41);
    step!(round_h, c, d, a, b, 3, 42);
    step!(round_h, b, c, d, a, 6, 43);
    step!(round_h, a, b, c, d, 9, 44);
    step!(round_h, d, a, b, c, 12, 45);
    step!(round_h, c, d, a, b, 15, 46);
    step!(round_h, b, c, d, a, 2, 47);

    step!(round_i, a, b, c, d, 0, 48);
    step!(round_i, d, a, b, c, 7, 49);
    step!(round_i, c, d, a, b, 14, 50);
    step!(round_i, b, c, d, a, 5, 51);
    step!(round_i, a, b, c, d, 12, 52);
    step!(round_i, d, a, b, c, 3, 53);
    step!(round_i, c, d, a, b, 10, 54);
    step!(round_i, b, c, d, a, 1, 55);
    step!(round_i, a, b, c, d, 8, 56);
    step!(round_i, d, a, b, c, 15, 57);
    step!(round_i, c, d, a, b, 6, 58);
    step!(round_i, b, c, d, a, 13, 59);
    step!(round_i, a, b, c, d, 4, 60);
    step!(round_i, d, a, b, c, 11, 61);
    step!(round_i, c, d, a, b, 2, 62);
    step!(round_i, b, c, d, a, 9, 63);
    state[0] = initial[0].wrapping_add(a);
    state[1] = initial[1].wrapping_add(b);
    state[2] = initial[2].wrapping_add(c);
    state[3] = initial[3].wrapping_add(d);
}

#[cfg(feature = "digest")]
#[inline]
pub(crate) fn compress_blocks(state: &mut [u32; 4], blocks: &[[u8; 64]]) {
    for block in blocks {
        compress_block(state, block);
    }
}

pub(crate) fn hash(input: &[u8]) -> [u8; 16] {
    let mut state = STATE_INIT;
    let mut chunks = input.chunks_exact(64);

    for chunk in &mut chunks {
        let block: &[u8; 64] = chunk.try_into().expect("64-byte chunk");
        compress_block(&mut state, block);
    }

    let tail = chunks.remainder();
    let mut final_blocks = [[0u8; 64]; 2];
    final_blocks[0][..tail.len()].copy_from_slice(tail);
    final_blocks[0][tail.len()] = 0x80;

    let bit_len = (input.len() as u64).wrapping_mul(8).to_le_bytes();
    let used = if tail.len() <= 55 {
        final_blocks[0][56..64].copy_from_slice(&bit_len);
        1
    } else {
        final_blocks[1][56..64].copy_from_slice(&bit_len);
        2
    };

    for block in &final_blocks[..used] {
        compress_block(&mut state, block);
    }

    state_to_bytes(state)
}

#[inline]
pub(crate) fn state_to_bytes(state: [u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (chunk, word) in out.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}
