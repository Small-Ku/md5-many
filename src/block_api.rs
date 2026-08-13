use core::fmt;

use digest::{
    HashMarker, Output,
    array::Array,
    block_api::{
        AlgorithmName, Block, BlockSizeUser, Buffer, BufferKindUser, Eager, FixedOutputCore,
        OutputSizeUser, Reset, UpdateCore,
    },
    typenum::{U16, U64, Unsigned},
};

use crate::{consts::STATE_INIT, scalar};

/// Block-level MD5 state used by the RustCrypto `digest` compatibility wrapper.
#[derive(Clone)]
pub struct Md5Core {
    block_len: u64,
    state: [u32; 4],
}

impl HashMarker for Md5Core {}

impl BlockSizeUser for Md5Core {
    type BlockSize = U64;
}

impl BufferKindUser for Md5Core {
    type BufferKind = Eager;
}

impl OutputSizeUser for Md5Core {
    type OutputSize = U16;
}

impl UpdateCore for Md5Core {
    #[inline]
    fn update_blocks(&mut self, blocks: &[Block<Self>]) {
        self.block_len = self.block_len.wrapping_add(blocks.len() as u64);
        let blocks = Array::cast_slice_to_core(blocks);
        scalar::compress_blocks(&mut self.state, blocks);
    }
}

impl FixedOutputCore for Md5Core {
    #[inline]
    fn finalize_fixed_core(&mut self, buffer: &mut Buffer<Self>, out: &mut Output<Self>) {
        let bit_len = self
            .block_len
            .wrapping_mul(Self::BlockSize::U64)
            .wrapping_add(buffer.get_pos() as u64)
            .wrapping_mul(8);
        let mut state = self.state;
        buffer.len64_padding_le(bit_len, |block| {
            scalar::compress_block(&mut state, &block.0)
        });
        for (chunk, word) in out.chunks_exact_mut(4).zip(state) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
    }
}

impl Default for Md5Core {
    fn default() -> Self {
        Self {
            block_len: 0,
            state: STATE_INIT,
        }
    }
}

impl Reset for Md5Core {
    fn reset(&mut self) {
        *self = Self::default();
    }
}

impl AlgorithmName for Md5Core {
    fn write_alg_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Md5")
    }
}
