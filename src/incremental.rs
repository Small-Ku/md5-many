use crate::{Md5Digest, consts::STATE_INIT, scalar, simd};
use fearless_simd::Level;

const MAX_LANES: usize = 16;

/// Incremental state for one MD5 message inside a multi-stream workload.
///
/// The state buffers at most one partial 64-byte block. It is intentionally
/// allocation-free so callers can keep any number of streams in an array,
/// `Vec`, slab, or another application-owned container.
#[derive(Clone, Copy, Debug)]
pub struct Md5State {
    state: [u32; 4],
    buffer: [u8; 64],
    buffer_len: u8,
    bytes: u64,
}

impl Md5State {
    /// Construct an empty incremental MD5 state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: STATE_INIT,
            buffer: [0; 64],
            buffer_len: 0,
            bytes: 0,
        }
    }

    /// Add bytes to this one stream using the optimized single-stream backend.
    ///
    /// When several streams receive data together, prefer
    /// [`crate::Md5Many::update_many`] so full blocks can occupy SIMD lanes.
    pub fn update(&mut self, input: &[u8]) {
        self.bytes = self.bytes.wrapping_add(input.len() as u64);
        let mut offset = 0;

        if self.buffer_len != 0 {
            let used = self.buffer_len as usize;
            let take = core::cmp::min(64 - used, input.len());
            self.buffer[used..used + take].copy_from_slice(&input[..take]);
            self.buffer_len += take as u8;
            offset += take;

            if self.buffer_len == 64 {
                scalar::compress_block(&mut self.state, &self.buffer);
                self.buffer_len = 0;
            } else {
                return;
            }
        }

        let direct = &input[offset..];
        let mut chunks = direct.chunks_exact(64);
        for chunk in &mut chunks {
            let block: &[u8; 64] = chunk.try_into().expect("64-byte chunk");
            scalar::compress_block(&mut self.state, block);
        }
        let tail = chunks.remainder();
        self.buffer[..tail.len()].copy_from_slice(tail);
        self.buffer_len = tail.len() as u8;
    }

    /// Return the digest without modifying the state.
    ///
    /// More data may be appended after this call. For several streams, prefer
    /// [`crate::Md5Many::finalize_many`] so their padding blocks are batched.
    #[must_use]
    pub fn finalize(&self) -> Md5Digest {
        let mut state = self.state;
        let mut final_blocks = [[0u8; 64]; 2];
        let tail_len = self.buffer_len as usize;
        final_blocks[0][..tail_len].copy_from_slice(&self.buffer[..tail_len]);
        final_blocks[0][tail_len] = 0x80;

        let bit_len = self.bytes.wrapping_mul(8).to_le_bytes();
        let used = if tail_len <= 55 {
            final_blocks[0][56..64].copy_from_slice(&bit_len);
            1
        } else {
            final_blocks[1][56..64].copy_from_slice(&bit_len);
            2
        };

        for block in &final_blocks[..used] {
            scalar::compress_block(&mut state, block);
        }
        scalar::state_to_bytes(state)
    }

    /// Reset this state to an empty MD5 message.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Return the number of bytes supplied so far, modulo `2^64`.
    #[must_use]
    pub const fn bytes_hashed(&self) -> u64 {
        self.bytes
    }

    /// Return whether no bytes have been supplied since construction/reset.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

impl Default for Md5State {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn compress_selected_blocks(
    level: Level,
    streams: &mut [Md5State],
    selected: &[usize],
    inputs: &[&[u8]],
) {
    debug_assert_eq!(selected.len(), inputs.len());
    debug_assert!(selected.len() <= MAX_LANES);
    if selected.is_empty() {
        return;
    }

    let mut states = [STATE_INIT; MAX_LANES];
    for (slot, &lane) in selected.iter().enumerate() {
        states[slot] = streams[lane].state;
    }
    simd::compress_many_blocks_with_level(level, &mut states[..selected.len()], inputs);
    for (slot, &lane) in selected.iter().enumerate() {
        streams[lane].state = states[slot];
    }
}

pub(crate) fn update_many_with_level(level: Level, streams: &mut [Md5State], inputs: &[&[u8]]) {
    assert_eq!(
        streams.len(),
        inputs.len(),
        "incremental state/input length mismatch"
    );
    if streams.is_empty() {
        return;
    }

    let lanes = core::cmp::max(1, simd::lanes_with_level(level));
    debug_assert!(lanes <= MAX_LANES);

    let mut group_start = 0;
    while group_start < streams.len() {
        let group_end = core::cmp::min(group_start + lanes, streams.len());
        let streams = &mut streams[group_start..group_end];
        let inputs = &inputs[group_start..group_end];
        let count = streams.len();

        let mut offsets = [0usize; MAX_LANES];
        let mut ready = [false; MAX_LANES];
        let mut ready_blocks = [[0u8; 64]; MAX_LANES];

        // First complete any partial blocks left by the previous update. Keep
        // those completed blocks together so 32+32-byte (etc.) streaming can
        // still use multi-buffer SIMD rather than degenerating to scalar MD5.
        for lane in 0..count {
            streams[lane].bytes = streams[lane].bytes.wrapping_add(inputs[lane].len() as u64);
            if streams[lane].buffer_len == 0 {
                continue;
            }

            let used = streams[lane].buffer_len as usize;
            let take = core::cmp::min(64 - used, inputs[lane].len());
            streams[lane].buffer[used..used + take].copy_from_slice(&inputs[lane][..take]);
            streams[lane].buffer_len += take as u8;
            offsets[lane] = take;

            if streams[lane].buffer_len == 64 {
                ready_blocks[lane] = streams[lane].buffer;
                streams[lane].buffer_len = 0;
                ready[lane] = true;
            }
        }

        let mut selected = [0usize; MAX_LANES];
        let mut selected_count = 0;
        for (lane, is_ready) in ready[..count].iter().copied().enumerate() {
            if is_ready {
                selected[selected_count] = lane;
                selected_count += 1;
            }
        }
        if selected_count != 0 {
            let mut block_refs: [&[u8]; MAX_LANES] = [&[]; MAX_LANES];
            for slot in 0..selected_count {
                block_refs[slot] = &ready_blocks[selected[slot]];
            }
            compress_selected_blocks(
                level,
                streams,
                &selected[..selected_count],
                &block_refs[..selected_count],
            );
        }

        // Consume direct full blocks. At each step we compact the streams that
        // still have block work, then process their largest common prefix. This
        // keeps SIMD occupancy high even when per-stream update sizes diverge.
        loop {
            let mut active_count = 0;
            let mut common_blocks = usize::MAX;
            for lane in 0..count {
                let blocks = (inputs[lane].len() - offsets[lane]) / 64;
                if blocks != 0 {
                    selected[active_count] = lane;
                    active_count += 1;
                    common_blocks = core::cmp::min(common_blocks, blocks);
                }
            }
            if active_count == 0 {
                break;
            }

            let bytes = common_blocks * 64;
            let mut block_refs: [&[u8]; MAX_LANES] = [&[]; MAX_LANES];
            for slot in 0..active_count {
                let lane = selected[slot];
                let start = offsets[lane];
                block_refs[slot] = &inputs[lane][start..start + bytes];
            }
            compress_selected_blocks(
                level,
                streams,
                &selected[..active_count],
                &block_refs[..active_count],
            );
            for &lane in &selected[..active_count] {
                offsets[lane] += bytes;
            }
        }

        for lane in 0..count {
            let tail = &inputs[lane][offsets[lane]..];
            debug_assert!(tail.len() < 64);
            if tail.is_empty() {
                continue;
            }
            debug_assert_eq!(streams[lane].buffer_len, 0);
            streams[lane].buffer[..tail.len()].copy_from_slice(tail);
            streams[lane].buffer_len = tail.len() as u8;
        }

        group_start = group_end;
    }
}

pub(crate) fn finalize_many_with_level(
    level: Level,
    streams: &[Md5State],
    outputs: &mut [Md5Digest],
) {
    assert!(
        outputs.len() >= streams.len(),
        "output slice is shorter than incremental state slice"
    );
    let outputs = &mut outputs[..streams.len()];
    if streams.is_empty() {
        return;
    }

    let lanes = core::cmp::max(1, simd::lanes_with_level(level));
    debug_assert!(lanes <= MAX_LANES);

    let mut group_start = 0;
    while group_start < streams.len() {
        let group_end = core::cmp::min(group_start + lanes, streams.len());
        let input_streams = &streams[group_start..group_end];
        let outputs = &mut outputs[group_start..group_end];
        let count = input_streams.len();

        // Work on copies so finalize_many is non-destructive and callers may
        // continue updating from the same checkpoint afterwards.
        let mut work = [Md5State::new(); MAX_LANES];
        let mut first_blocks = [[0u8; 64]; MAX_LANES];
        let mut first_refs: [&[u8]; MAX_LANES] = [&[]; MAX_LANES];
        let mut needs_second = [false; MAX_LANES];
        let selected: [usize; MAX_LANES] = core::array::from_fn(|i| i);

        for lane in 0..count {
            work[lane] = input_streams[lane];
            let tail_len = work[lane].buffer_len as usize;
            first_blocks[lane][..tail_len].copy_from_slice(&work[lane].buffer[..tail_len]);
            first_blocks[lane][tail_len] = 0x80;
            let bit_len = work[lane].bytes.wrapping_mul(8).to_le_bytes();
            if tail_len <= 55 {
                first_blocks[lane][56..64].copy_from_slice(&bit_len);
            } else {
                needs_second[lane] = true;
            }
        }
        for lane in 0..count {
            first_refs[lane] = &first_blocks[lane];
        }

        compress_selected_blocks(
            level,
            &mut work[..count],
            &selected[..count],
            &first_refs[..count],
        );

        let mut second_blocks = [[0u8; 64]; MAX_LANES];
        let mut second_refs: [&[u8]; MAX_LANES] = [&[]; MAX_LANES];
        let mut second_selected = [0usize; MAX_LANES];
        let mut second_count = 0;
        for lane in 0..count {
            if needs_second[lane] {
                second_blocks[second_count][56..64]
                    .copy_from_slice(&work[lane].bytes.wrapping_mul(8).to_le_bytes());
                second_selected[second_count] = lane;
                second_count += 1;
            }
        }
        for slot in 0..second_count {
            second_refs[slot] = &second_blocks[slot];
        }
        if second_count != 0 {
            compress_selected_blocks(
                level,
                &mut work[..count],
                &second_selected[..second_count],
                &second_refs[..second_count],
            );
        }

        for lane in 0..count {
            outputs[lane] = scalar::state_to_bytes(work[lane].state);
        }
        group_start = group_end;
    }
}
