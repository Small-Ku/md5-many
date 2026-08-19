use crate::{Md5Digest, consts::STATE_INIT, scalar, simd};
use fearless_simd::Level;

const MAX_LANES: usize = 48;

/// Incremental state for one MD5 message inside a multi-stream workload.
///
/// The state buffers at most one partial 64-byte block. It is intentionally
/// allocation-free so callers can keep any number of streams in an array,
/// `Vec`, slab, or another application-owned container.
#[derive(Clone, Copy, Debug)]
pub struct Md5State {
    pub(crate) state: [u32; 4],
    buffer: [u8; 64],
    bytes: u64,
}

impl Md5State {
    /// Construct an empty incremental MD5 state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: STATE_INIT,
            buffer: [0; 64],
            bytes: 0,
        }
    }

    /// Add bytes to this one stream using the optimized single-stream backend.
    ///
    /// When several streams receive data together, prefer
    /// [`crate::Md5Many::update_many`] so full blocks can occupy SIMD lanes.
    pub fn update(&mut self, input: &[u8]) {
        let buffered = (self.bytes & 63) as usize;
        self.bytes = self.bytes.wrapping_add(input.len() as u64);
        let mut offset = 0;

        if buffered != 0 {
            let take = core::cmp::min(64 - buffered, input.len());
            self.buffer[buffered..buffered + take].copy_from_slice(&input[..take]);
            offset = take;

            if buffered + take == 64 {
                scalar::compress_block(&mut self.state, &self.buffer);
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
    }

    /// Return the digest without modifying the state.
    ///
    /// More data may be appended after this call. For several streams, prefer
    /// [`crate::Md5Many::finalize_many`] so their padding blocks are batched.
    #[must_use]
    pub fn finalize(&self) -> Md5Digest {
        let mut state = self.state;
        let mut final_blocks = [[0u8; 64]; 2];
        let tail_len = (self.bytes & 63) as usize;
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
fn compress_all_blocks_exact<const N: usize>(
    level: Level,
    streams: &mut [Md5State],
    inputs: &[&[u8]],
) {
    debug_assert_eq!(streams.len(), N);
    debug_assert_eq!(inputs.len(), N);
    let mut states: [[u32; 4]; N] = core::array::from_fn(|lane| streams[lane].state);
    simd::compress_many_blocks_with_level(level, &mut states, inputs);
    for lane in 0..N {
        streams[lane].state = states[lane];
    }
}

#[inline]
fn compress_all_blocks(level: Level, streams: &mut [Md5State], inputs: &[&[u8]]) {
    debug_assert_eq!(streams.len(), inputs.len());
    debug_assert!(streams.len() <= MAX_LANES);
    match streams.len() {
        0 => {}
        4 => compress_all_blocks_exact::<4>(level, streams, inputs),
        8 => compress_all_blocks_exact::<8>(level, streams, inputs),
        12 => compress_all_blocks_exact::<12>(level, streams, inputs),
        16 => compress_all_blocks_exact::<16>(level, streams, inputs),
        24 => compress_all_blocks_exact::<24>(level, streams, inputs),
        32 => compress_all_blocks_exact::<32>(level, streams, inputs),
        48 => compress_all_blocks_exact::<48>(level, streams, inputs),
        _ => {
            let mut states = [[0u32; 4]; MAX_LANES];
            for (slot, stream) in streams.iter().enumerate() {
                states[slot] = stream.state;
            }
            simd::compress_many_blocks_with_level(level, &mut states[..streams.len()], inputs);
            for (slot, stream) in streams.iter_mut().enumerate() {
                stream.state = states[slot];
            }
        }
    }
}

#[inline(always)]
fn try_lockstep_aligned_exact<const N: usize>(
    streams: &mut [Md5State],
    inputs: &[&[u8]],
    input_len: usize,
) -> bool {
    debug_assert_eq!(streams.len(), N);
    debug_assert_eq!(inputs.len(), N);
    let add = input_len as u64;
    let mut checked = 0;
    while checked < N {
        if inputs[checked].len() != input_len || (streams[checked].bytes & 63) != 0 {
            for stream in &mut streams[..checked] {
                stream.bytes = stream.bytes.wrapping_sub(add);
            }
            return false;
        }
        streams[checked].bytes = streams[checked].bytes.wrapping_add(add);
        checked += 1;
    }
    true
}

#[inline(always)]
fn try_lockstep_aligned(streams: &mut [Md5State], inputs: &[&[u8]], input_len: usize) -> bool {
    match streams.len() {
        4 => try_lockstep_aligned_exact::<4>(streams, inputs, input_len),
        8 => try_lockstep_aligned_exact::<8>(streams, inputs, input_len),
        12 => try_lockstep_aligned_exact::<12>(streams, inputs, input_len),
        16 => try_lockstep_aligned_exact::<16>(streams, inputs, input_len),
        24 => try_lockstep_aligned_exact::<24>(streams, inputs, input_len),
        32 => try_lockstep_aligned_exact::<32>(streams, inputs, input_len),
        48 => try_lockstep_aligned_exact::<48>(streams, inputs, input_len),
        _ => false,
    }
}

#[inline]
fn update_equal_group(
    level: Level,
    native_lanes: usize,
    streams: &mut [Md5State],
    inputs: &[&[u8]],
) -> bool {
    if streams.is_empty()
        || !streams.len().is_multiple_of(native_lanes)
        || streams.len() > native_lanes * 3
    {
        return false;
    }

    let input_len = inputs[0].len();
    let buffered = (streams[0].bytes & 63) as usize;
    if !inputs.iter().all(|input| input.len() == input_len)
        || !streams
            .iter()
            .all(|stream| (stream.bytes & 63) as usize == buffered)
    {
        return false;
    }

    for stream in streams.iter_mut() {
        stream.bytes = stream.bytes.wrapping_add(input_len as u64);
    }

    if buffered == 0 && input_len.is_multiple_of(64) {
        simd::compress_md5_states_blocks_validated_with_level(level, streams, inputs);
        return true;
    }

    let mut offset = 0;
    if buffered != 0 {
        let used = buffered;
        let needed = 64 - used;
        let take = core::cmp::min(needed, input_len);
        for lane in 0..streams.len() {
            streams[lane].buffer[used..used + take].copy_from_slice(&inputs[lane][..take]);
        }
        offset = take;

        if take != needed {
            return true;
        }

        let mut block_refs: [&[u8]; MAX_LANES] = [&[]; MAX_LANES];
        for lane in 0..streams.len() {
            block_refs[lane] = &streams[lane].buffer;
        }
        let count = streams.len();
        let mut states = [STATE_INIT; MAX_LANES];
        for lane in 0..count {
            states[lane] = streams[lane].state;
        }
        simd::compress_many_blocks_with_level(level, &mut states[..count], &block_refs[..count]);
        for lane in 0..count {
            streams[lane].state = states[lane];
        }
    }

    let direct_len = ((input_len - offset) / 64) * 64;
    if direct_len != 0 {
        let mut direct_refs: [&[u8]; MAX_LANES] = [&[]; MAX_LANES];
        for lane in 0..streams.len() {
            direct_refs[lane] = &inputs[lane][offset..offset + direct_len];
        }
        let count = streams.len();
        compress_all_blocks(level, streams, &direct_refs[..count]);
        offset += direct_len;
    }

    let tail_len = input_len - offset;
    if tail_len != 0 {
        for lane in 0..streams.len() {
            streams[lane].buffer[..tail_len].copy_from_slice(&inputs[lane][offset..]);
        }
    }
    true
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

    let lanes = simd::lanes_with_level(level);
    debug_assert!((1..=MAX_LANES).contains(&lanes));

    // The dominant streaming shape is lockstep: one update call supplies the
    // same block-aligned fragment to one, two, or three native SIMD groups.
    // Handle the whole slice here so a 64-byte update does not pay the generic
    // group scheduler and repeat lane detection before entering the stateful
    // x86 kernels.
    if streams.len() == lanes || streams.len() == lanes * 2 || streams.len() == lanes * 3 {
        let input_len = inputs[0].len();
        if input_len != 0
            && input_len.is_multiple_of(64)
            && try_lockstep_aligned(streams, inputs, input_len)
        {
            simd::compress_md5_states_blocks_validated_with_level(level, streams, inputs);
            return;
        }
    }

    let mut group_start = 0;
    while group_start < streams.len() {
        let remaining = streams.len() - group_start;
        let mut handled_equal = false;
        for groups in [3usize, 2, 1] {
            let count = lanes * groups;
            if remaining < count {
                continue;
            }
            let group_end = group_start + count;
            if update_equal_group(
                level,
                lanes,
                &mut streams[group_start..group_end],
                &inputs[group_start..group_end],
            ) {
                group_start = group_end;
                handled_equal = true;
                break;
            }
        }
        if handled_equal {
            continue;
        }

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
            let used = (streams[lane].bytes & 63) as usize;
            streams[lane].bytes = streams[lane].bytes.wrapping_add(inputs[lane].len() as u64);
            if used == 0 {
                continue;
            }

            let take = core::cmp::min(64 - used, inputs[lane].len());
            streams[lane].buffer[used..used + take].copy_from_slice(&inputs[lane][..take]);
            offsets[lane] = take;

            if used + take == 64 {
                ready_blocks[lane] = streams[lane].buffer;
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
            streams[lane].buffer[..tail.len()].copy_from_slice(tail);
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

    let lanes = simd::lanes_with_level(level);
    debug_assert!((1..=MAX_LANES).contains(&lanes));

    let mut group_start = 0;
    while group_start < streams.len() {
        let remaining = streams.len() - group_start;
        let group_len = if remaining >= lanes * 3 {
            lanes * 3
        } else if remaining >= lanes * 2 {
            lanes * 2
        } else {
            core::cmp::min(lanes, remaining)
        };
        let group_end = group_start + group_len;
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
            let tail_len = (work[lane].bytes & 63) as usize;
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
