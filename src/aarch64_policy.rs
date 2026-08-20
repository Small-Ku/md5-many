//! Pure scheduling policy shared by the AArch64 backend and host-side tests.
//!
//! Keep hardware-specific NEON code out of this module so policy/invariant
//! changes are exercised by ordinary host `cargo test` runs as well as by the
//! native AArch64 CI runner.

/// Return whether measured AArch64 lane duplication is profitable.
#[inline(always)]
pub(crate) fn equal_len_padding_profitable(lanes: usize, len: usize) -> bool {
    let remainder = len & 63;
    match lanes {
        // Five lanes are one native NEON4 group plus one scalar residual in
        // the old scheduler. N2 measurements cross over at the 56-byte
        // padding boundary, are strongly positive for aligned blocks, and
        // remain positive from 120 bytes onward. Skip the weak 65..119 area.
        5 => len >= 120 || (len >= 56 && (remainder == 0 || remainder >= 56)),
        // These lane counts win at every measured point from 55 bytes upward.
        6 | 7 | 10 | 11 | 15 => len >= 55,
        // Nine lanes are native NEON8 plus one scalar residual today. The
        // padded NEON12 candidate is clearly positive on aligned blocks and
        // when padding spills into another block; for longer messages the
        // interleaved throughput advantage dominates. Avoid the measured
        // 55/119-byte counterexamples and other unmeasured short remainders.
        9 => len >= 256 || (len >= 56 && (remainder == 0 || remainder >= 56)),
        // Padding 13/14 lanes to 16 regressed every measured N2 workload.
        _ => false,
    }
}

/// Number of equal-length lanes consumed by the native AArch64 scheduler.
#[inline(always)]
pub(crate) fn equal_len_native_count(lanes: usize, len: usize) -> usize {
    if equal_len_padding_profitable(lanes, len) {
        lanes
    } else {
        lanes & !3usize
    }
}

#[cfg(test)]
mod tests {
    use super::{equal_len_native_count, equal_len_padding_profitable};

    #[test]
    fn measured_padding_boundaries_stay_explicit() {
        assert!(!equal_len_padding_profitable(5, 55));
        assert!(equal_len_padding_profitable(5, 56));
        assert!(equal_len_padding_profitable(5, 64));
        assert!(!equal_len_padding_profitable(5, 65));
        assert!(!equal_len_padding_profitable(5, 119));
        assert!(equal_len_padding_profitable(5, 120));

        assert!(!equal_len_padding_profitable(9, 55));
        assert!(equal_len_padding_profitable(9, 56));
        assert!(equal_len_padding_profitable(9, 64));
        assert!(!equal_len_padding_profitable(9, 119));
        assert!(equal_len_padding_profitable(9, 120));
        assert!(equal_len_padding_profitable(9, 256));

        for lanes in [6usize, 7, 10, 11, 15] {
            assert!(!equal_len_padding_profitable(lanes, 54));
            assert!(equal_len_padding_profitable(lanes, 55));
        }
        for lanes in [4usize, 8, 12, 13, 14, 16] {
            assert!(!equal_len_padding_profitable(lanes, 4096));
        }
    }

    #[test]
    fn processed_count_tracks_underfilled_policy() {
        // This is the contract that the native composition test used to
        // duplicate incorrectly as `lanes & !3`, which became stale when
        // lane duplication moved into production.
        assert_eq!(equal_len_native_count(5, 193), 5);
        assert_eq!(equal_len_native_count(8, 193), 8);
        assert_eq!(equal_len_native_count(9, 193), 8);
        assert_eq!(equal_len_native_count(12, 193), 12);
        assert_eq!(equal_len_native_count(13, 193), 12);
        assert_eq!(equal_len_native_count(15, 193), 15);
        assert_eq!(equal_len_native_count(20, 193), 20);
    }
}
