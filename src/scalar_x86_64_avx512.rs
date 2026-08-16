//! Single-stream AVX-512VL compressor.
//!
//! This is a Rust/intrinsics port of the CC0 `md5_block_avx512` schedule from
//! animetosho/md5-optimisation. It keeps one MD5 dependency chain in the low
//! `u32` lane of XMM registers while using the remaining lanes to cache four
//! message words plus round-constant deltas. AVX-512VL supplies ternary Boolean
//! operations and single-instruction rotates without widening the state to ZMM.

use core::sync::atomic::{AtomicU8, Ordering};

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_rol_epi32,
    _mm_srli_epi64, _mm_srli_si128, _mm_ternarylogic_epi32, _mm_unpackhi_epi64,
};

const BLOCK_SIZE: usize = 64;
const STATE_WORDS: usize = 4;

// The first 16 entries are the normal F-round K values. Later groups are
// deltas that turn the already-cached message+constant vectors into the G, H
// and I constants required by the next round family. This is the same table as
// the CC0 upstream AVX-512VL implementation.
const MD5_CONSTANTS: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0x124c_2332,
    0x0d56_6e0c,
    0xd8cf_331d,
    0x3317_3e99,
    0xf257_ec19,
    0x8ea7_4a33,
    0x1810_6d2d,
    0x6a28_6dd8,
    0xdbd9_7c15,
    0x969c_d637,
    0x0244_b8a2,
    0x9d01_8293,
    0x219a_3b68,
    0xac4b_7772,
    0x1cbd_c448,
    0x8eed_de60,
    0x00ea_6050,
    0xaea0_c4e2,
    0xc7bc_b26d,
    0xe01a_22fe,
    0x640a_d3e1,
    0x29cb_28e5,
    0x4447_69c5,
    0x8f4c_4887,
    0x4217_e194,
    0xb7f3_0253,
    0xbc7b_a81d,
    0x473f_06d1,
    0x59b1_4d5b,
    0x7eb7_95c1,
    0x3aae_3036,
    0x4700_9677,
    0x0987_fa4a,
    0xe0c5_738d,
    0x662b_7c56,
    0xba1d_9c0d,
    0xab74_aed9,
    0xfc99_66f7,
    0x9e79_260f,
    0x4c6f_b437,
    0xe836_87ce,
    0x11b2_0358,
    0x4130_380d,
    0x4f9d_9113,
    0x7e7f_bfde,
    0x256c_92db,
    0xadae_eb9b,
    0xde8a_69e8,
];

#[inline]
pub(crate) fn is_supported() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(all(target_feature = "avx512f", target_feature = "avx512vl"))
    }
}

static PREFERRED_CACHE: AtomicU8 = AtomicU8::new(0);

#[inline]
pub(crate) fn is_preferred() -> bool {
    match PREFERRED_CACHE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let preferred = is_supported() && is_genuine_intel();
            PREFERRED_CACHE.store(if preferred { 2 } else { 1 }, Ordering::Relaxed);
            preferred
        }
    }
}

#[inline]
fn is_genuine_intel() -> bool {
    // CPUID leaf 0 is available on every x86-64 CPU.
    let vendor = core::arch::x86_64::__cpuid(0);
    vendor.ebx == 0x756e_6547 && vendor.edx == 0x4965_6e69 && vendor.ecx == 0x6c65_746e
}

#[inline(always)]
fn load_message_group(block: &[u8; BLOCK_SIZE], word: usize) -> __m128i {
    debug_assert!(matches!(word, 0 | 4 | 8 | 12));
    // SAFETY: each load starts at one of four 16-byte group boundaries inside
    // the 64-byte block; unaligned loads are explicitly supported.
    unsafe { _mm_loadu_si128(block.as_ptr().add(word * 4).cast::<__m128i>()) }
}

#[inline(always)]
fn load_constant_group(index: usize) -> __m128i {
    debug_assert!(index <= 60 && index.is_multiple_of(4));
    // SAFETY: four u32 constants beginning at `index` are in the table, and
    // `_mm_loadu_si128` does not impose an alignment requirement.
    unsafe { _mm_loadu_si128(MD5_CONSTANTS.as_ptr().add(index).cast::<__m128i>()) }
}

macro_rules! round_x {
    ($truth:expr, $ia:expr, $a:ident, $b:expr, $c:expr, $input:expr, $shift:expr, $tmp2:ident) => {{
        $a = _mm_add_epi32($ia, $input);
        $tmp2 = _mm_ternarylogic_epi32::<$truth>($tmp2, $c, $b);
        $a = _mm_add_epi32($a, $tmp2);
        $a = _mm_rol_epi32::<$shift>($a);
        $tmp2 = $c;
        $a = _mm_add_epi32($a, $b);
    }};
}

macro_rules! round_h {
    ($a:ident, $b:expr, $input:expr, $shift:expr, $tmp1:ident, $tmp2:ident) => {{
        $tmp1 = _mm_add_epi32($a, $input);
        // `tmp2` carries the previous H result. XORing it with the old target
        // state and the new B state produces the next H input pair for free.
        $tmp2 = _mm_ternarylogic_epi32::<0x96>($tmp2, $a, $b);
        $a = _mm_add_epi32($tmp1, $tmp2);
        $a = _mm_rol_epi32::<$shift>($a);
        $a = _mm_add_epi32($a, $b);
    }};
}

macro_rules! rf4_first {
    ($v:expr, $a:ident, $b:ident, $c:ident, $d:ident, $ia:expr, $ib:expr, $ic:expr, $id:expr, $tmp1:ident, $tmp2:ident) => {{
        let v = $v;
        round_x!(0xd8, $ia, $a, $ib, $ic, v, 7, $tmp2);
        $tmp1 = _mm_srli_epi64::<32>(v);
        round_x!(0xd8, $id, $d, $a, $ib, $tmp1, 12, $tmp2);
        $tmp1 = _mm_unpackhi_epi64(v, v);
        round_x!(0xd8, $ic, $c, $d, $a, $tmp1, 17, $tmp2);
        $tmp1 = _mm_srli_epi64::<32>($tmp1);
        round_x!(0xd8, $ib, $b, $c, $d, $tmp1, 22, $tmp2);
    }};
}

macro_rules! rf4 {
    ($v:expr, $a:ident, $b:ident, $c:ident, $d:ident, $tmp1:ident, $tmp2:ident) => {{
        let v = $v;
        round_x!(0xd8, $a, $a, $b, $c, v, 7, $tmp2);
        $tmp1 = _mm_srli_epi64::<32>(v);
        round_x!(0xd8, $d, $d, $a, $b, $tmp1, 12, $tmp2);
        $tmp1 = _mm_unpackhi_epi64(v, v);
        round_x!(0xd8, $c, $c, $d, $a, $tmp1, 17, $tmp2);
        $tmp1 = _mm_srli_epi64::<32>($tmp1);
        round_x!(0xd8, $b, $b, $c, $d, $tmp1, 22, $tmp2);
    }};
}

macro_rules! rg4 {
    ($rs:expr, $r1:expr, $r2:expr, $a:ident, $b:ident, $c:ident, $d:ident, $tmp1:ident, $tmp2:ident) => {{
        let rs = $rs;
        let r1 = $r1;
        let r2 = $r2;
        $tmp1 = _mm_srli_epi64::<32>(rs);
        round_x!(0xac, $a, $a, $b, $c, $tmp1, 5, $tmp2);
        $tmp1 = _mm_unpackhi_epi64(r1, r1);
        round_x!(0xac, $d, $d, $a, $b, $tmp1, 9, $tmp2);
        $tmp1 = _mm_srli_si128::<12>(r2);
        round_x!(0xac, $c, $c, $d, $a, $tmp1, 14, $tmp2);
        round_x!(0xac, $b, $b, $c, $d, rs, 20, $tmp2);
    }};
}

macro_rules! rh4 {
    ($r1:expr, $rs:expr, $r2:expr, $a:ident, $b:ident, $c:ident, $d:ident, $tmp1:ident, $tmp2:ident) => {{
        let r1 = $r1;
        let rs = $rs;
        let r2 = $r2;
        $tmp1 = _mm_srli_epi64::<32>(r1);
        round_h!($a, $b, $tmp1, 4, $tmp1, $tmp2);
        round_h!($d, $a, rs, 11, $tmp1, $tmp2);
        $tmp1 = _mm_srli_si128::<12>(rs);
        round_h!($c, $d, $tmp1, 16, $tmp1, $tmp2);
        $tmp1 = _mm_unpackhi_epi64(r2, r2);
        round_h!($b, $c, $tmp1, 23, $tmp1, $tmp2);
    }};
}

macro_rules! ri4 {
    ($r1:expr, $rs:expr, $r2:expr, $a:ident, $b:ident, $c:ident, $d:ident, $tmp1:ident, $tmp2:ident) => {{
        let r1 = $r1;
        let rs = $rs;
        let r2 = $r2;
        round_x!(0x63, $a, $a, $b, $c, r1, 6, $tmp2);
        $tmp1 = _mm_srli_si128::<12>(rs);
        round_x!(0x63, $d, $d, $a, $b, $tmp1, 10, $tmp2);
        $tmp1 = _mm_unpackhi_epi64(r2, r2);
        round_x!(0x63, $c, $c, $d, $a, $tmp1, 15, $tmp2);
        $tmp1 = _mm_srli_epi64::<32>(rs);
        round_x!(0x63, $b, $b, $c, $d, $tmp1, 21, $tmp2);
    }};
}

/// Compress one block using XMM registers plus AVX-512F/VL instructions.
///
/// # Safety
///
/// The caller must ensure AVX-512F and AVX-512VL are available and enabled by
/// the operating system.
#[target_feature(enable = "avx512f,avx512vl")]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
#[allow(unused_assignments)]
pub(crate) unsafe fn compress_block(state: &mut [u32; STATE_WORDS], block: &[u8; BLOCK_SIZE]) {
    let ia = _mm_cvtsi32_si128(state[0] as i32);
    let ib = _mm_cvtsi32_si128(state[1] as i32);
    let ic = _mm_cvtsi32_si128(state[2] as i32);
    let id = _mm_cvtsi32_si128(state[3] as i32);

    let mut a: __m128i;
    let mut b: __m128i;
    let mut c: __m128i;
    let mut d: __m128i;
    let mut tmp1;
    let mut tmp2 = id;

    let mut in0 = _mm_add_epi32(load_message_group(block, 0), load_constant_group(0));
    let mut in4 = _mm_add_epi32(load_message_group(block, 4), load_constant_group(4));
    let mut in8 = _mm_add_epi32(load_message_group(block, 8), load_constant_group(8));
    let mut in12 = _mm_add_epi32(load_message_group(block, 12), load_constant_group(12));

    rf4_first!(in0, a, b, c, d, ia, ib, ic, id, tmp1, tmp2);
    rf4!(in4, a, b, c, d, tmp1, tmp2);
    rf4!(in8, a, b, c, d, tmp1, tmp2);
    rf4!(in12, a, b, c, d, tmp1, tmp2);

    in0 = _mm_add_epi32(in0, load_constant_group(16));
    in4 = _mm_add_epi32(in4, load_constant_group(20));
    in8 = _mm_add_epi32(in8, load_constant_group(24));
    rg4!(in0, in4, in8, a, b, c, d, tmp1, tmp2);
    in12 = _mm_add_epi32(in12, load_constant_group(28));
    rg4!(in4, in8, in12, a, b, c, d, tmp1, tmp2);
    rg4!(in8, in12, in0, a, b, c, d, tmp1, tmp2);
    rg4!(in12, in0, in4, a, b, c, d, tmp1, tmp2);

    in4 = _mm_add_epi32(in4, load_constant_group(36));
    tmp1 = _mm_srli_epi64::<32>(in4);
    a = _mm_add_epi32(a, tmp1);
    tmp2 = _mm_ternarylogic_epi32::<0x96>(tmp2, c, b);
    a = _mm_add_epi32(a, tmp2);
    a = _mm_rol_epi32::<4>(a);
    a = _mm_add_epi32(a, b);

    in8 = _mm_add_epi32(in8, load_constant_group(40));
    round_h!(d, a, in8, 11, tmp1, tmp2);
    tmp1 = _mm_srli_si128::<12>(in8);
    in12 = _mm_add_epi32(in12, load_constant_group(44));
    round_h!(c, d, tmp1, 16, tmp1, tmp2);
    tmp1 = _mm_unpackhi_epi64(in12, in12);
    round_h!(b, c, tmp1, 23, tmp1, tmp2);

    in0 = _mm_add_epi32(in0, load_constant_group(32));
    rh4!(in0, in4, in8, a, b, c, d, tmp1, tmp2);
    rh4!(in12, in0, in4, a, b, c, d, tmp1, tmp2);
    rh4!(in8, in12, in0, a, b, c, d, tmp1, tmp2);
    tmp2 = d;

    in0 = _mm_add_epi32(in0, load_constant_group(48));
    in4 = _mm_add_epi32(in4, load_constant_group(52));
    in12 = _mm_add_epi32(in12, load_constant_group(60));
    ri4!(in0, in4, in12, a, b, c, d, tmp1, tmp2);
    in8 = _mm_add_epi32(in8, load_constant_group(56));
    ri4!(in12, in0, in8, a, b, c, d, tmp1, tmp2);
    ri4!(in8, in12, in4, a, b, c, d, tmp1, tmp2);
    ri4!(in4, in8, in0, a, b, c, d, tmp1, tmp2);

    state[0] = _mm_cvtsi128_si32(_mm_add_epi32(a, ia)) as u32;
    state[1] = _mm_cvtsi128_si32(_mm_add_epi32(b, ib)) as u32;
    state[2] = _mm_cvtsi128_si32(_mm_add_epi32(c, ic)) as u32;
    state[3] = _mm_cvtsi128_si32(_mm_add_epi32(d, id)) as u32;
}
