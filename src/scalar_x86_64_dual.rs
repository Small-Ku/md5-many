// Dual-stream x86-64 scalar MD5 compressor.
//
// This interleaves two independent copies of the existing NoLEA + G-shortcut
// schedule round-by-round. MD5 is latency-bound; two GPR state chains let the
// out-of-order core overlap dependency bubbles without SIMD transpose or
// vector-rotate overhead. On the measured AMD Family 19h path, BMI1 ANDN
// also shortens the throughput-bound G/I rounds.

const BLOCK_SIZE: usize = 64;
const STATE_WORDS: usize = 4;

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
#[target_feature(enable = "bmi1")]
pub(crate) unsafe fn compress_block_pair_bmi1(
    states: &mut [[u32; STATE_WORDS]; 2],
    blocks: [&[u8; BLOCK_SIZE]; 2],
) {
    let [mut a0, mut b0, mut c0, mut d0] = states[0];
    let [mut a1, mut b1, mut c1, mut d1] = states[1];
    let initial0 = states[0];
    let initial1 = states[1];
    let m0 = blocks[0].as_ptr();
    let m1 = blocks[1].as_ptr();
    let mut t10: u32;
    let mut t11: u32;
    let mut t2: u32;

    // SAFETY: both pointers reference 64-byte blocks. Every memory operand is
    // a fixed 32-bit read within 0..64; all arithmetic state is register-only.
    unsafe {
        core::arch::asm!(
            "add {a0:e}, dword ptr [{m0}]",
            "mov {t10:e}, {d0:e}",
            "add {a1:e}, dword ptr [{m1}]",
            "mov {t11:e}, {d1:e}",
            "xor {t10:e}, {c0:e}\nadd {a0:e}, -680876936\nand {t10:e}, {b0:e}\nxor {t10:e}, {d0:e}\nadd {d0:e}, dword ptr [{m0} + 4]\nadd {a0:e}, {t10:e}\nrol {a0:e}, 7\nmov {t10:e}, {c0:e}\nadd {a0:e}, {b0:e}",
            "xor {t11:e}, {c1:e}\nadd {a1:e}, -680876936\nand {t11:e}, {b1:e}\nxor {t11:e}, {d1:e}\nadd {d1:e}, dword ptr [{m1} + 4]\nadd {a1:e}, {t11:e}\nrol {a1:e}, 7\nmov {t11:e}, {c1:e}\nadd {a1:e}, {b1:e}",
            "xor {t10:e}, {b0:e}\nadd {d0:e}, -389564586\nand {t10:e}, {a0:e}\nxor {t10:e}, {c0:e}\nadd {c0:e}, dword ptr [{m0} + 8]\nadd {d0:e}, {t10:e}\nrol {d0:e}, 12\nmov {t10:e}, {b0:e}\nadd {d0:e}, {a0:e}",
            "xor {t11:e}, {b1:e}\nadd {d1:e}, -389564586\nand {t11:e}, {a1:e}\nxor {t11:e}, {c1:e}\nadd {c1:e}, dword ptr [{m1} + 8]\nadd {d1:e}, {t11:e}\nrol {d1:e}, 12\nmov {t11:e}, {b1:e}\nadd {d1:e}, {a1:e}",
            "xor {t10:e}, {a0:e}\nadd {c0:e}, 606105819\nand {t10:e}, {d0:e}\nxor {t10:e}, {b0:e}\nadd {b0:e}, dword ptr [{m0} + 12]\nadd {c0:e}, {t10:e}\nrol {c0:e}, 17\nmov {t10:e}, {a0:e}\nadd {c0:e}, {d0:e}",
            "xor {t11:e}, {a1:e}\nadd {c1:e}, 606105819\nand {t11:e}, {d1:e}\nxor {t11:e}, {b1:e}\nadd {b1:e}, dword ptr [{m1} + 12]\nadd {c1:e}, {t11:e}\nrol {c1:e}, 17\nmov {t11:e}, {a1:e}\nadd {c1:e}, {d1:e}",
            "xor {t10:e}, {d0:e}\nadd {b0:e}, -1044525330\nand {t10:e}, {c0:e}\nxor {t10:e}, {a0:e}\nadd {a0:e}, dword ptr [{m0} + 16]\nadd {b0:e}, {t10:e}\nrol {b0:e}, 22\nmov {t10:e}, {d0:e}\nadd {b0:e}, {c0:e}",
            "xor {t11:e}, {d1:e}\nadd {b1:e}, -1044525330\nand {t11:e}, {c1:e}\nxor {t11:e}, {a1:e}\nadd {a1:e}, dword ptr [{m1} + 16]\nadd {b1:e}, {t11:e}\nrol {b1:e}, 22\nmov {t11:e}, {d1:e}\nadd {b1:e}, {c1:e}",
            "xor {t10:e}, {c0:e}\nadd {a0:e}, -176418897\nand {t10:e}, {b0:e}\nxor {t10:e}, {d0:e}\nadd {d0:e}, dword ptr [{m0} + 20]\nadd {a0:e}, {t10:e}\nrol {a0:e}, 7\nmov {t10:e}, {c0:e}\nadd {a0:e}, {b0:e}",
            "xor {t11:e}, {c1:e}\nadd {a1:e}, -176418897\nand {t11:e}, {b1:e}\nxor {t11:e}, {d1:e}\nadd {d1:e}, dword ptr [{m1} + 20]\nadd {a1:e}, {t11:e}\nrol {a1:e}, 7\nmov {t11:e}, {c1:e}\nadd {a1:e}, {b1:e}",
            "xor {t10:e}, {b0:e}\nadd {d0:e}, 1200080426\nand {t10:e}, {a0:e}\nxor {t10:e}, {c0:e}\nadd {c0:e}, dword ptr [{m0} + 24]\nadd {d0:e}, {t10:e}\nrol {d0:e}, 12\nmov {t10:e}, {b0:e}\nadd {d0:e}, {a0:e}",
            "xor {t11:e}, {b1:e}\nadd {d1:e}, 1200080426\nand {t11:e}, {a1:e}\nxor {t11:e}, {c1:e}\nadd {c1:e}, dword ptr [{m1} + 24]\nadd {d1:e}, {t11:e}\nrol {d1:e}, 12\nmov {t11:e}, {b1:e}\nadd {d1:e}, {a1:e}",
            "xor {t10:e}, {a0:e}\nadd {c0:e}, -1473231341\nand {t10:e}, {d0:e}\nxor {t10:e}, {b0:e}\nadd {b0:e}, dword ptr [{m0} + 28]\nadd {c0:e}, {t10:e}\nrol {c0:e}, 17\nmov {t10:e}, {a0:e}\nadd {c0:e}, {d0:e}",
            "xor {t11:e}, {a1:e}\nadd {c1:e}, -1473231341\nand {t11:e}, {d1:e}\nxor {t11:e}, {b1:e}\nadd {b1:e}, dword ptr [{m1} + 28]\nadd {c1:e}, {t11:e}\nrol {c1:e}, 17\nmov {t11:e}, {a1:e}\nadd {c1:e}, {d1:e}",
            "xor {t10:e}, {d0:e}\nadd {b0:e}, -45705983\nand {t10:e}, {c0:e}\nxor {t10:e}, {a0:e}\nadd {a0:e}, dword ptr [{m0} + 32]\nadd {b0:e}, {t10:e}\nrol {b0:e}, 22\nmov {t10:e}, {d0:e}\nadd {b0:e}, {c0:e}",
            "xor {t11:e}, {d1:e}\nadd {b1:e}, -45705983\nand {t11:e}, {c1:e}\nxor {t11:e}, {a1:e}\nadd {a1:e}, dword ptr [{m1} + 32]\nadd {b1:e}, {t11:e}\nrol {b1:e}, 22\nmov {t11:e}, {d1:e}\nadd {b1:e}, {c1:e}",
            "xor {t10:e}, {c0:e}\nadd {a0:e}, 1770035416\nand {t10:e}, {b0:e}\nxor {t10:e}, {d0:e}\nadd {d0:e}, dword ptr [{m0} + 36]\nadd {a0:e}, {t10:e}\nrol {a0:e}, 7\nmov {t10:e}, {c0:e}\nadd {a0:e}, {b0:e}",
            "xor {t11:e}, {c1:e}\nadd {a1:e}, 1770035416\nand {t11:e}, {b1:e}\nxor {t11:e}, {d1:e}\nadd {d1:e}, dword ptr [{m1} + 36]\nadd {a1:e}, {t11:e}\nrol {a1:e}, 7\nmov {t11:e}, {c1:e}\nadd {a1:e}, {b1:e}",
            "xor {t10:e}, {b0:e}\nadd {d0:e}, -1958414417\nand {t10:e}, {a0:e}\nxor {t10:e}, {c0:e}\nadd {c0:e}, dword ptr [{m0} + 40]\nadd {d0:e}, {t10:e}\nrol {d0:e}, 12\nmov {t10:e}, {b0:e}\nadd {d0:e}, {a0:e}",
            "xor {t11:e}, {b1:e}\nadd {d1:e}, -1958414417\nand {t11:e}, {a1:e}\nxor {t11:e}, {c1:e}\nadd {c1:e}, dword ptr [{m1} + 40]\nadd {d1:e}, {t11:e}\nrol {d1:e}, 12\nmov {t11:e}, {b1:e}\nadd {d1:e}, {a1:e}",
            "xor {t10:e}, {a0:e}\nadd {c0:e}, -42063\nand {t10:e}, {d0:e}\nxor {t10:e}, {b0:e}\nadd {b0:e}, dword ptr [{m0} + 44]\nadd {c0:e}, {t10:e}\nrol {c0:e}, 17\nmov {t10:e}, {a0:e}\nadd {c0:e}, {d0:e}",
            "xor {t11:e}, {a1:e}\nadd {c1:e}, -42063\nand {t11:e}, {d1:e}\nxor {t11:e}, {b1:e}\nadd {b1:e}, dword ptr [{m1} + 44]\nadd {c1:e}, {t11:e}\nrol {c1:e}, 17\nmov {t11:e}, {a1:e}\nadd {c1:e}, {d1:e}",
            "xor {t10:e}, {d0:e}\nadd {b0:e}, -1990404162\nand {t10:e}, {c0:e}\nxor {t10:e}, {a0:e}\nadd {a0:e}, dword ptr [{m0} + 48]\nadd {b0:e}, {t10:e}\nrol {b0:e}, 22\nmov {t10:e}, {d0:e}\nadd {b0:e}, {c0:e}",
            "xor {t11:e}, {d1:e}\nadd {b1:e}, -1990404162\nand {t11:e}, {c1:e}\nxor {t11:e}, {a1:e}\nadd {a1:e}, dword ptr [{m1} + 48]\nadd {b1:e}, {t11:e}\nrol {b1:e}, 22\nmov {t11:e}, {d1:e}\nadd {b1:e}, {c1:e}",
            "xor {t10:e}, {c0:e}\nadd {a0:e}, 1804603682\nand {t10:e}, {b0:e}\nxor {t10:e}, {d0:e}\nadd {d0:e}, dword ptr [{m0} + 52]\nadd {a0:e}, {t10:e}\nrol {a0:e}, 7\nmov {t10:e}, {c0:e}\nadd {a0:e}, {b0:e}",
            "xor {t11:e}, {c1:e}\nadd {a1:e}, 1804603682\nand {t11:e}, {b1:e}\nxor {t11:e}, {d1:e}\nadd {d1:e}, dword ptr [{m1} + 52]\nadd {a1:e}, {t11:e}\nrol {a1:e}, 7\nmov {t11:e}, {c1:e}\nadd {a1:e}, {b1:e}",
            "xor {t10:e}, {b0:e}\nadd {d0:e}, -40341101\nand {t10:e}, {a0:e}\nxor {t10:e}, {c0:e}\nadd {c0:e}, dword ptr [{m0} + 56]\nadd {d0:e}, {t10:e}\nrol {d0:e}, 12\nmov {t10:e}, {b0:e}\nadd {d0:e}, {a0:e}",
            "xor {t11:e}, {b1:e}\nadd {d1:e}, -40341101\nand {t11:e}, {a1:e}\nxor {t11:e}, {c1:e}\nadd {c1:e}, dword ptr [{m1} + 56]\nadd {d1:e}, {t11:e}\nrol {d1:e}, 12\nmov {t11:e}, {b1:e}\nadd {d1:e}, {a1:e}",
            "xor {t10:e}, {a0:e}\nadd {c0:e}, -1502002290\nand {t10:e}, {d0:e}\nxor {t10:e}, {b0:e}\nadd {b0:e}, dword ptr [{m0} + 60]\nadd {c0:e}, {t10:e}\nrol {c0:e}, 17\nmov {t10:e}, {a0:e}\nadd {c0:e}, {d0:e}",
            "xor {t11:e}, {a1:e}\nadd {c1:e}, -1502002290\nand {t11:e}, {d1:e}\nxor {t11:e}, {b1:e}\nadd {b1:e}, dword ptr [{m1} + 60]\nadd {c1:e}, {t11:e}\nrol {c1:e}, 17\nmov {t11:e}, {a1:e}\nadd {c1:e}, {d1:e}",
            "xor {t10:e}, {d0:e}\nadd {b0:e}, 1236535329\nand {t10:e}, {c0:e}\nxor {t10:e}, {a0:e}\nadd {a0:e}, dword ptr [{m0} + 4]\nadd {b0:e}, {t10:e}\nrol {b0:e}, 22\nadd {b0:e}, {c0:e}",
            "xor {t11:e}, {d1:e}\nadd {b1:e}, 1236535329\nand {t11:e}, {c1:e}\nxor {t11:e}, {a1:e}\nadd {a1:e}, dword ptr [{m1} + 4]\nadd {b1:e}, {t11:e}\nrol {b1:e}, 22\nadd {b1:e}, {c1:e}",
            "andn {t10:e}, {d0:e}, {c0:e}\nadd {a0:e}, -165796510\nmov {t2:e}, {d0:e}\nadd {d0:e}, dword ptr [{m0} + 24]\nadd {a0:e}, {t10:e}\nand {t2:e}, {b0:e}\nadd {a0:e}, {t2:e}\nrol {a0:e}, 5\nadd {a0:e}, {b0:e}",
            "andn {t11:e}, {d1:e}, {c1:e}\nadd {a1:e}, -165796510\nmov {t2:e}, {d1:e}\nadd {d1:e}, dword ptr [{m1} + 24]\nadd {a1:e}, {t11:e}\nand {t2:e}, {b1:e}\nadd {a1:e}, {t2:e}\nrol {a1:e}, 5\nadd {a1:e}, {b1:e}",
            "andn {t10:e}, {c0:e}, {b0:e}\nadd {d0:e}, -1069501632\nmov {t2:e}, {c0:e}\nadd {c0:e}, dword ptr [{m0} + 44]\nadd {d0:e}, {t10:e}\nand {t2:e}, {a0:e}\nadd {d0:e}, {t2:e}\nrol {d0:e}, 9\nadd {d0:e}, {a0:e}",
            "andn {t11:e}, {c1:e}, {b1:e}\nadd {d1:e}, -1069501632\nmov {t2:e}, {c1:e}\nadd {c1:e}, dword ptr [{m1} + 44]\nadd {d1:e}, {t11:e}\nand {t2:e}, {a1:e}\nadd {d1:e}, {t2:e}\nrol {d1:e}, 9\nadd {d1:e}, {a1:e}",
            "andn {t10:e}, {b0:e}, {a0:e}\nadd {c0:e}, 643717713\nmov {t2:e}, {b0:e}\nadd {b0:e}, dword ptr [{m0}]\nadd {c0:e}, {t10:e}\nand {t2:e}, {d0:e}\nadd {c0:e}, {t2:e}\nrol {c0:e}, 14\nadd {c0:e}, {d0:e}",
            "andn {t11:e}, {b1:e}, {a1:e}\nadd {c1:e}, 643717713\nmov {t2:e}, {b1:e}\nadd {b1:e}, dword ptr [{m1}]\nadd {c1:e}, {t11:e}\nand {t2:e}, {d1:e}\nadd {c1:e}, {t2:e}\nrol {c1:e}, 14\nadd {c1:e}, {d1:e}",
            "andn {t10:e}, {a0:e}, {d0:e}\nadd {b0:e}, -373897302\nmov {t2:e}, {a0:e}\nadd {a0:e}, dword ptr [{m0} + 20]\nadd {b0:e}, {t10:e}\nand {t2:e}, {c0:e}\nadd {b0:e}, {t2:e}\nrol {b0:e}, 20\nadd {b0:e}, {c0:e}",
            "andn {t11:e}, {a1:e}, {d1:e}\nadd {b1:e}, -373897302\nmov {t2:e}, {a1:e}\nadd {a1:e}, dword ptr [{m1} + 20]\nadd {b1:e}, {t11:e}\nand {t2:e}, {c1:e}\nadd {b1:e}, {t2:e}\nrol {b1:e}, 20\nadd {b1:e}, {c1:e}",
            "andn {t10:e}, {d0:e}, {c0:e}\nadd {a0:e}, -701558691\nmov {t2:e}, {d0:e}\nadd {d0:e}, dword ptr [{m0} + 40]\nadd {a0:e}, {t10:e}\nand {t2:e}, {b0:e}\nadd {a0:e}, {t2:e}\nrol {a0:e}, 5\nadd {a0:e}, {b0:e}",
            "andn {t11:e}, {d1:e}, {c1:e}\nadd {a1:e}, -701558691\nmov {t2:e}, {d1:e}\nadd {d1:e}, dword ptr [{m1} + 40]\nadd {a1:e}, {t11:e}\nand {t2:e}, {b1:e}\nadd {a1:e}, {t2:e}\nrol {a1:e}, 5\nadd {a1:e}, {b1:e}",
            "andn {t10:e}, {c0:e}, {b0:e}\nadd {d0:e}, 38016083\nmov {t2:e}, {c0:e}\nadd {c0:e}, dword ptr [{m0} + 60]\nadd {d0:e}, {t10:e}\nand {t2:e}, {a0:e}\nadd {d0:e}, {t2:e}\nrol {d0:e}, 9\nadd {d0:e}, {a0:e}",
            "andn {t11:e}, {c1:e}, {b1:e}\nadd {d1:e}, 38016083\nmov {t2:e}, {c1:e}\nadd {c1:e}, dword ptr [{m1} + 60]\nadd {d1:e}, {t11:e}\nand {t2:e}, {a1:e}\nadd {d1:e}, {t2:e}\nrol {d1:e}, 9\nadd {d1:e}, {a1:e}",
            "andn {t10:e}, {b0:e}, {a0:e}\nadd {c0:e}, -660478335\nmov {t2:e}, {b0:e}\nadd {b0:e}, dword ptr [{m0} + 16]\nadd {c0:e}, {t10:e}\nand {t2:e}, {d0:e}\nadd {c0:e}, {t2:e}\nrol {c0:e}, 14\nadd {c0:e}, {d0:e}",
            "andn {t11:e}, {b1:e}, {a1:e}\nadd {c1:e}, -660478335\nmov {t2:e}, {b1:e}\nadd {b1:e}, dword ptr [{m1} + 16]\nadd {c1:e}, {t11:e}\nand {t2:e}, {d1:e}\nadd {c1:e}, {t2:e}\nrol {c1:e}, 14\nadd {c1:e}, {d1:e}",
            "andn {t10:e}, {a0:e}, {d0:e}\nadd {b0:e}, -405537848\nmov {t2:e}, {a0:e}\nadd {a0:e}, dword ptr [{m0} + 36]\nadd {b0:e}, {t10:e}\nand {t2:e}, {c0:e}\nadd {b0:e}, {t2:e}\nrol {b0:e}, 20\nadd {b0:e}, {c0:e}",
            "andn {t11:e}, {a1:e}, {d1:e}\nadd {b1:e}, -405537848\nmov {t2:e}, {a1:e}\nadd {a1:e}, dword ptr [{m1} + 36]\nadd {b1:e}, {t11:e}\nand {t2:e}, {c1:e}\nadd {b1:e}, {t2:e}\nrol {b1:e}, 20\nadd {b1:e}, {c1:e}",
            "andn {t10:e}, {d0:e}, {c0:e}\nadd {a0:e}, 568446438\nmov {t2:e}, {d0:e}\nadd {d0:e}, dword ptr [{m0} + 56]\nadd {a0:e}, {t10:e}\nand {t2:e}, {b0:e}\nadd {a0:e}, {t2:e}\nrol {a0:e}, 5\nadd {a0:e}, {b0:e}",
            "andn {t11:e}, {d1:e}, {c1:e}\nadd {a1:e}, 568446438\nmov {t2:e}, {d1:e}\nadd {d1:e}, dword ptr [{m1} + 56]\nadd {a1:e}, {t11:e}\nand {t2:e}, {b1:e}\nadd {a1:e}, {t2:e}\nrol {a1:e}, 5\nadd {a1:e}, {b1:e}",
            "andn {t10:e}, {c0:e}, {b0:e}\nadd {d0:e}, -1019803690\nmov {t2:e}, {c0:e}\nadd {c0:e}, dword ptr [{m0} + 12]\nadd {d0:e}, {t10:e}\nand {t2:e}, {a0:e}\nadd {d0:e}, {t2:e}\nrol {d0:e}, 9\nadd {d0:e}, {a0:e}",
            "andn {t11:e}, {c1:e}, {b1:e}\nadd {d1:e}, -1019803690\nmov {t2:e}, {c1:e}\nadd {c1:e}, dword ptr [{m1} + 12]\nadd {d1:e}, {t11:e}\nand {t2:e}, {a1:e}\nadd {d1:e}, {t2:e}\nrol {d1:e}, 9\nadd {d1:e}, {a1:e}",
            "andn {t10:e}, {b0:e}, {a0:e}\nadd {c0:e}, -187363961\nmov {t2:e}, {b0:e}\nadd {b0:e}, dword ptr [{m0} + 32]\nadd {c0:e}, {t10:e}\nand {t2:e}, {d0:e}\nadd {c0:e}, {t2:e}\nrol {c0:e}, 14\nadd {c0:e}, {d0:e}",
            "andn {t11:e}, {b1:e}, {a1:e}\nadd {c1:e}, -187363961\nmov {t2:e}, {b1:e}\nadd {b1:e}, dword ptr [{m1} + 32]\nadd {c1:e}, {t11:e}\nand {t2:e}, {d1:e}\nadd {c1:e}, {t2:e}\nrol {c1:e}, 14\nadd {c1:e}, {d1:e}",
            "andn {t10:e}, {a0:e}, {d0:e}\nadd {b0:e}, 1163531501\nmov {t2:e}, {a0:e}\nadd {a0:e}, dword ptr [{m0} + 52]\nadd {b0:e}, {t10:e}\nand {t2:e}, {c0:e}\nadd {b0:e}, {t2:e}\nrol {b0:e}, 20\nadd {b0:e}, {c0:e}",
            "andn {t11:e}, {a1:e}, {d1:e}\nadd {b1:e}, 1163531501\nmov {t2:e}, {a1:e}\nadd {a1:e}, dword ptr [{m1} + 52]\nadd {b1:e}, {t11:e}\nand {t2:e}, {c1:e}\nadd {b1:e}, {t2:e}\nrol {b1:e}, 20\nadd {b1:e}, {c1:e}",
            "andn {t10:e}, {d0:e}, {c0:e}\nadd {a0:e}, -1444681467\nmov {t2:e}, {d0:e}\nadd {d0:e}, dword ptr [{m0} + 8]\nadd {a0:e}, {t10:e}\nand {t2:e}, {b0:e}\nadd {a0:e}, {t2:e}\nrol {a0:e}, 5\nadd {a0:e}, {b0:e}",
            "andn {t11:e}, {d1:e}, {c1:e}\nadd {a1:e}, -1444681467\nmov {t2:e}, {d1:e}\nadd {d1:e}, dword ptr [{m1} + 8]\nadd {a1:e}, {t11:e}\nand {t2:e}, {b1:e}\nadd {a1:e}, {t2:e}\nrol {a1:e}, 5\nadd {a1:e}, {b1:e}",
            "andn {t10:e}, {c0:e}, {b0:e}\nadd {d0:e}, -51403784\nmov {t2:e}, {c0:e}\nadd {c0:e}, dword ptr [{m0} + 28]\nadd {d0:e}, {t10:e}\nand {t2:e}, {a0:e}\nadd {d0:e}, {t2:e}\nrol {d0:e}, 9\nadd {d0:e}, {a0:e}",
            "andn {t11:e}, {c1:e}, {b1:e}\nadd {d1:e}, -51403784\nmov {t2:e}, {c1:e}\nadd {c1:e}, dword ptr [{m1} + 28]\nadd {d1:e}, {t11:e}\nand {t2:e}, {a1:e}\nadd {d1:e}, {t2:e}\nrol {d1:e}, 9\nadd {d1:e}, {a1:e}",
            "andn {t10:e}, {b0:e}, {a0:e}\nadd {c0:e}, 1735328473\nmov {t2:e}, {b0:e}\nadd {b0:e}, dword ptr [{m0} + 48]\nadd {c0:e}, {t10:e}\nand {t2:e}, {d0:e}\nadd {c0:e}, {t2:e}\nrol {c0:e}, 14\nadd {c0:e}, {d0:e}",
            "andn {t11:e}, {b1:e}, {a1:e}\nadd {c1:e}, 1735328473\nmov {t2:e}, {b1:e}\nadd {b1:e}, dword ptr [{m1} + 48]\nadd {c1:e}, {t11:e}\nand {t2:e}, {d1:e}\nadd {c1:e}, {t2:e}\nrol {c1:e}, 14\nadd {c1:e}, {d1:e}",
            "andn {t10:e}, {a0:e}, {d0:e}\nadd {b0:e}, -1926607734\nmov {t2:e}, {a0:e}\nadd {a0:e}, dword ptr [{m0} + 20]\nadd {b0:e}, {t10:e}\nand {t2:e}, {c0:e}\nadd {b0:e}, {t2:e}\nrol {b0:e}, 20\nmov {t10:e}, {d0:e}\nadd {b0:e}, {c0:e}",
            "andn {t11:e}, {a1:e}, {d1:e}\nadd {b1:e}, -1926607734\nmov {t2:e}, {a1:e}\nadd {a1:e}, dword ptr [{m1} + 20]\nadd {b1:e}, {t11:e}\nand {t2:e}, {c1:e}\nadd {b1:e}, {t2:e}\nrol {b1:e}, 20\nmov {t11:e}, {d1:e}\nadd {b1:e}, {c1:e}",
            "xor {t10:e}, {c0:e}\nadd {a0:e}, -378558\nadd {d0:e}, dword ptr [{m0} + 32]\nxor {t10:e}, {b0:e}\nadd {a0:e}, {t10:e}\nrol {a0:e}, 4\nmov {t10:e}, {c0:e}\nadd {a0:e}, {b0:e}",
            "xor {t11:e}, {c1:e}\nadd {a1:e}, -378558\nadd {d1:e}, dword ptr [{m1} + 32]\nxor {t11:e}, {b1:e}\nadd {a1:e}, {t11:e}\nrol {a1:e}, 4\nmov {t11:e}, {c1:e}\nadd {a1:e}, {b1:e}",
            "xor {t10:e}, {b0:e}\nadd {d0:e}, -2022574463\nadd {c0:e}, dword ptr [{m0} + 44]\nxor {t10:e}, {a0:e}\nadd {d0:e}, {t10:e}\nrol {d0:e}, 11\nmov {t10:e}, {b0:e}\nadd {d0:e}, {a0:e}",
            "xor {t11:e}, {b1:e}\nadd {d1:e}, -2022574463\nadd {c1:e}, dword ptr [{m1} + 44]\nxor {t11:e}, {a1:e}\nadd {d1:e}, {t11:e}\nrol {d1:e}, 11\nmov {t11:e}, {b1:e}\nadd {d1:e}, {a1:e}",
            "xor {t10:e}, {a0:e}\nadd {c0:e}, 1839030562\nadd {b0:e}, dword ptr [{m0} + 56]\nxor {t10:e}, {d0:e}\nadd {c0:e}, {t10:e}\nrol {c0:e}, 16\nmov {t10:e}, {a0:e}\nadd {c0:e}, {d0:e}",
            "xor {t11:e}, {a1:e}\nadd {c1:e}, 1839030562\nadd {b1:e}, dword ptr [{m1} + 56]\nxor {t11:e}, {d1:e}\nadd {c1:e}, {t11:e}\nrol {c1:e}, 16\nmov {t11:e}, {a1:e}\nadd {c1:e}, {d1:e}",
            "xor {t10:e}, {d0:e}\nadd {b0:e}, -35309556\nadd {a0:e}, dword ptr [{m0} + 4]\nxor {t10:e}, {c0:e}\nadd {b0:e}, {t10:e}\nrol {b0:e}, 23\nmov {t10:e}, {d0:e}\nadd {b0:e}, {c0:e}",
            "xor {t11:e}, {d1:e}\nadd {b1:e}, -35309556\nadd {a1:e}, dword ptr [{m1} + 4]\nxor {t11:e}, {c1:e}\nadd {b1:e}, {t11:e}\nrol {b1:e}, 23\nmov {t11:e}, {d1:e}\nadd {b1:e}, {c1:e}",
            "xor {t10:e}, {c0:e}\nadd {a0:e}, -1530992060\nadd {d0:e}, dword ptr [{m0} + 16]\nxor {t10:e}, {b0:e}\nadd {a0:e}, {t10:e}\nrol {a0:e}, 4\nmov {t10:e}, {c0:e}\nadd {a0:e}, {b0:e}",
            "xor {t11:e}, {c1:e}\nadd {a1:e}, -1530992060\nadd {d1:e}, dword ptr [{m1} + 16]\nxor {t11:e}, {b1:e}\nadd {a1:e}, {t11:e}\nrol {a1:e}, 4\nmov {t11:e}, {c1:e}\nadd {a1:e}, {b1:e}",
            "xor {t10:e}, {b0:e}\nadd {d0:e}, 1272893353\nadd {c0:e}, dword ptr [{m0} + 28]\nxor {t10:e}, {a0:e}\nadd {d0:e}, {t10:e}\nrol {d0:e}, 11\nmov {t10:e}, {b0:e}\nadd {d0:e}, {a0:e}",
            "xor {t11:e}, {b1:e}\nadd {d1:e}, 1272893353\nadd {c1:e}, dword ptr [{m1} + 28]\nxor {t11:e}, {a1:e}\nadd {d1:e}, {t11:e}\nrol {d1:e}, 11\nmov {t11:e}, {b1:e}\nadd {d1:e}, {a1:e}",
            "xor {t10:e}, {a0:e}\nadd {c0:e}, -155497632\nadd {b0:e}, dword ptr [{m0} + 40]\nxor {t10:e}, {d0:e}\nadd {c0:e}, {t10:e}\nrol {c0:e}, 16\nmov {t10:e}, {a0:e}\nadd {c0:e}, {d0:e}",
            "xor {t11:e}, {a1:e}\nadd {c1:e}, -155497632\nadd {b1:e}, dword ptr [{m1} + 40]\nxor {t11:e}, {d1:e}\nadd {c1:e}, {t11:e}\nrol {c1:e}, 16\nmov {t11:e}, {a1:e}\nadd {c1:e}, {d1:e}",
            "xor {t10:e}, {d0:e}\nadd {b0:e}, -1094730640\nadd {a0:e}, dword ptr [{m0} + 52]\nxor {t10:e}, {c0:e}\nadd {b0:e}, {t10:e}\nrol {b0:e}, 23\nmov {t10:e}, {d0:e}\nadd {b0:e}, {c0:e}",
            "xor {t11:e}, {d1:e}\nadd {b1:e}, -1094730640\nadd {a1:e}, dword ptr [{m1} + 52]\nxor {t11:e}, {c1:e}\nadd {b1:e}, {t11:e}\nrol {b1:e}, 23\nmov {t11:e}, {d1:e}\nadd {b1:e}, {c1:e}",
            "xor {t10:e}, {c0:e}\nadd {a0:e}, 681279174\nadd {d0:e}, dword ptr [{m0}]\nxor {t10:e}, {b0:e}\nadd {a0:e}, {t10:e}\nrol {a0:e}, 4\nmov {t10:e}, {c0:e}\nadd {a0:e}, {b0:e}",
            "xor {t11:e}, {c1:e}\nadd {a1:e}, 681279174\nadd {d1:e}, dword ptr [{m1}]\nxor {t11:e}, {b1:e}\nadd {a1:e}, {t11:e}\nrol {a1:e}, 4\nmov {t11:e}, {c1:e}\nadd {a1:e}, {b1:e}",
            "xor {t10:e}, {b0:e}\nadd {d0:e}, -358537222\nadd {c0:e}, dword ptr [{m0} + 12]\nxor {t10:e}, {a0:e}\nadd {d0:e}, {t10:e}\nrol {d0:e}, 11\nmov {t10:e}, {b0:e}\nadd {d0:e}, {a0:e}",
            "xor {t11:e}, {b1:e}\nadd {d1:e}, -358537222\nadd {c1:e}, dword ptr [{m1} + 12]\nxor {t11:e}, {a1:e}\nadd {d1:e}, {t11:e}\nrol {d1:e}, 11\nmov {t11:e}, {b1:e}\nadd {d1:e}, {a1:e}",
            "xor {t10:e}, {a0:e}\nadd {c0:e}, -722521979\nadd {b0:e}, dword ptr [{m0} + 24]\nxor {t10:e}, {d0:e}\nadd {c0:e}, {t10:e}\nrol {c0:e}, 16\nmov {t10:e}, {a0:e}\nadd {c0:e}, {d0:e}",
            "xor {t11:e}, {a1:e}\nadd {c1:e}, -722521979\nadd {b1:e}, dword ptr [{m1} + 24]\nxor {t11:e}, {d1:e}\nadd {c1:e}, {t11:e}\nrol {c1:e}, 16\nmov {t11:e}, {a1:e}\nadd {c1:e}, {d1:e}",
            "xor {t10:e}, {d0:e}\nadd {b0:e}, 76029189\nadd {a0:e}, dword ptr [{m0} + 36]\nxor {t10:e}, {c0:e}\nadd {b0:e}, {t10:e}\nrol {b0:e}, 23\nmov {t10:e}, {d0:e}\nadd {b0:e}, {c0:e}",
            "xor {t11:e}, {d1:e}\nadd {b1:e}, 76029189\nadd {a1:e}, dword ptr [{m1} + 36]\nxor {t11:e}, {c1:e}\nadd {b1:e}, {t11:e}\nrol {b1:e}, 23\nmov {t11:e}, {d1:e}\nadd {b1:e}, {c1:e}",
            "xor {t10:e}, {c0:e}\nadd {a0:e}, -640364487\nadd {d0:e}, dword ptr [{m0} + 48]\nxor {t10:e}, {b0:e}\nadd {a0:e}, {t10:e}\nrol {a0:e}, 4\nmov {t10:e}, {c0:e}\nadd {a0:e}, {b0:e}",
            "xor {t11:e}, {c1:e}\nadd {a1:e}, -640364487\nadd {d1:e}, dword ptr [{m1} + 48]\nxor {t11:e}, {b1:e}\nadd {a1:e}, {t11:e}\nrol {a1:e}, 4\nmov {t11:e}, {c1:e}\nadd {a1:e}, {b1:e}",
            "xor {t10:e}, {b0:e}\nadd {d0:e}, -421815835\nadd {c0:e}, dword ptr [{m0} + 60]\nxor {t10:e}, {a0:e}\nadd {d0:e}, {t10:e}\nrol {d0:e}, 11\nmov {t10:e}, {b0:e}\nadd {d0:e}, {a0:e}",
            "xor {t11:e}, {b1:e}\nadd {d1:e}, -421815835\nadd {c1:e}, dword ptr [{m1} + 60]\nxor {t11:e}, {a1:e}\nadd {d1:e}, {t11:e}\nrol {d1:e}, 11\nmov {t11:e}, {b1:e}\nadd {d1:e}, {a1:e}",
            "xor {t10:e}, {a0:e}\nadd {c0:e}, 530742520\nadd {b0:e}, dword ptr [{m0} + 8]\nxor {t10:e}, {d0:e}\nadd {c0:e}, {t10:e}\nrol {c0:e}, 16\nmov {t10:e}, {a0:e}\nadd {c0:e}, {d0:e}",
            "xor {t11:e}, {a1:e}\nadd {c1:e}, 530742520\nadd {b1:e}, dword ptr [{m1} + 8]\nxor {t11:e}, {d1:e}\nadd {c1:e}, {t11:e}\nrol {c1:e}, 16\nmov {t11:e}, {a1:e}\nadd {c1:e}, {d1:e}",
            "xor {t10:e}, {d0:e}\nadd {b0:e}, -995338651\nadd {a0:e}, dword ptr [{m0}]\nxor {t10:e}, {c0:e}\nadd {b0:e}, {t10:e}\nrol {b0:e}, 23\nadd {b0:e}, {c0:e}",
            "xor {t11:e}, {d1:e}\nadd {b1:e}, -995338651\nadd {a1:e}, dword ptr [{m1}]\nxor {t11:e}, {c1:e}\nadd {b1:e}, {t11:e}\nrol {b1:e}, 23\nadd {b1:e}, {c1:e}",
            "andn {t10:e}, {b0:e}, {d0:e}\nxor {t10:e}, {c0:e}\nadd {a0:e}, -198630845\nadd {d0:e}, dword ptr [{m0} + 28]\nsub {a0:e}, {t10:e}\nrol {a0:e}, 6\nadd {a0:e}, {b0:e}",
            "andn {t11:e}, {b1:e}, {d1:e}\nxor {t11:e}, {c1:e}\nadd {a1:e}, -198630845\nadd {d1:e}, dword ptr [{m1} + 28]\nsub {a1:e}, {t11:e}\nrol {a1:e}, 6\nadd {a1:e}, {b1:e}",
            "andn {t10:e}, {a0:e}, {c0:e}\nxor {t10:e}, {b0:e}\nadd {d0:e}, 1126891414\nadd {c0:e}, dword ptr [{m0} + 56]\nsub {d0:e}, {t10:e}\nrol {d0:e}, 10\nadd {d0:e}, {a0:e}",
            "andn {t11:e}, {a1:e}, {c1:e}\nxor {t11:e}, {b1:e}\nadd {d1:e}, 1126891414\nadd {c1:e}, dword ptr [{m1} + 56]\nsub {d1:e}, {t11:e}\nrol {d1:e}, 10\nadd {d1:e}, {a1:e}",
            "andn {t10:e}, {d0:e}, {b0:e}\nxor {t10:e}, {a0:e}\nadd {c0:e}, -1416354906\nadd {b0:e}, dword ptr [{m0} + 20]\nsub {c0:e}, {t10:e}\nrol {c0:e}, 15\nadd {c0:e}, {d0:e}",
            "andn {t11:e}, {d1:e}, {b1:e}\nxor {t11:e}, {a1:e}\nadd {c1:e}, -1416354906\nadd {b1:e}, dword ptr [{m1} + 20]\nsub {c1:e}, {t11:e}\nrol {c1:e}, 15\nadd {c1:e}, {d1:e}",
            "andn {t10:e}, {c0:e}, {a0:e}\nxor {t10:e}, {d0:e}\nadd {b0:e}, -57434056\nadd {a0:e}, dword ptr [{m0} + 48]\nsub {b0:e}, {t10:e}\nrol {b0:e}, 21\nadd {b0:e}, {c0:e}",
            "andn {t11:e}, {c1:e}, {a1:e}\nxor {t11:e}, {d1:e}\nadd {b1:e}, -57434056\nadd {a1:e}, dword ptr [{m1} + 48]\nsub {b1:e}, {t11:e}\nrol {b1:e}, 21\nadd {b1:e}, {c1:e}",
            "andn {t10:e}, {b0:e}, {d0:e}\nxor {t10:e}, {c0:e}\nadd {a0:e}, 1700485570\nadd {d0:e}, dword ptr [{m0} + 12]\nsub {a0:e}, {t10:e}\nrol {a0:e}, 6\nadd {a0:e}, {b0:e}",
            "andn {t11:e}, {b1:e}, {d1:e}\nxor {t11:e}, {c1:e}\nadd {a1:e}, 1700485570\nadd {d1:e}, dword ptr [{m1} + 12]\nsub {a1:e}, {t11:e}\nrol {a1:e}, 6\nadd {a1:e}, {b1:e}",
            "andn {t10:e}, {a0:e}, {c0:e}\nxor {t10:e}, {b0:e}\nadd {d0:e}, -1894986607\nadd {c0:e}, dword ptr [{m0} + 40]\nsub {d0:e}, {t10:e}\nrol {d0:e}, 10\nadd {d0:e}, {a0:e}",
            "andn {t11:e}, {a1:e}, {c1:e}\nxor {t11:e}, {b1:e}\nadd {d1:e}, -1894986607\nadd {c1:e}, dword ptr [{m1} + 40]\nsub {d1:e}, {t11:e}\nrol {d1:e}, 10\nadd {d1:e}, {a1:e}",
            "andn {t10:e}, {d0:e}, {b0:e}\nxor {t10:e}, {a0:e}\nadd {c0:e}, -1051524\nadd {b0:e}, dword ptr [{m0} + 4]\nsub {c0:e}, {t10:e}\nrol {c0:e}, 15\nadd {c0:e}, {d0:e}",
            "andn {t11:e}, {d1:e}, {b1:e}\nxor {t11:e}, {a1:e}\nadd {c1:e}, -1051524\nadd {b1:e}, dword ptr [{m1} + 4]\nsub {c1:e}, {t11:e}\nrol {c1:e}, 15\nadd {c1:e}, {d1:e}",
            "andn {t10:e}, {c0:e}, {a0:e}\nxor {t10:e}, {d0:e}\nadd {b0:e}, -2054922800\nadd {a0:e}, dword ptr [{m0} + 32]\nsub {b0:e}, {t10:e}\nrol {b0:e}, 21\nadd {b0:e}, {c0:e}",
            "andn {t11:e}, {c1:e}, {a1:e}\nxor {t11:e}, {d1:e}\nadd {b1:e}, -2054922800\nadd {a1:e}, dword ptr [{m1} + 32]\nsub {b1:e}, {t11:e}\nrol {b1:e}, 21\nadd {b1:e}, {c1:e}",
            "andn {t10:e}, {b0:e}, {d0:e}\nxor {t10:e}, {c0:e}\nadd {a0:e}, 1873313358\nadd {d0:e}, dword ptr [{m0} + 60]\nsub {a0:e}, {t10:e}\nrol {a0:e}, 6\nadd {a0:e}, {b0:e}",
            "andn {t11:e}, {b1:e}, {d1:e}\nxor {t11:e}, {c1:e}\nadd {a1:e}, 1873313358\nadd {d1:e}, dword ptr [{m1} + 60]\nsub {a1:e}, {t11:e}\nrol {a1:e}, 6\nadd {a1:e}, {b1:e}",
            "andn {t10:e}, {a0:e}, {c0:e}\nxor {t10:e}, {b0:e}\nadd {d0:e}, -30611745\nadd {c0:e}, dword ptr [{m0} + 24]\nsub {d0:e}, {t10:e}\nrol {d0:e}, 10\nadd {d0:e}, {a0:e}",
            "andn {t11:e}, {a1:e}, {c1:e}\nxor {t11:e}, {b1:e}\nadd {d1:e}, -30611745\nadd {c1:e}, dword ptr [{m1} + 24]\nsub {d1:e}, {t11:e}\nrol {d1:e}, 10\nadd {d1:e}, {a1:e}",
            "andn {t10:e}, {d0:e}, {b0:e}\nxor {t10:e}, {a0:e}\nadd {c0:e}, -1560198381\nadd {b0:e}, dword ptr [{m0} + 52]\nsub {c0:e}, {t10:e}\nrol {c0:e}, 15\nadd {c0:e}, {d0:e}",
            "andn {t11:e}, {d1:e}, {b1:e}\nxor {t11:e}, {a1:e}\nadd {c1:e}, -1560198381\nadd {b1:e}, dword ptr [{m1} + 52]\nsub {c1:e}, {t11:e}\nrol {c1:e}, 15\nadd {c1:e}, {d1:e}",
            "andn {t10:e}, {c0:e}, {a0:e}\nxor {t10:e}, {d0:e}\nadd {b0:e}, 1309151648\nadd {a0:e}, dword ptr [{m0} + 16]\nsub {b0:e}, {t10:e}\nrol {b0:e}, 21\nadd {b0:e}, {c0:e}",
            "andn {t11:e}, {c1:e}, {a1:e}\nxor {t11:e}, {d1:e}\nadd {b1:e}, 1309151648\nadd {a1:e}, dword ptr [{m1} + 16]\nsub {b1:e}, {t11:e}\nrol {b1:e}, 21\nadd {b1:e}, {c1:e}",
            "andn {t10:e}, {b0:e}, {d0:e}\nxor {t10:e}, {c0:e}\nadd {a0:e}, -145523071\nadd {d0:e}, dword ptr [{m0} + 44]\nsub {a0:e}, {t10:e}\nrol {a0:e}, 6\nadd {a0:e}, {b0:e}",
            "andn {t11:e}, {b1:e}, {d1:e}\nxor {t11:e}, {c1:e}\nadd {a1:e}, -145523071\nadd {d1:e}, dword ptr [{m1} + 44]\nsub {a1:e}, {t11:e}\nrol {a1:e}, 6\nadd {a1:e}, {b1:e}",
            "andn {t10:e}, {a0:e}, {c0:e}\nxor {t10:e}, {b0:e}\nadd {d0:e}, -1120210380\nadd {c0:e}, dword ptr [{m0} + 8]\nsub {d0:e}, {t10:e}\nrol {d0:e}, 10\nadd {d0:e}, {a0:e}",
            "andn {t11:e}, {a1:e}, {c1:e}\nxor {t11:e}, {b1:e}\nadd {d1:e}, -1120210380\nadd {c1:e}, dword ptr [{m1} + 8]\nsub {d1:e}, {t11:e}\nrol {d1:e}, 10\nadd {d1:e}, {a1:e}",
            "andn {t10:e}, {d0:e}, {b0:e}\nxor {t10:e}, {a0:e}\nadd {c0:e}, 718787258\nadd {b0:e}, dword ptr [{m0} + 36]\nsub {c0:e}, {t10:e}\nrol {c0:e}, 15\nadd {c0:e}, {d0:e}",
            "andn {t11:e}, {d1:e}, {b1:e}\nxor {t11:e}, {a1:e}\nadd {c1:e}, 718787258\nadd {b1:e}, dword ptr [{m1} + 36]\nsub {c1:e}, {t11:e}\nrol {c1:e}, 15\nadd {c1:e}, {d1:e}",
            "andn {t10:e}, {c0:e}, {a0:e}\nxor {t10:e}, {d0:e}\nadd {b0:e}, -343485552\nsub {b0:e}, {t10:e}\nrol {b0:e}, 21\nadd {b0:e}, {c0:e}",
            "andn {t11:e}, {c1:e}, {a1:e}\nxor {t11:e}, {d1:e}\nadd {b1:e}, -343485552\nsub {b1:e}, {t11:e}\nrol {b1:e}, 21\nadd {b1:e}, {c1:e}",
            a0 = inout(reg) a0,
            b0 = inout(reg) b0,
            c0 = inout(reg) c0,
            d0 = inout(reg) d0,
            a1 = inout(reg) a1,
            b1 = inout(reg) b1,
            c1 = inout(reg) c1,
            d1 = inout(reg) d1,
            t10 = out(reg) t10,
            t11 = out(reg) t11,
            t2 = out(reg) t2,
            m0 = in(reg) m0,
            m1 = in(reg) m1,
            options(nostack, readonly),
        );
    }
    let _ = (t10, t11, t2);

    states[0][0] = initial0[0].wrapping_add(a0);
    states[0][1] = initial0[1].wrapping_add(b0);
    states[0][2] = initial0[2].wrapping_add(c0);
    states[0][3] = initial0[3].wrapping_add(d0);
    states[1][0] = initial1[0].wrapping_add(a1);
    states[1][1] = initial1[1].wrapping_add(b1);
    states[1][2] = initial1[2].wrapping_add(c1);
    states[1][3] = initial1[3].wrapping_add(d1);
}

#[inline(always)]
fn padded_blocks_for_len(len: usize) -> usize {
    let full_blocks = len / BLOCK_SIZE;
    let tail = len & (BLOCK_SIZE - 1);
    full_blocks + if tail <= 55 { 1 } else { 2 }
}

#[inline(always)]
fn build_padded_block(input: &[u8], padded_blocks: usize, block_index: usize) -> [u8; 64] {
    let mut block = [0u8; BLOCK_SIZE];
    let base = block_index * BLOCK_SIZE;
    if base < input.len() {
        let count = core::cmp::min(BLOCK_SIZE, input.len() - base);
        block[..count].copy_from_slice(&input[base..base + count]);
    }
    if input.len() >= base && input.len() < base + BLOCK_SIZE {
        block[input.len() - base] = 0x80;
    }
    if block_index + 1 == padded_blocks {
        block[56..64].copy_from_slice(&(input.len() as u64).wrapping_mul(8).to_le_bytes());
    }
    block
}

/// Hash exactly two independent messages by pairing every compression block
/// that exists in both streams, then finishing any longer tail scalar.
///
/// # Safety
///
/// The caller must ensure BMI1 is available on the current CPU.
#[inline]
#[target_feature(enable = "bmi1")]
pub(crate) unsafe fn hash_pair_bmi1(inputs: [&[u8]; 2]) -> [[u8; 16]; 2] {
    let full_blocks = [inputs[0].len() / BLOCK_SIZE, inputs[1].len() / BLOCK_SIZE];
    let padded_blocks = [
        padded_blocks_for_len(inputs[0].len()),
        padded_blocks_for_len(inputs[1].len()),
    ];
    let paired_full = core::cmp::min(full_blocks[0], full_blocks[1]);
    let paired_total = core::cmp::min(padded_blocks[0], padded_blocks[1]);
    let mut states = [crate::consts::STATE_INIT; 2];

    for block_index in 0..paired_full {
        let offset = block_index * BLOCK_SIZE;
        let block0: &[u8; BLOCK_SIZE] = inputs[0][offset..offset + BLOCK_SIZE]
            .try_into()
            .expect("full MD5 block");
        let block1: &[u8; BLOCK_SIZE] = inputs[1][offset..offset + BLOCK_SIZE]
            .try_into()
            .expect("full MD5 block");
        unsafe { compress_block_pair_bmi1(&mut states, [block0, block1]) };
    }

    // At most the divergent edge blocks need materialization. For equally
    // sized inputs this is just the one or two MD5 finalization blocks.
    for block_index in paired_full..paired_total {
        let block0 = build_padded_block(inputs[0], padded_blocks[0], block_index);
        let block1 = build_padded_block(inputs[1], padded_blocks[1], block_index);
        unsafe { compress_block_pair_bmi1(&mut states, [&block0, &block1]) };
    }

    for lane in 0..2 {
        for block_index in paired_total..padded_blocks[lane] {
            if block_index < full_blocks[lane] {
                let offset = block_index * BLOCK_SIZE;
                let block: &[u8; BLOCK_SIZE] = inputs[lane][offset..offset + BLOCK_SIZE]
                    .try_into()
                    .expect("full MD5 block");
                crate::scalar_x86_64::compress_block(&mut states[lane], block);
            } else {
                let block = build_padded_block(inputs[lane], padded_blocks[lane], block_index);
                crate::scalar_x86_64::compress_block(&mut states[lane], &block);
            }
        }
    }

    [
        crate::scalar::state_to_bytes(states[0]),
        crate::scalar::state_to_bytes(states[1]),
    ]
}

#[cfg(test)]
mod tests {
    use super::compress_block_pair_bmi1;
    use std::vec;

    #[test]
    fn dual_hash_matches_single_stream_backend_across_boundaries() {
        if !std::is_x86_feature_detected!("bmi1") {
            return;
        }
        const PAIRS: &[(usize, usize)] = &[
            (0, 0),
            (1, 1),
            (55, 55),
            (55, 56),
            (56, 56),
            (63, 64),
            (64, 64),
            (64, 128),
            (119, 120),
            (127, 128),
            (1024, 1088),
            (1024, 4096),
            (1024, 65_536),
            (65_472, 65_536),
        ];

        for &(len0, len1) in PAIRS {
            let input0 = vec![0x39; len0];
            let input1 = vec![0xa7; len1];
            let actual = unsafe { super::hash_pair_bmi1([&input0, &input1]) };
            assert_eq!(
                actual[0],
                crate::scalar::hash(&input0),
                "len0={len0}, len1={len1}, lane=0"
            );
            assert_eq!(
                actual[1],
                crate::scalar::hash(&input1),
                "len0={len0}, len1={len1}, lane=1"
            );
        }
    }

    #[test]
    fn dual_compressor_matches_single_stream_backend() {
        if !std::is_x86_feature_detected!("bmi1") {
            return;
        }
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        for case in 0..256u32 {
            let mut blocks = [[0u8; 64]; 2];
            for block in &mut blocks {
                for byte in block {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *byte = seed as u8;
                }
            }
            let mut states = [
                [
                    0x6745_2301 ^ case,
                    0xefcd_ab89u32.wrapping_add(case),
                    0x98ba_dcfeu32.rotate_left(case & 31),
                    0x1032_5476u32.wrapping_sub(case),
                ],
                [
                    0x7654_3210 ^ case,
                    0x89ab_cdefu32.wrapping_sub(case),
                    0xfedc_ba98u32.rotate_right(case & 31),
                    0x0123_4567u32.wrapping_add(case),
                ],
            ];
            let mut expected = states;
            crate::scalar_x86_64::compress_block(&mut expected[0], &blocks[0]);
            crate::scalar_x86_64::compress_block(&mut expected[1], &blocks[1]);
            unsafe { compress_block_pair_bmi1(&mut states, [&blocks[0], &blocks[1]]) };
            assert_eq!(states, expected, "case={case}");
        }
    }
}
