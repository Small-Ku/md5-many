// Direct Rust/x86-64 port of animetosho/md5-optimisation's `md5_block_noleag`
// scheduling (commit 7cd4ad511f8cddbeed584c4087fb9506d94e8b87). Upstream
// is Public Domain / CC0-1.0. This port is generated from the MD5 round schedule
// and the upstream NoLEA + G dependency-shortcut strategy, not from another Rust port.

const BLOCK_SIZE: usize = 64;
const STATE_WORDS: usize = 4;

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
#[inline(always)]
pub(crate) fn compress_block(state: &mut [u32; STATE_WORDS], block: &[u8; BLOCK_SIZE]) {
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let initial = [a, b, c, d];
    let m = block.as_ptr();
    let mut t1: u32;
    let mut t2: u32;

    // SAFETY: the asm only reads the 64-byte block, keeps all state in registers,
    // and uses fixed offsets in 0..64. `out(reg)` prevents scratch registers from
    // overlapping the still-live state/input operands.
    unsafe {
        core::arch::asm!(
            // Seed F1 with M[0]; subsequent rounds preload the next message word.
            "add {a:e}, dword ptr [{m}]",
            "mov {t1:e}, {d:e}",
            // F rounds
            "xor {t1:e}, {c:e}\nadd {a:e}, -680876936\nand {t1:e}, {b:e}\nxor {t1:e}, {d:e}\nadd {d:e}, dword ptr [{m} + 4]\nadd {a:e}, {t1:e}\nrol {a:e}, 7\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "xor {t1:e}, {b:e}\nadd {d:e}, -389564586\nand {t1:e}, {a:e}\nxor {t1:e}, {c:e}\nadd {c:e}, dword ptr [{m} + 8]\nadd {d:e}, {t1:e}\nrol {d:e}, 12\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "xor {t1:e}, {a:e}\nadd {c:e}, 606105819\nand {t1:e}, {d:e}\nxor {t1:e}, {b:e}\nadd {b:e}, dword ptr [{m} + 12]\nadd {c:e}, {t1:e}\nrol {c:e}, 17\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "xor {t1:e}, {d:e}\nadd {b:e}, -1044525330\nand {t1:e}, {c:e}\nxor {t1:e}, {a:e}\nadd {a:e}, dword ptr [{m} + 16]\nadd {b:e}, {t1:e}\nrol {b:e}, 22\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "xor {t1:e}, {c:e}\nadd {a:e}, -176418897\nand {t1:e}, {b:e}\nxor {t1:e}, {d:e}\nadd {d:e}, dword ptr [{m} + 20]\nadd {a:e}, {t1:e}\nrol {a:e}, 7\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "xor {t1:e}, {b:e}\nadd {d:e}, 1200080426\nand {t1:e}, {a:e}\nxor {t1:e}, {c:e}\nadd {c:e}, dword ptr [{m} + 24]\nadd {d:e}, {t1:e}\nrol {d:e}, 12\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "xor {t1:e}, {a:e}\nadd {c:e}, -1473231341\nand {t1:e}, {d:e}\nxor {t1:e}, {b:e}\nadd {b:e}, dword ptr [{m} + 28]\nadd {c:e}, {t1:e}\nrol {c:e}, 17\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "xor {t1:e}, {d:e}\nadd {b:e}, -45705983\nand {t1:e}, {c:e}\nxor {t1:e}, {a:e}\nadd {a:e}, dword ptr [{m} + 32]\nadd {b:e}, {t1:e}\nrol {b:e}, 22\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "xor {t1:e}, {c:e}\nadd {a:e}, 1770035416\nand {t1:e}, {b:e}\nxor {t1:e}, {d:e}\nadd {d:e}, dword ptr [{m} + 36]\nadd {a:e}, {t1:e}\nrol {a:e}, 7\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "xor {t1:e}, {b:e}\nadd {d:e}, -1958414417\nand {t1:e}, {a:e}\nxor {t1:e}, {c:e}\nadd {c:e}, dword ptr [{m} + 40]\nadd {d:e}, {t1:e}\nrol {d:e}, 12\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "xor {t1:e}, {a:e}\nadd {c:e}, -42063\nand {t1:e}, {d:e}\nxor {t1:e}, {b:e}\nadd {b:e}, dword ptr [{m} + 44]\nadd {c:e}, {t1:e}\nrol {c:e}, 17\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "xor {t1:e}, {d:e}\nadd {b:e}, -1990404162\nand {t1:e}, {c:e}\nxor {t1:e}, {a:e}\nadd {a:e}, dword ptr [{m} + 48]\nadd {b:e}, {t1:e}\nrol {b:e}, 22\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "xor {t1:e}, {c:e}\nadd {a:e}, 1804603682\nand {t1:e}, {b:e}\nxor {t1:e}, {d:e}\nadd {d:e}, dword ptr [{m} + 52]\nadd {a:e}, {t1:e}\nrol {a:e}, 7\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "xor {t1:e}, {b:e}\nadd {d:e}, -40341101\nand {t1:e}, {a:e}\nxor {t1:e}, {c:e}\nadd {c:e}, dword ptr [{m} + 56]\nadd {d:e}, {t1:e}\nrol {d:e}, 12\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "xor {t1:e}, {a:e}\nadd {c:e}, -1502002290\nand {t1:e}, {d:e}\nxor {t1:e}, {b:e}\nadd {b:e}, dword ptr [{m} + 60]\nadd {c:e}, {t1:e}\nrol {c:e}, 17\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "xor {t1:e}, {d:e}\nadd {b:e}, 1236535329\nand {t1:e}, {c:e}\nxor {t1:e}, {a:e}\nadd {a:e}, dword ptr [{m} + 4]\nadd {b:e}, {t1:e}\nrol {b:e}, 22\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            // G rounds: dependency-shortened G from the CC0 implementation
            "not {t1:e}\nadd {a:e}, -165796510\nand {t1:e}, {c:e}\nmov {t2:e}, {d:e}\nadd {d:e}, dword ptr [{m} + 24]\nadd {a:e}, {t1:e}\nand {t2:e}, {b:e}\nadd {a:e}, {t2:e}\nrol {a:e}, 5\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "not {t1:e}\nadd {d:e}, -1069501632\nand {t1:e}, {b:e}\nmov {t2:e}, {c:e}\nadd {c:e}, dword ptr [{m} + 44]\nadd {d:e}, {t1:e}\nand {t2:e}, {a:e}\nadd {d:e}, {t2:e}\nrol {d:e}, 9\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "not {t1:e}\nadd {c:e}, 643717713\nand {t1:e}, {a:e}\nmov {t2:e}, {b:e}\nadd {b:e}, dword ptr [{m}]\nadd {c:e}, {t1:e}\nand {t2:e}, {d:e}\nadd {c:e}, {t2:e}\nrol {c:e}, 14\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "not {t1:e}\nadd {b:e}, -373897302\nand {t1:e}, {d:e}\nmov {t2:e}, {a:e}\nadd {a:e}, dword ptr [{m} + 20]\nadd {b:e}, {t1:e}\nand {t2:e}, {c:e}\nadd {b:e}, {t2:e}\nrol {b:e}, 20\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "not {t1:e}\nadd {a:e}, -701558691\nand {t1:e}, {c:e}\nmov {t2:e}, {d:e}\nadd {d:e}, dword ptr [{m} + 40]\nadd {a:e}, {t1:e}\nand {t2:e}, {b:e}\nadd {a:e}, {t2:e}\nrol {a:e}, 5\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "not {t1:e}\nadd {d:e}, 38016083\nand {t1:e}, {b:e}\nmov {t2:e}, {c:e}\nadd {c:e}, dword ptr [{m} + 60]\nadd {d:e}, {t1:e}\nand {t2:e}, {a:e}\nadd {d:e}, {t2:e}\nrol {d:e}, 9\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "not {t1:e}\nadd {c:e}, -660478335\nand {t1:e}, {a:e}\nmov {t2:e}, {b:e}\nadd {b:e}, dword ptr [{m} + 16]\nadd {c:e}, {t1:e}\nand {t2:e}, {d:e}\nadd {c:e}, {t2:e}\nrol {c:e}, 14\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "not {t1:e}\nadd {b:e}, -405537848\nand {t1:e}, {d:e}\nmov {t2:e}, {a:e}\nadd {a:e}, dword ptr [{m} + 36]\nadd {b:e}, {t1:e}\nand {t2:e}, {c:e}\nadd {b:e}, {t2:e}\nrol {b:e}, 20\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "not {t1:e}\nadd {a:e}, 568446438\nand {t1:e}, {c:e}\nmov {t2:e}, {d:e}\nadd {d:e}, dword ptr [{m} + 56]\nadd {a:e}, {t1:e}\nand {t2:e}, {b:e}\nadd {a:e}, {t2:e}\nrol {a:e}, 5\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "not {t1:e}\nadd {d:e}, -1019803690\nand {t1:e}, {b:e}\nmov {t2:e}, {c:e}\nadd {c:e}, dword ptr [{m} + 12]\nadd {d:e}, {t1:e}\nand {t2:e}, {a:e}\nadd {d:e}, {t2:e}\nrol {d:e}, 9\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "not {t1:e}\nadd {c:e}, -187363961\nand {t1:e}, {a:e}\nmov {t2:e}, {b:e}\nadd {b:e}, dword ptr [{m} + 32]\nadd {c:e}, {t1:e}\nand {t2:e}, {d:e}\nadd {c:e}, {t2:e}\nrol {c:e}, 14\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "not {t1:e}\nadd {b:e}, 1163531501\nand {t1:e}, {d:e}\nmov {t2:e}, {a:e}\nadd {a:e}, dword ptr [{m} + 52]\nadd {b:e}, {t1:e}\nand {t2:e}, {c:e}\nadd {b:e}, {t2:e}\nrol {b:e}, 20\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "not {t1:e}\nadd {a:e}, -1444681467\nand {t1:e}, {c:e}\nmov {t2:e}, {d:e}\nadd {d:e}, dword ptr [{m} + 8]\nadd {a:e}, {t1:e}\nand {t2:e}, {b:e}\nadd {a:e}, {t2:e}\nrol {a:e}, 5\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "not {t1:e}\nadd {d:e}, -51403784\nand {t1:e}, {b:e}\nmov {t2:e}, {c:e}\nadd {c:e}, dword ptr [{m} + 28]\nadd {d:e}, {t1:e}\nand {t2:e}, {a:e}\nadd {d:e}, {t2:e}\nrol {d:e}, 9\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "not {t1:e}\nadd {c:e}, 1735328473\nand {t1:e}, {a:e}\nmov {t2:e}, {b:e}\nadd {b:e}, dword ptr [{m} + 48]\nadd {c:e}, {t1:e}\nand {t2:e}, {d:e}\nadd {c:e}, {t2:e}\nrol {c:e}, 14\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "not {t1:e}\nadd {b:e}, -1926607734\nand {t1:e}, {d:e}\nmov {t2:e}, {a:e}\nadd {a:e}, dword ptr [{m} + 20]\nadd {b:e}, {t1:e}\nand {t2:e}, {c:e}\nadd {b:e}, {t2:e}\nrol {b:e}, 20\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            // H rounds
            "xor {t1:e}, {c:e}\nadd {a:e}, -378558\nadd {d:e}, dword ptr [{m} + 32]\nxor {t1:e}, {b:e}\nadd {a:e}, {t1:e}\nrol {a:e}, 4\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "xor {t1:e}, {b:e}\nadd {d:e}, -2022574463\nadd {c:e}, dword ptr [{m} + 44]\nxor {t1:e}, {a:e}\nadd {d:e}, {t1:e}\nrol {d:e}, 11\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "xor {t1:e}, {a:e}\nadd {c:e}, 1839030562\nadd {b:e}, dword ptr [{m} + 56]\nxor {t1:e}, {d:e}\nadd {c:e}, {t1:e}\nrol {c:e}, 16\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "xor {t1:e}, {d:e}\nadd {b:e}, -35309556\nadd {a:e}, dword ptr [{m} + 4]\nxor {t1:e}, {c:e}\nadd {b:e}, {t1:e}\nrol {b:e}, 23\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "xor {t1:e}, {c:e}\nadd {a:e}, -1530992060\nadd {d:e}, dword ptr [{m} + 16]\nxor {t1:e}, {b:e}\nadd {a:e}, {t1:e}\nrol {a:e}, 4\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "xor {t1:e}, {b:e}\nadd {d:e}, 1272893353\nadd {c:e}, dword ptr [{m} + 28]\nxor {t1:e}, {a:e}\nadd {d:e}, {t1:e}\nrol {d:e}, 11\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "xor {t1:e}, {a:e}\nadd {c:e}, -155497632\nadd {b:e}, dword ptr [{m} + 40]\nxor {t1:e}, {d:e}\nadd {c:e}, {t1:e}\nrol {c:e}, 16\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "xor {t1:e}, {d:e}\nadd {b:e}, -1094730640\nadd {a:e}, dword ptr [{m} + 52]\nxor {t1:e}, {c:e}\nadd {b:e}, {t1:e}\nrol {b:e}, 23\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "xor {t1:e}, {c:e}\nadd {a:e}, 681279174\nadd {d:e}, dword ptr [{m}]\nxor {t1:e}, {b:e}\nadd {a:e}, {t1:e}\nrol {a:e}, 4\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "xor {t1:e}, {b:e}\nadd {d:e}, -358537222\nadd {c:e}, dword ptr [{m} + 12]\nxor {t1:e}, {a:e}\nadd {d:e}, {t1:e}\nrol {d:e}, 11\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "xor {t1:e}, {a:e}\nadd {c:e}, -722521979\nadd {b:e}, dword ptr [{m} + 24]\nxor {t1:e}, {d:e}\nadd {c:e}, {t1:e}\nrol {c:e}, 16\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "xor {t1:e}, {d:e}\nadd {b:e}, 76029189\nadd {a:e}, dword ptr [{m} + 36]\nxor {t1:e}, {c:e}\nadd {b:e}, {t1:e}\nrol {b:e}, 23\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "xor {t1:e}, {c:e}\nadd {a:e}, -640364487\nadd {d:e}, dword ptr [{m} + 48]\nxor {t1:e}, {b:e}\nadd {a:e}, {t1:e}\nrol {a:e}, 4\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "xor {t1:e}, {b:e}\nadd {d:e}, -421815835\nadd {c:e}, dword ptr [{m} + 60]\nxor {t1:e}, {a:e}\nadd {d:e}, {t1:e}\nrol {d:e}, 11\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "xor {t1:e}, {a:e}\nadd {c:e}, 530742520\nadd {b:e}, dword ptr [{m} + 8]\nxor {t1:e}, {d:e}\nadd {c:e}, {t1:e}\nrol {c:e}, 16\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "xor {t1:e}, {d:e}\nadd {b:e}, -995338651\nadd {a:e}, dword ptr [{m}]\nxor {t1:e}, {c:e}\nadd {b:e}, {t1:e}\nrol {b:e}, 23\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            // I rounds
            "not {t1:e}\nadd {a:e}, -198630844\nadd {d:e}, dword ptr [{m} + 28]\nor {t1:e}, {b:e}\nxor {t1:e}, {c:e}\nadd {a:e}, {t1:e}\nrol {a:e}, 6\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "not {t1:e}\nadd {d:e}, 1126891415\nadd {c:e}, dword ptr [{m} + 56]\nor {t1:e}, {a:e}\nxor {t1:e}, {b:e}\nadd {d:e}, {t1:e}\nrol {d:e}, 10\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "not {t1:e}\nadd {c:e}, -1416354905\nadd {b:e}, dword ptr [{m} + 20]\nor {t1:e}, {d:e}\nxor {t1:e}, {a:e}\nadd {c:e}, {t1:e}\nrol {c:e}, 15\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "not {t1:e}\nadd {b:e}, -57434055\nadd {a:e}, dword ptr [{m} + 48]\nor {t1:e}, {c:e}\nxor {t1:e}, {d:e}\nadd {b:e}, {t1:e}\nrol {b:e}, 21\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "not {t1:e}\nadd {a:e}, 1700485571\nadd {d:e}, dword ptr [{m} + 12]\nor {t1:e}, {b:e}\nxor {t1:e}, {c:e}\nadd {a:e}, {t1:e}\nrol {a:e}, 6\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "not {t1:e}\nadd {d:e}, -1894986606\nadd {c:e}, dword ptr [{m} + 40]\nor {t1:e}, {a:e}\nxor {t1:e}, {b:e}\nadd {d:e}, {t1:e}\nrol {d:e}, 10\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "not {t1:e}\nadd {c:e}, -1051523\nadd {b:e}, dword ptr [{m} + 4]\nor {t1:e}, {d:e}\nxor {t1:e}, {a:e}\nadd {c:e}, {t1:e}\nrol {c:e}, 15\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "not {t1:e}\nadd {b:e}, -2054922799\nadd {a:e}, dword ptr [{m} + 32]\nor {t1:e}, {c:e}\nxor {t1:e}, {d:e}\nadd {b:e}, {t1:e}\nrol {b:e}, 21\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "not {t1:e}\nadd {a:e}, 1873313359\nadd {d:e}, dword ptr [{m} + 60]\nor {t1:e}, {b:e}\nxor {t1:e}, {c:e}\nadd {a:e}, {t1:e}\nrol {a:e}, 6\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "not {t1:e}\nadd {d:e}, -30611744\nadd {c:e}, dword ptr [{m} + 24]\nor {t1:e}, {a:e}\nxor {t1:e}, {b:e}\nadd {d:e}, {t1:e}\nrol {d:e}, 10\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "not {t1:e}\nadd {c:e}, -1560198380\nadd {b:e}, dword ptr [{m} + 52]\nor {t1:e}, {d:e}\nxor {t1:e}, {a:e}\nadd {c:e}, {t1:e}\nrol {c:e}, 15\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "not {t1:e}\nadd {b:e}, 1309151649\nadd {a:e}, dword ptr [{m} + 16]\nor {t1:e}, {c:e}\nxor {t1:e}, {d:e}\nadd {b:e}, {t1:e}\nrol {b:e}, 21\nmov {t1:e}, {d:e}\nadd {b:e}, {c:e}",
            "not {t1:e}\nadd {a:e}, -145523070\nadd {d:e}, dword ptr [{m} + 44]\nor {t1:e}, {b:e}\nxor {t1:e}, {c:e}\nadd {a:e}, {t1:e}\nrol {a:e}, 6\nmov {t1:e}, {c:e}\nadd {a:e}, {b:e}",
            "not {t1:e}\nadd {d:e}, -1120210379\nadd {c:e}, dword ptr [{m} + 8]\nor {t1:e}, {a:e}\nxor {t1:e}, {b:e}\nadd {d:e}, {t1:e}\nrol {d:e}, 10\nmov {t1:e}, {b:e}\nadd {d:e}, {a:e}",
            "not {t1:e}\nadd {c:e}, 718787259\nadd {b:e}, dword ptr [{m} + 36]\nor {t1:e}, {d:e}\nxor {t1:e}, {a:e}\nadd {c:e}, {t1:e}\nrol {c:e}, 15\nmov {t1:e}, {a:e}\nadd {c:e}, {d:e}",
            "not {t1:e}\nadd {b:e}, -343485551\nor {t1:e}, {c:e}\nxor {t1:e}, {d:e}\nadd {b:e}, {t1:e}\nrol {b:e}, 21\nadd {b:e}, {c:e}",
            a = inout(reg) a,
            b = inout(reg) b,
            c = inout(reg) c,
            d = inout(reg) d,
            t1 = out(reg) t1,
            t2 = out(reg) t2,
            m = in(reg) m,
            options(nostack, readonly),
        );
    }
    let _ = (t1, t2);

    state[0] = initial[0].wrapping_add(a);
    state[1] = initial[1].wrapping_add(b);
    state[2] = initial[2].wrapping_add(c);
    state[3] = initial[3].wrapping_add(d);
}
