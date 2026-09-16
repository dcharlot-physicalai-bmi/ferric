//! **The RISC-V ELF32 loader, against a real linker's output and an independent parser's verdict.**
//!
//! The fixture `fixtures/rv32_kernel.elf` is NOT hand-built — hand-building it would let the test keep
//! a copy of whatever the loader believes. It is compiled by LLVM from `fixtures/rv32_kernel.rs.txt`
//! and linked by `rust-lld`:
//!
//!   rustup target add riscv32imac-unknown-none-elf
//!   cargo build --release --target riscv32imac-unknown-none-elf   # 1204 bytes
//!   sha256 3ecdc659db793ef92c4c1c1a8112687e09afe9184b78a53efb13f21fa47eeed3
//!
//! The expected values come from `llvm-readobj --file-headers --program-headers` on that exact file —
//! a parser that shares no code with this one. The kernel source is shaped like a Tensix baby-core
//! kernel (read a mailbox word, transform it, write it back, spin) and deliberately carries three
//! section kinds so the loader meets every case: `.text`, `.rodata`, and a `.bss` whose segment has
//! **memsz 256 and filesz 0** — the zero-fill path a loader gets wrong silently.
//!
//! ⚠ This is NOT a Tenstorrent-toolchain ELF. SFPI's `riscv-tt-elf-g++` is the real producer and no
//! Tenstorrent hardware or toolchain has been available here. What this proves is that the parser
//! agrees with an independent one about a genuine ELF32 RISC-V image; what it does NOT prove is
//! anything about SFPI's particular section layout or relocations.

use ferric_tensor::tenstorrent::elf;

const ELF: &[u8] = include_bytes!("fixtures/rv32_kernel.elf");

/// Every number here was printed by llvm-readobj, not derived by me.
#[test]
fn parses_a_real_rv32_image_exactly_as_llvm_readobj_does() {
    let img = elf::parse(ELF).expect("the fixture must parse");
    assert_eq!(img.entry, 0x11104, "Entry: 0x11104");
    assert_eq!(img.loads.len(), 3, "llvm-readobj lists three PT_LOAD segments");

    // Offset / VirtualAddress / FileSize / MemSize, verbatim from the oracle.
    let want = [(0x0usize, 0x10000u32, 260usize, 260usize),
                (0x104, 0x11104, 52, 52),
                (0x138, 0x12138, 0, 256)];
    for (i, (off, addr, filesz, memsz)) in want.into_iter().enumerate() {
        let l = &img.loads[i];
        assert_eq!((l.file_off, l.addr, l.file_len, l.mem_len), (off, addr, filesz, memsz),
                   "PT_LOAD {i}");
    }

    // ⭐ The .bss segment is the one that matters: 256 bytes of ZERO that exist in no file byte.
    assert_eq!(img.loads[2].zero_len(), 256, "the .bss segment is pure zero-fill");
    assert_eq!(img.loads[2].bytes(ELF).len(), 0, "and it borrows nothing from the file");
    assert_eq!(img.loads[0].bytes(ELF).len(), 260);
    assert_eq!(img.mem_bytes(), 260 + 52 + 256);

    // The entry must land inside a loaded segment, or the core would start on unwritten memory.
    assert!(img.loads.iter().any(|l| img.entry >= l.addr
                && ((img.entry - l.addr) as usize) < l.mem_len),
            "entry 0x{:x} is not inside any loaded segment", img.entry);
}

/// ⭐ The rejections, each by MUTATING THE REAL FILE — so every one of them is reached through the
/// same parse path the good file takes, not through a hand-made buffer that might miss the check.
#[test]
fn it_refuses_every_image_it_cannot_faithfully_load() {
    let bad = |edit: &dyn Fn(&mut Vec<u8>)| -> String {
        let mut v = ELF.to_vec();
        edit(&mut v);
        elf::parse(&v).expect_err("this mutation must be refused")
    };
    assert!(bad(&|v| v[1] = b'X').contains("magic"), "corrupt magic");
    assert!(bad(&|v| v[4] = 2).contains("ELF32"), "EI_CLASS = 64-bit");
    assert!(bad(&|v| v[5] = 2).contains("little-endian"), "EI_DATA = big-endian");
    assert!(bad(&|v| v[18] = 0xF7).contains("EM_RISCV"), "a different machine");
    assert!(bad(&|v| v[42] = 56).contains("e_phentsize"), "64-bit program header size");
    assert!(bad(&|v| v.truncate(40)).contains("truncated"), "shorter than an ELF32 header");
    // ⚠ Program header 0 is PT_PHDR, NOT a load segment — an earlier draft of this test corrupted
    // its p_filesz and the parser "failed to catch" a mutation it is RIGHT to ignore, because those
    // bytes are never read. The first PT_LOAD is header 1. A mutation aimed at the wrong subject
    // proves nothing about the check it was meant to exercise.
    const PH: usize = 32;
    // ⚠ Two DIFFERENT defects live here and an earlier draft conflated them: raising only p_filesz
    // makes filesz > memsz, which a stricter, earlier check catches with its own message. Raise both
    // to reach the end-of-file bound, and assert each defect against the check that owns it.
    assert!(bad(&|v| {
        let phoff = u32::from_le_bytes([v[28], v[29], v[30], v[31]]) as usize;
        let seg = phoff + PH; // header 1 = the first PT_LOAD
        v[seg + 16..seg + 20].copy_from_slice(&0xFFFF_u32.to_le_bytes()); // p_filesz only
    }).contains("p_memsz"), "p_filesz larger than p_memsz");
    assert!(bad(&|v| {
        let phoff = u32::from_le_bytes([v[28], v[29], v[30], v[31]]) as usize;
        let seg = phoff + PH;
        v[seg + 16..seg + 20].copy_from_slice(&0xFFFF_u32.to_le_bytes()); // p_filesz
        v[seg + 20..seg + 24].copy_from_slice(&0xFFFF_u32.to_le_bytes()); // p_memsz too
    }).contains("file is"), "a PT_LOAD segment that runs past EOF");
    // And the same corruption on a NON-loaded header must be ignored, which is the other half of the
    // claim: this loader reads exactly the bytes it writes to a core, and nothing else.
    {
        let mut v = ELF.to_vec();
        let phoff = u32::from_le_bytes([v[28], v[29], v[30], v[31]]) as usize;
        v[phoff + 16..phoff + 20].copy_from_slice(&0xFFFF_u32.to_le_bytes()); // PT_PHDR's p_filesz
        assert!(elf::parse(&v).is_ok(), "a corrupt NON-load header is not this loader's business");
    }
    // ⛔ PT_DYNAMIC must be refused rather than skipped — silently loading an un-relocated image is
    // the exact failure that would present as an unexplainable hang on a core.
    assert!(bad(&|v| {
        let phoff = u32::from_le_bytes([v[28], v[29], v[30], v[31]]) as usize;
        v[phoff..phoff + 4].copy_from_slice(&2u32.to_le_bytes());
    }).contains("PT_DYNAMIC"), "a relocatable image");
    // ⛔ …and PT_DYNAMIC must be refused WHEREVER it sits, not only at index 0.
    assert!(bad(&|v| {
        let phoff = u32::from_le_bytes([v[28], v[29], v[30], v[31]]) as usize;
        v[phoff + 2 * PH..phoff + 2 * PH + 4].copy_from_slice(&2u32.to_le_bytes());
    }).contains("PT_DYNAMIC"), "a relocatable image, later header");

    // ⚠ THE CONTROL. Without it a parse() that returned Err for everything would pass all of the above.
    assert!(elf::parse(ELF).is_ok(), "the unmutated fixture must still parse");
}
