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

/// ⭐⭐ **THE REAL THING: an ELF from Tenstorrent's OWN compiler, for a real Tensix core.**
///
/// The test above proves the parser agrees with an independent parser about a genuine RISC-V image,
/// but it was LLVM's output, and the commit that added it said plainly that it proved nothing about
/// SFPI's layout. This closes that gap. `fixtures/tensix_wh_kernel.elf` was built with
/// **`riscv-tt-elf-g++` 15.1.0 from `tenstorrent/sfpi` 7.77.0**, targeting **`-mcpu=tt-wh`** — the
/// toolchain's own Wormhole multilib, not a generic `rv32`:
///
///   riscv-tt-elf-g++ -mcpu=tt-wh -mabi=ilp32 -O2 -ffreestanding -nostdlib -nostartfiles \
///                    -fno-exceptions -fno-rtti -T kernel.ld -o k_tt-wh.elf kernel.cc
///   4948 bytes, sha256 d9e81e6c1eabd0b5500b6b41d222f76e8f353fc177904fe3fadde35445bec3df
///
/// Source and linker script are beside it (`tensix_kernel.cc.txt`, `tensix_kernel.ld.txt`). Expected
/// values come from **SFPI's own `riscv-tt-elf-readelf`**, not from me. A Blackhole build
/// (`-mcpu=tt-bh`) was produced in the same run and parses the same way; one fixture is committed
/// because two would assert the same property twice.
///
/// ⭐ **This is the premise of the whole tier, demonstrated**: tt-metal JITs kernels at runtime, but
/// the artifact is an ordinary ELF and the compiler is an ordinary cross-compiler. **No Tenstorrent
/// hardware was involved in producing this** — so Ferric can ship tracked ELFs beside their sources
/// exactly as it ships `.ptx` beside `.cu`.
///
/// ⚠ Still unproven, and not claimed: that this kernel RUNS. Executing it needs a card.
#[test]
fn parses_a_kernel_built_by_tenstorrents_own_compiler_for_a_wormhole_core() {
    const WH: &[u8] = include_bytes!("fixtures/tensix_wh_kernel.elf");
    let img = elf::parse(WH).expect("an SFPI-built Tensix kernel must parse");

    // From `riscv-tt-elf-readelf -h`: Entry point address 0x10000, ELF32, little endian, RISC-V.
    assert_eq!(img.entry, 0x10000);
    // From `riscv-tt-elf-readelf -lW`:
    //   LOAD  0x001000 0x00010000 0x00010000 0x0005c 0x0015c RWE 0x1000
    assert_eq!(img.loads.len(), 1, "one LOAD segment");
    let l = &img.loads[0];
    assert_eq!((l.file_off, l.addr, l.file_len, l.mem_len), (0x1000, 0x10000, 0x5c, 0x15c));

    // ⭐ 0x15c - 0x5c = 256 bytes of .bss: the SCRATCH[64] array, which exists in NO file byte.
    // A loader that skips this leaves the core reading the previous kernel's leavings — a bug that
    // reproduces as "works the first time" and is why this fixture carries a .bss at all.
    assert_eq!(l.zero_len(), 256);
    assert_eq!(l.bytes(WH).len(), 0x5c);
    assert_eq!(img.mem_bytes(), 0x15c);

    // The entry is the first byte of the segment: _start leads the image, as a core expects.
    assert_eq!(img.entry, l.addr);

    // ⚠ This fixture's program header 0 is PT_RISCV_ATTRIBUTES with filesz 0x2a and memsz 0 — i.e.
    // filesz > memsz, which for a PT_LOAD is a hard error above. It is skipped, correctly, because
    // the loader reads only what it writes to a core. Recorded because a control aimed at THAT header
    // "passed" and briefly looked like the test was blind; the real control corrupts ph1's p_memsz
    // and does turn this test red.
    assert_eq!(img.loads.len(), 1, "only the PT_LOAD is taken, not the attributes header");
}
