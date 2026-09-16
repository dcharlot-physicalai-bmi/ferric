//! **Tenstorrent native tier, host side** — the `tt-kmd` ioctl UAPI, called directly.
//!
//! This is `cuda.rs`'s pattern on a third vendor: talk to the vendor's stable kernel ABI, probe at
//! runtime, and return `None` whenever anything is missing so the portable path stays the fallback.
//!
//! ⛔ **Why not `luwen`** (Tenstorrent's own Rust crate, Apache-2.0, on crates.io). It would work, and
//! it is the reference this module was written against. But it drags in **14 transitive crates**
//! (`prost`, `rust-embed`, `serde_json`, `bincode`, `tracing`, …) for what is, on the wire, a 20-entry
//! ioctl table. Ferric refused `cudarc` for the same reason and hand-declared the CUDA driver API;
//! this is that decision applied consistently. `luwen` remains the SPEC to read — as does tt-kmd's
//! own `ioctl.h`, which is `GPL-2.0-only WITH Linux-syscall-note`, i.e. explicitly callable from
//! non-GPL userspace.
//!
//! ⚠ **Nothing here has run against a Tenstorrent card.** No Wormhole or Blackhole device has been
//! available to this project. Every function returns `Option`/`Result` and the absent-device path IS
//! tested; the present-device path is **unverified** and says so at each call site. The ABI itself is
//! not guesswork — it is checked against the real header by a C compiler, see the tests below.
//!
//! ⭐ **Only the SYSCALLS are linux-gated; the ABI is not.** An earlier draft put the whole module
//! behind `#![cfg(target_os = "linux")]`, which would have hidden every checkable fact — the struct
//! layouts, the ioctl numbers, the argument rules — from the Mac where most commits are made. That is
//! vacuous-test mechanism #84, paid for once already today by a `Send` bound no local run could see.
//! These structs are fixed-size integer aggregates, so their `#[repr(C)]` layout is identical on every
//! mainstream 64-bit target: the layout test is *more* trustworthy for running in both places, not
//! less. `Device` and the `libc` externs stay linux-only, because `/dev/tenstorrent` does.

use std::ffi::c_ulong;
#[cfg(target_os = "linux")]
use std::ffi::{c_char, c_int, c_void, CString};

// ⛔ `ioctl` is variadic in C. Declaring it non-variadically happens to work on x86-64 SysV and is
// exactly the kind of "works on my arch" that this tier exists to avoid — tt hardware also ships on
// aarch64 hosts. Declared variadic, as C declares it.
#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
}
#[cfg(target_os = "linux")]
const O_RDWR: c_int = 2;
#[cfg(target_os = "linux")]
const O_CLOEXEC: c_int = 0o2000000;

/// `_IO(0xFA, nr)` — Linux's `_IOC(dir=_IOC_NONE, type, nr, size=0)` collapses to `(type << 8) | nr`.
/// Every tt-kmd ioctl is `_IO`, none is `_IOR`/`_IOW`: the structs are in/out in one buffer.
const fn io(nr: u32) -> c_ulong {
    ((0xFA_u32 << 8) | nr) as c_ulong
}
pub const GET_DEVICE_INFO: c_ulong = io(0);
pub const GET_HARVESTING: c_ulong = io(1);
pub const QUERY_MAPPINGS: c_ulong = io(2);
pub const ALLOCATE_DMA_BUF: c_ulong = io(3);
pub const FREE_DMA_BUF: c_ulong = io(4);
pub const GET_DRIVER_INFO: c_ulong = io(5);
pub const RESET_DEVICE: c_ulong = io(6);
pub const PIN_PAGES: c_ulong = io(7);
pub const LOCK_CTL: c_ulong = io(8);
pub const MAP_PEER_BAR: c_ulong = io(9);
pub const UNPIN_PAGES: c_ulong = io(10);
pub const ALLOCATE_TLB: c_ulong = io(11);
pub const FREE_TLB: c_ulong = io(12);
pub const CONFIGURE_TLB: c_ulong = io(13);
pub const SET_NOC_CLEANUP: c_ulong = io(14);
pub const SET_POWER_STATE: c_ulong = io(15);
pub const EXPORT_TLB_DMABUF: c_ulong = io(16);
pub const SMC_MSG: c_ulong = io(17);
pub const NOC_READ: c_ulong = io(18);
pub const NOC_WRITE: c_ulong = io(19);

/// `tenstorrent_mapping.mapping_id`. ⚠ These are IDs, not array indices — the header says so twice.
pub const MAPPING_UNUSED: u32 = 0;
pub const MAPPING_RESOURCE0_UC: u32 = 1;
pub const MAPPING_RESOURCE0_WC: u32 = 2;
pub const MAPPING_RESOURCE1_UC: u32 = 3;
pub const MAPPING_RESOURCE1_WC: u32 = 4;
pub const MAPPING_RESOURCE2_UC: u32 = 5;
pub const MAPPING_RESOURCE2_WC: u32 = 6;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DeviceInfoOut {
    pub output_size_bytes: u32,
    pub vendor_id: u16,
    pub device_id: u16,
    pub subsystem_vendor_id: u16,
    pub subsystem_id: u16,
    /// `[0:2]` function, `[3:7]` device, `[8:15]` bus.
    pub bus_dev_fn: u16,
    pub max_dma_buf_size_log2: u16,
    pub pci_domain: u16,
    pub reserved: u16,
}
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DeviceInfo {
    pub in_output_size_bytes: u32,
    pub out: DeviceInfoOut,
}
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DriverInfoOut {
    pub output_size_bytes: u32,
    pub driver_version: u32,
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
    pub reserved0: u8,
}
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DriverInfo {
    pub in_output_size_bytes: u32,
    pub out: DriverInfoOut,
}
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct Mapping {
    pub mapping_id: u32,
    pub reserved: u32,
    pub mapping_base: u64,
    pub mapping_size: u64,
}
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct QueryMappingsIn {
    pub output_mapping_count: u32,
    pub reserved: u32,
}
/// `TENSTORRENT_IOCTL_NOC_READ` / `_NOC_WRITE`. Kernel-mediated single-word access: slower than a
/// mapped TLB window, but it needs no mmap and the header calls it reset-safe, which makes it the
/// right first thing to reach for.
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct NocIo {
    pub argsz: u32,
    pub flags: u32,
    pub x: u16,
    pub y: u16,
    pub noc: u8,
    /// 1, 2, 4 or 8. `addr` must be aligned to it.
    pub width: u8,
    pub reserved0: [u8; 2],
    pub addr: u64,
    pub value: u64,
}

/// One open `/dev/tenstorrent/N`. ⚠ linux only — the device node does not exist elsewhere.
#[cfg(target_os = "linux")]
pub struct Device {
    fd: c_int,
    pub index: u32,
}
#[cfg(target_os = "linux")]
impl Drop for Device {
    fn drop(&mut self) {
        unsafe { close(self.fd) };
    }
}

#[cfg(target_os = "linux")]
impl Device {
    /// Opens `/dev/tenstorrent/{index}`. `None` when the node is absent (no card, or `tt-kmd` not
    /// loaded) — the same probe-and-decline contract as `cuda::driver()`.
    pub fn open(index: u32) -> Option<Device> {
        let path = CString::new(format!("/dev/tenstorrent/{index}")).ok()?;
        let fd = unsafe { open(path.as_ptr(), O_RDWR | O_CLOEXEC) };
        if fd < 0 {
            return None;
        }
        Some(Device { fd, index })
    }

    /// Every `/dev/tenstorrent/N` present, in index order. Empty on a machine with no Tenstorrent
    /// hardware, which is every machine this project has access to today.
    pub fn enumerate() -> Vec<Device> {
        let Ok(dir) = std::fs::read_dir("/dev/tenstorrent") else { return Vec::new() };
        let mut idx: Vec<u32> =
            dir.filter_map(|e| e.ok()?.file_name().to_str()?.parse::<u32>().ok()).collect();
        idx.sort_unstable();
        idx.into_iter().filter_map(Device::open).collect()
    }

    unsafe fn call<T>(&self, request: c_ulong, arg: &mut T) -> std::io::Result<()> {
        if unsafe { ioctl(self.fd, request, arg as *mut T as *mut c_void) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// ⚠ UNVERIFIED against hardware — no card has been available. The ABI is checked, the call is not.
    pub fn device_info(&self) -> std::io::Result<DeviceInfoOut> {
        let mut a = DeviceInfo {
            in_output_size_bytes: std::mem::size_of::<DeviceInfoOut>() as u32,
            ..Default::default()
        };
        unsafe { self.call(GET_DEVICE_INFO, &mut a)? };
        Ok(a.out)
    }

    /// ⚠ UNVERIFIED against hardware. `driver_version` is the IOCTL API version; this module was
    /// written against **2** (`TENSTORRENT_DRIVER_VERSION`).
    pub fn driver_info(&self) -> std::io::Result<DriverInfoOut> {
        let mut a = DriverInfo {
            in_output_size_bytes: std::mem::size_of::<DriverInfoOut>() as u32,
            ..Default::default()
        };
        unsafe { self.call(GET_DRIVER_INFO, &mut a)? };
        Ok(a.out)
    }

    /// The PCI BAR apertures, as `(mapping_id, base, size)`. ⚠ UNVERIFIED against hardware.
    ///
    /// ⛔ Two-pass by necessity: `tenstorrent_query_mappings_out` is a C flexible array member, so the
    /// caller supplies the count and the kernel fills that many. Asking for 0 does not report how many
    /// there are, so we ask for the 6 IDs the header defines and keep the ones that came back used.
    pub fn mappings(&self) -> std::io::Result<Vec<Mapping>> {
        const N: usize = 6;
        #[repr(C)]
        struct Buf {
            head: QueryMappingsIn,
            maps: [Mapping; N],
        }
        let mut b = Buf {
            head: QueryMappingsIn { output_mapping_count: N as u32, reserved: 0 },
            maps: [Mapping::default(); N],
        };
        unsafe { self.call(QUERY_MAPPINGS, &mut b)? };
        Ok(b.maps.into_iter().filter(|m| m.mapping_id != MAPPING_UNUSED).collect())
    }

    /// One kernel-mediated NOC read. `width` must be 1/2/4/8 and `addr` aligned to it.
    /// ⚠ UNVERIFIED against hardware.
    pub fn noc_read(&self, x: u16, y: u16, noc: u8, addr: u64, width: u8) -> std::io::Result<u64> {
        let mut a = noc_io(x, y, noc, addr, width, 0)?;
        unsafe { self.call(NOC_READ, &mut a)? };
        Ok(a.value)
    }

    /// One kernel-mediated NOC write. ⚠ UNVERIFIED against hardware.
    pub fn noc_write(
        &self, x: u16, y: u16, noc: u8, addr: u64, width: u8, value: u64,
    ) -> std::io::Result<()> {
        let mut a = noc_io(x, y, noc, addr, width, value)?;
        unsafe { self.call(NOC_WRITE, &mut a) }
    }
}

/// ⛔ The header's constraints, enforced HERE rather than discovered as an `EINVAL` from the kernel:
/// x and y are 0-63, noc is 0 or 1, width is 1/2/4/8, and addr is width-aligned.
fn noc_io(x: u16, y: u16, noc: u8, addr: u64, width: u8, value: u64) -> std::io::Result<NocIo> {
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidInput, m.to_string());
    if x > 63 || y > 63 {
        return Err(bad("NOC x/y must be 0-63"));
    }
    if noc > 1 {
        return Err(bad("NOC id must be 0 or 1"));
    }
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Err(bad("NOC width must be 1, 2, 4 or 8"));
    }
    if addr % width as u64 != 0 {
        return Err(bad("NOC addr must be aligned to width"));
    }
    Ok(NocIo {
        argsz: std::mem::size_of::<NocIo>() as u32,
        flags: 0,
        x,
        y,
        noc,
        width,
        reserved0: [0; 2],
        addr,
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// ⭐ **THE ABI ORACLE IS A C COMPILER, NOT MY ARITHMETIC.** Every number below was produced by
    /// compiling tt-kmd's real `ioctl.h` (fetched from `tenstorrent/tt-kmd@main`) into a program that
    /// prints `sizeof`/`offsetof`, then read off its output. Writing the expectations by hand from the
    /// struct definitions would be the classic vacuous test: the test would keep a copy of the same
    /// mistake the code made. The header is NOT vendored here — it is GPL-2.0, and Ferric does not
    /// take GPL source into its tree; only these measured facts about the ABI are recorded.
    ///
    /// ⚠ **Mutation-tested, and the results are worth writing down because two of them surprised me.**
    /// CAUGHT: swapping `bus_dev_fn` with `pci_domain` (a reorder), and widening `bus_dev_fn` to
    /// `u32` (a mid-struct width change). NOT caught, and correctly so: **dropping a TRAILING
    /// `reserved` field changes nothing** — `DeviceInfoOut` is 20 bytes either way because Rust pads
    /// to the 4-byte alignment, and `Mapping` is 24 either way because `mapping_base: u64` aligns to 8
    /// regardless. Those are ABI-benign, not blind spots. The layout is where the checkable truth
    /// lives, since the calls are untestable without a card — but it checks OFFSETS, so only changes
    /// that move a field are in scope.
    #[test]
    fn struct_layout_matches_the_c_compiler_on_tt_kmds_own_header() {
        assert_eq!(size_of::<DeviceInfoOut>(), 20);
        assert_eq!(offset_of!(DeviceInfoOut, vendor_id), 4);
        assert_eq!(offset_of!(DeviceInfoOut, bus_dev_fn), 12);
        assert_eq!(offset_of!(DeviceInfoOut, pci_domain), 16);
        assert_eq!(size_of::<DeviceInfo>(), 24);
        assert_eq!(offset_of!(DeviceInfo, out), 4);

        assert_eq!(size_of::<DriverInfoOut>(), 12);
        assert_eq!(offset_of!(DriverInfoOut, driver_version), 4);
        assert_eq!(size_of::<DriverInfo>(), 16);
        assert_eq!(offset_of!(DriverInfo, out), 4);

        assert_eq!(size_of::<Mapping>(), 24);
        assert_eq!(offset_of!(Mapping, mapping_base), 8);
        assert_eq!(offset_of!(Mapping, mapping_size), 16);
        assert_eq!(size_of::<QueryMappingsIn>(), 8);

        assert_eq!(size_of::<NocIo>(), 32);
        assert_eq!(offset_of!(NocIo, x), 8);
        assert_eq!(offset_of!(NocIo, noc), 12);
        assert_eq!(offset_of!(NocIo, addr), 16);
        assert_eq!(offset_of!(NocIo, value), 24);
    }

    /// The same C run printed these two, and `_IO(0xFA, n)` must give `0xFA00 | n` for every entry.
    #[test]
    fn ioctl_numbers_match_the_io_macro() {
        assert_eq!(GET_DEVICE_INFO, 64000); // printed by the C oracle
        assert_eq!(NOC_WRITE, 64019); // printed by the C oracle
        for (nr, got) in [
            (0, GET_DEVICE_INFO), (1, GET_HARVESTING), (2, QUERY_MAPPINGS), (3, ALLOCATE_DMA_BUF),
            (4, FREE_DMA_BUF), (5, GET_DRIVER_INFO), (6, RESET_DEVICE), (7, PIN_PAGES),
            (8, LOCK_CTL), (9, MAP_PEER_BAR), (10, UNPIN_PAGES), (11, ALLOCATE_TLB),
            (12, FREE_TLB), (13, CONFIGURE_TLB), (14, SET_NOC_CLEANUP), (15, SET_POWER_STATE),
            (16, EXPORT_TLB_DMABUF), (17, SMC_MSG), (18, NOC_READ), (19, NOC_WRITE),
        ] {
            assert_eq!(got, 0xFA00 | nr as c_ulong, "ioctl nr {nr}");
        }
    }

    /// ⭐ The argument checks are the one piece of BEHAVIOUR testable with no hardware, so they are
    /// written as rejections rather than left for the kernel to discover as `EINVAL`.
    #[test]
    fn noc_io_refuses_what_the_header_forbids() {
        assert!(noc_io(64, 0, 0, 0, 4, 0).is_err(), "x > 63");
        assert!(noc_io(0, 64, 0, 0, 4, 0).is_err(), "y > 63");
        assert!(noc_io(0, 0, 2, 0, 4, 0).is_err(), "noc must be 0 or 1");
        assert!(noc_io(0, 0, 0, 0, 3, 0).is_err(), "width 3 is not 1/2/4/8");
        assert!(noc_io(0, 0, 0, 2, 4, 0).is_err(), "addr 2 is not 4-aligned");
        // ⚠ And the mutation that keeps the four above honest: a VALID call must still build, or a
        // function that rejected everything would pass every assertion here.
        let ok = noc_io(63, 63, 1, 8, 8, 0xdead).expect("a legal request must build");
        assert_eq!(ok.argsz, 32);
        assert_eq!((ok.width, ok.addr, ok.value), (8, 8, 0xdead));
    }

    /// ⛔ This project has NO Tenstorrent hardware, so the honest test is the absent path: probing a
    /// machine without a card must decline quietly, never panic and never block, exactly as
    /// `cuda::driver()` does without an NVIDIA driver. On a machine that DOES have one this reports
    /// what it found rather than asserting, because a lock nobody has recorded is not evidence.
    #[cfg(target_os = "linux")]
    #[test]
    fn probing_a_machine_without_a_card_declines_quietly() {
        let found = Device::enumerate();
        if found.is_empty() {
            assert!(Device::open(0).is_none(), "no /dev/tenstorrent/0, so open must be None");
            eprintln!("  no Tenstorrent device on this host — absent path exercised");
        } else {
            for d in &found {
                eprintln!(
                    "  /dev/tenstorrent/{} opened; device_info={:?} driver_info={:?}",
                    d.index,
                    d.device_info(),
                    d.driver_info()
                );
            }
        }
    }
}
