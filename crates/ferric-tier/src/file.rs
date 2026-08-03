//! A file-backed [`Backing`]: positional reads, no seeking, no shared cursor.
//!
//! Positional I/O (`pread` on unix, `seek_read` on Windows) rather than seek-then-read is what makes this
//! `Sync` and therefore usable from [`crate::PrefetchCache`]'s worker thread. A `File` behind a mutex with
//! a shared cursor would serialise every read and defeat the overlap it exists to enable; positional reads
//! carry their offset as an argument, so any number of threads can read any number of ranges at once.
//!
//! This is also why the crate does not use `mmap`. Two reasons, both learned from engines that stream
//! terabyte checkpoints:
//!
//!  1. **Honest memory accounting.** Buffer-owned pages never become file-backed mappings, so peak RSS
//!     reflects what is actually resident rather than what has been mapped. A budget you cannot measure
//!     is not a budget.
//!  2. **The budget stays enforceable.** With `mmap` the kernel decides residency; the plan in
//!     [`crate::plan_layers`] would become advisory.

use crate::{Backing, TierError};
use std::fs::File;
use std::path::Path;

/// Read-only positional access to a file.
#[derive(Debug)]
pub struct FileBacking {
    f: File,
    len: u64,
}

impl FileBacking {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TierError> {
        let p = path.as_ref();
        let f = File::open(p).map_err(|e| TierError::Io(format!("open {}: {e}", p.display())))?;
        let len = f
            .metadata()
            .map_err(|e| TierError::Io(format!("stat {}: {e}", p.display())))?
            .len();
        Ok(Self { f, len })
    }

    pub fn len(&self) -> u64 { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl Backing for FileBacking {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
        // Range check first, so reading past EOF is a clear error rather than a silent short read that a
        // caller might mistake for a valid all-zero weight.
        let end = offset.checked_add(dst.len() as u64).ok_or_else(|| {
            TierError::Io(format!("read range overflows u64 at offset {offset}"))
        })?;
        if end > self.len {
            return Err(TierError::ShortRead {
                want: dst.len(),
                got: self.len.saturating_sub(offset) as usize,
            });
        }

        // Loop: a positional read is permitted to return fewer bytes than asked for, and treating a short
        // return as success is exactly how a cache slot ends up holding half a weight.
        let mut done = 0usize;
        while done < dst.len() {
            let n = read_at_impl(&self.f, offset + done as u64, &mut dst[done..])
                .map_err(|e| TierError::Io(format!("read at {}: {e}", offset + done as u64)))?;
            if n == 0 {
                return Err(TierError::ShortRead { want: dst.len(), got: done });
            }
            done += n;
        }
        Ok(())
    }
}

#[cfg(unix)]
#[inline]
fn read_at_impl(f: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, offset)
}

#[cfg(windows)]
#[inline]
fn read_at_impl(f: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    // `seek_read` moves the handle's cursor as a side effect, but since every read supplies its own
    // absolute offset the cursor is never consulted, so concurrent readers still cannot interfere.
    f.seek_read(buf, offset)
}

#[cfg(not(any(unix, windows)))]
#[inline]
fn read_at_impl(_f: &File, _offset: u64, _buf: &mut [u8]) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "positional file reads are not available on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("ferric-tier-{name}-{}.bin", std::process::id()));
        let mut f = File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
        p
    }

    #[test]
    fn reads_exact_ranges() {
        let data: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let p = tmp("exact", &data);
        let b = FileBacking::open(&p).unwrap();
        assert_eq!(b.len(), 4096);
        for (off, len) in [(0u64, 16usize), (100, 256), (4080, 16), (1, 4095)] {
            let mut got = vec![0u8; len];
            b.read_at(off, &mut got).unwrap();
            assert_eq!(got, &data[off as usize..off as usize + len], "range ({off},{len})");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn reading_past_the_end_is_an_error_not_zeros() {
        // The failure this prevents: a truncated or mis-indexed checkpoint silently yielding all-zero
        // weights, which a model happily runs and which looks like a quality problem, not a bug.
        let p = tmp("eof", &[7u8; 64]);
        let b = FileBacking::open(&p).unwrap();
        let mut got = vec![0u8; 128];
        let e = b.read_at(0, &mut got).unwrap_err();
        assert!(matches!(e, TierError::ShortRead { want: 128, got: 64 }), "got {e:?}");
        let e = b.read_at(60, &mut vec![0u8; 8]).unwrap_err();
        assert!(matches!(e, TierError::ShortRead { .. }), "got {e:?}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn is_usable_from_several_threads_at_once() {
        // The property that makes the prefetch worker possible: positional reads share no cursor, so
        // concurrent reads of different ranges cannot interfere.
        let data: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let p = tmp("threads", &data);
        let b = std::sync::Arc::new(FileBacking::open(&p).unwrap());
        let mut hs = Vec::new();
        for t in 0..8u64 {
            let b = std::sync::Arc::clone(&b);
            let expect = data.clone();
            hs.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    let off = (t * 137) % 3000;
                    let mut got = vec![0u8; 512];
                    b.read_at(off, &mut got).unwrap();
                    assert_eq!(got, &expect[off as usize..off as usize + 512]);
                }
            }));
        }
        for h in hs { h.join().unwrap(); }
        let _ = std::fs::remove_file(&p);
    }
}
