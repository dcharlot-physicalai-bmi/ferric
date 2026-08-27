//! **Reading one weight from several devices at once** — Colibri's weighted mirror, with the
//! silent-corruption mode it does not mention closed off.
//!
//! Colibri reports that "weighted mirror/split routing" aggregates bandwidth, measuring a
//! 9 GB/s + 3 GB/s pair reading experts ~33% faster than the fast drive alone. The shape is: every
//! device holds a **full copy**, and each read is divided between them in proportion to what they
//! can actually deliver. No device holds a fraction of the model, so losing one costs speed and not
//! correctness — which is why a mirror is the right structure for a checkpoint you already have room
//! for twice.
//!
//! ## ⛔ The failure mode a mirror introduces, which nothing above it can detect
//!
//! If the copies are not identical — a stale mirror, an interrupted copy, two different quantisations
//! of the same model — then splitting a read across them returns a buffer stitched from two
//! versions. Every length is right, every offset is right, and the bytes are wrong in the middle.
//! That is precisely the corruption [`crate::Backing`]'s contract exists to prevent
//! ("the same `(offset, len)` must yield the same bytes on every call"), and a composite backing can
//! violate it while every component honours it.
//!
//! So construction takes probes and [`MirroredBacking::verify`] reads them from **every** device and
//! compares. It is not free and it is not optional: a mirror nobody checked is a checkpoint you are
//! trusting twice for no reason.
//!
//! ## Why the split here is by BYTES, not by [`crate::split_n`]
//!
//! `split_n` divides *indivisible items* — experts — and its greedy exists because you cannot send
//! half an expert. Inside a single read the unit is a byte, and bytes at megabyte granularity are
//! divisible enough that the continuous answer `L_i = L·B_i/ΣB` is simply correct. Reaching for the
//! integer solver here would be slower and no more accurate. The two granularities are genuinely
//! different problems and this file uses the one that fits.
//!
//! ⚠ **Aggregation requires the reads to actually overlap.** Issued one after another, a "split"
//! read is the same total bytes through the same devices in sequence and buys nothing at all — it is
//! strictly worse than not splitting, because of the extra seeks. The reads run on scoped threads,
//! which is why this module is not built for wasm.

use crate::{Backing, TierError};

/// Reads below this are served whole by the fastest device. Splitting a small read pays two seeks
/// and a thread spawn to save microseconds of transfer — the overhead is not a rounding error at
/// this size, it is the entire cost.
pub const MIN_SPLIT_BYTES: usize = 1 << 20;

/// Several full copies of the same data, read in parallel and weighted by measured bandwidth.
pub struct MirroredBacking {
    devices: Vec<Box<dyn Backing + Sync>>,
    /// Measured bytes/second per device. Same order as `devices`.
    bandwidths: Vec<f64>,
    verified: bool,
}

impl MirroredBacking {
    /// Build from devices and their **measured** rates.
    ///
    /// Refuses a rate that is not a rate, for the reason [`crate::FabricProfile::measured`] does: a
    /// plausible constant here produces a split that looks reasonable and sends most of the read to
    /// the wrong device, and nothing downstream can tell.
    pub fn new(devices: Vec<Box<dyn Backing + Sync>>, bandwidths: Vec<f64>)
        -> Result<MirroredBacking, String>
    {
        if devices.len() != bandwidths.len() {
            return Err(format!("{} devices but {} bandwidths", devices.len(), bandwidths.len()));
        }
        if devices.is_empty() { return Err("a mirror needs at least one device".into()) }
        for (i, b) in bandwidths.iter().enumerate() {
            if !(*b > 0.0) || !b.is_finite() {
                return Err(format!("device {i} bandwidth is {b}, which is not a rate a split can \
                                    be built from"));
            }
        }
        Ok(MirroredBacking { devices, bandwidths, verified: false })
    }

    pub fn len(&self) -> usize { self.devices.len() }
    pub fn is_empty(&self) -> bool { self.devices.is_empty() }
    /// Whether [`Self::verify`] has been run and passed. A read on an unverified mirror still works;
    /// this is here so a caller can refuse to proceed without one.
    pub fn is_verified(&self) -> bool { self.verified }

    /// Read each probe from **every** device and require agreement.
    ///
    /// ⚠ Probes must be chosen to cover the file, not just its start. Two copies that diverge only
    /// past the first gigabyte agree perfectly on any probe near zero — and a truncated or
    /// interrupted copy is exactly the case that diverges late. [`Self::spread_probes`] builds a set
    /// that cannot be fooled that way.
    pub fn verify(&mut self, probes: &[(u64, usize)]) -> Result<(), TierError> {
        if probes.is_empty() {
            // Refuse rather than pass: a verification with no probes returns Ok and proves nothing,
            // which is worse than never having called it because it sets `verified`.
            return Err(TierError::MirrorMismatch("verify() called with no probes: it would set the mirror \
                                       verified while checking nothing".into()));
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        for &(off, len) in probes {
            a.clear(); a.resize(len, 0);
            self.devices[0].read_at(off, &mut a)?;
            for (i, d) in self.devices.iter().enumerate().skip(1) {
                b.clear(); b.resize(len, 0);
                d.read_at(off, &mut b)?;
                if a != b {
                    let at = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(0);
                    return Err(TierError::MirrorMismatch(format!(
                        "mirror device {i} disagrees with device 0 at offset {} (probe {off}+{len}, \
                         first differing byte {}): these are not copies of the same data, and \
                         splitting a read across them would return a buffer stitched from two \
                         versions with the right length and the wrong bytes",
                        off + at as u64, at)));
                }
            }
        }
        self.verified = true;
        Ok(())
    }

    /// Probe offsets spread across `total_len`, including the very last bytes.
    ///
    /// The final probe is deliberately flush with the end: a truncated copy is the most likely way a
    /// mirror goes wrong, and it is invisible to every probe that stops short.
    pub fn spread_probes(total_len: u64, n: usize, probe_len: usize) -> Vec<(u64, usize)> {
        if total_len == 0 || n == 0 || probe_len == 0 { return Vec::new() }
        let plen = probe_len.min(total_len as usize);
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let off = if n == 1 { 0 } else {
                (total_len - plen as u64).saturating_mul(i as u64) / (n as u64 - 1)
            };
            v.push((off, plen));
        }
        v.dedup();
        v
    }

    /// Byte ranges per device for a read of `len`, proportional to measured bandwidth.
    ///
    /// Returned as `(device, start, end)` with `start..end` relative to the read. Exposed because
    /// the split is the interesting part and a test that can only observe the assembled buffer
    /// cannot tell a good split from a degenerate one that sent everything to device 0.
    pub fn plan(&self, len: usize) -> Vec<(usize, usize, usize)> {
        if len == 0 { return Vec::new() }
        // Small reads go whole to the fastest device — see MIN_SPLIT_BYTES.
        if len < MIN_SPLIT_BYTES || self.devices.len() == 1 {
            let fastest = self.bandwidths.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);
            return vec![(fastest, 0, len)];
        }
        let total: f64 = self.bandwidths.iter().sum();
        let mut out = Vec::with_capacity(self.devices.len());
        let mut cur = 0usize;
        for (i, b) in self.bandwidths.iter().enumerate() {
            // The LAST device takes whatever remains, so rounding can never lose or duplicate a
            // byte. Distributing the remainder instead would be a second place to get it wrong.
            let end = if i + 1 == self.bandwidths.len() { len }
                      else { (cur + ((len as f64 * b / total).round() as usize)).min(len) };
            if end > cur { out.push((i, cur, end)); }
            cur = end;
        }
        debug_assert_eq!(out.last().map(|r| r.2), Some(len), "the plan must cover the whole read");
        out
    }
}

impl Backing for MirroredBacking {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
        let plan = self.plan(dst.len());
        if plan.len() <= 1 {
            return match plan.first() {
                Some(&(d, _, _)) => self.devices[d].read_at(offset, dst),
                None => Ok(()), // zero-length read
            };
        }
        // Hand each device a disjoint &mut slice of the caller's buffer. Disjointness is what makes
        // the parallel fill sound, and `split_at_mut` proves it to the compiler rather than to a
        // reviewer — the same argument `prefetch.rs` makes for its ownership transfer.
        let mut rest: &mut [u8] = dst;
        let mut parts: Vec<(usize, u64, &mut [u8])> = Vec::with_capacity(plan.len());
        let mut consumed = 0usize;
        for &(dev, start, end) in &plan {
            debug_assert_eq!(start, consumed, "plan ranges must be contiguous");
            let (head, tail) = rest.split_at_mut(end - start);
            parts.push((dev, offset + start as u64, head));
            rest = tail;
            consumed = end;
        }
        let devices = &self.devices;
        let mut errs: Vec<Result<(), TierError>> = Vec::new();
        std::thread::scope(|s| {
            let hs: Vec<_> = parts.into_iter()
                .map(|(dev, off, buf)| s.spawn(move || devices[dev].read_at(off, buf)))
                .collect();
            for h in hs { errs.push(h.join().unwrap_or_else(|_| Err(TierError::Io(
                "a mirror read thread panicked; the destination buffer is partially filled and \
                 must not be used".into())))); }
        });
        // Report the FIRST failure rather than the last: with several devices failing at once the
        // last one to be joined is an arbitrary choice and makes the cause harder to find.
        for e in errs { e?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A deterministic device: byte `i` is `(i * 31 + salt) as u8`, so two devices with the same
    /// salt are true copies and different salts are a corrupt mirror. It also counts bytes served,
    /// which is the only way a test can see the split rather than just the result.
    struct Fake { salt: u64, served: AtomicU64, len: u64 }
    impl Fake {
        fn new(salt: u64, len: u64) -> Fake { Fake { salt, served: AtomicU64::new(0), len } }
        fn byte(&self, i: u64) -> u8 { (i.wrapping_mul(31).wrapping_add(self.salt) % 251) as u8 }
    }
    impl Backing for Fake {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
            if offset + dst.len() as u64 > self.len {
                return Err(TierError::Io(format!("read past end: {offset}+{}", dst.len())));
            }
            for (j, b) in dst.iter_mut().enumerate() { *b = self.byte(offset + j as u64) }
            self.served.fetch_add(dst.len() as u64, Ordering::Relaxed);
            Ok(())
        }
    }

    const LEN: u64 = 64 << 20;

    fn mirror(salts: &[u64], bw: &[f64]) -> MirroredBacking {
        let devs: Vec<Box<dyn Backing + Sync>> =
            salts.iter().map(|&s| Box::new(Fake::new(s, LEN)) as Box<dyn Backing + Sync>).collect();
        MirroredBacking::new(devs, bw.to_vec()).unwrap()
    }

    /// The crate's whole guarantee, on a new axis: which device served a byte must not be observable
    /// in the byte. Checked against a single-device read of the same range.
    #[test]
    fn a_striped_read_is_byte_identical_to_a_single_device_read() {
        let m = mirror(&[7, 7, 7], &[9.0e9, 3.0e9, 1.0e9]);
        let solo = Fake::new(7, LEN);
        for (off, len) in [(0u64, 4 << 20), (1 << 20, 8 << 20), (12345, (3 << 20) + 777),
                           (LEN - (2 << 20), 2 << 20), (0, 100), (999, 0)] {
            let mut a = vec![0u8; len];
            let mut b = vec![0u8; len];
            m.read_at(off, &mut a).expect("striped");
            solo.read_at(off, &mut b).expect("solo");
            assert_eq!(a, b, "striped read at {off}+{len} differs from a single-device read");
        }
    }

    /// ⛔ THE SILENT CORRUPTION. Different salts = different data at the same offsets. A split read
    /// returns the right LENGTH and the wrong BYTES, and only verification catches it.
    #[test]
    fn a_mismatched_mirror_is_caught_by_verify_and_would_otherwise_corrupt_silently() {
        let mut m = mirror(&[7, 8], &[9.0e9, 3.0e9]);
        let e = m.verify(&MirroredBacking::spread_probes(LEN, 4, 4096)).unwrap_err();
        let msg = format!("{e:?}");
        assert!(msg.contains("disagrees"), "unhelpful mismatch report: {msg}");
        assert!(!m.is_verified());

        // And demonstrate what verification is FOR: the read succeeds and is wrong.
        let mut got = vec![0u8; 8 << 20];
        m.read_at(0, &mut got).expect("the corrupt read still succeeds — that is the danger");
        let solo = Fake::new(7, LEN);
        let mut want = vec![0u8; 8 << 20];
        solo.read_at(0, &mut want).unwrap();
        assert_ne!(got, want, "the fixture is supposed to produce a stitched buffer");
        // A matched mirror passes.
        let mut good = mirror(&[7, 7], &[9.0e9, 3.0e9]);
        good.verify(&MirroredBacking::spread_probes(LEN, 4, 4096)).expect("identical copies");
        assert!(good.is_verified());
    }

    /// ⚠ A truncated copy diverges LATE, so probes clustered at the start cannot see it. The probe
    /// set must reach the final bytes — this pins that it does.
    #[test]
    fn probes_reach_the_end_of_the_file_where_a_truncated_copy_diverges() {
        let p = MirroredBacking::spread_probes(LEN, 5, 4096);
        assert_eq!(p.len(), 5);
        assert_eq!(p[0].0, 0, "first probe should start at 0");
        assert_eq!(p[4].0 + p[4].1 as u64, LEN, "last probe must be flush with the end of the file");
        assert!(p.windows(2).all(|w| w[0].0 < w[1].0), "probes must be spread, not repeated");
        // Degenerate inputs give no probes rather than a probe that reads out of range.
        assert!(MirroredBacking::spread_probes(0, 4, 4096).is_empty());
        assert!(MirroredBacking::spread_probes(LEN, 0, 4096).is_empty());
        assert_eq!(MirroredBacking::spread_probes(100, 2, 4096)[0].1, 100, "probe clamps to the file");
    }

    /// ⛔ A verification with nothing to check must REFUSE, not pass. Returning Ok would set
    /// `verified` while proving nothing — worse than never calling it.
    #[test]
    fn verifying_with_no_probes_is_refused() {
        let mut m = mirror(&[7, 8], &[9.0e9, 3.0e9]);
        assert!(m.verify(&[]).is_err());
        assert!(!m.is_verified(), "a refused verification must not mark the mirror verified");
    }

    /// The split must actually be PROPORTIONAL. A composite that quietly sent everything to device 0
    /// would pass the byte-identity test perfectly while aggregating nothing.
    #[test]
    fn bytes_are_divided_in_proportion_to_measured_bandwidth() {
        let m = mirror(&[7, 7], &[9.0e9, 3.0e9]);
        let len = 12 << 20;
        let plan = m.plan(len);
        assert_eq!(plan.len(), 2, "both devices should be used for a {len}-byte read");
        let d0 = plan.iter().find(|p| p.0 == 0).unwrap();
        let share = (d0.2 - d0.1) as f64 / len as f64;
        assert!((share - 0.75).abs() < 0.01, "9 of 12 GB/s should take ~75%, took {share:.3}");
        assert_eq!(plan.iter().map(|p| p.2 - p.1).sum::<usize>(), len, "the plan must cover the read");
        // Contiguous and in order, or the assembled buffer is a permutation of the right bytes.
        assert_eq!(plan[0].1, 0);
        for w in plan.windows(2) { assert_eq!(w[0].2, w[1].1, "plan ranges must be contiguous") }
    }

    /// Small reads are not worth two seeks and a thread. This pins the guard, since without it every
    /// 4 KB metadata read would fan out.
    #[test]
    fn a_small_read_goes_whole_to_the_fastest_device() {
        let m = mirror(&[7, 7], &[3.0e9, 9.0e9]); // fastest is device 1
        let plan = m.plan(4096);
        assert_eq!(plan, vec![(1, 0, 4096)], "a 4 KB read should go whole to the fastest device");
        assert!(MIN_SPLIT_BYTES > 4096);
    }

    #[test]
    fn a_mirror_refuses_bandwidths_that_are_not_rates() {
        let d = || -> Vec<Box<dyn Backing + Sync>> { vec![Box::new(Fake::new(7, LEN))] };
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(MirroredBacking::new(d(), vec![bad]).is_err(), "accepted {bad}");
        }
        assert!(MirroredBacking::new(d(), vec![1.0, 2.0]).is_err(), "count mismatch accepted");
        assert!(MirroredBacking::new(Vec::new(), Vec::new()).is_err(), "empty mirror accepted");
    }

    /// One device must behave exactly like not having a mirror at all — no split, no threads.
    #[test]
    fn a_single_device_mirror_is_a_passthrough() {
        let m = mirror(&[7], &[5.0e9]);
        let mut a = vec![0u8; 8 << 20];
        m.read_at(1 << 20, &mut a).unwrap();
        let solo = Fake::new(7, LEN);
        let mut b = vec![0u8; 8 << 20];
        solo.read_at(1 << 20, &mut b).unwrap();
        assert_eq!(a, b);
        assert_eq!(m.plan(8 << 20), vec![(0, 0, 8 << 20)]);
    }
}
