//! **What can this machine actually measure?** Prints the probed meter table and what `best()` picks.
//!
//! Worth running on any new machine before trusting an energy number from it: every row is a REAL
//! probe, so `[available]` means the meter was constructed here, not that it might exist.
//!
//!   cargo run -p ferric-joule --example energy_capability
fn main() {
    println!("{}", ferric_joule::capability_report());
    match ferric_joule::best() {
        Some(m) => println!("best() -> {} · class {:?} · boundary {:?}", m.source(), m.class(), m.boundary()),
        None => println!("best() -> None — every energy claim on this machine routes to the refusal path"),
    }
}
