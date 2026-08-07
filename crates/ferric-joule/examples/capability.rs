//! What can this machine actually measure? Run this before trusting any energy number from it.
fn main() {
    print!("{}", ferric_joule::capability_report());
    match ferric_joule::best() {
        Some(m) => println!("\n  best available: {} [{}]", m.source(), m.class().label()),
        None => println!("\n  No real meter. Any joule figure produced here would be arithmetic, not measurement."),
    }
}
