//! Check the acceptance corpus in this repository.
//!
//! `cargo run -p ferric-accept --example check`
//!
//! Exit 0 means the documents, schemas, and ledger obey the receipt rules.
//! It does not mean a model, a fabric, or a joule figure passed.

fn main() {
    let root = ferric_accept::repo_root();
    let errs = ferric_accept::check_corpus(&root);
    if errs.is_empty() {
        println!("ferric-accept: corpus ok ({})", root.display());
        return;
    }
    for e in &errs {
        eprintln!("ferric-accept: {e}");
    }
    eprintln!("ferric-accept: {} problem(s)", errs.len());
    std::process::exit(1);
}
