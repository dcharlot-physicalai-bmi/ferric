// Pure-Rust proto compilation (protox, no protoc binary) → prost types for the ONNX schema.
fn main() {
    let fds = protox::compile(["onnx.proto"], ["."]).expect("compile onnx.proto");
    let mut cfg = prost_build::Config::new();
    cfg.compile_fds(fds).expect("prost gen");
    println!("cargo:rerun-if-changed=onnx.proto");
}
