import CFerric

/// Ferric — a pure-Rust cross-fabric AI runtime, on-device from Swift (Apple Silicon / Metal).
public final class Ferric {
    private let handle: OpaquePointer
    public init?(model: String) {
        guard let h = ferric_load(model) else { return nil }
        self.handle = h
    }
    /// Free-text completion.
    public func generate(_ prompt: String, maxTokens: UInt32 = 128) -> String {
        guard let p = ferric_generate(handle, prompt, maxTokens) else { return "" }
        defer { ferric_free_string(p) }
        return String(cString: p)
    }
    /// Schema-constrained generation — guaranteed-conformant JSON via guided decoding.
    public func generateJSON(_ prompt: String, schema: String, maxTokens: UInt32 = 128) -> String {
        guard let p = ferric_generate_json(handle, prompt, schema, maxTokens) else { return "" }
        defer { ferric_free_string(p) }
        return String(cString: p)
    }
    deinit { ferric_free(handle) }
}
