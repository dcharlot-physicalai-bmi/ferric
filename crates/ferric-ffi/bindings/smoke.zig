// Zig calls Ferric with ZERO binding code — @cImport the cbindgen header, link -lferric.
//   zig build-exe smoke.zig -I<crate> -L<target> -lferric -lc && ./smoke <model.gguf>
// (Verified working on zig 0.16 with a hardcoded path; adapt argv per your zig version.)
const std = @import("std");
const c = @cImport(@cInclude("libferric.h"));
pub fn main() void {
    const model = "path/to/model.gguf"; // or read from args per your zig version
    const h = c.ferric_load(model);
    if (h == null) { std.debug.print("load failed\n", .{}); return; }
    const g = c.ferric_generate(h, "The capital of France is", 8);
    std.debug.print("ZIG GEN : {s}\n", .{g});
    c.ferric_free_string(g);
    const j = c.ferric_generate_json(h, "Invent a person.",
        "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}}}", 40);
    std.debug.print("ZIG JSON: {s}\n", .{j});
    c.ferric_free_string(j);
    c.ferric_free(h);
}
