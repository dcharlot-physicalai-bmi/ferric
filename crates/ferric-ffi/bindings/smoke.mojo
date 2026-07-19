# Mojo has TWO clean routes to Ferric.
#
# (A) Python interop — Mojo is a Python superset, so our `ferric` PyPI package works as-is:
from python import Python
fn main() raises:
    var ferric = Python.import_module("ferric")
    var llm = ferric.Ferric("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    print(llm.generate("The capital of France is"))
    print(llm.generate_object("Invent a person.",
        Python.evaluate("{'type':'object','properties':{'name':{'type':'string'},'age':{'type':'integer'}}}")))

# (B) Zero-overhead native — load libferric via DLHandle and call the C ABI directly:
#   from sys.ffi import DLHandle
#   var lib = DLHandle("libferric.dylib")
#   var load = lib.get_function[fn(UnsafePointer[UInt8]) -> UnsafePointer[NoneType]]("ferric_load")
#   var gen  = lib.get_function[fn(UnsafePointer[NoneType], UnsafePointer[UInt8], UInt32) -> UnsafePointer[UInt8]]("ferric_generate")
#   ... same ferric_load / ferric_generate / ferric_generate_json / ferric_free_string / ferric_free.
