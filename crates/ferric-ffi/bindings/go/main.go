package main
/*
#include "libferric.h"
#include <stdlib.h>
*/
import "C"
import ("fmt"; "os"; "unsafe")
func main() {
    model := C.CString(os.Args[1]); defer C.free(unsafe.Pointer(model))
    h := C.ferric_load(model)
    if h == nil { fmt.Println("load failed"); return }
    defer C.ferric_free(h)
    p := C.CString("The capital of France is"); defer C.free(unsafe.Pointer(p))
    g := C.ferric_generate(h, p, 8)
    fmt.Println("GO  GEN :", C.GoString(g)); C.ferric_free_string(g)
    pj := C.CString("Invent a person."); defer C.free(unsafe.Pointer(pj))
    sc := C.CString(`{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}`); defer C.free(unsafe.Pointer(sc))
    j := C.ferric_generate_json(h, pj, sc, 40)
    fmt.Println("GO  JSON:", C.GoString(j)); C.ferric_free_string(j)
}
