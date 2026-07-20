//! Guided decoding — a byte-level **incremental JSON acceptor**. The sampler clones it per candidate
//! token (it is `Copy`, fixed 32-deep stack, no alloc) and asks "does feeding this token's bytes keep
//! the output a valid JSON prefix?"; tokens that don't are masked to -inf. `can_stop` is true once a
//! complete top-level value has been consumed, so the model auto-terminates on valid JSON. This is the
//! core of Ferric's differentiator: constrained decoding done in-runtime, deterministic across fabrics.

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase { Value, ObjKey, ObjKeyReq, Colon, ObjComma, ArrValue, ArrValueReq, ArrComma, End }

/// JSON number sub-state (proper grammar: `-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?`). `completable`
/// marks states where the number is a valid stopping point (so a non-numeric byte ends it cleanly).
#[derive(Clone, Copy, PartialEq, Eq)]
enum NumSt { Neg, IntZero, Int, Dot, Frac, Exp, ExpSign, ExpDig }
impl NumSt { fn completable(self) -> bool { matches!(self, NumSt::IntZero | NumSt::Int | NumSt::Frac | NumSt::ExpDig) } }

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lex { None, Str, StrEsc, StrU(u8), Num(NumSt), Kw(u8, u8) } // Kw(kind 0=true 1=false 2=null, matched)

#[derive(Clone, Copy)]
pub struct Json {
    stack: [bool; 32], // true = array, false = object
    depth: usize,
    phase: Phase,
    lex: Lex,
    str_key: bool,       // the string currently being lexed is an object key
    require_object: bool, // top-level value must be an object (OpenAI json_object mode)
}

impl Json {
    pub fn new() -> Self { Json { stack: [false; 32], depth: 0, phase: Phase::Value, lex: Lex::None, str_key: false, require_object: false } }
    /// json_object mode: the whole response must be a single JSON object `{…}`.
    pub fn object() -> Self { let mut j = Self::new(); j.require_object = true; j }

    /// A complete top-level value has been consumed and nothing is mid-token — safe to emit EOS.
    pub fn can_stop(&self) -> bool { self.lex == Lex::None && self.phase == Phase::End && self.depth == 0 }

    /// Feed one byte. Returns false (and leaves self unchanged-enough to discard) if the byte would
    /// make the output an invalid JSON prefix.
    pub fn step(&mut self, b: u8) -> bool {
        let ws = b == b' ' || b == b'\t' || b == b'\n' || b == b'\r';
        match self.lex {
            Lex::Str => {
                if b == b'"' { self.lex = Lex::None; if self.str_key { self.phase = Phase::Colon; } else { self.complete_value(); } true }
                else if b == b'\\' { self.lex = Lex::StrEsc; true }
                else if b < 0x20 { false } // unescaped control char
                else { true }             // any other byte (incl UTF-8 continuation)
            }
            Lex::StrEsc => match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => { self.lex = Lex::Str; true }
                b'u' => { self.lex = Lex::StrU(0); true }
                _ => false,
            },
            Lex::StrU(n) => { if b.is_ascii_hexdigit() { self.lex = if n == 3 { Lex::Str } else { Lex::StrU(n + 1) }; true } else { false } }
            Lex::Num(st) => {
                let dig = b.is_ascii_digit();
                let next = match st {
                    NumSt::Neg => if b == b'0' { Some(NumSt::IntZero) } else if (b'1'..=b'9').contains(&b) { Some(NumSt::Int) } else { None },
                    NumSt::IntZero => if b == b'.' { Some(NumSt::Dot) } else if b == b'e' || b == b'E' { Some(NumSt::Exp) } else { None }, // no leading-zero digits
                    NumSt::Int => if dig { Some(NumSt::Int) } else if b == b'.' { Some(NumSt::Dot) } else if b == b'e' || b == b'E' { Some(NumSt::Exp) } else { None },
                    NumSt::Dot => if dig { Some(NumSt::Frac) } else { None },
                    NumSt::Frac => if dig { Some(NumSt::Frac) } else if b == b'e' || b == b'E' { Some(NumSt::Exp) } else { None },
                    NumSt::Exp => if b == b'+' || b == b'-' { Some(NumSt::ExpSign) } else if dig { Some(NumSt::ExpDig) } else { None },
                    NumSt::ExpSign => if dig { Some(NumSt::ExpDig) } else { None },
                    NumSt::ExpDig => if dig { Some(NumSt::ExpDig) } else { None },
                };
                match next {
                    Some(s2) => { self.lex = Lex::Num(s2); true }
                    None if st.completable() => { self.lex = Lex::None; self.complete_value(); self.step_struct(b, ws) } // number ended cleanly
                    None => false, // e.g. "1." then a non-digit — malformed, reject
                }
            }
            Lex::Kw(kind, m) => {
                let word: &[u8] = match kind { 0 => b"true", 1 => b"false", _ => b"null" };
                if (m as usize) < word.len() && word[m as usize] == b {
                    let nm = m + 1;
                    if nm as usize == word.len() { self.lex = Lex::None; self.complete_value(); } else { self.lex = Lex::Kw(kind, nm); }
                    true
                } else { false }
            }
            Lex::None => self.step_struct(b, ws),
        }
    }

    fn complete_value(&mut self) {
        self.phase = if self.depth == 0 { Phase::End } else if self.stack[self.depth - 1] { Phase::ArrComma } else { Phase::ObjComma };
    }

    fn step_struct(&mut self, b: u8, ws: bool) -> bool {
        if ws { return true; } // whitespace is allowed between structural tokens
        match self.phase {
            Phase::Value | Phase::ArrValue | Phase::ArrValueReq => self.begin_value(b),
            Phase::ObjKey => { if b == b'"' { self.lex = Lex::Str; self.str_key = true; true } else if b == b'}' { self.close(false) } else { false } }
            Phase::ObjKeyReq => { if b == b'"' { self.lex = Lex::Str; self.str_key = true; true } else { false } }
            Phase::Colon => { if b == b':' { self.phase = Phase::Value; true } else { false } }
            Phase::ObjComma => { if b == b',' { self.phase = Phase::ObjKeyReq; true } else if b == b'}' { self.close(false) } else { false } }
            Phase::ArrComma => { if b == b',' { self.phase = Phase::ArrValueReq; true } else if b == b']' { self.close(true) } else { false } }
            Phase::End => false,
        }
    }

    fn begin_value(&mut self, b: u8) -> bool {
        // json_object mode: the top-level value must be an object.
        if self.depth == 0 && self.require_object && b != b'{' { return false; }
        match b {
            b'{' => { if self.depth >= 32 { return false; } self.stack[self.depth] = false; self.depth += 1; self.phase = Phase::ObjKey; true }
            b'[' => { if self.depth >= 32 { return false; } self.stack[self.depth] = true; self.depth += 1; self.phase = Phase::ArrValue; true }
            b'"' => { self.lex = Lex::Str; self.str_key = false; true }
            b'-' => { self.lex = Lex::Num(NumSt::Neg); true }
            b'0' => { self.lex = Lex::Num(NumSt::IntZero); true }
            b'1'..=b'9' => { self.lex = Lex::Num(NumSt::Int); true }
            b't' => { self.lex = Lex::Kw(0, 1); true }
            b'f' => { self.lex = Lex::Kw(1, 1); true }
            b'n' => { self.lex = Lex::Kw(2, 1); true }
            b']' if self.phase == Phase::ArrValue => self.close(true), // empty array
            _ => false,
        }
    }

    fn close(&mut self, array: bool) -> bool {
        if self.depth == 0 || self.stack[self.depth - 1] != array { return false; }
        self.depth -= 1;
        self.complete_value();
        true
    }
}

// ===== JSON-Schema-conformant decoding =====
// A schema compiles to a flat template: fixed structural literals (`{"key":`, `,"key":`, `}`) with
// typed value slots between them, properties emitted in declaration order (compact, no free
// whitespace). This covers the dominant "extract these fields" contract (Pydantic-style models):
// object with scalar/enum/nested-object properties; arrays and unknown types fall back to a free
// JSON value (embedded `Json`). The acceptor (`Schema`) is Copy — it borrows the compiled program.

/// One template element: a fixed literal to emit, or a typed value slot.
#[derive(Clone)]
pub enum Item { Lit(Vec<u8>), Str, Int, Num, Bool, Enum(Vec<Vec<u8>>), Any, Arr(Box<Item>) }

/// Compile a JSON-Schema object to a template. Returns None if it isn't a supported object schema.
pub fn compile(schema: &serde_json::Value) -> Option<Vec<Item>> {
    if schema["type"].as_str()? != "object" { return None; }
    let mut prog = Vec::new();
    compile_object(schema, &mut prog);
    Some(prog)
}

/// Compile from a JSON-Schema *string* — convenience for callers that don't pull in serde_json
/// (e.g. the WASM/browser binding). Empty/invalid/non-object → None (caller falls back to json_object).
pub fn compile_str(schema_json: &str) -> Option<Vec<Item>> {
    if schema_json.trim().is_empty() { return None; }
    compile(&serde_json::from_str::<serde_json::Value>(schema_json).ok()?)
}

fn value_items(sub: &serde_json::Value, prog: &mut Vec<Item>) {
    if let Some(opts) = sub["enum"].as_array() {
        // enum of literals → each option as its exact JSON token bytes.
        let bytes: Vec<Vec<u8>> = opts.iter().map(|o| serde_json::to_vec(o).unwrap_or_default()).collect();
        prog.push(Item::Enum(bytes));
        return;
    }
    match sub["type"].as_str().unwrap_or("") {
        "string" => prog.push(Item::Str),
        "integer" => prog.push(Item::Int),
        "number" => prog.push(Item::Num),
        "boolean" => prog.push(Item::Bool),
        "object" => compile_object(sub, prog),
        // Typed array of a scalar/enum element → enforce `[`, elements, commas, `]`. Arrays of objects
        // or nested arrays (no single-Item element) fall back to a free JSON value.
        "array" => match scalar_item(&sub["items"]) {
            Some(el) => prog.push(Item::Arr(Box::new(el))),
            None => prog.push(Item::Any),
        },
        _ => prog.push(Item::Any), // null / unknown → free JSON value
    }
}

/// A sub-schema that compiles to a single scalar/enum `Item` (the allowed element of a typed array).
fn scalar_item(sub: &serde_json::Value) -> Option<Item> {
    if let Some(opts) = sub["enum"].as_array() {
        return Some(Item::Enum(opts.iter().map(|o| serde_json::to_vec(o).unwrap_or_default()).collect()));
    }
    match sub["type"].as_str().unwrap_or("") {
        "string" => Some(Item::Str),
        "integer" => Some(Item::Int),
        "number" => Some(Item::Num),
        "boolean" => Some(Item::Bool),
        _ => None,
    }
}

fn compile_object(schema: &serde_json::Value, prog: &mut Vec<Item>) {
    let empty = serde_json::Map::new();
    let props = schema["properties"].as_object().unwrap_or(&empty);
    if props.is_empty() { prog.push(Item::Lit(b"{}".to_vec())); return; }
    for (i, (key, sub)) in props.iter().enumerate() {
        let head = if i == 0 { format!("{{{}:", serde_json::to_string(key).unwrap()) } else { format!(",{}:", serde_json::to_string(key).unwrap()) };
        prog.push(Item::Lit(head.into_bytes()));
        value_items(sub, prog);
    }
    prog.push(Item::Lit(b"}".to_vec()));
}

#[derive(Clone, Copy)]
enum Val { Fresh, Str(u8), Num(NumSt), Bool(u8, u8), Enum(u64, usize), Any(Json), Arr(u8, ArrElem) } // Str(u8): 0 body,1 esc,2..=5 in \u; Arr(phase, element-substate)

/// In-progress state of the current element inside a typed array (mirrors the scalar `Val` variants,
/// but non-recursive so `Val::Arr` stays `Copy`). Enum's option list lives in the array's `Item`.
#[derive(Clone, Copy)]
enum ArrElem { Fresh, Str(u8), Num(NumSt), Bool(u8, u8), Enum(u64, usize) }

/// Outcome of feeding a byte to a typed-array element acceptor.
enum EO { Reject, Consumed, Completed, EndedBefore } // EndedBefore: byte isn't the element's (number closed by lookahead) — reprocess as structural

#[derive(Clone, Copy)]
pub struct Schema<'a> { prog: &'a [Item], pos: usize, loff: usize, val: Val }

impl<'a> Schema<'a> {
    pub fn new(prog: &'a [Item]) -> Self { Schema { prog, pos: 0, loff: 0, val: Val::Fresh } }
    pub fn can_stop(&self) -> bool { self.pos >= self.prog.len() }

    fn advance(&mut self) { self.pos += 1; self.loff = 0; self.val = Val::Fresh; }

    pub fn step(&mut self, b: u8) -> bool {
        if self.pos >= self.prog.len() { return false; }
        match &self.prog[self.pos] {
            Item::Lit(bytes) => {
                if self.loff < bytes.len() && bytes[self.loff] == b { self.loff += 1; if self.loff == bytes.len() { self.advance(); } true } else { false }
            }
            Item::Str => self.step_str(b),
            Item::Int => self.step_num(b, true),
            Item::Num => self.step_num(b, false),
            Item::Bool => self.step_bool(b),
            Item::Enum(opts) => self.step_enum(b, opts),
            Item::Any => self.step_any(b),
            Item::Arr(elem) => self.step_arr(elem, b),
        }
    }

    /// Typed array: `[` (elem (`,` elem)*)? `]`, each elem constrained to `elem`'s scalar/enum type.
    /// Phases: 0 need `[`, 1 after `[` (elem or `]`), 2 in elem, 3 after elem (`,` or `]`), 4 after `,`.
    fn step_arr(&mut self, elem: &Item, b: u8) -> bool {
        let (phase, mut ev) = match self.val { Val::Arr(p, e) => (p, e), Val::Fresh => (0u8, ArrElem::Fresh), _ => return false };
        match phase {
            0 => if b == b'[' { self.val = Val::Arr(1, ArrElem::Fresh); true } else { false },
            1 | 4 => {
                if phase == 1 && b == b']' { self.advance(); return true; } // empty array (only right after `[`)
                match step_elem(elem, &mut ev, b) {
                    EO::Consumed => { self.val = Val::Arr(2, ev); true }
                    EO::Completed => { self.val = Val::Arr(3, ArrElem::Fresh); true }
                    EO::Reject | EO::EndedBefore => false,
                }
            }
            2 => match step_elem(elem, &mut ev, b) {
                EO::Consumed => { self.val = Val::Arr(2, ev); true }
                EO::Completed => { self.val = Val::Arr(3, ArrElem::Fresh); true }
                EO::EndedBefore => { self.val = Val::Arr(3, ArrElem::Fresh); self.step_arr(elem, b) } // number closed — reprocess byte as `,`/`]`
                EO::Reject => false,
            },
            3 => {
                if b == b',' { self.val = Val::Arr(4, ArrElem::Fresh); true }
                else if b == b']' { self.advance(); true }
                else { false }
            }
            _ => false,
        }
    }

    fn step_str(&mut self, b: u8) -> bool {
        match self.val {
            Val::Fresh => { if b == b'"' { self.val = Val::Str(0); true } else { false } }
            Val::Str(0) => { if b == b'"' { self.advance(); true } else if b == b'\\' { self.val = Val::Str(1); true } else if b < 0x20 { false } else { true } }
            Val::Str(1) => { if matches!(b, b'"'|b'\\'|b'/'|b'b'|b'f'|b'n'|b'r'|b't') { self.val = Val::Str(0); true } else if b == b'u' { self.val = Val::Str(2); true } else { false } }
            Val::Str(n) => { if b.is_ascii_hexdigit() { self.val = if n == 5 { Val::Str(0) } else { Val::Str(n + 1) }; true } else { false } }
            _ => false,
        }
    }

    fn step_num(&mut self, b: u8, int_only: bool) -> bool {
        let st = match self.val { Val::Num(s) => s, Val::Fresh => { // start
            let s = match b { b'-' => NumSt::Neg, b'0' => NumSt::IntZero, b'1'..=b'9' => NumSt::Int, _ => return false };
            self.val = Val::Num(s); return true;
        } _ => return false };
        let dig = b.is_ascii_digit();
        let next = match st {
            NumSt::Neg => if b == b'0' { Some(NumSt::IntZero) } else if (b'1'..=b'9').contains(&b) { Some(NumSt::Int) } else { None },
            NumSt::IntZero => if !int_only && b == b'.' { Some(NumSt::Dot) } else if !int_only && (b == b'e' || b == b'E') { Some(NumSt::Exp) } else { None },
            NumSt::Int => if dig { Some(NumSt::Int) } else if !int_only && b == b'.' { Some(NumSt::Dot) } else if !int_only && (b == b'e' || b == b'E') { Some(NumSt::Exp) } else { None },
            NumSt::Dot => if dig { Some(NumSt::Frac) } else { None },
            NumSt::Frac => if dig { Some(NumSt::Frac) } else if b == b'e' || b == b'E' { Some(NumSt::Exp) } else { None },
            NumSt::Exp => if b == b'+' || b == b'-' { Some(NumSt::ExpSign) } else if dig { Some(NumSt::ExpDig) } else { None },
            NumSt::ExpSign => if dig { Some(NumSt::ExpDig) } else { None },
            NumSt::ExpDig => if dig { Some(NumSt::ExpDig) } else { None },
        };
        match next {
            Some(s2) => { self.val = Val::Num(s2); true }
            None if st.completable() => { self.advance(); self.step(b) } // number ended — reprocess byte against next item
            None => false,
        }
    }

    fn step_bool(&mut self, b: u8) -> bool {
        let (kind, m) = match self.val { Val::Bool(k, m) => (k, m), Val::Fresh => {
            match b { b't' => { self.val = Val::Bool(0, 1); return true } b'f' => { self.val = Val::Bool(1, 1); return true } _ => return false }
        } _ => return false };
        let word: &[u8] = if kind == 0 { b"true" } else { b"false" };
        if (m as usize) < word.len() && word[m as usize] == b {
            let nm = m + 1; if nm as usize == word.len() { self.advance(); } else { self.val = Val::Bool(kind, nm); } true
        } else { false }
    }

    fn step_enum(&mut self, b: u8, opts: &[Vec<u8>]) -> bool {
        let (mut mask, off) = match self.val { Val::Enum(m, o) => (m, o), Val::Fresh => ((1u64 << opts.len().min(64)) - 1, 0), _ => return false };
        // keep only options whose byte `off` == b
        let mut any = false;
        for (i, o) in opts.iter().enumerate().take(64) {
            if mask & (1 << i) != 0 { if off < o.len() && o[off] == b { any = true; } else { mask &= !(1 << i); } }
        }
        if !any { return false; }
        let noff = off + 1;
        // if any surviving option is fully matched at noff, that option completes the value
        let done = opts.iter().enumerate().take(64).any(|(i, o)| mask & (1 << i) != 0 && o.len() == noff);
        if done { self.advance(); } else { self.val = Val::Enum(mask, noff); }
        true
    }

    fn step_any(&mut self, b: u8) -> bool {
        let mut j = match self.val { Val::Any(j) => j, Val::Fresh => Json::new(), _ => return false };
        if !j.step(b) {
            // value may have ended (e.g. a number closed by the next literal) — if complete, reprocess.
            if j.can_stop() { self.advance(); return self.step(b); }
            return false;
        }
        // a complete top-level value in the embedded acceptor → this slot is done, but Json can't tell
        // mid-stream for objects/strings; it reports can_stop after closing. Advance when it can stop AND
        // the char just consumed closed it (handled by re-entry above for numbers). For strings/objects,
        // detect completion:
        if j.can_stop() { self.advance(); } else { self.val = Val::Any(j); }
        true
    }
}

/// Feed one byte to a typed-array element acceptor (`item` is Str/Int/Num/Bool/Enum), tracking the
/// element's in-progress state in `ev`. Mirrors the scalar `Schema` steppers but reports completion
/// (via `EO`) instead of advancing the program, so the array loop owns the repetition.
fn step_elem(item: &Item, ev: &mut ArrElem, b: u8) -> EO {
    match item {
        Item::Str => match *ev {
            ArrElem::Fresh => if b == b'"' { *ev = ArrElem::Str(0); EO::Consumed } else { EO::Reject },
            ArrElem::Str(0) => if b == b'"' { EO::Completed } else if b == b'\\' { *ev = ArrElem::Str(1); EO::Consumed } else if b < 0x20 { EO::Reject } else { EO::Consumed },
            ArrElem::Str(1) => if matches!(b, b'"'|b'\\'|b'/'|b'b'|b'f'|b'n'|b'r'|b't') { *ev = ArrElem::Str(0); EO::Consumed } else if b == b'u' { *ev = ArrElem::Str(2); EO::Consumed } else { EO::Reject },
            ArrElem::Str(n) => if b.is_ascii_hexdigit() { *ev = if n == 5 { ArrElem::Str(0) } else { ArrElem::Str(n + 1) }; EO::Consumed } else { EO::Reject },
            _ => EO::Reject,
        },
        Item::Int | Item::Num => {
            let int_only = matches!(item, Item::Int);
            match *ev {
                ArrElem::Fresh => { let s = match b { b'-' => NumSt::Neg, b'0' => NumSt::IntZero, b'1'..=b'9' => NumSt::Int, _ => return EO::Reject }; *ev = ArrElem::Num(s); EO::Consumed }
                ArrElem::Num(st) => {
                    let dig = b.is_ascii_digit();
                    let next = match st {
                        NumSt::Neg => if b == b'0' { Some(NumSt::IntZero) } else if (b'1'..=b'9').contains(&b) { Some(NumSt::Int) } else { None },
                        NumSt::IntZero => if !int_only && b == b'.' { Some(NumSt::Dot) } else if !int_only && (b == b'e' || b == b'E') { Some(NumSt::Exp) } else { None },
                        NumSt::Int => if dig { Some(NumSt::Int) } else if !int_only && b == b'.' { Some(NumSt::Dot) } else if !int_only && (b == b'e' || b == b'E') { Some(NumSt::Exp) } else { None },
                        NumSt::Dot => if dig { Some(NumSt::Frac) } else { None },
                        NumSt::Frac => if dig { Some(NumSt::Frac) } else if b == b'e' || b == b'E' { Some(NumSt::Exp) } else { None },
                        NumSt::Exp => if b == b'+' || b == b'-' { Some(NumSt::ExpSign) } else if dig { Some(NumSt::ExpDig) } else { None },
                        NumSt::ExpSign => if dig { Some(NumSt::ExpDig) } else { None },
                        NumSt::ExpDig => if dig { Some(NumSt::ExpDig) } else { None },
                    };
                    match next { Some(s2) => { *ev = ArrElem::Num(s2); EO::Consumed } None if st.completable() => EO::EndedBefore, None => EO::Reject }
                }
                _ => EO::Reject,
            }
        }
        Item::Bool => match *ev {
            ArrElem::Fresh => match b { b't' => { *ev = ArrElem::Bool(0, 1); EO::Consumed } b'f' => { *ev = ArrElem::Bool(1, 1); EO::Consumed } _ => EO::Reject },
            ArrElem::Bool(k, m) => { let word: &[u8] = if k == 0 { b"true" } else { b"false" }; if (m as usize) < word.len() && word[m as usize] == b { let nm = m + 1; if nm as usize == word.len() { EO::Completed } else { *ev = ArrElem::Bool(k, nm); EO::Consumed } } else { EO::Reject } }
            _ => EO::Reject,
        },
        Item::Enum(opts) => {
            let (mask, off) = match *ev { ArrElem::Enum(m, o) => (m, o), ArrElem::Fresh => ((1u64 << opts.len().min(64)) - 1, 0), _ => return EO::Reject };
            let mut m2 = mask; let mut any = false;
            for (i, o) in opts.iter().enumerate().take(64) {
                if m2 & (1 << i) != 0 { if off < o.len() && o[off] == b { any = true; } else { m2 &= !(1 << i); } }
            }
            if !any { return EO::Reject; }
            let noff = off + 1;
            let done = opts.iter().enumerate().take(64).any(|(i, o)| m2 & (1 << i) != 0 && o.len() == noff);
            if done { EO::Completed } else { *ev = ArrElem::Enum(m2, noff); EO::Consumed }
        }
        _ => EO::Reject,
    }
}

/// A guided-decoding constraint: free-form-but-valid JSON, or schema-conformant JSON.
#[derive(Clone, Copy)]
pub enum Guide<'a> { Json(Json), Schema(Schema<'a>) }
impl<'a> Guide<'a> {
    pub fn step(&mut self, b: u8) -> bool { match self { Guide::Json(j) => j.step(b), Guide::Schema(s) => s.step(b) } }
    pub fn can_stop(&self) -> bool { match self { Guide::Json(j) => j.can_stop(), Guide::Schema(s) => s.can_stop() } }
}

#[cfg(test)]
mod tests {
    use super::{compile, Json, Schema};
    fn sch_accepts(schema: &str, out: &str) -> bool {
        let v: serde_json::Value = serde_json::from_str(schema).unwrap();
        let prog = compile(&v).expect("compile");
        let mut s = Schema::new(&prog);
        for &b in out.as_bytes() { if !s.step(b) { return false; } }
        s.can_stop()
    }
    #[test]
    fn schema_object() {
        let sc = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}"#;
        assert!(sch_accepts(sc, r#"{"name":"Bob","age":42}"#));
        assert!(!sch_accepts(sc, r#"{"name":"Bob","age":4.2}"#));   // integer slot rejects a float
        assert!(!sch_accepts(sc, r#"{"age":42,"name":"Bob"}"#));    // wrong key order
        assert!(!sch_accepts(sc, r#"{"name":"Bob"}"#));             // missing required field
        assert!(!sch_accepts(sc, r#"{"name":42,"age":42}"#));       // string slot rejects a number
    }
    #[test]
    fn schema_enum_and_types() {
        let sc = r#"{"type":"object","properties":{"color":{"enum":["red","green"]},"ok":{"type":"boolean"},"n":{"type":"number"}}}"#;
        assert!(sch_accepts(sc, r#"{"color":"red","ok":true,"n":-3.5e2}"#));
        assert!(sch_accepts(sc, r#"{"color":"green","ok":false,"n":0}"#));
        assert!(!sch_accepts(sc, r#"{"color":"blue","ok":true,"n":1}"#)); // not in enum
        assert!(!sch_accepts(sc, r#"{"color":"red","ok":yes,"n":1}"#));   // bad boolean
    }
    #[test]
    fn schema_nested() {
        let sc = r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"object","properties":{"c":{"type":"string"}}}}}"#;
        assert!(sch_accepts(sc, r#"{"a":1,"b":{"c":"hi"}}"#));
        assert!(!sch_accepts(sc, r#"{"a":1,"b":{"c":9}}"#)); // nested string slot rejects number
    }
    #[test]
    fn schema_typed_arrays() {
        let sc = r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}},"nums":{"type":"array","items":{"type":"integer"}}}}"#;
        assert!(sch_accepts(sc, r#"{"tags":["a","bb","c"],"nums":[1,2,30]}"#));
        assert!(sch_accepts(sc, r#"{"tags":[],"nums":[]}"#));            // empty arrays
        assert!(sch_accepts(sc, r#"{"tags":["x"],"nums":[-5]}"#));
        assert!(!sch_accepts(sc, r#"{"tags":["a",1],"nums":[]}"#));      // string array rejects a number element
        assert!(!sch_accepts(sc, r#"{"tags":[],"nums":[1,2.5]}"#));      // integer array rejects a float
        assert!(!sch_accepts(sc, r#"{"tags":["a",],"nums":[]}"#));       // trailing comma rejected
        assert!(!sch_accepts(sc, r#"{"tags":"a","nums":[]}"#));          // array slot rejects a bare string
        assert!(!sch_accepts(sc, r#"{"tags":[,"nums":[]}"#));            // leading comma / no element
    }
    #[test]
    fn schema_array_of_enum_and_bool() {
        let sc = r#"{"type":"object","properties":{"flags":{"type":"array","items":{"type":"boolean"}},"cols":{"type":"array","items":{"enum":["r","g","b"]}}}}"#;
        assert!(sch_accepts(sc, r#"{"flags":[true,false,true],"cols":["r","b"]}"#));
        assert!(!sch_accepts(sc, r#"{"flags":[true,1],"cols":["r"]}"#));  // bool array rejects a number
        assert!(!sch_accepts(sc, r#"{"flags":[],"cols":["x"]}"#));        // enum array rejects out-of-set
    }

    fn accepts(s: &str) -> bool {
        let mut j = Json::new();
        for &b in s.as_bytes() { if !j.step(b) { return false; } }
        j.can_stop()
    }
    #[test]
    fn valid() {
        for s in ["{}", "[]", "{\"a\":1}", "{\"a\": [1, 2, 3], \"b\": {\"c\": true}}", "  {\n\"x\": \"hi\\n\", \"y\": -3.14e2, \"z\": null}  ", "[true,false,null,\"s\"]"] {
            assert!(accepts(s), "should accept: {s}");
        }
    }
    #[test]
    fn invalid() {
        for s in ["{", "{\"a\"}", "{\"a\":}", "[1,]", "{,}", "{\"a\":1,}", "truue", "{'a':1}", "[1 2]",
                  "1.", "01", "1e", "1.e5", "-", "{\"a\": 1.}", "[1.]"] {
            assert!(!accepts(s), "should reject: {s}");
        }
    }
    #[test]
    fn valid_numbers() {
        for s in ["[0]", "[1.5]", "[-3]", "[1e10]", "[1.2e-3]", "[0.5]", "[123]"] { assert!(accepts(s), "num: {s}"); }
    }
    #[test]
    fn object_mode() {
        let ok = |s: &str, obj: bool| { let mut j = if obj { Json::object() } else { Json::new() }; s.bytes().all(|b| j.step(b)) && j.can_stop() };
        assert!(ok("{\"a\":1}", true));   // object accepted in object mode
        assert!(!ok("[1,2]", true));      // array rejected in object mode
        assert!(!ok("42", true));         // bare number rejected in object mode
        assert!(ok("[1,2]", false));      // array fine in general mode
    }
}

#[cfg(test)]
pub(crate) fn accepts_top(s: &str) -> bool {
    let mut j = Json::new();
    for &b in s.as_bytes() { if !j.step(b) { return false; } }
    j.can_stop()
}
