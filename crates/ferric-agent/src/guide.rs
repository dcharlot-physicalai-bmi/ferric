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

/// One template element: a fixed literal to emit, or a typed value slot. Objects with optional/required
/// properties compile to an `ObjOpen … (Prop <value>)* ObjEnd` run (properties emitted contiguously); the
/// acceptor walks it, skipping optional props and enforcing that every required one appears.
#[derive(Clone)]
pub enum Item {
    Lit(Vec<u8>), Str(u32, u32), Int, Num, Bool, Enum(Vec<Vec<u8>>), Any, Arr(Box<Item>, u32, u32), // Str(minLength, maxLength); Arr(element, minItems, maxItems)
    IntRange(i64, i64), // an integer constrained to [minimum, maximum] (inclusive)
    ObjOpen { unsat0: u32, close: u32 }, // matches `{`; unsat0 = bits of the required props; close = prog idx of the matching ObjEnd
    Prop(Vec<Cand>),                     // a property boundary (the object's cursor): choose a candidate key, or `,`/`}`
    ObjEnd,                              // matches `}`
}

/// A candidate property at a `Prop` boundary: `key` = the exact `"name":` bytes (prefix-free), `target`
/// = prog idx of this property's value, `bit` = its `1<<idx` (cleared from the frame's required set).
#[derive(Clone)]
pub struct Cand { key: Vec<u8>, target: u32, bit: u32 }

/// Compile a JSON-Schema object to a template. Returns None if it isn't a supported object schema.
pub fn compile(schema: &serde_json::Value) -> Option<Vec<Item>> {
    if schema["type"].as_str()? != "object" { return None; }
    let mut prog = Vec::new();
    compile_object(schema, &mut prog, 0);
    Some(prog)
}

/// Compile from a JSON-Schema *string* — convenience for callers that don't pull in serde_json
/// (e.g. the WASM/browser binding). Empty/invalid/non-object → None (caller falls back to json_object).
pub fn compile_str(schema_json: &str) -> Option<Vec<Item>> {
    if schema_json.trim().is_empty() { return None; }
    compile(&serde_json::from_str::<serde_json::Value>(schema_json).ok()?)
}

fn value_items(sub: &serde_json::Value, prog: &mut Vec<Item>, depth: usize) {
    if let Some(opts) = sub["enum"].as_array() {
        // enum of literals → each option as its exact JSON token bytes.
        let bytes: Vec<Vec<u8>> = opts.iter().map(|o| serde_json::to_vec(o).unwrap_or_default()).collect();
        prog.push(Item::Enum(bytes));
        return;
    }
    match sub["type"].as_str().unwrap_or("") {
        "string" => prog.push(str_item(sub)),
        "integer" => prog.push(int_item(sub)),
        "number" => prog.push(Item::Num),
        "boolean" => prog.push(Item::Bool),
        "object" => compile_object(sub, prog, depth + 1),
        // Typed array of a scalar/enum element → enforce `[`, elements, commas, `]`. Arrays of objects
        // or nested arrays (no single-Item element) fall back to a free JSON value.
        "array" => match scalar_item(&sub["items"]) {
            Some(el) => {
                let min = sub["minItems"].as_u64().unwrap_or(0) as u32;
                let max = sub["maxItems"].as_u64().map(|v| v as u32).unwrap_or(u32::MAX);
                prog.push(Item::Arr(Box::new(el), min, max));
            }
            None => prog.push(Item::Any),
        },
        _ => prog.push(Item::Any), // null / unknown → free JSON value
    }
}

/// Build an `Item::Str` carrying the schema's `minLength`/`maxLength` (code points). Absent ⇒ 0 / unbounded.
fn str_item(sub: &serde_json::Value) -> Item {
    let min = sub["minLength"].as_u64().unwrap_or(0) as u32;
    let max = sub["maxLength"].as_u64().map(|v| v as u32).unwrap_or(u32::MAX);
    Item::Str(min, max)
}

/// Read an integer bound: an integer as-is, or a float rounded INTO the allowed range (ceil for the lower
/// bound, floor for the upper) so `minimum: 0.0` / `maximum: 5.5` still constrain. Out-of-i64 → None.
fn int_bound(v: &serde_json::Value, is_min: bool) -> Option<i64> {
    if let Some(i) = v.as_i64() { return Some(i); }
    let f = v.as_f64()?;
    let x = if is_min { f.ceil() } else { f.floor() };
    if (i64::MIN as f64..=i64::MAX as f64).contains(&x) { Some(x as i64) } else { None }
}

/// Build an integer `Item`, honoring inclusive `minimum`/`maximum`. Both absent ⇒ plain `Item::Int`
/// (unbounded); a nonsensical `min > max` also falls back to `Item::Int` so the acceptor never deadlocks.
/// (`exclusiveMinimum`/`exclusiveMaximum` and float `number` bounds are not yet enforced.)
fn int_item(sub: &serde_json::Value) -> Item {
    let lo = int_bound(&sub["minimum"], true);
    let hi = int_bound(&sub["maximum"], false);
    if lo.is_none() && hi.is_none() { return Item::Int; }
    let (lo, hi) = (lo.unwrap_or(i64::MIN), hi.unwrap_or(i64::MAX));
    if lo > hi { Item::Int } else { Item::IntRange(lo, hi) }
}

/// A sub-schema that compiles to a single scalar/enum `Item` (the allowed element of a typed array).
fn scalar_item(sub: &serde_json::Value) -> Option<Item> {
    if let Some(opts) = sub["enum"].as_array() {
        return Some(Item::Enum(opts.iter().map(|o| serde_json::to_vec(o).unwrap_or_default()).collect()));
    }
    match sub["type"].as_str().unwrap_or("") {
        "string" => Some(str_item(sub)),
        "integer" => Some(Item::Int),
        "number" => Some(Item::Num),
        "boolean" => Some(Item::Bool),
        _ => None,
    }
}

const MAX_PROPS: usize = 32; // required/emitted bitmasks are u32; wider objects fall back to free JSON

/// Compile an object schema to `ObjOpen … (Prop <value>)* ObjEnd`, with required/optional handling.
fn compile_object(schema: &serde_json::Value, prog: &mut Vec<Item>, depth: usize) {
    let empty = serde_json::Map::new();
    // NB: relies on serde_json's `preserve_order` feature (see Cargo.toml) so `props.keys()` yields the
    // schema's DECLARATION order. Without it the Map is a BTreeMap and the acceptor would silently enforce
    // ALPHABETICAL property order instead — breaking every schema whose fields aren't declared sorted.
    let props = schema["properties"].as_object().unwrap_or(&empty);
    // Beyond the runtime object-stack depth or the bitmask width, fall back to free-but-valid JSON so the
    // Copy acceptor can never overflow — never a panic or silent truncation.
    if depth >= MAX_OBJ_DEPTH || props.len() > MAX_PROPS { prog.push(Item::Any); return; }
    // POLICY: an absent `required` means "all properties required" (the extract-these-fields contract,
    // keeps legacy schemas byte-compatible). `required:[]` or a subset makes the remaining props optional.
    let req_list = schema["required"].as_array();
    let reqd: Vec<bool> = props.keys()
        .map(|n| match req_list { Some(a) => a.iter().any(|v| v.as_str() == Some(n.as_str())), None => true })
        .collect();
    let names: Vec<&str> = props.keys().map(|s| s.as_str()).collect();
    // Pass 1: ObjOpen placeholder, then each Prop placeholder + its value, then ObjEnd.
    let open = prog.len();
    prog.push(Item::ObjOpen { unsat0: 0, close: 0 });
    let mut pp = Vec::with_capacity(names.len());
    for sub in props.values() {
        pp.push(prog.len());
        prog.push(Item::Prop(Vec::new()));
        value_items(sub, prog, depth);
    }
    let close = prog.len();
    prog.push(Item::ObjEnd);
    // Pass 2: backpatch now that all indices are known.
    let mut unsat0 = 0u32;
    for (i, &r) in reqd.iter().enumerate() { if r { unsat0 |= 1 << i; } }
    prog[open] = Item::ObjOpen { unsat0, close: close as u32 };
    for j in 0..names.len() {
        let mut cands = Vec::new();
        for m in j..names.len() {
            let mut key = serde_json::to_string(names[m]).unwrap().into_bytes();
            key.push(b':');
            cands.push(Cand { key, target: (pp[m] + 1) as u32, bit: 1 << m });
            if reqd[m] { break; } // stop AFTER the first required — a required prop is never a skip target
        }
        prog[pp[j]] = Item::Prop(cands);
    }
}

#[derive(Clone, Copy)]
enum Val { Fresh, Str(u8, u16), Num(NumSt), Bool(u8, u8), Enum(u64, usize), Any(Json), Arr(u8, u32, ArrElem), Key(u32, u16), IntR { neg: bool, digits: u8, lead0: bool, mag: i128 } } // Str(state, code-points-so-far): state 0 body,1 esc,2..=5 in \u; Arr(phase, elements-so-far, element-substate); Key(surviving-candidate mask, key bytes matched); IntR = bounded integer (sign + magnitude accumulated)

/// One open object level (bounded, no heap): `unsat` = required props not yet emitted; `close` = prog idx
/// of this object's ObjEnd. Nesting ≤ 8 (compile falls deeper objects back to free JSON).
#[derive(Clone, Copy)]
struct Frame { unsat: u32, close: u32 }
const MAX_OBJ_DEPTH: usize = 8;

/// In-progress state of the current element inside a typed array (mirrors the scalar `Val` variants,
/// but non-recursive so `Val::Arr` stays `Copy`). Enum's option list lives in the array's `Item`.
#[derive(Clone, Copy)]
enum ArrElem { Fresh, Str(u8, u16), Num(NumSt), Bool(u8, u8), Enum(u64, usize) }

/// Outcome of feeding a byte to a typed-array element acceptor.
enum EO { Reject, Consumed, Completed, EndedBefore } // EndedBefore: byte isn't the element's (number closed by lookahead) — reprocess as structural

#[derive(Clone, Copy)]
pub struct Schema<'a> { prog: &'a [Item], pos: usize, loff: usize, val: Val, stack: [Frame; MAX_OBJ_DEPTH], depth: u8, emitted: u16 }

impl<'a> Schema<'a> {
    pub fn new(prog: &'a [Item]) -> Self { Schema { prog, pos: 0, loff: 0, val: Val::Fresh, stack: [Frame { unsat: 0, close: 0 }; MAX_OBJ_DEPTH], depth: 0, emitted: 0 } }
    /// Safe to emit EOS: the whole program is consumed AND we're not sitting inside an open object.
    pub fn can_stop(&self) -> bool { self.depth == 0 && self.pos >= self.prog.len() }

    fn advance(&mut self) { self.pos += 1; self.loff = 0; self.val = Val::Fresh; }

    pub fn step(&mut self, b: u8) -> bool {
        if self.pos >= self.prog.len() { return false; }
        let prog = self.prog; // decouple the &[Item] borrow from &mut self used by the steppers
        match &prog[self.pos] {
            Item::Lit(bytes) => {
                if self.loff < bytes.len() && bytes[self.loff] == b { self.loff += 1; if self.loff == bytes.len() { self.advance(); } true } else { false }
            }
            Item::Str(min, max) => self.step_str(b, *min, *max),
            Item::Int => self.step_num(b, true),
            Item::IntRange(lo, hi) => self.step_intrange(b, *lo, *hi),
            Item::Num => self.step_num(b, false),
            Item::Bool => self.step_bool(b),
            Item::Enum(opts) => self.step_enum(b, opts),
            Item::Any => self.step_any(b),
            Item::Arr(elem, min, max) => self.step_arr(elem, *min, *max, b),
            Item::ObjOpen { unsat0, close } => self.step_objopen(*unsat0, *close, b),
            Item::Prop(cands) => self.step_prop(cands, b),
            Item::ObjEnd => self.step_objend(b),
        }
    }

    /// `{` opens an object level: push a frame with its required-set, land on the first `Prop` (or `ObjEnd`).
    fn step_objopen(&mut self, unsat0: u32, close: u32, b: u8) -> bool {
        if !matches!(self.val, Val::Fresh) || b != b'{' || self.depth as usize >= MAX_OBJ_DEPTH { return false; }
        self.stack[self.depth as usize] = Frame { unsat: unsat0, close };
        self.emitted &= !(1 << self.depth); // this level hasn't emitted a property yet
        self.depth += 1;
        self.advance();
        true
    }

    /// At a property boundary: choose a candidate key (`"name":`), or `,` before the next key, or `}` to
    /// close (only if no required prop is still missing). `cands` = [current .. first-required inclusive].
    fn step_prop(&mut self, cands: &[Cand], b: u8) -> bool {
        let d = (self.depth - 1) as usize;
        let full = if cands.len() >= 32 { u32::MAX } else { (1u32 << cands.len()) - 1 };
        match self.val {
            Val::Fresh => {
                let em = self.emitted & (1 << (self.depth - 1)) != 0;
                if b == b'}' {
                    if self.stack[d].unsat != 0 { return false; } // a required property is still missing
                    self.depth -= 1;
                    self.pos = self.stack[d].close as usize + 1; self.loff = 0; self.val = Val::Fresh;
                    true
                } else if b == b',' {
                    if !em { return false; } // comma only between properties, never leading
                    self.val = Val::Key(full, 0); true
                } else if b == b'"' {
                    if em { return false; } // first property: no leading comma
                    self.key_feed(cands, full, 0, b) // this `"` is the first key byte
                } else { false }
            }
            Val::Key(mask, koff) => self.key_feed(cands, mask, koff, b),
            _ => false,
        }
    }

    /// Match one key byte against the surviving candidates (prefix-free → ≤1 completes). On completion,
    /// clear the property's required bit, mark this level as having emitted, and jump to its value.
    fn key_feed(&mut self, cands: &[Cand], mask: u32, koff: u16, b: u8) -> bool {
        let ko = koff as usize;
        let mut nm = 0u32;
        for (i, c) in cands.iter().enumerate().take(32) {
            if mask & (1 << i) != 0 && ko < c.key.len() && c.key[ko] == b { nm |= 1 << i; }
        }
        if nm == 0 { return false; } // wrong order / duplicate / unknown key → no candidate survives
        let noff = ko + 1;
        for (i, c) in cands.iter().enumerate().take(32) {
            if nm & (1 << i) != 0 && c.key.len() == noff {
                let d = (self.depth - 1) as usize;
                self.stack[d].unsat &= !c.bit;         // required satisfied once the key appears
                self.emitted |= 1 << (self.depth - 1);
                self.pos = c.target as usize; self.loff = 0; self.val = Val::Fresh;
                return true;
            }
        }
        self.val = Val::Key(nm, noff as u16); true
    }

    /// `}` reached after the last emitted property's value (or on an empty object). Close if satisfied.
    fn step_objend(&mut self, b: u8) -> bool {
        if !matches!(self.val, Val::Fresh) { return false; }
        let d = (self.depth - 1) as usize;
        if b == b'}' && self.stack[d].unsat == 0 { self.depth -= 1; self.advance(); true } else { false }
    }

    /// Typed array: `[` (elem (`,` elem)*)? `]`, each elem constrained to `elem`'s scalar/enum type,
    /// element count in `[min, max]`. Phases: 0 need `[`, 1 after `[` (elem or `]`), 2 in elem, 3 after
    /// elem (`,` or `]`), 4 after `,`. `count` = completed elements so far.
    fn step_arr(&mut self, elem: &Item, min: u32, max: u32, b: u8) -> bool {
        let (phase, count, mut ev) = match self.val { Val::Arr(p, c, e) => (p, c, e), Val::Fresh => (0u8, 0u32, ArrElem::Fresh), _ => return false };
        match phase {
            0 => if b == b'[' { self.val = Val::Arr(1, 0, ArrElem::Fresh); true } else { false },
            1 | 4 => {
                if phase == 1 && b == b']' { if count >= min { self.advance(); return true; } return false; } // empty/short array
                if count >= max { return false; } // maxItems reached — no more elements
                match step_elem(elem, &mut ev, b) {
                    EO::Consumed => { self.val = Val::Arr(2, count, ev); true }
                    EO::Completed => { self.val = Val::Arr(3, count + 1, ArrElem::Fresh); true }
                    EO::Reject | EO::EndedBefore => false,
                }
            }
            2 => match step_elem(elem, &mut ev, b) {
                EO::Consumed => { self.val = Val::Arr(2, count, ev); true }
                EO::Completed => { self.val = Val::Arr(3, count + 1, ArrElem::Fresh); true }
                EO::EndedBefore => { self.val = Val::Arr(3, count + 1, ArrElem::Fresh); self.step_arr(elem, min, max, b) } // number closed — reprocess byte as `,`/`]`
                EO::Reject => false,
            },
            3 => {
                if b == b',' && count < max { self.val = Val::Arr(4, count, ArrElem::Fresh); true }
                else if b == b']' && count >= min { self.advance(); true }
                else { false }
            }
            _ => false,
        }
    }

    // `min`/`max` = minLength/maxLength in Unicode code points; `c` = code points *completed* so far;
    // `max` = u32::MAX means unbounded. A new code point may START only while `c < max`, and its count is
    // incremented when it COMPLETES — so a multibyte char can never be half-emitted and then closed (which
    // would orphan a UTF-8 lead byte into invalid output). State byte: 0 body, 1 after `\`, 2..=5 in `\uXXXX`,
    // 6/7/8 mid-multibyte char with 1/2/3 continuation bytes still to come. Closing `"` is legal only in
    // state 0 (never mid-char), which also forces a clean close once the length cap is hit.
    fn step_str(&mut self, b: u8, min: u32, max: u32) -> bool {
        match self.val {
            Val::Fresh => { if b == b'"' { self.val = Val::Str(0, 0); true } else { false } }
            Val::Str(0, c) => {
                let at_max = (c as u32) >= max;
                if b == b'"' { if (c as u32) < min { return false; } self.advance(); true }              // close: minLength gate
                else if b == b'\\' { if at_max { return false; } self.val = Val::Str(1, c); true }        // escape = one code point
                else if b < 0x20 { false }                                                                // raw control char: invalid JSON
                else if b < 0x80 { if at_max { return false; } self.val = Val::Str(0, c.saturating_add(1)); true } // ASCII code point
                else if b < 0xC0 { false }                                                                // stray continuation with no lead
                else if b < 0xE0 { if at_max { return false; } self.val = Val::Str(6, c); true }          // 2-byte lead → 1 more
                else if b < 0xF0 { if at_max { return false; } self.val = Val::Str(7, c); true }          // 3-byte lead → 2 more
                else if b < 0xF8 { if at_max { return false; } self.val = Val::Str(8, c); true }          // 4-byte lead → 3 more
                else { false }                                                                            // invalid lead byte
            }
            Val::Str(1, c) => { if matches!(b, b'"'|b'\\'|b'/'|b'b'|b'f'|b'n'|b'r'|b't') { self.val = Val::Str(0, c.saturating_add(1)); true } else if b == b'u' { self.val = Val::Str(2, c); true } else { false } }
            Val::Str(n @ 2..=5, c) => { if b.is_ascii_hexdigit() { self.val = if n == 5 { Val::Str(0, c.saturating_add(1)) } else { Val::Str(n + 1, c) }; true } else { false } }
            Val::Str(6, c) => { if (0x80..0xC0).contains(&b) { self.val = Val::Str(0, c.saturating_add(1)); true } else { false } } // last continuation → code point complete
            Val::Str(7, c) => { if (0x80..0xC0).contains(&b) { self.val = Val::Str(6, c); true } else { false } }
            Val::Str(8, c) => { if (0x80..0xC0).contains(&b) { self.val = Val::Str(7, c); true } else { false } }
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

    /// Bounded integer `[lo, hi]` (inclusive). Accepts a digit iff the value can still terminate in range
    /// OR be extended into range (`int_feasible`); the terminator (any non-digit) closes the number and is
    /// reprocessed against the next item, valid iff the value landed in range. JSON integer grammar:
    /// optional `-`, then `0` or `[1-9][0-9]*` (no leading zeros). Value = sign + i128 magnitude.
    fn step_intrange(&mut self, b: u8, lo: i64, hi: i64) -> bool {
        match self.val {
            Val::Fresh => {
                if b == b'-' {
                    if lo < 0 { self.val = Val::IntR { neg: true, digits: 0, lead0: false, mag: 0 }; true } else { false } // '-' only if a negative value is in range
                } else if b.is_ascii_digit() { self.intr_first_digit(b, false, lo, hi) } else { false }
            }
            Val::IntR { neg, digits, lead0, mag } => {
                if b.is_ascii_digit() {
                    if lead0 { return false; }                                    // no digit after a leading `0`/`-0`
                    if digits == 0 { self.intr_first_digit(b, neg, lo, hi) }       // first magnitude digit after `-`
                    else {
                        let mag2 = mag * 10 + (b - b'0') as i128;
                        if int_feasible(neg, mag2, lo, hi) { self.val = Val::IntR { neg, digits: digits + 1, lead0: false, mag: mag2 }; true } else { false }
                    }
                } else if digits >= 1 && in_range(int_value(neg, mag), lo, hi) { self.advance(); self.step(b) } // number complete + in range → reprocess terminator
                else { false }                                                    // incomplete (`-` alone) or out of range
            }
            _ => false,
        }
    }

    /// First digit of a bounded integer (magnitude 0 ⇒ `0`/`-0`, which JSON forbids extending).
    fn intr_first_digit(&mut self, b: u8, neg: bool, lo: i64, hi: i64) -> bool {
        let d = (b - b'0') as i128;
        if d == 0 {
            if in_range(0, lo, hi) { self.val = Val::IntR { neg, digits: 1, lead0: true, mag: 0 }; true } else { false }
        } else if int_feasible(neg, d, lo, hi) {
            self.val = Val::IntR { neg, digits: 1, lead0: false, mag: d }; true
        } else { false }
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

// ===== bounded-integer feasibility (used by step_intrange) =====
fn int_value(neg: bool, mag: i128) -> i128 { if neg { -mag } else { mag } }
fn in_range(v: i128, lo: i64, hi: i64) -> bool { v >= lo as i128 && v <= hi as i128 }

/// Can a partial integer of sign `neg` and magnitude `mag` still reach `[lo, hi]` — either by stopping
/// now, or by appending more digits? Appending `k ≥ 1` digits gives magnitude in
/// `[mag·10^k, (mag+1)·10^k − 1]`; feasible iff some `k`'s value-interval intersects `[lo, hi]`.
fn int_feasible(neg: bool, mag: i128, lo: i64, hi: i64) -> bool {
    in_range(int_value(neg, mag), lo, hi) || int_can_extend(neg, mag, lo, hi)
}
fn int_can_extend(neg: bool, mag: i128, lo: i64, hi: i64) -> bool {
    let (lo, hi) = (lo as i128, hi as i128);
    let mut pk: i128 = 1; // 10^k, k = 1,2,…
    for _ in 0..20 {
        pk = match pk.checked_mul(10) { Some(x) => x, None => break };
        let lo_mag = match mag.checked_mul(pk) { Some(x) => x, None => break };      // mag·10^k
        let hi_mag = match lo_mag.checked_add(pk - 1) { Some(x) => x, None => break }; // (mag+1)·10^k − 1
        let (vlo, vhi) = if neg { (-hi_mag, -lo_mag) } else { (lo_mag, hi_mag) };
        if vlo <= hi && vhi >= lo { return true; }                                    // interval ∩ [lo,hi] ≠ ∅
        if !neg && lo_mag > hi { break; }                                             // non-neg values only grow past hi
        if neg && -lo_mag < lo { break; }                                             // neg values only shrink past lo
    }
    false
}

/// Feed one byte to a typed-array element acceptor (`item` is Str/Int/Num/Bool/Enum), tracking the
/// element's in-progress state in `ev`. Mirrors the scalar `Schema` steppers but reports completion
/// (via `EO`) instead of advancing the program, so the array loop owns the repetition.
fn step_elem(item: &Item, ev: &mut ArrElem, b: u8) -> EO {
    match item {
        Item::Str(min, max) => match *ev {
            ArrElem::Fresh => if b == b'"' { *ev = ArrElem::Str(0, 0); EO::Consumed } else { EO::Reject },
            ArrElem::Str(0, c) => {
                let at_max = (c as u32) >= *max;
                if b == b'"' { if (c as u32) < *min { EO::Reject } else { EO::Completed } }              // minLength gate
                else if b == b'\\' { if at_max { EO::Reject } else { *ev = ArrElem::Str(1, c); EO::Consumed } }
                else if b < 0x20 { EO::Reject }
                else if b < 0x80 { if at_max { EO::Reject } else { *ev = ArrElem::Str(0, c.saturating_add(1)); EO::Consumed } } // ASCII
                else if b < 0xC0 { EO::Reject }                                                          // stray continuation
                else if b < 0xE0 { if at_max { EO::Reject } else { *ev = ArrElem::Str(6, c); EO::Consumed } } // 2-byte lead
                else if b < 0xF0 { if at_max { EO::Reject } else { *ev = ArrElem::Str(7, c); EO::Consumed } } // 3-byte lead
                else if b < 0xF8 { if at_max { EO::Reject } else { *ev = ArrElem::Str(8, c); EO::Consumed } } // 4-byte lead
                else { EO::Reject }
            }
            ArrElem::Str(1, c) => if matches!(b, b'"'|b'\\'|b'/'|b'b'|b'f'|b'n'|b'r'|b't') { *ev = ArrElem::Str(0, c.saturating_add(1)); EO::Consumed } else if b == b'u' { *ev = ArrElem::Str(2, c); EO::Consumed } else { EO::Reject },
            ArrElem::Str(n @ 2..=5, c) => if b.is_ascii_hexdigit() { *ev = if n == 5 { ArrElem::Str(0, c.saturating_add(1)) } else { ArrElem::Str(n + 1, c) }; EO::Consumed } else { EO::Reject },
            ArrElem::Str(6, c) => if (0x80..0xC0).contains(&b) { *ev = ArrElem::Str(0, c.saturating_add(1)); EO::Consumed } else { EO::Reject },
            ArrElem::Str(7, c) => if (0x80..0xC0).contains(&b) { *ev = ArrElem::Str(6, c); EO::Consumed } else { EO::Reject },
            ArrElem::Str(8, c) => if (0x80..0xC0).contains(&b) { *ev = ArrElem::Str(7, c); EO::Consumed } else { EO::Reject },
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
    fn sch_accepts(schema: &str, out: &str) -> bool { sch_accepts_bytes(schema, out.as_bytes()) }


    fn sch_accepts_bytes(schema: &str, out: &[u8]) -> bool {
        let v: serde_json::Value = serde_json::from_str(schema).unwrap();
        let prog = compile(&v).expect("compile");
        let mut s = Schema::new(&prog);
        for &b in out { if !s.step(b) { return false; } }
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
    fn schema_array_bounds() {
        let sc = r#"{"type":"object","properties":{"xs":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3}}}"#;
        assert!(sch_accepts(sc, r#"{"xs":[1,2]}"#));       // min
        assert!(sch_accepts(sc, r#"{"xs":[1,2,3]}"#));     // max
        assert!(!sch_accepts(sc, r#"{"xs":[1]}"#));        // below minItems
        assert!(!sch_accepts(sc, r#"{"xs":[1,2,3,4]}"#));  // above maxItems (4th element / comma rejected)
        assert!(!sch_accepts(sc, r#"{"xs":[]}"#));         // empty below min
    }
    #[test]
    fn schema_array_of_enum_and_bool() {
        let sc = r#"{"type":"object","properties":{"flags":{"type":"array","items":{"type":"boolean"}},"cols":{"type":"array","items":{"enum":["r","g","b"]}}}}"#;
        assert!(sch_accepts(sc, r#"{"flags":[true,false,true],"cols":["r","b"]}"#));
        assert!(!sch_accepts(sc, r#"{"flags":[true,1],"cols":["r"]}"#));  // bool array rejects a number
        assert!(!sch_accepts(sc, r#"{"flags":[],"cols":["x"]}"#));        // enum array rejects out-of-set
    }
    #[test]
    fn schema_required_optional() {
        // `required` = subset; non-required props may be omitted, in declaration order.
        let sc = r#"{"type":"object","properties":{"id":{"type":"integer"},"name":{"type":"string"}},"required":["id"]}"#;
        assert!(sch_accepts(sc, r#"{"id":1,"name":"x"}"#));   // both
        assert!(sch_accepts(sc, r#"{"id":1}"#));              // optional name omitted
        assert!(!sch_accepts(sc, r#"{"name":"x"}"#));         // required id missing
        assert!(!sch_accepts(sc, r#"{}"#));                   // required id missing
        assert!(!sch_accepts(sc, r#"{"name":"x","id":1}"#));  // wrong declaration order
        assert!(!sch_accepts(sc, r#"{"id":1,}"#));            // trailing comma
        assert!(!sch_accepts(sc, r#"{"id":4.2}"#));           // integer slot rejects a float
    }
    #[test]
    fn schema_all_optional() {
        let sc = r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},"required":[]}"#;
        assert!(sch_accepts(sc, r#"{}"#));
        assert!(sch_accepts(sc, r#"{"a":1}"#));
        assert!(sch_accepts(sc, r#"{"b":true}"#));
        assert!(sch_accepts(sc, r#"{"a":1,"b":true}"#));
        assert!(!sch_accepts(sc, r#"{"b":true,"a":1}"#));     // wrong order
        assert!(!sch_accepts(sc, r#"{"a":1,"a":2}"#));        // duplicate
        assert!(!sch_accepts(sc, r#"{,"a":1}"#));             // leading comma
    }
    #[test]
    fn schema_optional_before_required() {
        // required is in the MIDDLE — the optional before it can be skipped, but never past `b`.
        let sc = r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},"required":["b"]}"#;
        assert!(sch_accepts(sc, r#"{"b":2}"#));               // skip optional a, required b, skip optional c
        assert!(sch_accepts(sc, r#"{"a":1,"b":2,"c":3}"#));   // all
        assert!(!sch_accepts(sc, r#"{"c":3}"#));              // required b skipped
        assert!(!sch_accepts(sc, r#"{"a":1,"c":3}"#));        // required b skipped (can't jump a→c)
        assert!(!sch_accepts(sc, r#"{"a":1,"b":2,}"#));       // trailing comma after last prop
    }
    #[test]
    fn schema_nested_required_optional() {
        let sc = r#"{"type":"object","properties":{"id":{"type":"integer"},"addr":{"type":"object","properties":{"city":{"type":"string"},"zip":{"type":"string"}},"required":["city"]}},"required":["id","addr"]}"#;
        assert!(sch_accepts(sc, r#"{"id":1,"addr":{"city":"x"}}"#));            // nested optional zip omitted
        assert!(sch_accepts(sc, r#"{"id":1,"addr":{"city":"x","zip":"y"}}"#));  // nested full
        assert!(!sch_accepts(sc, r#"{"id":1,"addr":{}}"#));                     // nested required city missing
        assert!(!sch_accepts(sc, r#"{"id":1}"#));                               // required addr missing
        assert!(!sch_accepts(sc, r#"{"id":1,"addr":{"zip":"y"}}"#));            // nested required city skipped
        assert!(!sch_accepts(sc, r#"{"id":1,"addr":{"city":"x"},}"#));          // outer trailing comma
        assert!(!sch_accepts(sc, r#"{"id":1,"addr":{"city":"x",}}"#));          // inner trailing comma
    }
    #[test]
    fn schema_prefix_free_keys() {
        // "age" vs "agent" — prefix-free key matcher must keep them distinct.
        let sc = r#"{"type":"object","properties":{"age":{"type":"integer"},"agent":{"type":"string"}},"required":[]}"#;
        assert!(sch_accepts(sc, r#"{"agent":"x"}"#));         // skip age, match agent
        assert!(sch_accepts(sc, r#"{"age":5,"agent":"x"}"#));
        assert!(sch_accepts(sc, r#"{"age":5}"#));
    }
    #[test]
    fn schema_optional_array_prop() {
        let sc = r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}},"n":{"type":"integer"}},"required":["n"]}"#;
        assert!(sch_accepts(sc, r#"{"n":3}"#));               // optional array omitted
        assert!(sch_accepts(sc, r#"{"tags":["a"],"n":3}"#));
        assert!(!sch_accepts(sc, r#"{"tags":["a"]}"#));       // required n missing
    }
    #[test]
    fn schema_empty_props() {
        assert!(sch_accepts(r#"{"type":"object","properties":{}}"#, r#"{}"#));
        assert!(!sch_accepts(r#"{"type":"object","properties":{}}"#, r#"{"x":1}"#));
    }
    #[test]
    fn schema_optional_object_skipped() {
        // An OBJECT-typed property that's optional must be skippable without corrupting the value stream.
        let sc = r#"{"type":"object","properties":{"a":{"type":"integer"},"addr":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},"required":[]}"#;
        assert!(sch_accepts(sc, r#"{}"#));                                // skip both
        assert!(sch_accepts(sc, r#"{"a":1}"#));                          // skip the object prop
        assert!(sch_accepts(sc, r#"{"addr":{"city":"x"}}"#));           // skip scalar, take the object
        assert!(sch_accepts(sc, r#"{"a":1,"addr":{"city":"x"}}"#));     // both
        assert!(!sch_accepts(sc, r#"{"addr":{"city":"x"},"a":1}"#));    // wrong order
        assert!(!sch_accepts(sc, r#"{"addr":{}}"#));                    // nested required city missing
    }
    #[test]
    fn schema_prefix_name() {
        // One property name is a strict prefix of another. The closing-quote delimiter keeps the compiled
        // keys prefix-free ("a": vs "ab":), so the matcher never confuses them.
        let sc = r#"{"type":"object","properties":{"a":{"type":"integer"},"ab":{"type":"integer"}},"required":[]}"#;
        assert!(sch_accepts(sc, r#"{"ab":2}"#));            // skip "a", match "ab"
        assert!(sch_accepts(sc, r#"{"a":1}"#));             // match "a", stop
        assert!(sch_accepts(sc, r#"{"a":1,"ab":2}"#));      // both
        assert!(!sch_accepts(sc, r#"{"ab":2,"a":1}"#));     // wrong order
    }
    #[test]
    fn schema_colon_in_name() {
        // A property name containing ':' — the colon lives INSIDE the quotes; the terminator is still `":`.
        let sc = r#"{"type":"object","properties":{"a:b":{"type":"integer"},"c":{"type":"string"}},"required":["a:b"]}"#;
        assert!(sch_accepts(sc, r#"{"a:b":1,"c":"x"}"#));
        assert!(sch_accepts(sc, r#"{"a:b":1}"#));           // optional c omitted
        assert!(!sch_accepts(sc, r#"{"c":"x"}"#));          // required "a:b" missing
    }
    #[test]
    fn schema_deep_nesting_falls_back() {
        // Nesting past MAX_OBJ_DEPTH compiles the deepest object to Item::Any (free-but-valid JSON),
        // so the Copy acceptor never overflows. Verify a 10-deep instance still parses as valid JSON.
        let mut sc = String::new();
        let mut close = String::new();
        for _ in 0..10 { sc.push_str(r#"{"type":"object","properties":{"n":"#); close.push_str("}}"); }
        sc.push_str(r#"{"type":"integer"}"#);
        sc.push_str(&close);
        let mut inst = String::new();
        let mut ci = String::new();
        for _ in 0..10 { inst.push_str(r#"{"n":"#); ci.push('}'); }
        inst.push('1');
        inst.push_str(&ci);
        assert!(sch_accepts(&sc, &inst));                   // deep-but-valid accepted via the Any fallback
    }
    #[test]
    fn schema_str_maxlength() {
        let sc = r#"{"type":"object","properties":{"s":{"type":"string","maxLength":3}},"required":["s"]}"#;
        assert!(sch_accepts(sc, r#"{"s":"abc"}"#));         // exactly max
        assert!(sch_accepts(sc, r#"{"s":""}"#));            // empty ok (no minLength)
        assert!(sch_accepts(sc, r#"{"s":"ab"}"#));          // under max
        assert!(!sch_accepts(sc, r#"{"s":"abcd"}"#));       // over max
    }
    #[test]
    fn schema_str_minlength() {
        let sc = r#"{"type":"object","properties":{"s":{"type":"string","minLength":2}},"required":["s"]}"#;
        assert!(sch_accepts(sc, r#"{"s":"ab"}"#));          // exactly min
        assert!(sch_accepts(sc, r#"{"s":"abcdef"}"#));      // over min, no max
        assert!(!sch_accepts(sc, r#"{"s":"a"}"#));          // under min
        assert!(!sch_accepts(sc, r#"{"s":""}"#));           // empty under min
    }
    #[test]
    fn schema_str_min_and_max() {
        let sc = r#"{"type":"object","properties":{"s":{"type":"string","minLength":2,"maxLength":4}}}"#;
        assert!(sch_accepts(sc, r#"{"s":"ab"}"#));
        assert!(sch_accepts(sc, r#"{"s":"abcd"}"#));
        assert!(!sch_accepts(sc, r#"{"s":"a"}"#));
        assert!(!sch_accepts(sc, r#"{"s":"abcde"}"#));
    }
    #[test]
    fn schema_str_length_counts_escapes_as_one() {
        let sc = r#"{"type":"object","properties":{"s":{"type":"string","maxLength":1}},"required":["s"]}"#;
        assert!(sch_accepts(sc, r#"{"s":"\n"}"#));          // one escape = one code point
        assert!(sch_accepts(sc, r#"{"s":"é"}"#));      // one \u escape = one code point
        assert!(!sch_accepts(sc, r#"{"s":"\n\n"}"#));       // two escapes = two code points
        assert!(!sch_accepts(sc, r#"{"s":"a\n"}"#));        // char + escape = two
    }
    #[test]
    fn schema_str_length_counts_codepoints_not_bytes() {
        // "é" is two UTF-8 bytes but one code point — maxLength:1 must accept it, reject two of them.
        let sc = r#"{"type":"object","properties":{"s":{"type":"string","maxLength":1}}}"#;
        assert!(sch_accepts(sc, "{\"s\":\"é\"}"));           // 1 code point
        assert!(!sch_accepts(sc, "{\"s\":\"éé\"}"));         // 2 code points (4 bytes)
    }
    #[test]
    fn schema_str_utf8_wellformed() {
        // The acceptor must enforce well-formed UTF-8 inside strings so a length cap can never orphan a
        // multibyte char into invalid output (the bug the live test caught).
        let sc = r#"{"type":"object","properties":{"s":{"type":"string"}},"required":["s"]}"#;
        // A 4-byte emoji (😀 = F0 9F 98 80) is one code point — accepted.
        assert!(sch_accepts(sc, "{\"s\":\"😀\"}"));
        // A lead byte followed immediately by a closing quote (orphaned char) must be REJECTED.
        assert!(!sch_accepts_bytes(sc, b"{\"s\":\"\xC3\"}"));      // 2-byte lead, then close — incomplete
        assert!(!sch_accepts_bytes(sc, b"{\"s\":\"\xF0\x9F\"}"));  // 4-byte lead + 1 cont, then close — incomplete
        // A stray continuation byte with no lead must be REJECTED.
        assert!(!sch_accepts_bytes(sc, b"{\"s\":\"\xA9\"}"));
    }
    #[test]
    fn schema_str_maxlength_multibyte_boundary() {
        // maxLength:2 with two 2-byte chars: exactly 2 code points (4 bytes) accepted; a 3rd rejected, and
        // the string is FORCED to close cleanly at the cap rather than orphaning a lead byte.
        let sc = r#"{"type":"object","properties":{"s":{"type":"string","maxLength":2}},"required":["s"]}"#;
        assert!(sch_accepts(sc, "{\"s\":\"éé\"}"));      // 2 code points, 4 bytes
        assert!(!sch_accepts(sc, "{\"s\":\"ééé\"}"));    // 3 code points > max
        assert!(sch_accepts(sc, "{\"s\":\"😀a\"}"));     // 4-byte char + ascii = 2 code points
    }
    #[test]
    fn schema_int_range_exhaustive() {
        // Brute force: for many [lo,hi], EVERY integer in a window must be accepted iff it's in range.
        let bounds = [(0i64, 10), (1, 5), (-5, 5), (-10, -1), (0, 0), (-1, 1), (3, 3), (0, 100),
                      (-100, 100), (7, 250), (0, i64::MAX / 2), (i64::MIN / 2, 0), (i64::MIN / 2, i64::MAX / 2)];
        for &(lo, hi) in &bounds {
            let sc = format!(r#"{{"type":"object","properties":{{"x":{{"type":"integer","minimum":{lo},"maximum":{hi}}}}},"required":["x"]}}"#);
            for v in -400i64..=400 {
                let inst = format!(r#"{{"x":{v}}}"#);
                assert_eq!(sch_accepts(&sc, &inst), v >= lo && v <= hi, "lo={lo} hi={hi} v={v}");
            }
        }
    }
    #[test]
    fn schema_int_range_boundary_values() {
        // Exact i64-scale boundaries: accept the endpoints, reject just outside.
        let sc = r#"{"type":"object","properties":{"x":{"type":"integer","minimum":1000,"maximum":9999}},"required":["x"]}"#;
        assert!(sch_accepts(sc, r#"{"x":1000}"#));
        assert!(sch_accepts(sc, r#"{"x":9999}"#));
        assert!(sch_accepts(sc, r#"{"x":5000}"#));
        assert!(!sch_accepts(sc, r#"{"x":999}"#));   // one below (also fewer digits)
        assert!(!sch_accepts(sc, r#"{"x":10000}"#)); // one above (also more digits) — the runaway guard
        assert!(!sch_accepts(sc, r#"{"x":0}"#));
    }
    #[test]
    fn schema_int_range_syntax() {
        let sc = r#"{"type":"object","properties":{"x":{"type":"integer","minimum":-50,"maximum":50}},"required":["x"]}"#;
        assert!(sch_accepts(sc, r#"{"x":-50}"#));
        assert!(sch_accepts(sc, r#"{"x":0}"#));
        assert!(!sch_accepts(sc, r#"{"x":00}"#));    // leading zero
        assert!(!sch_accepts(sc, r#"{"x":01}"#));    // leading zero
        assert!(!sch_accepts(sc, r#"{"x":1.5}"#));   // not an integer
        assert!(!sch_accepts(sc, r#"{"x":-}"#));     // lone minus, no digit
        assert!(!sch_accepts(sc, r#"{"x":5e2}"#));   // exponent not an integer literal
    }
    #[test]
    fn schema_int_only_max_or_min() {
        // Only maximum given → minimum unbounded (very negative allowed); only minimum → unbounded above.
        let mx = r#"{"type":"object","properties":{"x":{"type":"integer","maximum":100}},"required":["x"]}"#;
        assert!(sch_accepts(mx, r#"{"x":100}"#));
        assert!(sch_accepts(mx, r#"{"x":-999999}"#));
        assert!(!sch_accepts(mx, r#"{"x":101}"#));
        let mn = r#"{"type":"object","properties":{"x":{"type":"integer","minimum":100}},"required":["x"]}"#;
        assert!(sch_accepts(mn, r#"{"x":100}"#));
        assert!(sch_accepts(mn, r#"{"x":100000000}"#));
        assert!(!sch_accepts(mn, r#"{"x":99}"#));
        assert!(!sch_accepts(mn, r#"{"x":-1}"#));
    }
    #[test]
    fn schema_array_of_bounded_strings() {
        let sc = r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string","maxLength":2}}}}"#;
        assert!(sch_accepts(sc, r#"{"tags":["ab","cd"]}"#));
        assert!(sch_accepts(sc, r#"{"tags":["a",""]}"#));
        assert!(!sch_accepts(sc, r#"{"tags":["abc"]}"#));    // element over maxLength
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
