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

#[cfg(test)]
mod tests {
    use super::Json;
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
