//! **The model's own chat template, rendered the way its authors render it.**
//!
//! A GGUF carries `tokenizer.chat_template`, the Jinja template the model was trained to read. The server
//! used to load it and never use it (rustc warned "field `template` is never read"): four families were
//! guessed from vocabulary markers and anything else fell to `role: content` concatenation, so gemma-4 and
//! DeepSeek-V2 received prompts in a format they were never trained on. 12.5 of 14 serving peers render the
//! template.
//!
//! The reference is Hugging Face's `apply_chat_template` — jinja2's `ImmutableSandboxedEnvironment` with
//! `trim_blocks=True, lstrip_blocks=True`, the loop-controls extension, the globals `raise_exception` and
//! `strftime_now`, and a `tojson` overridden to Python's `json.dumps(x, ensure_ascii=False)`. Each is
//! reproduced here, and `scripts/chat_template_conformance.sh` renders every template on the machine
//! through both engines and compares the strings byte for byte.
use minijinja::{Environment, Error, ErrorKind, Value as J};
use serde_json::Value;

pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
    /// The template reads `tools` itself (Qwen, Llama-3.1+, Mistral…): tools go to it, not into a
    /// system prompt of Ferric's own making.
    pub handles_tools: bool,
}

/// Python's `json.dumps(x, ensure_ascii=False, indent=indent)`: `", "` and `": "` separators when not
/// indenting, `","` + newline when indenting; non-ASCII kept; floats as Python writes them.
fn py_json(v: &Value, indent: Option<usize>, depth: usize, out: &mut String) {
    let nl = |out: &mut String, d: usize| if let Some(n) = indent { out.push('\n'); out.push_str(&" ".repeat(n * d)); };
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(f) = n.as_f64().filter(|_| n.is_f64()) {
                // Python prints a float with ".0" when integral, `repr` precision otherwise.
                if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 { out.push_str(&format!("{f:.1}")) }
                else if f.is_nan() { out.push_str("NaN") }
                else if f.is_infinite() { out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" }) }
                else { out.push_str(&format!("{f}")) }
            } else { out.push_str(&n.to_string()) }
        }
        Value::String(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"), '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"), '\u{8}' => out.push_str("\\b"),
                    '\u{c}' => out.push_str("\\f"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Value::Array(a) => {
            if a.is_empty() { out.push_str("[]"); return; }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 { out.push(','); if indent.is_none() { out.push(' '); } }
                nl(out, depth + 1);
                py_json(x, indent, depth + 1, out);
            }
            nl(out, depth);
            out.push(']');
        }
        Value::Object(m) => {
            if m.is_empty() { out.push_str("{}"); return; }
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 { out.push(','); if indent.is_none() { out.push(' '); } }
                nl(out, depth + 1);
                py_json(&Value::String(k.clone()), indent, depth + 1, out);
                out.push_str(": ");
                py_json(x, indent, depth + 1, out);
            }
            nl(out, depth);
            out.push('}');
        }
    }
}

fn tojson(v: J, kwargs: minijinja::value::Kwargs) -> Result<J, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let _: Option<bool> = kwargs.get("ensure_ascii")?; // HF's override accepts it; non-ASCII is always kept
    kwargs.assert_all_used()?;
    let j: Value = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut s = String::new();
    py_json(&j, indent, 0, &mut s);
    Ok(J::from_safe_string(s))
}

/// `strftime_now(fmt)` — the local time, formatted with the strftime codes templates use (Llama-3.x asks
/// for `"%d %b %Y"`). A template that reads `date_string` from the context takes the caller's instead.
fn strftime_now(fmt: String) -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let (days, rem) = (now.div_euclid(86_400), now.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    let wd = (days + 4).rem_euclid(7); // 1970-01-01 was a Thursday
    const MON: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const DAY: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' { out.push(c); continue; }
        match it.next() {
            Some('d') => out.push_str(&format!("{d:02}")), Some('m') => out.push_str(&format!("{m:02}")),
            Some('Y') => out.push_str(&format!("{y}")), Some('y') => out.push_str(&format!("{:02}", y % 100)),
            Some('b') => out.push_str(&MON[(m - 1) as usize][..3]), Some('B') => out.push_str(MON[(m - 1) as usize]),
            Some('a') => out.push_str(&DAY[wd as usize][..3]), Some('A') => out.push_str(DAY[wd as usize]),
            Some('H') => out.push_str(&format!("{:02}", rem / 3600)), Some('M') => out.push_str(&format!("{:02}", rem % 3600 / 60)),
            Some('S') => out.push_str(&format!("{:02}", rem % 60)), Some('%') => out.push('%'),
            Some(o) => { out.push('%'); out.push(o); }
            None => out.push('%'),
        }
    }
    out
}

/// HF's `AssistantTracker` extension adds `{% generation %}…{% endgeneration %}` (MiMo's template uses it): it
/// renders its body unchanged and only records where the assistant's text lies. Rewritten to an
/// `if true` block with the SAME whitespace-control markers, so trim/lstrip behave exactly as before.
fn strip_generation_tags(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(p) = rest.find("{%") {
        out.push_str(&rest[..p]);
        let tag = &rest[p..];
        let Some(end) = tag.find("%}") else { out.push_str(tag); return out };
        let inner = &tag[2..end];
        let (lead, body) = match inner.chars().next() { Some(c @ ('-' | '+')) => (c.to_string(), &inner[1..]), _ => (String::new(), inner) };
        let (body, trail) = match body.chars().last() { Some(c @ ('-' | '+')) => (&body[..body.len() - 1], c.to_string()), _ => (body, String::new()) };
        match body.trim() {
            "generation" => out.push_str(&format!("{{%{lead} if true {trail}%}}")),
            "endgeneration" => out.push_str(&format!("{{%{lead} endif {trail}%}}")),
            _ => out.push_str(&tag[..end + 2]),
        }
        rest = &tag[end + 2..];
    }
    out.push_str(rest);
    out
}

impl ChatTemplate {
    pub fn compile(src: &str, bos: &str, eos: &str) -> Result<ChatTemplate, String> {
        let src = &strip_generation_tags(src);
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", |msg: String| -> Result<J, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chat", src.to_string()).map_err(|e| format!("chat template does not compile: {e}"))?;
        Ok(ChatTemplate { env, bos: bos.to_string(), eos: eos.to_string(), handles_tools: src.contains("tools") })
    }

    /// Render `messages` (OpenAI-shaped; content already text) with the context HF passes: `messages`,
    /// `tools` (when given), `add_generation_prompt`, the special tokens, and any `chat_template_kwargs`
    /// (`enable_thinking`, `date_string`, …).
    pub fn render(&self, messages: &[Value], tools: Option<&[Value]>, add_generation_prompt: bool,
                  kwargs: &serde_json::Map<String, Value>) -> Result<String, String> {
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), Value::Array(messages.to_vec()));
        if let Some(t) = tools { ctx.insert("tools".into(), Value::Array(t.to_vec())); }
        ctx.insert("add_generation_prompt".into(), Value::Bool(add_generation_prompt));
        ctx.insert("bos_token".into(), Value::String(self.bos.clone()));
        ctx.insert("eos_token".into(), Value::String(self.eos.clone()));
        for (k, v) in kwargs { ctx.insert(k.clone(), v.clone()); }
        let t = self.env.get_template("chat").map_err(|e| e.to_string())?;
        t.render(J::from_serialize(&Value::Object(ctx))).map_err(|e| {
            // A template's own raise_exception (e.g. "roles must alternate") is the model's rule; say so.
            format!("the model's chat template refused this conversation: {e}")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tojson_writes_what_python_json_dumps_writes() {
        let v = json!({"name": "get_weather", "parameters": {"type": "object", "required": ["city"], "n": 1.0, "x": 0.5}, "é": "ü\n"});
        let mut s = String::new();
        py_json(&v, None, 0, &mut s);
        assert_eq!(s, r#"{"name": "get_weather", "parameters": {"type": "object", "required": ["city"], "n": 1.0, "x": 0.5}, "é": "ü\n"}"#);
        let mut p = String::new();
        py_json(&json!({"a": [1, 2]}), Some(2), 0, &mut p);
        assert_eq!(p, "{\n  \"a\": [\n    1,\n    2\n  ]\n}");
    }

    #[test]
    fn trim_and_lstrip_blocks_and_python_methods_behave_like_hf() {
        let t = ChatTemplate::compile(
            "{% for m in messages %}\n  {% if m.role == 'user' %}<u>{{ m.content.strip() }}</u>\n  {% endif %}\n{% endfor %}{% if add_generation_prompt %}<a>{% endif %}",
            "<s>", "</s>").unwrap();
        let out = t.render(&[json!({"role": "user", "content": "  hi  "})], None, true, &Default::default()).unwrap();
        assert_eq!(out, "<u>hi</u>\n<a>");
    }

    #[test]
    fn generation_tags_render_their_body_with_the_same_whitespace_control() {
        assert_eq!(strip_generation_tags("a{%- generation %}b{% endgeneration -%}c"), "a{%- if true %}b{% endif -%}c");
        let t = ChatTemplate::compile("x{% generation %}{{ 1 + 1 }}{% endgeneration %}y", "", "").unwrap();
        assert_eq!(t.render(&[], None, true, &Default::default()).unwrap(), "x2y");
    }

    #[test]
    fn a_templates_own_exception_is_an_error_naming_it() {
        let t = ChatTemplate::compile("{{ raise_exception('roles must alternate') }}", "", "").unwrap();
        assert!(t.render(&[], None, true, &Default::default()).unwrap_err().contains("roles must alternate"));
    }
}
