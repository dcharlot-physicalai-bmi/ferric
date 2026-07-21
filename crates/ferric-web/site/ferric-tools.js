// On-device tool-calling for Ferric — the agentic loop, in a browser tab. Given a loaded FerricModel,
// a set of tools, and a JS `execute(name, args)` callback, it advertises the tools to the model in the
// Hermes/qwen format, parses the model's <tool_call> emissions, runs the tools locally, feeds the
// results back, and loops until the model answers. Same protocol as Ferric's native server — no server,
// no API key, nothing leaves the tab.

/** Hermes/qwen tool system prompt (mirrors ferric-agent::tools::hermes_prompt). */
export function hermesSystem(tools) {
  let s = 'You are a function-calling AI. You are given function signatures inside <tools></tools>. ' +
    'To call a function, emit a JSON object {"name": <name>, "arguments": <args>} inside <tool_call></tool_call> tags. ' +
    'You may emit multiple <tool_call> blocks. Only call a function when it is needed.\n<tools>\n';
  for (const t of tools) s += JSON.stringify(t) + '\n';
  s += '</tools>';
  return s;
}

/** Top-level balanced `{…}` spans (string-aware, so braces inside JSON strings don't confuse it). */
function balancedObjects(s) {
  const out = []; let depth = 0, start = 0, inStr = false, esc = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inStr) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') inStr = true;
    else if (c === '{') { if (depth === 0) start = i; depth++; }
    else if (c === '}') { depth--; if (depth === 0) out.push(s.slice(start, i + 1)); }
  }
  return out;
}

/** Parse tool calls. Pass 1: well-formed <tool_call>{json}</tool_call> tags (Hermes emits multiple
 *  concatenated tags). Pass 2 (fallback, like the native parser): any balanced {…} carrying both
 *  `name` and `arguments` — small models often emit the bare JSON without the tags. */
export function parseToolCalls(text) {
  const calls = [];
  const re = /<tool_call>([\s\S]*?)<\/tool_call>/g;
  let m;
  while ((m = re.exec(text))) {
    try { const v = JSON.parse(m[1].trim()); if (v && v.name) calls.push({ name: v.name, arguments: v.arguments || {} }); }
    catch { /* skip a malformed tag */ }
  }
  if (!calls.length) {
    for (const obj of balancedObjects(text)) {
      try { const v = JSON.parse(obj); if (v && v.name !== undefined && v.arguments !== undefined) calls.push({ name: v.name, arguments: v.arguments || {} }); }
      catch { /* not JSON */ }
    }
  }
  return calls;
}

/**
 * Run an on-device tool-calling loop.
 *   fm       — a loaded FerricModel
 *   tools    — [{ name, description, parameters }] (JSON-schema `parameters`)
 *   execute  — async (name, args) => result   (your local tool implementations)
 *   user     — the user request
 * Returns { text, trace } where trace lists each tool call + its result.
 */
export async function runToolLoop(fm, { user, tools, execute, system = '', steps = 160, maxRounds = 4, onToken = () => {} }) {
  // Plain user:/assistant: transcript. (ChatML's <|im_start|> are special tokens the browser
  // `generate` doesn't encode as such, so a literal-text ChatML prompt tokenizes wrong — this format
  // is what works with raw generate.) The model's tool_call emission is fed back verbatim + the
  // tool_response, and it synthesizes the answer from the results.
  let convo = hermesSystem(tools) + (system ? `\n${system}` : '') + `\nuser: ${user}\nassistant:`;
  const trace = [];
  for (let round = 0; round < maxRounds; round++) {
    const out = await fm.generate(convo, steps, onToken);
    const calls = parseToolCalls(out);
    if (!calls.length) return { text: out.trim(), trace };
    convo += out;
    for (const c of calls) {
      const result = await execute(c.name, c.arguments);
      trace.push({ name: c.name, args: c.arguments, result });
      convo += `\n<tool_response>${JSON.stringify(result)}</tool_response>`;
    }
    convo += '\nassistant:';
  }
  return { text: (await fm.generate(convo, steps, onToken)).trim(), trace };
}
