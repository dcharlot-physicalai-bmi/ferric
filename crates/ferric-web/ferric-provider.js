// Vercel AI SDK provider for Ferric — makes on-device (WASM/WebGPU) Ferric a drop-in `LanguageModelV2`.
// generateObject/streamObject flow through Ferric's guided decoding, so structured output is
// schema-conformant AND deterministic, running entirely in the browser tab. No incumbent does this.
import init, { bonsai_generate_json, bonsai_stream } from './pkg/ferric_web.js';

let ready = null;
export async function ensureReady() { if (!ready) ready = init(); return ready; }

// The AI SDK v5 prompt is an array of { role, content }, content = string | [{type:'text',text}].
function flattenPrompt(prompt) {
  const parts = [];
  for (const m of prompt || []) {
    const c = m.content;
    const text = typeof c === 'string' ? c : Array.isArray(c) ? c.filter(p => p.type === 'text').map(p => p.text).join('') : '';
    if (text) parts.push(`${m.role}: ${text}`);
  }
  return parts.join('\n') + '\nassistant:';
}

const USAGE = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };

/** Create a Ferric LanguageModelV2 from raw GGUF bytes. `opts.steps` caps generated tokens. */
export function ferric(modelBytes, opts = {}) {
  const steps = opts.steps ?? 128;
  const modelId = opts.modelId ?? 'ferric-bonsai';
  return {
    specificationVersion: 'v2',
    provider: 'ferric',
    modelId,
    supportedUrls: {},

    async doGenerate(options) {
      await ensureReady();
      const prompt = flattenPrompt(options.prompt);
      const rf = options.responseFormat;
      let text;
      if (rf && rf.type === 'json') {
        // generateObject → guided decoding: schema-conformant if a schema is present, else valid JSON.
        const schema = rf.schema ? JSON.stringify(rf.schema) : '';
        text = await bonsai_generate_json(modelBytes, prompt, steps, schema, () => {});
      } else {
        let acc = '';
        await bonsai_stream(modelBytes, prompt, steps, (kind, payload) => { if (kind === 'token') acc += payload; });
        text = acc;
      }
      return { content: [{ type: 'text', text }], finishReason: 'stop', usage: USAGE, warnings: [] };
    },

    async doStream(options) {
      await ensureReady();
      const prompt = flattenPrompt(options.prompt);
      const rf = options.responseFormat;
      const stream = new ReadableStream({
        async start(c) {
          c.enqueue({ type: 'stream-start', warnings: [] });
          c.enqueue({ type: 'text-start', id: '0' });
          const onTok = (kind, payload) => { if (kind === 'token') c.enqueue({ type: 'text-delta', id: '0', delta: payload }); };
          if (rf && rf.type === 'json') {
            const schema = rf.schema ? JSON.stringify(rf.schema) : '';
            await bonsai_generate_json(modelBytes, prompt, steps, schema, onTok);
          } else {
            await bonsai_stream(modelBytes, prompt, steps, onTok);
          }
          c.enqueue({ type: 'text-end', id: '0' });
          c.enqueue({ type: 'finish', finishReason: 'stop', usage: USAGE });
          c.close();
        },
      });
      return { stream };
    },
  };
}
