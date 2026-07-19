// Vercel AI SDK provider for Ferric — makes on-device (WASM/WebGPU) Ferric a drop-in `LanguageModelV2`.
// generateObject/streamObject flow through Ferric's guided decoding, so structured output is
// schema-conformant AND deterministic, running entirely in the browser tab. No incumbent does this.
//
// Usage: const fm = await loadFerric(ggufBytes);  // load ONCE (weights uploaded to GPU once)
//        const model = ferric(fm);                // reuse across every generateText/Object/streamText
import init, { FerricModel } from './pkg/ferric_web.js';

let ready = null;
export async function ensureReady() { if (!ready) ready = init(); return ready; }

/** Load a GGUF once into a reusable Ferric model handle. */
export async function loadFerric(ggufBytes) {
  await ensureReady();
  return await FerricModel.load(ggufBytes);
}

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

/** Wrap a loaded `FerricModel` handle as an AI SDK v2 language model. `opts.steps` caps tokens. */
export function ferric(fm, opts = {}) {
  const steps = opts.steps ?? 128;
  return {
    specificationVersion: 'v2',
    provider: 'ferric',
    modelId: opts.modelId ?? 'ferric-bonsai',
    supportedUrls: {},

    async doGenerate(options) {
      const prompt = flattenPrompt(options.prompt);
      const rf = options.responseFormat;
      let text;
      if (rf && rf.type === 'json') {
        text = await fm.generate_json(prompt, steps, rf.schema ? JSON.stringify(rf.schema) : '', () => {});
      } else {
        text = await fm.generate(prompt, steps, () => {});
      }
      return { content: [{ type: 'text', text }], finishReason: 'stop', usage: USAGE, warnings: [] };
    },

    async doStream(options) {
      const prompt = flattenPrompt(options.prompt);
      const rf = options.responseFormat;
      const stream = new ReadableStream({
        async start(c) {
          c.enqueue({ type: 'stream-start', warnings: [] });
          c.enqueue({ type: 'text-start', id: '0' });
          const onTok = (kind, payload) => { if (kind === 'token') c.enqueue({ type: 'text-delta', id: '0', delta: payload }); };
          if (rf && rf.type === 'json') await fm.generate_json(prompt, steps, rf.schema ? JSON.stringify(rf.schema) : '', onTok);
          else await fm.generate(prompt, steps, onTok);
          c.enqueue({ type: 'text-end', id: '0' });
          c.enqueue({ type: 'finish', finishReason: 'stop', usage: USAGE });
          c.close();
        },
      });
      return { stream };
    },
  };
}
