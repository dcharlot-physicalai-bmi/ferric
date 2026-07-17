// ferric-flow RAG inside a Cloudflare Worker — the local proof of "option A": a serialized HNSW graph
// is loaded once per isolate and queried in wasm, with no managed vector DB. This is the exact path
// the live Assistant would take to retire env.VECTORIZE.query (there the graph comes from R2/KV, built
// on /ingest from bge-m3 embeddings; here it's bundled, built offline from the real corpus).
import init, { Rag } from '../../pkg-web/ferric_flow_wasm.js';
import wasmModule from '../../pkg-web/ferric_flow_wasm_bg.wasm';
import fixture from '../fixture.json';

// Instantiate the wasm module with the async init (the Workers-correct path — top-level sync
// instantiation isn't available during module eval), then rehydrate the serialized graph once per
// isolate. Cached across requests via the module-scope `rag`.
let rag: Rag | null = null;
async function getRag(): Promise<Rag> {
  if (!rag) {
    await init(wasmModule);
    rag = new Rag(Uint8Array.from(atob(fixture.indexB64), (c) => c.charCodeAt(0)));
  }
  return rag;
}

export default {
  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const titles = fixture.titles as Record<string, string>;

    if (url.pathname === '/health') {
      return Response.json({ ok: true, engine: 'ferric-flow (wasm-in-Worker)', liveDocs: (await getRag()).liveLen() });
    }
    if (url.pathname === '/search') {
      const r = await getRag();
      const qi = Math.max(0, Math.min(fixture.queries.length - 1, parseInt(url.searchParams.get('qi') || '0', 10)));
      const query = fixture.queries[qi];
      const hits = JSON.parse(r.query(new Float32Array(query.vec), 5)) as { id: string; dist: number }[];
      return Response.json({
        engine: 'ferric-flow (wasm-in-Worker)',
        liveDocs: r.liveLen(),
        query: query.q,
        hits: hits.map((h) => ({ title: titles[h.id] || h.id, id: h.id, dist: h.dist })),
      });
    }
    return new Response(
      'ferric-flow Worker POC — the serialized HNSW graph, queried in wasm.\n' +
        'GET /health\nGET /search?qi=0..' + (fixture.queries.length - 1) + '\n',
      { headers: { 'content-type': 'text/plain' } },
    );
  },
};
