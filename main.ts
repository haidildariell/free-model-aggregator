// free-model-aggregator-01 entry: one server, two client wires, ten free-tier providers.
//
// Built on the kit starter's frame (brief, catalog probe, register, poll) with its routing
// replaced: every free model that meets a request's requirements is ranked by the quality
// table's rating_milli; a plan made at load spreads the manifest over the providers under a
// per-provider cap so no single budget carries the trial; a refused call (402, 429, 5xx)
// fails the request over to the next candidate instead of ending it; and all three upstream
// wires (openai, anthropic, gemini) are spoken, pivoting through the OpenAI shape.
//
// Every choice derives from the brief, the manifest and the providers' answers: no clock and
// no randomness in any decision, so a replay of the same trial reproduces the same calls.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicResponseToOpenAI,
  anthropicStreamToOpenAI,
  geminiResponseToOpenAI,
  geminiStreamToOpenAI,
  openAIRequestToAnthropic,
  openAIRequestToGemini,
} from './lib/upstream.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Planned calls per provider before the planner prefers the next-best model elsewhere. */
const PLAN_CAP = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  buildPlan();
}

// ---------------------------------------------------------------------------
// Candidates, planning, provider health
// ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

interface ProviderState {
  dead: boolean;
  cooldownUntil: number; // manifest index through which a 429/5xx provider rests
}

const providerState = new Map<string, ProviderState>();
const plan = new Map<number, Candidate>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { dead: false, cooldownUntil: -1 };
    providerState.set(providerId, state);
  }
  return state;
}

function allCandidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

const byRank = (a: Candidate, b: Candidate): number =>
  b.rating - a.rating ||
  (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
  (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);

/** Every free candidate that meets the requirements, best first. */
function rankFor(requires: Requirement): Candidate[] {
  return allCandidates()
    .filter((c) => meetsRequirements(requires, capabilitiesOf(c.model, ratingOfModel(brief.ratings, c.probe.provider.id, c.model.id))))
    .sort(byRank);
}

/**
 * The plan: scarcest requests first (fewest candidates, then heaviest), each taking its best
 * candidate whose provider is still under the cap. It reserves the only servers of a scarce
 * request from being spent on requests that have other homes.
 */
function buildPlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  const ranked = manifest.map((entry) => rankFor(entry.requires));
  const order = manifest
    .map((entry, index) => ({ entry, index, options: ranked[index].length }))
    .sort((a, b) => a.options - b.options || b.entry.weight - a.entry.weight || a.index - b.index);
  const planned = new Map<string, number>();
  for (const { index } of order) {
    const options = ranked[index];
    if (options.length === 0) continue;
    const pick = options.find((c) => (planned.get(c.probe.provider.id) ?? 0) < PLAN_CAP) ?? options[0];
    plan.set(index, pick);
    planned.set(pick.probe.provider.id, (planned.get(pick.probe.provider.id) ?? 0) + 1);
  }
}

/** The candidates to try for one request, in order: the plan first, then rank, resting providers last. */
function tryOrder(index: number, requires: Requirement): Candidate[] {
  const ranked = rankFor(requires);
  const first = plan.get(index);
  const ordered: Candidate[] = [];
  if (first !== undefined && ranked.some((c) => c.probe === first.probe && c.model === first.model)) ordered.push(first);
  for (const c of ranked) if (c !== first) ordered.push(c);
  const ready = ordered.filter((c) => {
    const s = stateOf(c.probe.provider.id);
    return !s.dead && s.cooldownUntil < index;
  });
  const resting = ordered.filter((c) => {
    const s = stateOf(c.probe.provider.id);
    return !s.dead && s.cooldownUntil >= index;
  });
  return [...ready, ...resting];
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Client request handling
// ---------------------------------------------------------------------------

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, index: number): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

interface UpstreamAnswer {
  ok: boolean;
  status: number;
  /** The answer in the canonical OpenAI shape (JSON), when not streaming. */
  json: Json | null;
  /** The answer as canonical OpenAI SSE text, when streaming. */
  sse: string | null;
  /** The provider's raw error body, for forwarding when every candidate fails. */
  rawError: string;
}

/** Build the upstream request for one candidate from the canonical body. */
function upstreamCall(candidate: Candidate, canonical: Json, stream: boolean): { url: string; headers: Record<string, string>; body: Json } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    return {
      url: `${base}/v1/chat/completions`,
      headers,
      body: { ...canonical, model: candidate.model.id, ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}) },
    };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    return { url: `${base}/v1/messages`, headers, body: openAIRequestToAnthropic(canonical, candidate.model.id) };
  }
  const verb = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return { url: `${base}/v1beta/models/${candidate.model.id}:${verb}`, headers, body: openAIRequestToGemini(canonical, candidate.model.id) };
}

/** One provider call, its answer canonicalised. Never throws: failures come back as answers. */
async function callCandidate(candidate: Candidate, canonical: Json, stream: boolean): Promise<UpstreamAnswer> {
  const provider = candidate.probe.provider;
  const { url, headers, body } = upstreamCall(candidate, canonical, stream);
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, json: null, sse: null, rawError: (err as Error).message };
  }
  if (!upstream.ok) {
    return { ok: false, status: upstream.status, json: null, sse: null, rawError: await upstream.text() };
  }
  if (stream) {
    const text = await upstream.text();
    if (text.trimStart().startsWith('{')) {
      // A JSON body on a stream request is an error spoken late: treat it as a failed call.
      return { ok: false, status: upstream.status, json: null, sse: null, rawError: text };
    }
    const sse =
      provider.wire === 'openai' ? text : provider.wire === 'anthropic' ? anthropicStreamToOpenAI(text, candidate.model.id) : geminiStreamToOpenAI(text, candidate.model.id);
    return { ok: true, status: 200, json: null, sse, rawError: '' };
  }
  const text = await upstream.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, status: upstream.status, json: null, sse: null, rawError: text };
  }
  if (!isRecord(parsed)) return { ok: false, status: upstream.status, json: null, sse: null, rawError: text };
  const json =
    provider.wire === 'openai'
      ? parsed
      : provider.wire === 'anthropic'
        ? anthropicResponseToOpenAI(parsed, candidate.model.id)
        : geminiResponseToOpenAI(parsed, candidate.model.id);
  return { ok: true, status: 200, json, sse: null, rawError: '' };
}

/** Note a failed call in the provider's health: 402 spends the budget, 429/5xx rest it briefly. */
function noteFailure(candidate: Candidate, status: number, index: number): void {
  const state = stateOf(candidate.probe.provider.id);
  if (status === 402) state.dead = true;
  else if (status === 429 || status === 0 || status >= 500) state.cooldownUntil = index + 2;
}

/** A one-chunk stream over a whole SSE document, for the Anthropic event converter. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}

/** Serve one client request on either wire, failing over across the ranked candidates. */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const index = manifestIndex;
  manifestIndex += 1;
  const requires = requirementsFor(body, index);
  const stream = body.stream === true;
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  const canonical: Json = clientWire === 'openai' ? { ...body } : anthropicToOpenAI(body, requestedModel);
  canonical.stream = stream;
  const candidates = tryOrder(index, requires);
  let lastFailure: UpstreamAnswer | null = null;
  for (const candidate of candidates) {
    const answer = await callCandidate(candidate, canonical, stream);
    if (!answer.ok) {
      noteFailure(candidate, answer.status, index);
      lastFailure = answer;
      continue;
    }
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (clientWire === 'openai') {
        res.end(answer.sse ?? '');
      } else {
        for await (const event of anthropicEventsFromOpenAI(streamOf(answer.sse ?? ''), requestedModel)) res.write(event);
        res.end();
      }
      return;
    }
    const json = answer.json ?? {};
    if (clientWire === 'openai') sendJson(res, 200, json);
    else sendJson(res, 200, openAIToAnthropic(json, requestedModel));
    return;
  }
  const status = lastFailure !== null && lastFailure.status >= 400 ? lastFailure.status : 503;
  const detail = lastFailure?.rawError.slice(0, 300) ?? 'no free model on the directory meets this request';
  if (clientWire === 'openai') sendError(res, 'openai', status, 'upstream_error', `every provider refused: ${detail}`);
  else sendError(res, 'anthropic', status, 'api_error', `every provider refused: ${detail}`);
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
