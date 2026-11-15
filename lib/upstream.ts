// Upstream wires the baseline lacks: Anthropic and Gemini providers, pivoting through the
// OpenAI Chat Completions shape as the canonical form. Client request -> canonical (main.ts,
// via wire.ts for Anthropic clients) -> the provider's wire; the provider's answer -> canonical
// -> the client's wire (wire.ts again). Pure functions of their arguments: no clock, no
// randomness, so a replay reproduces the same bytes.

import { isRecord, type Json } from './providers.ts';

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

// ---------------------------------------------------------------------------
// Canonical (OpenAI) request -> Anthropic Messages request
// ---------------------------------------------------------------------------

export function openAIRequestToAnthropic(body: Json, model: string): Json {
  let system = '';
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const pushUser = (blocks: Json[]): void => {
    const last = messages[messages.length - 1];
    if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) {
      (last.content as Json[]).push(...blocks);
    } else {
      messages.push({ role: 'user', content: blocks });
    }
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (role === 'tool') {
      pushUser([
        {
          type: 'tool_result',
          tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
          content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
        },
      ]);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const t = textOf(raw.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
      continue;
    }
    // user (or anything else): text and image parts become blocks
    const blocks: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]+);base64,(.*)$/.exec(url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    pushUser(blocks.length > 0 ? blocks : [{ type: 'text', text: '' }]);
  }
  const out: Json = {
    model,
    messages,
    max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : 4096,
  };
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'auto' || choice === 'none') out.tool_choice = { type: choice };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Anthropic Messages response -> canonical (OpenAI) response
// ---------------------------------------------------------------------------

function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-anthropic',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: finishReasonOf(body.stop_reason),
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

// ---------------------------------------------------------------------------
// Canonical (OpenAI) request -> Gemini generateContent request
// ---------------------------------------------------------------------------

export function openAIRequestToGemini(body: Json, model: string): Json {
  let system = '';
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: typeof raw.name === 'string' ? raw.name : '',
              response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
            },
          },
        ],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const data = /^data:([^;,]+);base64,(.*)$/.exec(part.image_url.url);
          if (data !== null) parts.push({ inlineData: { mimeType: data[1], data: data[2] } });
          else parts.push({ fileData: { fileUri: part.image_url.url } });
        }
      }
    }
    if (role === 'assistant') {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      contents.push({ role: 'model', parts: parts.length > 0 ? parts : [{ text: '' }] });
      continue;
    }
    contents.push({ role: 'user', parts: parts.length > 0 ? parts : [{ text: '' }] });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  else if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Gemini generateContent response -> canonical (OpenAI) response / chunks
// ---------------------------------------------------------------------------

function geminiPartsOf(body: Json): { text: string; toolCalls: Json[] } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  return { text, toolCalls };
}

function geminiFinishOf(body: Json): string | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const reason = first.finishReason;
  if (typeof reason !== 'string') return null;
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST') return 'content_filter';
  return 'stop';
}

export function geminiResponseToOpenAI(body: Json, model: string): Json {
  const { text, toolCalls } = geminiPartsOf(body);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: geminiFinishOf(body) ?? 'stop',
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

// ---------------------------------------------------------------------------
// SSE conversion: provider streams -> canonical OpenAI chunk frames (text)
// ---------------------------------------------------------------------------

export function sseFrame(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

function openAIChunk(model: string, delta: Json, finish: string | null, usage?: Json): Json {
  const chunk: Json = {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
  if (usage !== undefined) chunk.usage = usage;
  return chunk;
}

/** Parse an SSE body (already fully read) into its data payloads, in order. */
export function ssePayloads(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) out.push(trimmed.slice(5).trim());
    }
  }
  return out;
}

/** An Anthropic SSE body as canonical OpenAI SSE text (chunks + [DONE]). */
export function anthropicStreamToOpenAI(text: string, model: string): string {
  const frames: string[] = [];
  let currentEvent = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: string | null = null;
  const toolIndexByBlock = new Map<number, number>();
  let nextTool = 0;
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    let data = '';
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('event:')) currentEvent = trimmed.slice(6).trim();
      else if (trimmed.startsWith('data:')) data = trimmed.slice(5).trim();
    }
    if (data === '') continue;
    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(payload)) continue;
    if (currentEvent === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) {
      inputTokens = numberOrZero(payload.message.usage.input_tokens);
    } else if (currentEvent === 'content_block_start' && isRecord(payload.content_block) && payload.content_block.type === 'tool_use') {
      const blockIndex = typeof payload.index === 'number' ? payload.index : 0;
      const toolIndex = nextTool;
      nextTool += 1;
      toolIndexByBlock.set(blockIndex, toolIndex);
      frames.push(
        sseFrame(
          openAIChunk(model, {
            tool_calls: [
              {
                index: toolIndex,
                id: typeof payload.content_block.id === 'string' ? payload.content_block.id : `toolu_${toolIndex}`,
                type: 'function',
                function: { name: typeof payload.content_block.name === 'string' ? payload.content_block.name : '', arguments: '' },
              },
            ],
          }, null),
        ),
      );
    } else if (currentEvent === 'content_block_delta' && isRecord(payload.delta)) {
      if (payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') {
        frames.push(sseFrame(openAIChunk(model, { content: payload.delta.text }, null)));
      } else if (payload.delta.type === 'input_json_delta' && typeof payload.delta.partial_json === 'string') {
        const blockIndex = typeof payload.index === 'number' ? payload.index : 0;
        const toolIndex = toolIndexByBlock.get(blockIndex) ?? 0;
        frames.push(
          sseFrame(openAIChunk(model, { tool_calls: [{ index: toolIndex, function: { arguments: payload.delta.partial_json } }] }, null)),
        );
      }
    } else if (currentEvent === 'message_delta') {
      if (isRecord(payload.delta)) finish = finishReasonOf(payload.delta.stop_reason);
      if (isRecord(payload.usage)) outputTokens = numberOrZero(payload.usage.output_tokens);
    }
  }
  frames.push(
    sseFrame(
      openAIChunk(model, {}, finish ?? 'stop', {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      }),
    ),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A Gemini SSE body as canonical OpenAI SSE text (chunks + [DONE]). */
export function geminiStreamToOpenAI(text: string, model: string): string {
  const frames: string[] = [];
  let usage: Json | undefined;
  let finish: string | null = null;
  for (const payloadText of ssePayloads(text)) {
    if (payloadText === '[DONE]') continue;
    let payload: unknown;
    try {
      payload = JSON.parse(payloadText);
    } catch {
      continue;
    }
    if (!isRecord(payload)) continue;
    const { text: deltaText, toolCalls } = geminiPartsOf(payload);
    const delta: Json = {};
    if (deltaText !== '') delta.content = deltaText;
    if (toolCalls.length > 0) {
      delta.tool_calls = toolCalls.map((call, i) => ({
        index: i,
        id: call.id,
        type: 'function',
        function: (call as Json).function,
      }));
    }
    const f = geminiFinishOf(payload);
    if (f !== null) finish = f;
    if (isRecord(payload.usageMetadata)) {
      usage = {
        prompt_tokens: numberOrZero(payload.usageMetadata.promptTokenCount),
        completion_tokens: numberOrZero(payload.usageMetadata.candidatesTokenCount),
        total_tokens: numberOrZero(payload.usageMetadata.totalTokenCount),
      };
    }
    if (Object.keys(delta).length > 0) frames.push(sseFrame(openAIChunk(model, delta, null)));
  }
  frames.push(sseFrame(openAIChunk(model, {}, finish ?? 'stop', usage)));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}
