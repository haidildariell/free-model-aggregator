# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/haidildariell/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/haidildariell/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

An entry for the make.fun bounty **free-model-aggregator-01** — "The best free AI model aggregator".

One Node server that speaks both client wires — OpenAI Chat Completions and Anthropic
Messages, JSON and streaming — and routes every request to the best free model across ten
free-tier providers, each spoken to in its own wire (OpenAI, Anthropic, Gemini).

- Every free model meeting a request's requirements (tools, vision, JSON mode, context) is
  ranked by the quality table's rating; a plan made at startup spreads the workload over the
  providers under a per-provider cap so no single free budget carries the trial.
- A refused call (402 budget spent, 429 rate limit, 5xx) fails the request over to the next
  candidate instead of ending it; a 402 marks the provider spent for the rest of the trial.
- All three upstream wires are implemented, pivoting through the OpenAI shape: system
  prompts, images, tool calls and their results, stop reasons, usage and streaming events
  are translated in both directions.
- Every routing choice derives from the brief, the workload manifest and the providers'
  answers — no clock, no randomness — so a replayed trial reproduces the same calls.

Local practice grades (kit `grade-local`, 3 instances): **score 99 / 100** — quality 99,
served 100, wire 100, failover 100, catalog 100.

Bounty: https://make.fun/bounty/free-model-aggregator-01

## Run

```
node main.ts            # with MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE set by the trial
node main.ts --list     # print the provider catalog JSON
```

MIT licensed.
