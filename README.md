# Hermes Skills

Personal skill tap for [Hermes Agent](https://hermes-agent.nousresearch.com/).

## Install

Add this tap as a skill source, then install the skills you need:

```bash
# Register the tap (one-time)
hermes skills tap add countrvl/hermes-skills

# Install individual skills
hermes skills install countrvl/hermes-skills/prompt-optimizer

# Install prompt-optimizer runtime dependencies
pip install -r prompt-optimizer/requirements.txt
```

`tap add` only registers the source — it does **not** auto-install everything.
Pick and install only the skills you want. `prompt-optimizer` also needs its Python
runtime dependencies installed explicitly; at minimum that is `httpx` for LLM API
requests. If your Hermes Agent installs skills into a separate skill directory, run
`pip install -r requirements.txt` from inside the installed `prompt-optimizer` directory.

## Skills

| Name | Description |
|------|-------------|
| [prompt-optimizer](prompt-optimizer/) | Multi-agent prompt optimizer — detects contradictions, format gaps, and few-shot inconsistencies, then surgically rewrites prompts. Two modes: build from scratch or analyze existing. Based on OpenAI Cookbook methodology. Powered by Python runtime. |

## How it works

`prompt-optimizer` always runs contradiction and format checkers in parallel;
the few-shot consistency checker runs only when `examples` are provided. After
checks complete, a surgical rewriter can apply fixes. Each checker is a separate
LLM call.

**The skill reuses your Hermes model/provider only when that provider exposes an
OpenAI-compatible chat completions API** (`/chat/completions`). It reads
`config.yaml` → `model.provider` + `model.default` and credentials from Hermes
`.env`/`auth.json`, but the runtime is not a universal adapter for every Hermes
provider. Native APIs such as Anthropic Messages are not called directly; use an
OpenAI-compatible proxy/base URL via `PROMPT_OPTIMIZER_API_BASE` and set the
matching `PROMPT_OPTIMIZER_MODEL`, or choose one of the supported compatible
providers (DeepSeek, OpenAI, OpenRouter, Google OpenAI-compatible endpoint, Groq,
or xAI).
