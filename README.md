# Hermes Skills

Personal skill tap for [Hermes Agent](https://hermes-agent.nousresearch.com/).

## Install

Add this tap as a skill source, then install the skills you need:

```bash
# Register the tap (one-time)
hermes skills tap add countrvl/hermes-skills

# Install individual skills
hermes skills install countrvl/hermes-skills/prompt-optimizer
```

`tap add` only registers the source — it does **not** auto-install everything.
Pick and install only the skills you want.

## Skills

| Name | Description |
|------|-------------|
| [prompt-optimizer](prompt-optimizer/) | Multi-agent prompt optimizer — detects contradictions, format gaps, and few-shot inconsistencies, then surgically rewrites prompts. Two modes: build from scratch or analyze existing. Based on OpenAI Cookbook methodology. Powered by Python runtime. |

## How it works

`prompt-optimizer` always runs contradiction and format checkers in parallel;
the few-shot consistency checker runs only when `examples` are provided. After
checks complete, a surgical rewriter can apply fixes. Each checker is a separate
LLM call.

**The skill uses the same model and provider configured in your Hermes Agent**
(`config.yaml` → `model.provider` + `model.default`). API credentials are read
from your Hermes `.env` automatically. No extra setup needed.
