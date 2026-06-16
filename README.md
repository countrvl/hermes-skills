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
