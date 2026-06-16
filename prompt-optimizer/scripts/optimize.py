#!/usr/bin/env python3
"""
Prompt Optimizer — multi-agent prompt analysis and rewriting.
Based on OpenAI Cookbook "Optimize Prompts" methodology.

Usage:
  python optimize.py check --prompt "..." [--examples '[...]']
  python optimize.py rewrite --prompt "..." --check-result '{"contradiction":...}' [--examples '[...]']

Environment:
  PROMPT_OPTIMIZER_API_KEY  — API key (falls back to OPENAI_API_KEY, HERMES_API_KEY)
  PROMPT_OPTIMIZER_API_BASE — API base URL (default: https://api.deepseek.com/v1)
  PROMPT_OPTIMIZER_MODEL    — Model name (default: deepseek-chat)
"""

import os
import sys
import json
import asyncio
import argparse
import difflib
from typing import Any, Optional

# ── Configuration ──────────────────────────────────────────────
#
# Priority for API key and settings:
#   1. Explicit env vars: PROMPT_OPTIMIZER_API_KEY / _API_BASE / _MODEL
#   2. Hermes config.yaml → current provider + model
#   3. Hermes auth.json → provider's base_url and env-var name for key
#   4. Hermes .env file → API key by provider's env-var name
#   5. Standard fallback env vars (OPENAI_API_KEY, HERMES_API_KEY)

# Known provider defaults (used when auth.json is unavailable)
_PROVIDER_DEFAULTS = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "model": "claude-sonnet-4-20250514"},
    "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "xai": {"base_url": "https://api.x.ai/v1", "model": "grok-3-beta"},
}


def _first_env_path(*names: str) -> str:
    """Return the first configured environment path, expanded to an absolute path."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return os.path.abspath(os.path.expanduser(value))
    return ""


def _hermes_dir() -> str:
    """
    Locate the Hermes Agent configuration directory.

    This mirrors README.md: prompt-optimizer reuses the provider/model from
    Hermes config.yaml and credentials from Hermes .env/auth.json. Search order:
      1. Official Hermes home env vars, with HERMES_HOME first.
      2. Windows LOCALAPPDATA layout: <LOCALAPPDATA>/hermes.
      3. Unix/XDG layout: <XDG_CONFIG_HOME or ~/.config>/hermes.
    """
    hermes_home = _first_env_path("HERMES_HOME", "HERMES_AGENT_HOME")
    if hermes_home:
        return hermes_home

    local_app_data = _first_env_path("LOCALAPPDATA")
    if local_app_data:
        return os.path.join(local_app_data, "hermes")

    xdg_config_home = _first_env_path("XDG_CONFIG_HOME")
    if xdg_config_home:
        return os.path.join(xdg_config_home, "hermes")

    return os.path.join(os.path.expanduser("~"), ".config", "hermes")


def _read_config_yaml() -> dict:
    """Read provider and model from Hermes config.yaml (no YAML dependency)."""
    config_path = os.path.join(_hermes_dir(), "config.yaml")
    result = {}
    try:
        with open(config_path) as f:
            in_model = False
            for line in f:
                raw_line = line.rstrip("\n")
                stripped = raw_line.strip()
                indent = len(raw_line) - len(raw_line.lstrip(" \t"))

                if stripped.startswith("model:") and indent == 0:
                    in_model = True
                    continue

                if in_model:
                    if stripped and indent == 0:
                        in_model = False  # left model section
                        continue
                    if stripped.startswith("provider:"):
                        result["provider"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                    elif stripped.startswith("default:"):
                        result["model"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
    except (OSError, IOError):
        pass
    return result


def _read_auth_json() -> dict:
    """Read provider credentials from Hermes auth.json."""
    auth_path = os.path.join(_hermes_dir(), "auth.json")
    try:
        with open(auth_path) as f:
            data = json.load(f)
        pool = data.get("credential_pool", {})
        # Return first credential for each provider
        result = {}
        for provider, creds in pool.items():
            if creds and isinstance(creds, list):
                cred = creds[0]
                result[provider] = {
                    "base_url": cred.get("base_url", ""),
                    "label": cred.get("label", ""),  # env var name
                }
        return result
    except (OSError, IOError, json.JSONDecodeError):
        return {}


def _read_env_value(env_path: str, key: str) -> str:
    """Read a specific KEY=VALUE from a .env-style file."""
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    value = v.strip().strip('"').strip("'")
                    if value and not value.startswith("#"):
                        return value
    except (OSError, IOError):
        pass
    return ""


def _resolve_config() -> tuple[str, str, str]:
    """
    Resolve (api_key, base_url, model) from Hermes configuration.
    Returns empty strings for missing values.
    """
    # 1. Explicit overrides (user set these intentionally)
    api_key = os.environ.get("PROMPT_OPTIMIZER_API_KEY", "")
    base_url = os.environ.get("PROMPT_OPTIMIZER_API_BASE", "")
    model = os.environ.get("PROMPT_OPTIMIZER_MODEL", "")

    # 2. Read Hermes config for provider/model
    config = _read_config_yaml()
    provider = config.get("provider", "deepseek")

    # 3. Read auth.json for this provider's base_url and env-var label
    auth = _read_auth_json()
    provider_auth = auth.get(provider, {})

    if not base_url:
        base_url = provider_auth.get("base_url", "")
    if not base_url:
        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        base_url = defaults.get("base_url", "https://api.deepseek.com/v1")

    if not model:
        model = config.get("model", "")
    if not model:
        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        model = defaults.get("model", "deepseek-chat")

    # 4. API key: use auth.json label to find the right env var
    if not api_key:
        label = provider_auth.get("label", "")
        if label:
            api_key = os.environ.get(label, "")
        if not api_key and label:
            # Try .env file
            env_path = os.path.join(_hermes_dir(), ".env")
            api_key = _read_env_value(env_path, label)

    # 5. Standard fallbacks
    if not api_key:
        for var in ("OPENAI_API_KEY", "HERMES_API_KEY"):
            api_key = os.environ.get(var, "")
            if api_key:
                break
    if not api_key:
        env_path = os.path.join(_hermes_dir(), ".env")
        for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            api_key = _read_env_value(env_path, var)
            if api_key:
                break

    return api_key, base_url.rstrip("/"), model


API_KEY, API_BASE, MODEL = _resolve_config()
MAX_RETRIES = 2
TIMEOUT = 60

# ── Checker & Rewriter System Prompts ─────────────────────────

CONTRADICTION_CHECKER_PROMPT = """Ты — **Dev-Contradiction-Checker**. Твоя задача — найти логические противоречия внутри промпта (developer message).

**Определения:**
- Противоречие = две инструкции, которые невозможно выполнить одновременно.
- Пересечения или избыточность НЕ являются противоречиями. Флагай только взаимоисключающие требования.

**Что ты ДОЛЖЕН сделать:**
1. Сравни каждое требование и запрет со всеми остальными.
2. Выпиши максимум ПЯТЬ противоречий, каждое одним предложением.
3. Если противоречий нет — честно скажи об этом.

**Важно:** Будь консервативен. Если сомневаешься — НЕ флагай. Лучше пропустить слабый сигнал, чем выдать ложное срабатывание.

**Формат ответа — строгий JSON:**
```json
{
  "has_issues": true,
  "issues": [
    "Инструкция требует «отвечай только на английском», но также говорит «nunca respondas en inglés» — невозможно выполнить оба требования одновременно",
    "..."
  ]
}
```
Поле `has_issues` = true ТОЛЬКО если массив `issues` непустой. Никаких дополнительных ключей или текста вне JSON."""

FORMAT_CHECKER_PROMPT = """Ты — **Format-Checker**. Твоя задача — проверить, насколько чётко в промпте описан формат вывода.

**Алгоритм:**
1. Определи, требует ли промпт структурированный вывод: JSON, CSV, XML, Markdown-таблицу, список с определёнными полями, или любой формат с явной структурой.
2. Если промпт чисто разговорный (свободный текст без требований к структуре) — форматных проблем нет.
3. Если структура требуется — проверь:
   - Указаны ли все обязательные поля/ключи?
   - Понятны ли типы данных для каждого поля (строка, число, массив)?
   - Ясно ли описана вложенность и отношения между полями?
   - Есть ли неоднозначности в описании формата?
4. Выпиши максимум ПЯТЬ проблем.

**Важно:** Будь консервативен. Если формат описан достаточно — не придумывай придирок. Не флагай отсутствие формата, если промпт не требует структурированного вывода.

**Формат ответа — строгий JSON:**
```json
{
  "has_issues": true,
  "issues": [
    "Промпт требует JSON, но не указывает обязательные поля — модель может вернуть произвольную структуру",
    "..."
  ]
}
```
Поле `has_issues` = true ТОЛЬКО если массив `issues` непустой."""

FEWSHOT_CHECKER_PROMPT = """Ты — **Few-Shot-Consistency-Checker**. Твоя задача — проверить, соответствуют ли few-shot примеры правилам из промпта.

**Что ты получишь:**
- `DEVELOPER_MESSAGE` — промпт с инструкциями и правилами
- `EXAMPLES` — список примеров (роль + содержание)

**Алгоритм проверки:**
1. Извлеки ВСЕ правила и требования из `DEVELOPER_MESSAGE`.
2. Для каждого ответа ассистента в примерах проверь:
   - Соблюдает ли он формат вывода, указанный в промпте?
   - Выполняет ли он все обязательные требования?
   - Не нарушает ли он явные запреты?
3. Выпиши максимум ПЯТЬ несоответствий, указывая номер примера и конкретное правило, которое нарушено.
4. Если примеры полностью соответствуют промпту — проблем нет.

**Важно:** Не флагай стилистические расхождения. Только явные нарушения правил и требований.

**Формат ответа — строгий JSON:**
```json
{
  "has_issues": true,
  "issues": [
    "Пример №1: ассистент даёт развёрнутый ответ с объяснением, но промпт требует «только да или нет без пояснений»",
    "..."
  ]
}
```
Поле `has_issues` = true ТОЛЬКО если массив `issues` непустой."""

REWRITER_PROMPT = """Ты — **Prompt-Rewriter**. Твоя задача — исправить промпт, устранив найденные проблемы, и при этом СОХРАНИТЬ всё, что работает.

**Что ты получишь:**
- `ORIGINAL_PROMPT` — исходный текст промпта
- `ISSUES` — список найденных проблем (противоречия, проблемы формата, несоответствия примеров)

**Правила переписывания:**
1. Исправляй ТОЛЬКО то, на что указывают `ISSUES`. Остальной текст не трогай.
2. При противоречиях: разреши конфликт в пользу более важного/фундаментального требования. Если непонятно, какое важнее — сохрани более строгое.
3. При проблемах формата: добавь недостающие поля, уточни типы, опиши структуру.
4. Сохрани оригинальный стиль, тон и структуру промпта. Не переписывай хорошо работающие секции.
5. Не удаляй контент, не связанный с проблемами. Будь хирургичен.
6. Не оборачивай результат в --- или другие markdown-разделители. new_prompt — это чистый текст промпта, а не документ.

**Формат ответа — строгий JSON:**
```json
{
  "new_prompt": "полный текст исправленного промпта"
}
```"""

FEWSHOT_REWRITER_PROMPT = """Ты — **Few-Shot-Rewriter**. Твоя задача — исправить few-shot примеры так, чтобы они полностью соответствовали правилам из промпта.

**Что ты получишь:**
- `NEW_PROMPT` — уже исправленный промпт (после рерайта)
- `ORIGINAL_EXAMPLES` — исходные примеры
- `FEWSHOT_ISSUES` — список найденных несоответствий

**Правила:**
1. Исправь ТОЛЬКО ответы ассистента, которые нарушают правила. Сообщения пользователя не трогай.
2. Каждый исправленный ответ должен строго следовать ВСЕМ правилам из `NEW_PROMPT`.
3. Сохрани смысл ответа, где это возможно. Если ответ полностью противоречит правилам — замени на корректный, но близкий по духу.
4. Не меняй количество примеров и порядок сообщений.

**Формат ответа — строгий JSON:**
```json
{
  "new_examples": [
    {"role": "user", "content": "исходный вопрос"},
    {"role": "assistant", "content": "исправленный ответ"},
    ...
  ]
}
```"""

# ── LLM API Client ────────────────────────────────────────────

async def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> dict:
    """Call OpenAI-compatible chat completions API, parse JSON from response."""
    import httpx

    if not API_KEY:
        raise RuntimeError(
            "No API key found. Set PROMPT_OPTIMIZER_API_KEY, OPENAI_API_KEY, or HERMES_API_KEY."
        )

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

            # Extract JSON from response (may have markdown fences)
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove opening fence
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove closing fence
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            return json.loads(content)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            last_error = f"Failed to parse response (attempt {attempt + 1}): {e}"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
                continue
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2)
                continue

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES + 1} attempts: {last_error}")


# ── Checkers ───────────────────────────────────────────────────

async def check_contradictions(prompt: str) -> dict:
    """Check prompt for logical contradictions."""
    try:
        result = await call_llm(
            CONTRADICTION_CHECKER_PROMPT,
            f"Проверь этот промпт на противоречия:\n\n---\n{prompt}\n---",
        )
        return {
            "status": "ok",
            "ok": True,
            "has_issues": result.get("has_issues", False),
            "issues": result.get("issues", []),
        }
    except Exception as e:
        return {
            "status": "error",
            "ok": False,
            "has_issues": None,
            "issues": [],
            "error": str(e),
        }


async def check_format(prompt: str) -> dict:
    """Check prompt for unclear format specifications."""
    try:
        result = await call_llm(
            FORMAT_CHECKER_PROMPT,
            f"Проверь этот промпт на чёткость формата вывода:\n\n---\n{prompt}\n---",
        )
        return {
            "status": "ok",
            "ok": True,
            "has_issues": result.get("has_issues", False),
            "issues": result.get("issues", []),
        }
    except Exception as e:
        return {
            "status": "error",
            "ok": False,
            "has_issues": None,
            "issues": [],
            "error": str(e),
        }


async def check_fewshot(prompt: str, examples: list[dict]) -> dict:
    """Check few-shot examples for consistency with prompt rules."""
    try:
        # Format examples for the checker
        examples_text = json.dumps(examples, ensure_ascii=False, indent=2)
        user_msg = (
            f"DEVELOPER_MESSAGE:\n---\n{prompt}\n---\n\n"
            f"EXAMPLES:\n{examples_text}"
        )
        result = await call_llm(FEWSHOT_CHECKER_PROMPT, user_msg)
        return {
            "status": "ok",
            "ok": True,
            "has_issues": result.get("has_issues", False),
            "issues": result.get("issues", []),
        }
    except Exception as e:
        return {
            "status": "error",
            "ok": False,
            "has_issues": None,
            "issues": [],
            "error": str(e),
        }


# ── Rewriters ──────────────────────────────────────────────────

async def rewrite_prompt(prompt: str, issues: dict) -> str:
    """Rewrite prompt to fix detected issues."""
    issues_text = json.dumps(issues, ensure_ascii=False, indent=2)
    user_msg = (
        f"ORIGINAL_PROMPT:\n---\n{prompt}\n---\n\n"
        f"ISSUES:\n{issues_text}"
    )
    result = await call_llm(REWRITER_PROMPT, user_msg, temperature=0.2)
    return result.get("new_prompt", prompt)


async def rewrite_examples(
    examples: list[dict],
    new_prompt: str,
    fewshot_issues: dict,
) -> list[dict]:
    """Rewrite few-shot examples to match the new prompt."""
    examples_text = json.dumps(examples, ensure_ascii=False, indent=2)
    issues_text = json.dumps(fewshot_issues, ensure_ascii=False, indent=2)
    user_msg = (
        f"NEW_PROMPT:\n---\n{new_prompt}\n---\n\n"
        f"ORIGINAL_EXAMPLES:\n{examples_text}\n\n"
        f"FEWSHOT_ISSUES:\n{issues_text}"
    )
    result = await call_llm(FEWSHOT_REWRITER_PROMPT, user_msg, temperature=0.2)
    return result.get("new_examples", examples)


# ── Diff ───────────────────────────────────────────────────────

def generate_diff(original: str, rewritten: str, label: str = "prompt") -> str:
    """Generate unified diff between original and rewritten text."""
    original_lines = original.splitlines(keepends=True)
    rewritten_lines = rewritten.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        rewritten_lines,
        fromfile=f"original_{label}",
        tofile=f"optimized_{label}",
    )
    return "".join(diff)


# ── Commands ───────────────────────────────────────────────────

async def cmd_check(prompt: str, examples: Optional[list[dict]] = None) -> dict:
    """Run all three checkers in parallel."""
    tasks = [
        check_contradictions(prompt),
        check_format(prompt),
    ]

    if examples:
        tasks.append(check_fewshot(prompt, examples))

    results = await asyncio.gather(*tasks)

    contradiction = results[0]
    format_issues = results[1]
    fewshot = results[2] if len(results) > 2 else {
        "status": "ok",
        "ok": True,
        "has_issues": False,
        "issues": [],
    }

    has_any = (
        contradiction.get("has_issues", False)
        or format_issues.get("has_issues", False)
        or fewshot.get("has_issues", False)
    )
    has_errors = (
        not contradiction.get("ok", False)
        or not format_issues.get("ok", False)
        or not fewshot.get("ok", False)
    )

    return {
        "contradiction": contradiction,
        "format": format_issues,
        "fewshot": fewshot,
        "has_any_issues": has_any,
        "has_errors": has_errors,
    }


async def cmd_rewrite(
    prompt: str,
    check_result: dict,
    examples: Optional[list[dict]] = None,
) -> dict:
    """Rewrite prompt (and optionally examples) based on check results."""
    contradiction = check_result.get("contradiction", {})
    format_issues = check_result.get("format", {})
    fewshot = check_result.get("fewshot", {})

    has_prompt_issues = (
        contradiction.get("has_issues", False)
        or format_issues.get("has_issues", False)
    )
    has_fewshot_issues = fewshot.get("has_issues", False)

    new_prompt = prompt
    new_examples = examples

    if has_prompt_issues:
        all_issues = []
        if contradiction.get("has_issues"):
            for issue in contradiction.get("issues", []):
                all_issues.append(f"[Противоречие] {issue}")
        if format_issues.get("has_issues"):
            for issue in format_issues.get("issues", []):
                all_issues.append(f"[Формат] {issue}")

        issues_payload = {"issues": all_issues}
        new_prompt = await rewrite_prompt(prompt, issues_payload)

    if has_fewshot_issues and examples:
        fewshot_payload = {"issues": fewshot.get("issues", [])}
        new_examples = await rewrite_examples(examples, new_prompt, fewshot_payload)

    result = {
        "new_prompt": new_prompt,
        "diff": generate_diff(prompt, new_prompt, "prompt"),
    }

    if examples:
        result["new_examples"] = new_examples
        if new_examples != examples:
            result["examples_diff"] = generate_diff(
                json.dumps(examples, ensure_ascii=False, indent=2),
                json.dumps(new_examples, ensure_ascii=False, indent=2),
                "examples",
            )

    return result


# ── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prompt Optimizer — multi-agent prompt analysis and rewriting"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # check
    p_check = sub.add_parser("check", help="Run checkers on a prompt")
    p_check.add_argument("--prompt", required=True, help="The prompt text to check")
    p_check.add_argument(
        "--examples",
        default=None,
        help='Optional JSON array of few-shot examples [{"role":"user","content":"..."},...]',
    )

    # rewrite
    p_rewrite = sub.add_parser("rewrite", help="Rewrite prompt based on check results")
    p_rewrite.add_argument("--prompt", required=True, help="The original prompt text")
    p_rewrite.add_argument(
        "--check-result",
        required=True,
        help="JSON output from 'check' command",
    )
    p_rewrite.add_argument(
        "--examples",
        default=None,
        help="Optional JSON array of few-shot examples",
    )

    args = parser.parse_args()

    if not API_KEY:
        print(
            "ERROR: No API key found. Set one of: "
            "PROMPT_OPTIMIZER_API_KEY, OPENAI_API_KEY, HERMES_API_KEY",
            file=sys.stderr,
        )
        sys.exit(1)

    examples = None
    if hasattr(args, "examples") and args.examples:
        try:
            examples = json.loads(args.examples)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid --examples JSON: {e}", file=sys.stderr)
            sys.exit(1)

    async def run():
        if args.mode == "check":
            result = await cmd_check(args.prompt, examples)
        else:  # rewrite
            try:
                check_result = json.loads(args.check_result)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid --check-result JSON: {e}", file=sys.stderr)
                sys.exit(1)
            result = await cmd_rewrite(args.prompt, check_result, examples)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(run())


if __name__ == "__main__":
    main()
