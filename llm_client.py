import json
import logging
import re
import threading
import time
from typing import Optional

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY,
    GOOGLE_API_KEY,
    FIREWORKS_API_KEY,
    RATE_LIMITS,
    TARGET_TEMPERATURE,
)

logger = logging.getLogger("pih.llm_model")

# Base back-off (seconds) after an OpenRouter "in_flight_budget_exhausted" 402; multiplied by the attempt number.
IN_FLIGHT_BUDGET_RETRY_DELAY = 40.0

_next_slot_time: dict[str, float] = {}
_rate_limit_lock = threading.Lock()


def _rate_limit(provider: str) -> None:
    """
    Per-provider request spacing. The lock is held only to reserve a time slot; the sleep happens
    outside it, so a thread waiting on one provider never blocks threads calling another provider.
    """
    limit = RATE_LIMITS.get(provider, 60)
    if limit <= 0:
        return
    min_interval = 60.0 / limit
    with _rate_limit_lock:
        now = time.time()
        slot = max(now, _next_slot_time.get(provider, 0.0))
        _next_slot_time[provider] = slot + min_interval
    wait = slot - time.time()
    if wait > 0:
        time.sleep(wait)


def split_reasoning_and_output(raw_content: str, raw_reasoning: str = "") -> tuple[str, str]:
    """
    Splits an LLM response into (think_trace, final_answer).
    Handles:
      1. Explicit <think>...</think> tags inside raw_content.
      2. Unclosed <think>... tags due to max_token truncation.
      3. Provider API-level reasoning_content (e.g. OpenRouter, DeepSeek-R1).
    """
    raw_content = (raw_content or "").strip()
    raw_reasoning = (raw_reasoning or "").strip()

    # 1. Check for complete <think>...</think> in content
    match = re.search(r"<think>(.*?)</think>(.*)", raw_content, flags=re.DOTALL)
    if match:
        think_inside = match.group(1).strip()
        final_ans = match.group(2).strip()
        combined_think = f"{raw_reasoning}\n{think_inside}".strip() if raw_reasoning else think_inside
        return combined_think, final_ans

    # 2. Check for unclosed <think> tag (truncated generation)
    unclosed_match = re.search(r"<think>(.*)", raw_content, flags=re.DOTALL)
    if unclosed_match:
        think_inside = unclosed_match.group(1).strip()
        combined_think = f"{raw_reasoning}\n{think_inside}".strip() if raw_reasoning else think_inside
        return combined_think, ""

    # 3. If reasoning was provided separately via delta.reasoning_content
    if raw_reasoning:
        return raw_reasoning, raw_content

    # 4. Default: No reasoning detected
    return "", raw_content


def _sampling_kwargs(top_p: Optional[float]) -> dict:
    """Optional sampling parameters; omitted entirely when unset so provider defaults apply."""
    return {"top_p": top_p} if top_p is not None else {}


def _stream_openai_compatible(
    client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
) -> dict:
    """
    Streams OpenAI-compatible completions while accumulating delta.reasoning_content
    and delta.content into separate buffers before splitting.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    reasoning_buffer: list[str] = []
    content_buffer: list[str] = []
    finish_reason = "stop"
    model_returned = model_id
    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **_sampling_kwargs(top_p),
        )

        for chunk in stream:
            if hasattr(chunk, "model") and chunk.model:
                model_returned = chunk.model
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = {
                    "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(chunk.usage, "total_tokens", 0) or 0,
                }
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta:
                # Capture reasoning_content (e.g. DeepSeek-R1, QwQ, Kimi on OpenRouter)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_buffer.append(delta.reasoning_content)
                elif hasattr(delta, "reasoning") and delta.reasoning:
                    reasoning_buffer.append(delta.reasoning)
                if hasattr(delta, "content") and delta.content:
                    content_buffer.append(delta.content)

        raw_reasoning = "".join(reasoning_buffer)
        raw_content = "".join(content_buffer)
        think_trace, final_answer = split_reasoning_and_output(raw_content, raw_reasoning)

        return {
            "content": raw_content,
            "raw_content": raw_content,
            "raw_reasoning": raw_reasoning,
            "think_trace": think_trace,
            "final_answer": final_answer,
            "usage": usage_data,
            "model": model_returned,
            "finish_reason": finish_reason,
        }

    except Exception as e:
        logger.debug(f"Streaming failed ({e}), falling back to non-streaming call...")
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **_sampling_kwargs(top_p),
        )
        msg = response.choices[0].message
        raw_content = msg.content or ""
        raw_reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        if not raw_reasoning and hasattr(msg, "model_extra") and msg.model_extra:
            raw_reasoning = (
                msg.model_extra.get("reasoning_content", "") or msg.model_extra.get("reasoning", "") or ""
            )
        think_trace, final_answer = split_reasoning_and_output(raw_content, raw_reasoning)
        usage = response.usage
        return {
            "content": raw_content,
            "raw_content": raw_content,
            "raw_reasoning": raw_reasoning,
            "think_trace": think_trace,
            "final_answer": final_answer,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
        }


def _call_openai(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    return _stream_openai_compatible(
        client=client,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )


def _call_anthropic(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    request = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        **_sampling_kwargs(top_p),
    }
    if system_prompt:
        request["system"] = system_prompt
    response = client.messages.create(**request)
    raw_content = ""
    raw_reasoning = ""
    for block in response.content:
        if getattr(block, "type", "") == "text":
            raw_content += getattr(block, "text", "")
        elif getattr(block, "type", "") == "thinking":
            raw_reasoning += getattr(block, "thinking", "")

    think_trace, final_answer = split_reasoning_and_output(raw_content, raw_reasoning)

    return {
        "content": raw_content,
        "raw_content": raw_content,
        "raw_reasoning": raw_reasoning,
        "think_trace": think_trace,
        "final_answer": final_answer,
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
        "model": response.model,
        "finish_reason": response.stop_reason,
    }


def _call_openrouter(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
) -> dict:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    return _stream_openai_compatible(
        client=client,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )


def _call_fireworks(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
) -> dict:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=FIREWORKS_API_KEY,
    )
    return _stream_openai_compatible(
        client=client,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )


def _call_google(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=model_id,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **_sampling_kwargs(top_p),
        ),
    )
    raw_content = response.text or ""
    think_trace, final_answer = split_reasoning_and_output(raw_content, "")
    return {
        "content": raw_content,
        "raw_content": raw_content,
        "raw_reasoning": "",
        "think_trace": think_trace,
        "final_answer": final_answer,
        "usage": {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
        },
        "model": model_id,
        "finish_reason": "stop",
    }


_PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "openrouter": _call_openrouter,
    "fireworks": _call_fireworks,
    "google": _call_google,
}


def apply_prompt_format(model_config: dict, system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """
    Applies per-model prompt-format rules. Models configured with "use_system_prompt": False
    (e.g. DeepSeek-R1, per its official usage recommendations) receive the system prompt
    prepended to the start of the user turn instead of as a separate system message.
    """
    if system_prompt and model_config.get("use_system_prompt", True) is False:
        return "", f"{system_prompt}\n\n{user_prompt}"
    return system_prompt, user_prompt


def call_llm(
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    top_p: Optional[float] = None,
) -> dict:
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    call_fn = _PROVIDERS.get(provider)

    if call_fn is None:
        return {
            "content": "",
            "raw_content": "",
            "raw_reasoning": "",
            "think_trace": "",
            "final_answer": "",
            "usage": {},
            "model": model_id,
            "finish_reason": "error",
            "error": f"Unknown provider: {provider}",
        }

    system_prompt, user_prompt = apply_prompt_format(model_config, system_prompt, user_prompt)

    for attempt in range(max_retries):
        try:
            _rate_limit(provider)
            result = call_fn(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            logger.debug(
                f"[{model_config['display_name']}] OK "
                f"{result['usage'].get('total_tokens', '?')} tokens"
            )
            return result

        except Exception as e:
            delay = retry_delay * (2 ** attempt)
            if "in_flight_budget_exhausted" in str(e):
                # OpenRouter 402: the reserved cost of the requests still running (max_tokens x price) exceeds the
                # balance. It clears when they finish (Retry-After is 120 s), so back off much longer than usual.
                delay = IN_FLIGHT_BUDGET_RETRY_DELAY * (attempt + 1)
            logger.warning(
                f"[{model_config['display_name']}] Attempt {attempt + 1}/{max_retries} "
                f"failed: {e}. Retrying in {delay:.1f}s"
            )
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error(
                    f"[{model_config['display_name']}] All {max_retries} attempts failed."
                )
                return {
                    "content": "",
                    "raw_content": "",
                    "raw_reasoning": "",
                    "think_trace": "",
                    "final_answer": "",
                    "usage": {},
                    "model": model_id,
                    "finish_reason": "error",
                    "error": str(e),
                }


def target_sampling_params(model_config: dict) -> dict:
    """Sampling parameters for a target model: per-model overrides, else global defaults."""
    return {
        "temperature": model_config.get("temperature", TARGET_TEMPERATURE),
        "top_p": model_config.get("top_p"),
    }


def call_target_model(model_config: dict, system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    """call_llm with the target model's own sampling settings and prompt-format rules."""
    params = target_sampling_params(model_config)
    return call_llm(
        model_config=model_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=max_tokens,
    )


# LaTeX commands whose first letter collides with a legal JSON escape (\b \f \n \r \t).
# json.loads silently accepts "\times" as TAB + "imes", "\frac" as FORMFEED + "rac", "\beta" as
# BACKSPACE + "eta", "\nu" as NEWLINE + "u", "\rho" as CR + "ho" — no error is raised, so the text is
# corrupted without any recovery tier ever running. These are re-escaped before every parse attempt.
# Trade-off: a genuine \n or \t escape immediately followed by one of these exact whole words
# (e.g. a newline followed by "u = 0.3") would be misread as LaTeX; that is rare in this pipeline's JSON.
_LATEX_JSON_ESCAPE_COLLISIONS = (
    # \t...
    "times", "text", "textrm", "textbf", "textit", "textup", "theta", "tau", "tan", "tanh",
    "tilde", "to", "top", "triangle", "tfrac",
    # \f...
    "frac", "dfrac", "forall", "flat",
    # \b...
    "beta", "bar", "boxed", "bf", "big", "bigl", "bigr", "bigg", "binom", "bullet", "bmod", "boldsymbol",
    # \n...
    "nu", "nabla", "neq", "ne", "neg", "not", "nolimits", "ni",
    # \r...
    "rho", "right", "rightarrow", "rm", "rangle", "rceil", "rfloor",
)
_LATEX_COLLISION_RE = re.compile(
    r"(?<!\\)\\(?=(?:"
    + "|".join(sorted(_LATEX_JSON_ESCAPE_COLLISIONS, key=len, reverse=True))
    + r")(?![A-Za-z]))"
)
# Group 1 is an existing "\\" pair, which must be kept as it is; the second alternative is any other backslash that
# does not start a legal JSON escape (\" \/ \b \f \n \r \t \uXXXX). Pairs are consumed left to right, so the second
# backslash of a valid "\\mathrm" is never mistaken for a lone invalid one. Use with
# `_INVALID_JSON_ESCAPE_RE.sub(lambda m: m.group(1) or "\\\\", s)`.
_INVALID_JSON_ESCAPE_RE = re.compile(r'(\\\\)|\\(?![/"bfnrt]|u[0-9a-fA-F]{4})')


def protect_latex_escapes(s: str) -> str:
    """Doubles the backslash of LaTeX commands that json.loads would otherwise decode as control characters."""
    return _LATEX_COLLISION_RE.sub(r"\\\\", s)


def extract_json_from_response(text: str) -> Optional[dict]:
    if not text:
        return None

    # If text contains reasoning tags, look primarily in final answer portion
    _, clean_text = split_reasoning_and_output(text)
    if clean_text:
        text_to_parse = clean_text
    else:
        text_to_parse = text

    def try_parse(s: str) -> Optional[dict]:
        s = protect_latex_escapes(s)
        try:
            return json.loads(s.strip())
        except Exception:
            # Fallback 1: Escape remaining invalid backslashes from LaTeX (e.g. \mathrm, \alpha, \underline)
            s_fixed = _INVALID_JSON_ESCAPE_RE.sub(lambda m: m.group(1) or "\\\\", s)
            # Fallback 2: Fix tuple syntax in values (e.g. "answer": (13.97, 2.408) -> "answer": [13.97, 2.408])
            s_fixed = re.sub(r":\s*\(([^)]+)\)", r": [\1]", s_fixed)
            # Fallback 3: Remove trailing commas before closing braces/brackets
            s_fixed = re.sub(r",\s*([\]}])", r"\1", s_fixed)
            try:
                return json.loads(s_fixed.strip())
            except Exception:
                return None

    direct = try_parse(text_to_parse)
    if direct:
        return direct

    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    ]
    for target in [text_to_parse, text]:
        for pattern in patterns:
            match = re.search(pattern, target, re.DOTALL)
            if match:
                candidate = match.group(1) if match.lastindex else match.group(0)
                parsed = try_parse(candidate)
                if parsed:
                    return parsed

    return None
