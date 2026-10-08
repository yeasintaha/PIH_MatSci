import json
import time
import logging
import threading
from typing import Optional

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY,
    GOOGLE_API_KEY,
    FIREWORKS_API_KEY,
    RATE_LIMITS,
)

logger = logging.getLogger("pih.llm_model")

_last_request_time: dict[str, float] = {}
_rate_limit_lock = threading.Lock()


def _rate_limit(provider: str) -> None:
    limit = RATE_LIMITS.get(provider, 60)
    if limit <= 0:
        return
    min_interval = 60.0 / limit
    with _rate_limit_lock:
        now = time.time()
        last = _last_request_time.get(provider, 0)
        wait = min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        _last_request_time[provider] = time.time()


def _call_openai(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "model": response.model,
        "finish_reason": response.choices[0].finish_reason,
    }


def _call_anthropic(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_id,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "content": response.content[0].text,
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
) -> dict:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = response.usage
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
        "model": response.model,
        "finish_reason": response.choices[0].finish_reason,
    }


def _call_google(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=model_id,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return {
        "content": response.text,
        "usage": {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
        },
        "model": model_id,
        "finish_reason": "stop",
    }


def _call_fireworks(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=FIREWORKS_API_KEY,
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = response.usage
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
        "model": response.model,
        "finish_reason": response.choices[0].finish_reason,
    }


_PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "openrouter": _call_openrouter,
    "fireworks": _call_fireworks,
    "google": _call_google,
}


def call_llm(
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    call_fn = _PROVIDERS.get(provider)

    if call_fn is None:
        return {
            "content": "",
            "usage": {},
            "model": model_id,
            "finish_reason": "error",
            "error": f"Unknown provider: {provider}",
        }

    for attempt in range(max_retries):
        try:
            _rate_limit(provider)
            result = call_fn(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.debug(
                f"[{model_config['display_name']}] OK"
                f"{result['usage'].get('total_tokens', '?')} tokens"
            )
            return result

        except Exception as e:
            delay = retry_delay * (2 ** attempt)
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
                    "usage": {},
                    "model": model_id,
                    "finish_reason": "error",
                    "error": str(e),
                }


def extract_json_from_response(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    import re
    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate.strip())
            except (json.JSONDecodeError, IndexError):
                continue

    return None
