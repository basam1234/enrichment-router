from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models import EnrichmentRequest, FieldName, ResolvedField
from ..pricing import modeled_cost_usd

# Tier 2 uses an LLM for high-quality enrichment. Confidence is set to
# 0.9 — lower than 1.0 to acknowledge that LLMs can hallucinate
# industry categories, country names, or public/private status, and the
# prompt does not cite sources. The declared cost is a conservative
# placeholder; actual cost is computed per-call from token usage against
# Claude Haiku 4.5 rates via modeled_cost_usd.
TIER2_CONFIDENCE: float = 0.9
TIER2_DECLARED_COST_USD: float = 0.001
TIER2_DECLARED_LATENCY_MS: float = 2000.0


@dataclass(frozen=True)
class LLMProviderConfig:
    """Immutable configuration for an OpenAI-compatible LLM provider.

    Frozen to prevent accidental mutation after construction — the
    GroqLLMClient caches the OpenAI client handle at init time, and
    changing base_url or api_key after the fact would silently have
    no effect on the already-constructed client.

    The Groq endpoint is used for development/testing because it
    offers free access to openai/gpt-oss-20b. Production would swap
    this for a direct Anthropic (Claude Haiku) or another paid
    provider. The pricing model in pricing.py always assumes Claude
    Haiku 4.5 rates regardless of which provider is configured here.
    """

    base_url: str
    api_key_env: str
    model: str


DEFAULT_GROQ_CONFIG = LLMProviderConfig(
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    model="llama-3.1-8b-instant",
)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> LLMResponse: ...


class GroqLLMClient:
    def __init__(self, config: LLMProviderConfig = DEFAULT_GROQ_CONFIG):
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing env var {config.api_key_env}; cannot construct GroqLLMClient"
            )
        self._client = OpenAI(base_url=config.base_url, api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system: str, prompt: str) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


class FakeLLMClient:
    def __init__(self, scripts: dict[str, LLMResponse]):
        self.scripts = dict(scripts)

    def complete(self, system: str, prompt: str) -> LLMResponse:
        for substring, response in self.scripts.items():
            if substring in prompt:
                return response
        raise KeyError(
            f"FakeLLMClient: no scripted response matches prompt (first 80 chars): "
            f"{prompt[:80]!r}"
        )


@dataclass
class LLMResult:
    resolved: dict[FieldName, ResolvedField] = field(default_factory=dict)
    cost_usd: float = 0.0
    measured_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _parse_llm_json(text: str) -> dict:
    cleaned = _strip_markdown_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_prompt(request: EnrichmentRequest, needed: list[FieldName]) -> str:
    fields_list = ", ".join(needed)
    return (
        f"Company name: {request.name}\n"
        f"Domain (if known): {request.domain or 'unknown'}\n\n"
        f"Fill in ONLY these fields as a JSON object: {fields_list}.\n"
        f"Use null for any field you do not know. Do not include any "
        f"other keys. Do not include explanations or markdown.\n"
        f'Example shape: {{"industry": "Technology", '
        f'"country": null, "is_public": true, '
        f'"short_description": "A short summary."}}'
    )


_SYSTEM_PROMPT = (
    "You are a company enrichment assistant. Given a company name and "
    "optional domain, fill in only the requested fields. Use null for "
    "anything you do not know. Respond with a single JSON object and "
    "nothing else."
)


def enrich(request: EnrichmentRequest, client: LLMClient) -> LLMResult:
    """Run tier-2 LLM enrichment against request.fields_needed."""
    needed = sorted(request.fields_needed)
    if not needed:
        return LLMResult()

    prompt = _build_prompt(request, needed)
    start = time.perf_counter()
    response = client.complete(_SYSTEM_PROMPT, prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    parsed = _parse_llm_json(response.text)
    resolved: dict[FieldName, ResolvedField] = {}
    for f in needed:
        if f in parsed and parsed[f] is not None:
            resolved[f] = ResolvedField(
                name=f,
                value=parsed[f],
                tier=2,
                confidence=TIER2_CONFIDENCE,
            )

    cost = modeled_cost_usd(response.prompt_tokens, response.completion_tokens)
    return LLMResult(
        resolved=resolved,
        cost_usd=cost,
        measured_latency_ms=elapsed_ms,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )
