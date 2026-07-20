from __future__ import annotations

# Sourced from https://platform.claude.com/docs/en/about-claude/pricing
# — verify before relying on this for real budgeting, as Anthropic updates
# pricing periodically (e.g., Sonnet has a scheduled rate change on
# 2026-09-01). The Groq call used to exercise this code path in
# development/testing is free; this constant models what the equivalent
# enrichment would cost against production-grade Claude Haiku.
#
# NOTE: modeled_cost_usd for this tier is a conservative upper-bound
# estimate, not an exact prediction. Even at low reasoning effort,
# Groq's openai/gpt-oss-20b incurs some hidden chain-of-thought
# token overhead that a non-thinking Claude Haiku call wouldn't.
# Actual Claude Haiku 4.5 costs for the same prompt would likely be
# slightly lower than the modeled cost reported here.
CLAUDE_HAIKU_4_5_INPUT_PER_MTOK_USD: float = 1.00
CLAUDE_HAIKU_4_5_OUTPUT_PER_MTOK_USD: float = 5.00


def modeled_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Compute USD cost at Claude Haiku 4.5 rates for given token usage."""
    return (prompt_tokens / 1_000_000.0) * CLAUDE_HAIKU_4_5_INPUT_PER_MTOK_USD + (
        completion_tokens / 1_000_000.0
    ) * CLAUDE_HAIKU_4_5_OUTPUT_PER_MTOK_USD
