"""Judge provider profiles: protocol-specific options and known tariffs.

Unknown endpoints retain the portable OpenAI-compatible request shape and an
unknown tariff. Explicit cost caps reject uncached calls without a tariff.
"""
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class JudgeProfile:
    input_usd_per_million: str
    output_usd_per_million: str
    cost_basis: str
    json_mode: bool = False
    reasoning_effort: str | None = None
    short_max_tokens: int = 64

    def rates(self):
        million = Decimal('1000000')
        return (Decimal(self.input_usd_per_million) / million,
                Decimal(self.output_usd_per_million) / million, self.cost_basis)


PROFILES = {
    ('http://127.0.0.1:11434/v1/chat/completions', 'qwen3:1.7b'):
        JudgeProfile('0', '0', 'local_ollama_qwen3_1_7b', True),
    ('https://openrouter.ai/api/v1/chat/completions', 'openai/gpt-oss-120b:free'):
        JudgeProfile('0', '0', 'openrouter_gpt_oss_120b_free'),
    ('https://openrouter.ai/api/v1/chat/completions', 'google/gemini-2.5-pro'):
        JudgeProfile('1.25', '10', 'openrouter_gemini_2_5_pro'),
    ('https://api.groq.com/openai/v1/chat/completions', 'openai/gpt-oss-120b'):
        JudgeProfile('0.15', '0.60', 'groq_gpt_oss_120b', True, 'low', 512),
    ('https://api.groq.com/openai/v1/chat/completions', 'openai/gpt-oss-20b'):
        JudgeProfile('0.075', '0.30', 'groq_gpt_oss_20b', True, 'low', 512),
}


def profile_for(endpoint, model):
    return PROFILES.get((endpoint, model))


def judge_rates(endpoint, model):
    profile = profile_for(endpoint, model)
    return profile.rates() if profile else None


def request_options(endpoint, model, *, semantic=False):
    profile = profile_for(endpoint, model)
    options = {'max_tokens': 4096 if semantic else (profile.short_max_tokens if profile else 64)}
    if profile and profile.json_mode:
        options['response_format'] = {'type': 'json_object'}
    if profile and profile.reasoning_effort:
        options['reasoning_effort'] = profile.reasoning_effort
    return options
