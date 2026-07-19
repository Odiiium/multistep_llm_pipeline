from data_models import (
    Criticality,
    FallbackMode,
    FallbackStep,
    IntentLMAnswer,
    JudgedLMResult,
    StepPolicy,
)
from promts import (
    FALLBACK_EXTRACTION_PROMPT,
    FALLBACK_FINAL_ANSWER_PROMPT,
    FALLBACK_INTENT_PROMPT,
    FALLBACK_SELF_CHECK_PROMPT,
    FALLBACK_SENSE_EXTRACTION_PROMPT,
)

def ladder(first : float, second : float, third : float, fallback_prompt : str) -> list[FallbackStep]:
    return [
        FallbackStep(level=0, mode=FallbackMode.normal, temperature=first),
        FallbackStep(level=1, mode=FallbackMode.repair, temperature=second),
        FallbackStep(level=2, mode=FallbackMode.fallback_prompt, temperature=third, prompt=fallback_prompt),
    ]

STEP_POLICIES = {

    "extract_sense": StepPolicy(
        name="extract_sense",
        model=None,
        min_chars=10,
        soft_max_chars=500,
        hard_max_chars=2000,
        criticality=Criticality.optional,
        retry_budget=3,
        temperature_ladder=[0.95, 0.3, 0.0],
        fallback_chain=ladder(0.95, 0.3, 0.0, FALLBACK_SENSE_EXTRACTION_PROMPT),
    ),

    "intent": StepPolicy(
        name="intent",
        model=IntentLMAnswer,
        min_chars=2,
        soft_max_chars=80,
        hard_max_chars=250,
        criticality=Criticality.degradable,
        retry_budget=3,
        temperature_ladder=[0.01, 0.0, 0.0],
        fallback_chain=ladder(0.01, 0.0, 0.0, FALLBACK_INTENT_PROMPT),
    ),

    "extract_fields": StepPolicy(
        name="extract_fields",
        model=None,
        min_chars=2,
        soft_max_chars=500,
        hard_max_chars=2000,
        criticality=Criticality.degradable,
        retry_budget=3,
        temperature_ladder=[0.01, 0.0, 0.0],
        fallback_chain=ladder(0.01, 0.0, 0.0, FALLBACK_EXTRACTION_PROMPT),
    ),

    "final_answer": StepPolicy(
        name="final_answer",
        model=None,
        min_chars=20,
        soft_max_chars=900,
        hard_max_chars=3000,
        criticality=Criticality.required,
        retry_budget=3,
        temperature_ladder=[0.95, 0.4, 0.1],
        fallback_chain=ladder(0.95, 0.4, 0.1, FALLBACK_FINAL_ANSWER_PROMPT),
    ),

    "judge": StepPolicy(
        name="judge",
        model=JudgedLMResult,
        min_chars=2,
        soft_max_chars=300,
        hard_max_chars=1000,
        criticality=Criticality.optional,
        retry_budget=3,
        temperature_ladder=[0.01, 0.0, 0.0],
        fallback_chain=ladder(0.01, 0.0, 0.0, FALLBACK_SELF_CHECK_PROMPT),
    ),
}

def for_extraction(model_type) -> StepPolicy:
    base = STEP_POLICIES["extract_fields"]

    return StepPolicy(
        name=base.name,
        model=model_type,
        min_chars=base.min_chars,
        soft_max_chars=base.soft_max_chars,
        hard_max_chars=base.hard_max_chars,
        criticality=base.criticality,
        retry_budget=base.retry_budget,
        temperature_ladder=list(base.temperature_ladder),
        fallback_chain=list(base.fallback_chain),
    )
