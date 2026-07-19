import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import openai

from data_models import (
    FailureKind, FallbackMode, FallbackStep, IntentLMAnswer, JudgedLMResult,
    RawResponse, StepPolicy, SupportFields,
)
from llm_core.errors import ErrorClass, classify_exception
from llm_core.fallback import classify_intent_offline, condense_message, empty_fields_for
from llm_core.guards import GuardedLLMClient
from llm_core.policies import STEP_POLICIES, ladder
from llm_core.retry import RetryPolicy, compute_delay, retry_after_seconds
from llm_core.validation import (
    sanitize_json_text, to_strict_schema, truncate_at_sentence, validate_response,
)
from tests.fake_client import FakeLLMClient, Raise, RecordingLogger, Return

PASSED, FAILED = [], []

GOOD_INTENT = '{"intent": "support", "confidence": 4, "reason": "user reports a bug"}'
GOOD_JUDGE = '{"passed": true, "score": 9, "issues": []}'

def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    mark = "  ok  " if condition else " FAIL "
    print(f"[{mark}] {name}" + (f"  -> {detail}" if detail and not condition else ""))

def http_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://api.example/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("boom", response=response, body=None)

def text_policy(**kw):
    base = dict(name="t", model=None, min_chars=1, soft_max_chars=None,
                hard_max_chars=None, retry_budget=3, temperature_ladder=[])
    base.update(kw)
    return StepPolicy(**base)

def guarded(script, policy, retry=None):
    client = FakeLLMClient(script)
    logger = RecordingLogger()
    guard = GuardedLLMClient(client, policy, logger,
                             retry_policy=retry or RetryPolicy(base_delay=0.0, jitter=0.0),
                             sleep=lambda _: None)
    return client, logger, guard

def test_error_classification():
    print("\n--- error classification ---")

    for label, exc in [("401", http_error(openai.AuthenticationError, 401)),
                       ("403", http_error(openai.PermissionDeniedError, 403)),
                       ("404", http_error(openai.NotFoundError, 404)),
                       ("400", http_error(openai.BadRequestError, 400))]:
        cls, kind = classify_exception(exc)
        check(f"{label} -> fatal", cls is ErrorClass.fatal and kind is FailureKind.fatal)

    for label, exc in [("429", http_error(openai.RateLimitError, 429)),
                       ("500", http_error(openai.InternalServerError, 500)),
                       ("timeout", openai.APITimeoutError(request=httpx.Request("POST", "https://x")))]:
        cls, kind = classify_exception(exc)
        check(f"{label} -> transient", cls is ErrorClass.transient and kind is FailureKind.api_error)

    check("TypeError -> fatal", classify_exception(TypeError("bug"))[0] is ErrorClass.fatal)

def test_backoff():
    print("\n--- backoff ---")

    policy = RetryPolicy(base_delay=1.0, cap=30.0, jitter=0.0)
    delays = [compute_delay(a, policy) for a in range(1, 8)]
    check("exponential growth", delays[:4] == [1.0, 2.0, 4.0, 8.0], str(delays))
    check("cap respected", all(d <= 30.0 for d in delays))

    jittered = RetryPolicy(base_delay=1.0, jitter=0.5)
    samples = {round(compute_delay(3, jittered), 6) for _ in range(50)}
    check("jitter spreads attempts", len(samples) > 20, str(len(samples)))

    exc = http_error(openai.RateLimitError, 429, headers={"retry-after": "7"})
    check("Retry-After parsed", retry_after_seconds(exc) == 7.0)
    check("Retry-After wins over backoff", compute_delay(1, policy, retry_after=7.0) == 7.0)

def test_validation_order():
    print("\n--- validation order ---")

    policy = text_policy(model=IntentLMAnswer)

    check("empty -> empty", validate_response(RawResponse(content="  "), policy)["failure"] is FailureKind.empty)
    check("refusal -> empty",
          validate_response(RawResponse(content="", refusal="no"), policy)["failure"] is FailureKind.empty)

    cut = validate_response(RawResponse(content='{"intent": "support", "reas', finish_reason="length"), policy)
    check("max_tokens cut -> truncated, not invalid_json", cut["failure"] is FailureKind.truncated, str(cut["failure"]))

    broken = validate_response(RawResponse(content='{"intent": "support"', finish_reason="stop"), policy)
    check("broken json -> invalid_json", broken["failure"] is FailureKind.invalid_json)

    wrong = validate_response(RawResponse(content='{"intent": "banana", "confidence": 4, "reason": "ok reason"}'), policy)
    check("bad enum -> schema", wrong["failure"] is FailureKind.schema)

    good = validate_response(RawResponse(content=GOOD_INTENT), policy)
    check("valid passes", good["failure"] is None and good["parsed"] is not None)

def test_length_limits():
    print("\n--- length limits ---")

    policy = text_policy(soft_max_chars=50, hard_max_chars=100)

    check("under soft -> ok", validate_response(RawResponse(content="x" * 30), policy)["failure"] is None)

    soft = validate_response(RawResponse(content="x" * 70), policy)
    check("between soft and hard -> accepted and flagged",
          soft["failure"] is None and soft["soft_limit_hit"])

    check("over hard -> too_long",
          validate_response(RawResponse(content="x" * 200), policy)["failure"] is FailureKind.too_long)
    check("under min_chars -> empty",
          validate_response(RawResponse(content="ab"), text_policy(min_chars=10))["failure"] is FailureKind.empty)

def test_sanitizer_and_truncation():
    print("\n--- sanitizer and sentence truncation ---")

    fenced, changed = sanitize_json_text('```json\n{"a": 1}\n```')
    check("strips json fence", fenced == '{"a": 1}' and changed)
    check("leaves clean json", sanitize_json_text('{"a": 1}') == ('{"a": 1}', False))
    check("strips preamble", sanitize_json_text('Here is the JSON:\n{"a": 1}')[0] == '{"a": 1}')
    check("strips trailer", sanitize_json_text('{"a": 1}\n\nHope this helps!')[0] == '{"a": 1}')

    cut = truncate_at_sentence("First sentence here. Second sentence follows. Third trails", 46)
    check("cuts on sentence boundary", cut.endswith("."), repr(cut))
    check("short text untouched", truncate_at_sentence("short", 100) == "short")

def test_strict_schema():
    print("\n--- strict schema ---")

    raw = SupportFields.model_json_schema()
    strict = to_strict_schema(raw)

    check("source not strict-ready", raw.get("required") in (None, []))
    check("all props required", set(strict["required"]) == set(strict["properties"].keys()))
    check("additionalProperties false", strict["additionalProperties"] is False)
    check("defaults stripped", all("default" not in p for p in strict["properties"].values()))
    check("$defs kept", "$defs" in strict)
    check("source not mutated", "additionalProperties" not in raw)

def test_retry_transient_and_fatal():
    print("\n--- retry on transient, stop on fatal ---")

    client, logger, guard = guarded(
        [Raise(http_error(openai.RateLimitError, 429)),
         Raise(openai.APITimeoutError(request=httpx.Request("POST", "https://x"))),
         Return(GOOD_INTENT)],
        text_policy(model=IntentLMAnswer))

    result = guard.call([])
    check("recovers after 429 + timeout", result.ok)
    check("exactly 3 attempts", result.attempts == 3, str(result.attempts))
    check("outcomes logged", logger.outcomes() == ["api_error", "api_error", "ok"], str(logger.outcomes()))

    client, logger, guard = guarded([Raise(http_error(openai.AuthenticationError, 401))], text_policy())
    result = guard.call([])
    check("401 fails immediately", not result.ok and result.failure is FailureKind.fatal)
    check("no retries on fatal", client.call_count == 1, str(client.call_count))

    client, logger, guard = guarded([Raise(http_error(openai.InternalServerError, 500))], text_policy())
    result = guard.call([])
    check("budget exhausted -> api_error", result.failure is FailureKind.api_error)
    check("calls equal budget", client.call_count == 3, str(client.call_count))

def test_fallback_ladder_levels():
    print("\n--- fallback ladder: level escalation ---")

    policy = text_policy(model=IntentLMAnswer,
                         fallback_chain=ladder(0.9, 0.3, 0.0, "FALLBACK PROMPT BODY"))

    client, logger, guard = guarded([Return(""), Return("not json"), Return(GOOD_INTENT)], policy)
    result = guard.call([])

    check("recovers at level 2", result.ok and result.fallback_level == 2, str(result.fallback_level))
    check("levels escalate 0->1->2", logger.levels() == [0, 1, 2], str(logger.levels()))
    check("modes follow the ladder",
          logger.modes() == ["normal", "repair", "fallback_prompt"], str(logger.modes()))
    check("temperature descends", client.temperatures == [0.9, 0.3, 0.0], str(client.temperatures))

def test_fallback_repair_message():
    print("\n--- fallback L1: repair message ---")

    policy = text_policy(model=IntentLMAnswer,
                         fallback_chain=ladder(0.9, 0.3, 0.0, "FALLBACK PROMPT BODY"))

    client, logger, guard = guarded([Return("not json at all"), Return(GOOD_INTENT)], policy)
    result = guard.call([])

    check("recovers at level 1", result.ok and result.fallback_level == 1, str(result.fallback_level))
    check("first attempt has no repair message", len(client.calls[0]["messages"]) == 0)

    repair = client.message_texts(1)
    check("second attempt carries repair message", len(repair) == 1, str(len(repair)))
    check("repair names the failure", "invalid_json" in repair[0], repair[0][:80])
    check("repair echoes the bad answer", "not json at all" in repair[0])
    check("repair gives an instruction", "valid JSON" in repair[0] or "JSON object" in repair[0])

def test_fallback_prompt_override():
    print("\n--- fallback L2: prompt override ---")

    policy = text_policy(model=IntentLMAnswer,
                         fallback_chain=ladder(0.9, 0.3, 0.0, "FALLBACK PROMPT BODY"))

    client, logger, guard = guarded([Return(""), Return("still bad"), Return(GOOD_INTENT)], policy)
    client.sys_message = None
    result = guard.call([])

    check("fallback prompt applied on level 2",
          client.system_prompts[2] == "FALLBACK PROMPT BODY", str(client.system_prompts))
    check("earlier attempts keep original prompt",
          client.system_prompts[:2] == [None, None], str(client.system_prompts[:2]))
    check("prompt restored after the call", client.sys_message is None)

def test_repair_skipped_without_content():
    print("\n--- repair degrades to normal when there is nothing to repair ---")

    policy = text_policy(model=IntentLMAnswer,
                         fallback_chain=ladder(0.9, 0.3, 0.0, "FALLBACK PROMPT BODY"))

    client, logger, guard = guarded(
        [Raise(http_error(openai.InternalServerError, 500)), Return(GOOD_INTENT)], policy)
    result = guard.call([])

    check("recovers after api error", result.ok)
    check("mode falls back to normal", logger.modes()[1] == "normal", str(logger.modes()))
    check("no repair message appended", len(client.calls[1]["messages"]) == 0)

def test_deterministic_fallbacks():
    print("\n--- fallback L3: deterministic, no LLM ---")

    cases = [
        ("The application crashes every time I try to upload a file.", "support"),
        ("How much does your premium plan cost?", "sales"),
        ("I am very disappointed because my subscription was canceled.", "complaint"),
        ("I would like to suggest adding dark mode.", "feedback"),
        ("What is the capital of France?", "general_question"),
    ]
    for text, expected in cases:
        got = classify_intent_offline(text)
        check(f"offline intent: {expected}", got == expected, f"got {got}")

    check("empty text -> general_question", classify_intent_offline("") == "general_question")

    empty = empty_fields_for("support")
    check("empty fields model is valid", empty is not None and empty.intent.value == "support")
    check("empty fields are all None", empty.problem is None and empty.summary is None)

    long_text = "word " * 500
    condensed = condense_message(long_text, limit=100)
    check("condense respects limit", len(condensed) <= 101, str(len(condensed)))
    check("condense does not split words", not condensed.replace("…", "").endswith("wor"))

def test_policies_wired():
    print("\n--- step policies carry ladders ---")

    for name, policy in STEP_POLICIES.items():
        check(f"{name}: ladder has 3 levels", len(policy.fallback_chain) == 3, str(len(policy.fallback_chain)))
        check(f"{name}: level 2 has fallback prompt", bool(policy.fallback_chain[2].prompt))
        check(f"{name}: budget matches ladder", policy.budget() == 3, str(policy.budget()))

    check("final_answer is required", STEP_POLICIES["final_answer"].criticality.value == "required")
    check("judge is optional", STEP_POLICIES["judge"].criticality.value == "optional")

def test_truncated_and_metadata():
    print("\n--- truncated diagnosis and call metadata ---")

    client, logger, guard = guarded(
        [Return('{"intent": "support", "reason": "cut', finish_reason="length")],
        text_policy(model=IntentLMAnswer, retry_budget=1))

    result = guard.call([])
    check("failure is truncated", result.failure is FailureKind.truncated, str(result.failure))
    check("finish_reason propagated", result.finish_reason == "length")

    client, logger, guard = guarded(
        [Return("a fine free text answer", usage={"prompt_tokens": 120, "completion_tokens": 34})],
        text_policy(retry_budget=1))

    result = guard.call([])
    check("usage kept", result.usage == {"prompt_tokens": 120, "completion_tokens": 34})
    check("tokens logged", logger.calls[0]["prompt_tokens"] == 120)

    client, logger, guard = guarded([Return('```json\n' + GOOD_JUDGE + '\n```')],
                                    text_policy(model=JudgedLMResult, retry_budget=1))
    result = guard.call([])
    check("fenced json accepted", result.ok)
    check("sanitizer flag surfaced", result.sanitized is True and logger.calls[0]["sanitized"] is True)

def main():
    test_error_classification()
    test_backoff()
    test_validation_order()
    test_length_limits()
    test_sanitizer_and_truncation()
    test_strict_schema()
    test_retry_transient_and_fatal()
    test_fallback_ladder_levels()
    test_fallback_repair_message()
    test_fallback_prompt_override()
    test_repair_skipped_without_content()
    test_deterministic_fallbacks()
    test_policies_wired()
    test_truncated_and_metadata()

    print("\n" + "=" * 60)
    print(f"passed: {len(PASSED)}   failed: {len(FAILED)}")
    for name in FAILED:
        print(f"  FAIL: {name}")
    print("=" * 60)

    return 1 if FAILED else 0

if __name__ == "__main__":
    sys.exit(main())
