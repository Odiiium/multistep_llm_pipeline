import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import openai

from data_models import (
    Criticality, FailureKind, IntentLMAnswer, JudgedLMResult, RawResponse, StepPolicy,
)
from llm_core.errors import ErrorClass, classify_exception
from llm_core.guards import GuardedLLMClient
from llm_core.retry import RetryPolicy, compute_delay, retry_after_seconds
from llm_core.validation import (
    sanitize_json_text, to_strict_schema, truncate_at_sentence, validate_response,
)
from tests.fake_client import FakeLLMClient, Raise, RecordingLogger, Return


PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    mark = "  ok  " if condition else " FAIL "
    print(f"[{mark}] {name}" + (f"  -> {detail}" if detail and not condition else ""))


def _http_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://api.example/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("boom", response=response, body=None)


def _guarded(script, policy, budget=None, retry=None):
    if budget is not None:
        policy.retry_budget = budget
    client = FakeLLMClient(script)
    logger = RecordingLogger()
    guarded = GuardedLLMClient(client, policy, logger,
                               retry_policy=retry or RetryPolicy(base_delay=0.0, jitter=0.0),
                               sleep=lambda _: None)          # тесты не спят по-настоящему
    return client, logger, guarded


def text_policy(**kw):
    base = dict(name="t", model=None, min_chars=1, soft_max_chars=None,
                hard_max_chars=None, retry_budget=3, temperature_ladder=[])
    base.update(kw)
    return StepPolicy(**base)


# ---------------------------------------------------------------- errors

def test_error_classification():
    print("\n--- классификация ошибок ---")

    fatal = [
        ("401 AuthenticationError", _http_error(openai.AuthenticationError, 401)),
        ("403 PermissionDenied",    _http_error(openai.PermissionDeniedError, 403)),
        ("404 NotFound",            _http_error(openai.NotFoundError, 404)),
        ("400 BadRequest",          _http_error(openai.BadRequestError, 400)),
    ]
    for label, exc in fatal:
        cls, kind = classify_exception(exc)
        check(f"{label} -> fatal", cls is ErrorClass.fatal and kind is FailureKind.fatal, f"{cls}/{kind}")

    transient = [
        ("429 RateLimit",       _http_error(openai.RateLimitError, 429)),
        ("500 InternalServer",  _http_error(openai.InternalServerError, 500)),
        ("APITimeout",          openai.APITimeoutError(request=httpx.Request("POST", "https://x"))),
    ]
    for label, exc in transient:
        cls, kind = classify_exception(exc)
        check(f"{label} -> transient", cls is ErrorClass.transient and kind is FailureKind.api_error, f"{cls}/{kind}")

    # Баг в нашем коде не должен ретраиться три раза подряд.
    cls, _ = classify_exception(TypeError("bug in our code"))
    check("TypeError -> fatal (не ретраим свой баг)", cls is ErrorClass.fatal)


def test_backoff():
    print("\n--- backoff ---")

    policy = RetryPolicy(base_delay=1.0, cap=30.0, jitter=0.0)
    delays = [compute_delay(a, policy) for a in (1, 2, 3, 4, 5, 6, 7)]
    check("экспоненциальный рост", delays[:4] == [1.0, 2.0, 4.0, 8.0], str(delays))
    check("потолок соблюдается", all(d <= 30.0 for d in delays), str(delays))

    jittered = RetryPolicy(base_delay=1.0, jitter=0.5)
    samples = {round(compute_delay(3, jittered), 6) for _ in range(50)}
    check("jitter разводит попытки", len(samples) > 20, f"уникальных значений: {len(samples)}")

    exc = _http_error(openai.RateLimitError, 429, headers={"retry-after": "7"})
    check("Retry-After читается", retry_after_seconds(exc) == 7.0, str(retry_after_seconds(exc)))
    check("Retry-After приоритетнее backoff",
          compute_delay(1, policy, retry_after=7.0) == 7.0)


# ------------------------------------------------------------ validation

def test_validation_order():
    print("\n--- порядок проверок ---")

    policy = text_policy(model=IntentLMAnswer)

    empty = validate_response(RawResponse(content="   "), policy)
    check("пустой ответ -> empty (а не 'успех')", empty["failure"] is FailureKind.empty)

    refusal = validate_response(RawResponse(content="", refusal="I cannot help"), policy)
    check("refusal -> empty", refusal["failure"] is FailureKind.empty)

    # Ключевой кейс: оборванный JSON с finish_reason=="length" должен
    # диагностироваться как truncated, иначе чинить будут промпт вместо max_tokens.
    truncated = validate_response(
        RawResponse(content='{"intent": "support", "confidence": 4, "reas',
                    finish_reason="length"), policy)
    check("обрыв по max_tokens -> truncated, НЕ invalid_json",
          truncated["failure"] is FailureKind.truncated, str(truncated["failure"]))

    broken = validate_response(RawResponse(content='{"intent": "support"', finish_reason="stop"), policy)
    check("битый JSON без finish_reason=length -> invalid_json",
          broken["failure"] is FailureKind.invalid_json, str(broken["failure"]))

    wrong = validate_response(
        RawResponse(content='{"intent": "banana", "confidence": 4, "reason": "ok reason here"}'), policy)
    check("значение вне enum -> schema", wrong["failure"] is FailureKind.schema, str(wrong["failure"]))

    good = validate_response(
        RawResponse(content='{"intent": "support", "confidence": 4, "reason": "user reports a bug"}'), policy)
    check("валидный ответ проходит", good["failure"] is None and good["parsed"] is not None)


def test_length_limits():
    print("\n--- ограничения длины ---")

    policy = text_policy(soft_max_chars=50, hard_max_chars=100)

    ok = validate_response(RawResponse(content="x" * 30), policy)
    check("в пределах soft -> ok", ok["failure"] is None and not ok["soft_limit_hit"])

    soft = validate_response(RawResponse(content="x" * 70), policy)
    check("между soft и hard -> принято, но помечено",
          soft["failure"] is None and soft["soft_limit_hit"], str(soft))

    hard = validate_response(RawResponse(content="x" * 200), policy)
    check("выше hard -> too_long", hard["failure"] is FailureKind.too_long)

    short = validate_response(RawResponse(content="ab"), text_policy(min_chars=10))
    check("короче min_chars -> empty", short["failure"] is FailureKind.empty)

    # Длина считается по полям разобранной модели, а не по длине JSON-строки.
    long_field = json.dumps({"intent": "support", "confidence": 4, "reason": "y" * 300})
    res = validate_response(RawResponse(content=long_field),
                            text_policy(model=IntentLMAnswer, hard_max_chars=250))
    check("длинное поле внутри JSON ловится", res["failure"] in (FailureKind.too_long, FailureKind.schema))


def test_sanitizer():
    print("\n--- санитайзер ---")

    fenced, changed = sanitize_json_text('```json\n{"a": 1}\n```')
    check("снимает ```json fence", fenced == '{"a": 1}' and changed, fenced)

    plain, changed2 = sanitize_json_text('{"a": 1}')
    check("чистый JSON не трогает", plain == '{"a": 1}' and not changed2, plain)

    pre, _ = sanitize_json_text('Here is the JSON:\n{"a": 1}')
    check("срезает короткую преамбулу", pre == '{"a": 1}', pre)

    tail, _ = sanitize_json_text('{"a": 1}\n\nHope this helps!')
    check("срезает хвост после JSON", tail == '{"a": 1}', tail)


def test_truncate_at_sentence():
    print("\n--- обрезка по границе предложения ---")

    text = "First sentence here. Second sentence follows. Third one trails off"
    cut = truncate_at_sentence(text, 46)
    check("режет по границе предложения", cut.endswith("."), repr(cut))
    check("не режет посреди слова", not cut.endswith("Sec"), repr(cut))

    check("короткий текст не трогает", truncate_at_sentence("short", 100) == "short")

    no_period = truncate_at_sentence("aaaa bbbb cccc dddd eeee", 12)
    check("без точки - по границе слова", " " not in no_period[-1:] and "…" in no_period, repr(no_period))


def test_strict_schema():
    print("\n--- strict-совместимость схемы ---")

    from data_models import SupportFields

    raw = SupportFields.model_json_schema()
    strict = to_strict_schema(raw)

    check("исходная схема не strict-совместима", raw.get("required") in (None, []), str(raw.get("required")))
    check("все поля попали в required",
          set(strict["required"]) == set(strict["properties"].keys()),
          str(strict.get("required")))
    check("additionalProperties: false", strict["additionalProperties"] is False)
    check("default удалён", all("default" not in p for p in strict["properties"].values()))
    check("$defs сохранены", "$defs" in strict)
    check("исходная схема не мутирована", "additionalProperties" not in raw)


# -------------------------------------------------------------- guarded

def test_retry_on_transient():
    print("\n--- ретраи на временных ошибках ---")

    good = '{"intent": "support", "confidence": 4, "reason": "user reports a bug"}'
    client, logger, guarded = _guarded(
        [Raise(_http_error(openai.RateLimitError, 429)),
         Raise(openai.APITimeoutError(request=httpx.Request("POST", "https://x"))),
         Return(good)],
        text_policy(model=IntentLMAnswer), budget=3)

    result = guarded.call([])
    check("после 429 и таймаута доходит до успеха", result.ok, str(result.failure))
    check("сделано ровно 3 попытки", result.attempts == 3, str(result.attempts))
    check("каждая попытка залогирована", len(logger.calls) == 3, str(len(logger.calls)))
    check("outcome по попыткам верный",
          logger.outcomes() == ["api_error", "api_error", "ok"], str(logger.outcomes()))


def test_no_retry_on_fatal():
    print("\n--- fatal не ретраится ---")

    client, logger, guarded = _guarded(
        [Raise(_http_error(openai.AuthenticationError, 401))],
        text_policy(), budget=3)

    result = guarded.call([])
    check("401 -> провал сразу", not result.ok and result.failure is FailureKind.fatal, str(result.failure))
    check("сетевой вызов ровно один", client.call_count == 1, f"вызовов: {client.call_count}")
    check("ретраев не было", result.attempts == 1, str(result.attempts))


def test_retry_exhaustion():
    print("\n--- исчерпание бюджета ---")

    client, logger, guarded = _guarded(
        [Raise(_http_error(openai.InternalServerError, 500))],
        text_policy(), budget=3)

    result = guarded.call([])
    check("после бюджета -> api_error", not result.ok and result.failure is FailureKind.api_error)
    check("вызовов ровно по бюджету", client.call_count == 3, f"вызовов: {client.call_count}")


def test_bad_output_retry_and_ladder():
    print("\n--- ретрай плохого ответа + лестница температур ---")

    good = '{"intent": "sales", "confidence": 5, "reason": "asks about pricing"}'
    policy = text_policy(model=IntentLMAnswer, temperature_ladder=[0.9, 0.3, 0.0])
    client, logger, guarded = _guarded([Return(""), Return("not json at all"), Return(good)],
                                       policy, budget=3)

    result = guarded.call([])
    check("пустой -> битый -> валидный: доходит до успеха", result.ok, str(result.failure))
    check("температура понижается на повторах",
          client.temperatures == [0.9, 0.3, 0.0], str(client.temperatures))
    check("причины отказов зафиксированы",
          logger.outcomes() == ["empty", "invalid_json", "ok"], str(logger.outcomes()))
    check("сырые плохие ответы сохранены", len(logger.failures) == 2, str(len(logger.failures)))
    check("в failures лежит сам текст ответа",
          logger.failures[1]["raw_response"] == "not json at all", str(logger.failures[1]))


def test_truncated_is_reported_correctly():
    print("\n--- обрыв по max_tokens в guarded ---")

    client, logger, guarded = _guarded(
        [Return('{"intent": "support", "confidence": 4, "reason": "trunc', finish_reason="length")],
        text_policy(model=IntentLMAnswer), budget=1)

    result = guarded.call([])
    check("failure == truncated (не invalid_json)",
          result.failure is FailureKind.truncated, str(result.failure))
    check("finish_reason доехал до результата", result.finish_reason == "length", str(result.finish_reason))


def test_usage_and_metadata():
    print("\n--- метаданные вызова ---")

    client, logger, guarded = _guarded(
        [Return("a perfectly fine free-text answer", usage={"prompt_tokens": 120, "completion_tokens": 34})],
        text_policy(), budget=1)

    result = guarded.call([])
    check("usage сохранён", result.usage == {"prompt_tokens": 120, "completion_tokens": 34}, str(result.usage))
    check("latency измерен", result.latency_ms >= 0)
    entry = logger.calls[0]
    check("токены в логе", entry["prompt_tokens"] == 120 and entry["completion_tokens"] == 34, str(entry))
    check("длина ответа в логе", entry["response_len"] == len("a perfectly fine free-text answer"), str(entry))


def test_sanitized_flag_surfaces():
    print("\n--- флаг санитайзера виден в метриках ---")

    good = '```json\n{"passed": true, "score": 9, "issues": []}\n```'
    client, logger, guarded = _guarded([Return(good)], text_policy(model=JudgedLMResult), budget=1)

    result = guarded.call([])
    check("ответ в ```json``` принимается", result.ok, str(result.failure))
    check("но факт чистки помечен", result.sanitized is True)
    check("и виден в логе", logger.calls[0]["sanitized"] is True)


def main():
    test_error_classification()
    test_backoff()
    test_validation_order()
    test_length_limits()
    test_sanitizer()
    test_truncate_at_sentence()
    test_strict_schema()
    test_retry_on_transient()
    test_no_retry_on_fatal()
    test_retry_exhaustion()
    test_bad_output_retry_and_ladder()
    test_truncated_is_reported_correctly()
    test_usage_and_metadata()
    test_sanitized_flag_surfaces()

    print("\n" + "=" * 60)
    print(f"passed: {len(PASSED)}   failed: {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
    print("=" * 60)

    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
