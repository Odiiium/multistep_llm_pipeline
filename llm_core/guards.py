import time
from contextlib import contextmanager
from promts import build_repair_message
from .errors import ErrorClass, classify_exception, describe_exception
from .retry import RetryPolicy, compute_delay, retry_after_seconds
from .validation import validate_response
from data_models import (
    FailureKind,
    FallbackMode,
    LLMCallResult,
    Message,
    StepPolicy,
    SystemMessage,
    UserMessage,
)

REPAIRABLE_FAILURES = frozenset({
    FailureKind.empty,
    FailureKind.invalid_json,
    FailureKind.schema,
    FailureKind.too_long,
    FailureKind.truncated,
})

class GuardedLLMClient:
    def __init__(self, client, policy : StepPolicy, logger, retry_policy : RetryPolicy | None = None, sleep = time.sleep):
        self.client = client
        self.policy = policy
        self.logger = logger
        self.retry = retry_policy or RetryPolicy(max_attempts=policy.budget())
        self.sleep = sleep

    @property
    def model_name(self) -> str:
        return getattr(self.client, "model_name", None) or \
               getattr(getattr(self.client, "payload_settings", None), "model_name", "unknown")

    @contextmanager
    def overridden_prompt(self, prompt : str | None):
        if not prompt:
            yield
            return

        previous = getattr(self.client, "sys_message", None)
        self.client.sys_message = SystemMessage(content=prompt)

        try:
            yield
        finally:
            self.client.sys_message = previous

    def can_repair(self, last : LLMCallResult | None) -> bool:
        return last is not None and last.failure in REPAIRABLE_FAILURES

    def build_messages(self, messages : list[Message], mode : FallbackMode, last : LLMCallResult | None) -> list[Message]:
        if mode is not FallbackMode.repair:
            return messages

        repair = build_repair_message(last.failure.value, last.detail, last.content)

        return list(messages) + [UserMessage(content=repair)]

    def effective_mode(self, plan, last : LLMCallResult | None) -> FallbackMode:
        if plan.mode is FallbackMode.repair and not self.can_repair(last):
            return FallbackMode.normal

        return plan.mode

    def call(self, messages : list[Message]) -> LLMCallResult:
        policy = self.policy
        budget = policy.budget()
        last : LLMCallResult | None = None

        for attempt in range(1, budget + 1):
            plan = policy.plan_for(attempt)
            mode = self.effective_mode(plan, last)
            temperature = plan.temperature
            attempt_messages = self.build_messages(messages, mode, last)
            prompt_override = plan.prompt if mode is FallbackMode.fallback_prompt else None
            started = time.perf_counter()

            try:
                with self.overridden_prompt(prompt_override):
                    raw = self.client.generate_raw(attempt_messages, temperature=temperature)
            except Exception as exc:
                latency = int((time.perf_counter() - started) * 1000)
                error_class, kind = classify_exception(exc)
                detail = describe_exception(exc)

                self.logger.log_call(step=policy.name,
                                     model=self.model_name,
                                     attempt=attempt,
                                     fallback_level=plan.level,
                                     fallback_mode=mode.value,
                                     outcome=kind.value,
                                     error_class=error_class.value,
                                     error_type=type(exc).__name__,
                                     detail=detail,
                                     temperature=temperature,
                                     latency_ms=latency)

                if error_class is ErrorClass.fatal:
                    self.logger.error("%s: non-retryable error, aborting - %s", policy.name, detail)
                    return LLMCallResult.failed(FailureKind.fatal,
                                                attempts=attempt,
                                                fallback_level=plan.level,
                                                latency_ms=latency,
                                                raw_error=detail,
                                                detail=detail)

                last = LLMCallResult.failed(FailureKind.api_error,
                                            attempts=attempt,
                                            fallback_level=plan.level,
                                            latency_ms=latency,
                                            raw_error=detail,
                                            detail=detail)

                if attempt >= budget:
                    self.logger.error("%s: retries exhausted (%d) - %s", policy.name, budget, detail)
                    return last

                delay = compute_delay(attempt, self.retry, retry_after_seconds(exc))
                self.logger.warning("%s: %s, retry %d/%d in %.1fs", policy.name, detail, attempt + 1, budget, delay)
                self.sleep(delay)
                continue

            latency = int((time.perf_counter() - started) * 1000)
            checked = validate_response(raw, policy)
            failure = checked["failure"]

            self.logger.log_call(step=policy.name,
                                 model=self.model_name,
                                 attempt=attempt,
                                 fallback_level=plan.level,
                                 fallback_mode=mode.value,
                                 outcome="ok" if failure is None else failure.value,
                                 detail=checked["detail"],
                                 temperature=temperature,
                                 finish_reason=raw.finish_reason,
                                 response_len=len(raw.content) if raw.content else 0,
                                 sanitized=checked["sanitized"],
                                 soft_limit_hit=checked["soft_limit_hit"],
                                 latency_ms=latency,
                                 prompt_tokens=(raw.usage or {}).get("prompt_tokens"),
                                 completion_tokens=(raw.usage or {}).get("completion_tokens"))

            if failure is None:
                if checked["soft_limit_hit"]:
                    self.logger.warning("%s: soft length limit - %s", policy.name, checked["detail"])

                if plan.level > 0:
                    self.logger.info("%s: recovered at fallback level %d (%s)", policy.name, plan.level, mode.value)

                return LLMCallResult.success(checked["content"],
                                             parsed=checked["parsed"],
                                             attempts=attempt,
                                             fallback_level=plan.level,
                                             finish_reason=raw.finish_reason,
                                             usage=raw.usage,
                                             latency_ms=latency,
                                             sanitized=checked["sanitized"],
                                             soft_limit_hit=checked["soft_limit_hit"],
                                             detail=checked["detail"])

            self.logger.log_failure(raw.content,
                                    step=policy.name,
                                    model=self.model_name,
                                    attempt=attempt,
                                    fallback_level=plan.level,
                                    fallback_mode=mode.value,
                                    outcome=failure.value,
                                    detail=checked["detail"],
                                    finish_reason=raw.finish_reason)

            last = LLMCallResult.failed(failure,
                                        content=checked["content"] or raw.content,
                                        attempts=attempt,
                                        fallback_level=plan.level,
                                        finish_reason=raw.finish_reason,
                                        usage=raw.usage,
                                        latency_ms=latency,
                                        sanitized=checked["sanitized"],
                                        detail=checked["detail"])

            if attempt >= budget:
                self.logger.error("%s: validation failed after %d attempts - %s",
                                  policy.name, budget, checked["detail"])
                return last

            next_plan = policy.plan_for(attempt + 1)
            self.logger.warning("%s: %s (%s), escalating to fallback level %d",
                                policy.name, failure.value, checked["detail"], next_plan.level)

            self.sleep(self.retry.bad_output_delay)

        return last or LLMCallResult.failed(FailureKind.api_error, attempts=budget)
