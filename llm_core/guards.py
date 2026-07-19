import time
from data_models import FailureKind, LLMCallResult, Message, StepPolicy
from .errors import ErrorClass, classify_exception, describe_exception
from .retry import RetryPolicy, compute_delay, retry_after_seconds
from .validation import validate_response

class GuardedLLMClient:
    def __init__(self, client, policy : StepPolicy, logger, retry_policy : RetryPolicy | None = None, sleep = time.sleep):
        self.client = client
        self.policy = policy
        self.logger = logger
        self.retry = retry_policy or RetryPolicy(max_attempts=policy.retry_budget)
        self.sleep = sleep

    @property
    def model_name(self) -> str:
        return getattr(self.client, "model_name", None) or \
               getattr(getattr(self.client, "payload_settings", None), "model_name", "unknown")

    def call(self, messages : list[Message]) -> LLMCallResult:
        policy = self.policy
        budget = max(1, policy.retry_budget)
        last : LLMCallResult | None = None

        for attempt in range(1, budget + 1):
            temperature = policy.temperature_for(attempt)
            started = time.perf_counter()

            try:
                raw = self.client.generate_raw(messages, temperature=temperature)
            except BaseException as exc:
                latency = int((time.perf_counter() - started) * 1000)
                error_class, kind = classify_exception(exc)
                detail = describe_exception(exc)

                self.logger.log_call(step=policy.name,
                                     model=self.model_name,
                                     attempt=attempt,
                                     outcome=kind.value,
                                     error_class=error_class.value,
                                     error_type=type(exc).__name__,
                                     detail=detail,
                                     temperature=temperature,
                                     latency_ms=latency)

                if error_class is ErrorClass.fatal:
                    self.logger.error("%s: non-retryable error, refusal - %s", policy.name, detail)
                    return LLMCallResult.failed(FailureKind.fatal,
                                                attempts=attempt,
                                                latency_ms=latency,
                                                raw_error=detail,
                                                detail=detail)

                last = LLMCallResult.failed(FailureKind.api_error,
                                            attempts=attempt,
                                            latency_ms=latency,
                                            raw_error=detail,
                                            detail=detail)

                if attempt >= budget:
                    self.logger.error("%s: retries exhausted (%d) - %s", policy.name, budget, detail)
                    return last

                delay = compute_delay(attempt, self.retry, retry_after_seconds(exc))
                self.logger.warning("%s: %s, retry %d/%d in %.1fс",
                                    policy.name, detail, attempt + 1, budget, delay)
                self.sleep(delay)
                continue

            latency = int((time.perf_counter() - started) * 1000)
            checked = validate_response(raw, policy)
            failure = checked["failure"]

            self.logger.log_call(step=policy.name,
                                 model=self.model_name,
                                 attempt=attempt,
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

                return LLMCallResult.success(checked["content"],
                                             parsed=checked["parsed"],
                                             attempts=attempt,
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
                                    outcome=failure.value,
                                    detail=checked["detail"],
                                    finish_reason=raw.finish_reason)

            last = LLMCallResult.failed(failure,
                                        content=raw.content,
                                        attempts=attempt,
                                        finish_reason=raw.finish_reason,
                                        usage=raw.usage,
                                        latency_ms=latency,
                                        sanitized=checked["sanitized"],
                                        detail=checked["detail"])

            if attempt >= budget:
                self.logger.error("%s: answer hasn't been validated after %d retries - %s",
                                  policy.name, budget, checked["detail"])
                return last

            self.logger.warning("%s: %s (%s), retry %d/%d",
                                policy.name, failure.value, checked["detail"], attempt + 1, budget)

            self.sleep(self.retry.bad_output_delay)

        return last or LLMCallResult.failed(FailureKind.api_error, attempts=budget)