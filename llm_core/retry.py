import email.utils
import random
import time
from dataclasses import dataclass

@dataclass
class RetryPolicy:
    max_attempts : int = 3
    base_delay   : float = 1.0
    cap          : float = 30.0
    jitter       : float = 0.5
    bad_output_delay : float = 0.2

def compute_delay(attempt : int, policy : RetryPolicy, retry_after : float | None = None) -> float:
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, policy.cap)

    raw = policy.base_delay * (2 ** (attempt - 1))
    factor = random.uniform(1.0 - policy.jitter, 1.0 + policy.jitter)

    return max(0.0, min(raw, policy.cap) * factor)

def retry_after_seconds(exc : BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if not headers:
        return None

    value = headers.get("retry-after") or headers.get("Retry-After")

    if not value:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except Exception:
        return None
