import enum
from data_models import FailureKind

class ErrorClass(enum.Enum):
    transient   = "transient"
    bad_output  = "bad_output"
    fatal       = "fatal"

try:
    import openai
    
    FATAL_TYPES = (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.BadRequestError,
        openai.UnprocessableEntityError
    )
    TRANSIENT_TYPES = (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
    API_ERROR_BASE = (openai.APIError,)
except Exception:
    FATAL_TYPES = ()
    TRANSIENT_TYPES = ()
    API_ERROR_BASE = ()

def get_status_code(exception) -> int | None:
    return getattr(exception, "status_code", None) or getattr(getattr(exception, "response", None), "status_code", None)

def classify_exception(exception : BaseException) -> tuple[ErrorClass, FailureKind]:
    if FATAL_TYPES and isinstance(exception, FATAL_TYPES):
        return ErrorClass.fatal, FailureKind.fatal

    if TRANSIENT_TYPES and isinstance(exception, TRANSIENT_TYPES):
        return ErrorClass.transient, FailureKind.api_error

    code = get_status_code(exception)
    if code is not None:
        if code in (408, 409, 429) or code >= 500:
            return ErrorClass.transient, FailureKind.api_error
        if 400 <= code < 500:
            return ErrorClass.fatal, FailureKind.fatal

    if API_ERROR_BASE and isinstance(exception, API_ERROR_BASE):
        return ErrorClass.transient, FailureKind.api_error

    if isinstance(exception, (TimeoutError, ConnectionError)):
        return ErrorClass.transient, FailureKind.api_error

    return ErrorClass.fatal, FailureKind.fatal


def describe_exception(exc : BaseException) -> str:
    code = get_status_code(exc)
    prefix = f"HTTP {code} " if code else ""
    text = str(exc).strip().replace("\n", " ")
    return f"{prefix}{type(exc).__name__}: {text[:300]}"
