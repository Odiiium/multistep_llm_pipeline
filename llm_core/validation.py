import copy
import json
import re
from data_models import FailureKind, RawResponse, StepPolicy

FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)
PREAMBLE_RE = re.compile(r"^[^{\[]{0,80}?(?=[{\[])", re.DOTALL)
SENTENCE_END_RE = re.compile(r"[.!?…](?:\s|$)")

def sanitize_json_text(raw : str) -> tuple[str, bool]:
    if raw is None:
        return "", False

    original = raw
    text = raw.strip()

    fence = FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()

    if text and text[0] not in "{[":
        cut = PREAMBLE_RE.match(text)
        if cut and cut.end() > 0:
            text = text[cut.end():].strip()

    if text and text[0] in "{[":
        closing = "}" if text[0] == "{" else "]"
        last = text.rfind(closing)
        if last != -1 and last < len(text) - 1:
            text = text[:last + 1]

    return text, text != original.strip()


def truncate_at_sentence(text : str, limit : int) -> str:
    if text is None or len(text) <= limit:
        return text

    window = text[:limit]

    ends = [m.end() for m in SENTENCE_END_RE.finditer(window)]
    
    if ends and ends[-1] >= limit * 0.5:
        return window[:ends[-1]].strip()

    space = window.rfind(" ")
    if space > 0:
        return window[:space].strip() + "…"

    return window.strip() + "…"

def _string_lengths(value, path="") -> list[tuple[str, int]]:
    out = []

    if isinstance(value, str):
        out.append((path or "<root>", len(value)))
    elif isinstance(value, dict):
        for k, v in value.items():
            out.extend(_string_lengths(v, f"{path}.{k}" if path else k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.extend(_string_lengths(v, f"{path}[{i}]"))

    return out

def validate_response(raw : RawResponse, policy : StepPolicy) -> dict:
    results = {
        "failure": None,
        "detail": None,
        "parsed": None,
        "content": raw.content,
        "sanitized": False,
        "soft_limit_hit": False,
    }

    # empty check
    if raw.refusal:
        results["failure"] = FailureKind.empty
        results["detail"] = f"model refusal: {str(raw.refusal)[:200]}"
        return results

    if raw.content is None or not raw.content.strip():
        results["failure"] = FailureKind.empty
        results["detail"] = "empty or whitespace-only content"
        return results

    content = raw.content.strip()

    if len(content) < max(1, policy.min_chars):
        results["failure"] = FailureKind.empty
        results["detail"] = f"content shorter than min_chars: {len(content)} < {policy.min_chars}"
        return results

    # max_tokens check
    if raw.finish_reason == "length":
        results["failure"] = FailureKind.truncated
        results["detail"] = "finish_reason == 'length': answer truncated by max_tokens"
        return results

    # Pydantic
    if policy.model is not None:
        cleaned, was_sanitized = sanitize_json_text(content)
        results["sanitized"] = was_sanitized
        results["content"] = cleaned

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as ex:
            results["failure"] = FailureKind.invalid_json
            results["detail"] = f"JSONDecodeError: {ex}"
            return results

        if not isinstance(data, dict):
            results["failure"] = FailureKind.invalid_json
            results["detail"] = f"Expected JSON, received : {type(data).__name__}"
            return results

        try:
            parsed = policy.model(**data)
        except Exception as ex:
            results["failure"] = FailureKind.schema
            results["detail"] = f"{type(ex).__name__}: {str(ex)[:400]}"
            return results

        results["parsed"] = parsed
        measured = _string_lengths(parsed.model_dump(mode="json"))
    else:
        measured = [("<text>", len(content))]

    # length check
    if policy.hard_max_chars is not None:
        over = [(f, n) for f, n in measured if n > policy.hard_max_chars]
        if over:
            field, length = max(over, key=lambda x: x[1])
            results["failure"] = FailureKind.too_long
            results["detail"] = f"field '{field}': {length} has symbols, than greater than hard_max {policy.hard_max_chars}"
            return results

    if policy.soft_max_chars is not None:
        soft = [(f, n) for f, n in measured if n > policy.soft_max_chars]
        if soft:
            field, length = max(soft, key=lambda x: x[1])
            results["soft_limit_hit"] = True
            results["detail"] = f"field '{field}': {length} as symbols, than greater than soft_max {policy.soft_max_chars}"

    return results

STRIP_KEYS = ("default", "$comment")

def to_strict_schema(schema : dict) -> dict:
    out = copy.deepcopy(schema)
    strictify(out)
    return out

def strictify(node):
    if isinstance(node, dict):
        for key in STRIP_KEYS:
            node.pop(key, None)

        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties")
            if isinstance(props, dict):
                node["required"] = list(props.keys())
                node["additionalProperties"] = False

        for value in node.values():
            strictify(value)

    elif isinstance(node, list):
        for item in node:
            strictify(item)