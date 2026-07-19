import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_models import RawResponse

class Scripted:
    pass

class Return(Scripted):
    def __init__(self, content, finish_reason="stop", usage=None, refusal=None):
        self.response = RawResponse(content=content,
                                    finish_reason=finish_reason,
                                    usage=usage or {"prompt_tokens": 10, "completion_tokens": 20},
                                    refusal=refusal)

class Raise(Scripted):
    def __init__(self, exc):
        self.exc = exc

class FakeLLMClient:
    def __init__(self, script : list, model_name : str = "fake-model"):
        self.script = list(script)
        self.model_name = model_name
        self.sys_message = None
        self.calls = []

    def set_response_schema_format(self, response_schema):
        self.response_schema = response_schema

    def generate_raw(self, messages, temperature=None):
        index = min(len(self.calls), len(self.script) - 1)
        step = self.script[index]

        self.calls.append({
            "messages": list(messages),
            "temperature": temperature,
            "sys_message": getattr(self.sys_message, "content", None),
        })

        if isinstance(step, Raise):
            raise step.exc

        return step.response

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def temperatures(self) -> list:
        return [c["temperature"] for c in self.calls]

    @property
    def system_prompts(self) -> list:
        return [c["sys_message"] for c in self.calls]

    def message_texts(self, index : int) -> list:
        return [getattr(m, "content", "") for m in self.calls[index]["messages"]]

class RecordingLogger:
    def __init__(self):
        self.calls = []
        self.failures = []
        self.messages = []
        self.run_id = "test"
        self.calls_path = "<memory>"

    def log_call(self, **fields):
        self.calls.append(fields)
        return fields

    def log_failure(self, raw_response=None, **fields):
        self.failures.append({**fields, "raw_response": raw_response})
        return fields

    def info(self, msg, *args):
        self.messages.append(("info", msg % args if args else msg))

    def warning(self, msg, *args):
        self.messages.append(("warning", msg % args if args else msg))

    def error(self, msg, *args):
        self.messages.append(("error", msg % args if args else msg))

    def context(self, **fields):
        from contextlib import nullcontext
        return nullcontext(self)

    def new_trace(self) -> str:
        return "trace"

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "total_calls": len(self.calls),
            "ok_calls": sum(1 for c in self.calls if c.get("outcome") == "ok"),
            "ok_rate": 0.0,
            "by_outcome": {},
            "calls_log": self.calls_path,
            "failures_log": "<memory>",
        }

    def outcomes(self) -> list:
        return [c.get("outcome") for c in self.calls]

    def levels(self) -> list:
        return [c.get("fallback_level") for c in self.calls]

    def modes(self) -> list:
        return [c.get("fallback_mode") for c in self.calls]

    def text(self) -> str:
        return "\n".join(m for _, m in self.messages)
