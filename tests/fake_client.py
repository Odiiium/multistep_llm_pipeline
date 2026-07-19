import sys
import os

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
        self.calls = []

    def generate_raw(self, messages, temperature=None):
        index = min(len(self.calls), len(self.script) - 1)
        step = self.script[index]

        self.calls.append({"messages": messages, "temperature": temperature})

        if isinstance(step, Raise):
            raise step.exc

        return step.response

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def temperatures(self) -> list:
        return [c["temperature"] for c in self.calls]


class RecordingLogger:
    def __init__(self):
        self.calls = []
        self.failures = []
        self.messages = []

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

    def outcomes(self) -> list:
        return [c.get("outcome") for c in self.calls]
