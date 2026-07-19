import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

DEFAULT_LOG_DIR = "itgrind_transformers/llm_logs"

_console = logging.getLogger("llm_pipeline")

def _setup_console(level : int = logging.INFO):
    if _console.handlers:
        return _console

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    _console.addHandler(handler)
    _console.setLevel(level)
    _console.propagate = False

    return _console

def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    return str(value)

class PipelineLogger:
    def __init__(self, log_dir : str = DEFAULT_LOG_DIR, console_level : int = logging.INFO, run_id : str | None = None):
        os.makedirs(log_dir, exist_ok=True)

        self.log_dir = log_dir
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.calls_path = os.path.join(log_dir, "pipeline.jsonl")
        self.failures_path = os.path.join(log_dir, "failures.jsonl")
        self.console = _setup_console(console_level)

        self._lock = threading.Lock()
        self._ctx = {}
        self._counters = {}

    @contextmanager
    def context(self, **fields):
        previous = dict(self._ctx)
        self._ctx.update(fields)
        try:
            yield self
        finally:
            self._ctx = previous

    def new_trace(self) -> str:
        return uuid.uuid4().hex[:8]

    def _write(self, path : str, record : dict):
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _base(self) -> dict:
        return {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            **{k: _jsonable(v) for k, v in self._ctx.items()},
        }

    def log_call(self, **fields):
        record = self._base()
        record.update({k: _jsonable(v) for k, v in fields.items()})
        self._write(self.calls_path, record)

        outcome = record.get("outcome")
        self._counters[outcome] = self._counters.get(outcome, 0) + 1

        return record

    def log_failure(self, raw_response : str | None, **fields):
        record = self._base()
        record.update({k: _jsonable(v) for k, v in fields.items()})
        record["raw_response"] = raw_response
        self._write(self.failures_path, record)

        return record

    def info(self, msg, *args):
        self.console.info(self._prefix() + msg, *args)

    def warning(self, msg, *args):
        self.console.warning(self._prefix() + msg, *args)

    def error(self, msg, *args):
        self.console.error(self._prefix() + msg, *args)

    def _prefix(self) -> str:
        trace = self._ctx.get("trace_id")
        step = self._ctx.get("step")

        if trace and step:
            return f"[{trace} {step}] "
        if trace:
            return f"[{trace}] "

        return ""

    @property
    def counters(self) -> dict:
        return dict(self._counters)

    def summary(self) -> dict:
        total = sum(self._counters.values())
        ok = self._counters.get("ok", 0)

        return {
            "run_id": self.run_id,
            "total_calls": total,
            "ok_calls": ok,
            "ok_rate": round(ok / total, 3) if total else 0.0,
            "by_outcome": dict(sorted(self._counters.items(), key=lambda kv: -kv[1])),
            "calls_log": self.calls_path,
            "failures_log": self.failures_path,
        }


_default_logger : PipelineLogger | None = None


def get_logger(**kw) -> PipelineLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = PipelineLogger(**kw)
    return _default_logger
