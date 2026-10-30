"""Per-run request logging, isolated between concurrent Streamlit sessions."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from inspect import signature
from time import perf_counter
from uuid import uuid4

_scope = ContextVar("call_scope", default={})
_collector = ContextVar("call_collector", default=None)


@contextmanager
def collect_calls(on_call=None):
    calls = []
    token = _collector.set((calls, on_call))
    try:
        yield calls
    finally:
        _collector.reset(token)


def current_calls():
    collector = _collector.get()
    return collector[0] if collector else []


def _persist(record):
    collector = _collector.get()
    if collector and collector[1]:
        try:
            collector[1](record)
            record.pop("persistence_error", None)
        except Exception as exc:
            # Preserve the response and keep the in-memory log downloadable.
            record["persistence_error"] = f"{type(exc).__name__}: {exc}"


def audit_call(method):
    sig = signature(method)

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        args_bound = sig.bind(self, *args, **kwargs)
        args_bound.apply_defaults()
        request = args_bound.arguments
        self.reset_usage()
        started = perf_counter()
        record = {"call_id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
                  **_scope.get(), "provider": self.provider, "model": request["model"],
                  "system_prompt": request["system_prompt"], "prompt": request["user_message"],
                  "response": "", "status": "success", "error": None}
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            record["request_id"] = getattr(exc, "request_id", None)
            raise
        finally:
            record.update(self.get_last_response())
            record["usage"] = self.get_last_usage()
            record["duration_ms"] = round((perf_counter() - started) * 1000)
            self.last_call = record
            collector = _collector.get()
            if collector is not None:
                collector[0].append(record)
                _persist(record)
    return wrapped


def invoke(provider, method, agent, phase, cycle=None, **kwargs):
    token = _scope.set({"agent": agent, "phase": phase, "cycle": cycle})
    try:
        return getattr(provider, method)(**kwargs)
    finally:
        _scope.reset(token)


def mark_validation_error(provider, error):
    record = provider.last_call
    if record:
        record.update(status="error", error=f"Response validation: {error}")
        _persist(record)


def call_errors(calls):
    return [{key: call.get(key) for key in
             ("call_id", "agent", "phase", "cycle", "provider", "model", "error", "stop_reason")}
            for call in calls if call.get("status") == "error"]
