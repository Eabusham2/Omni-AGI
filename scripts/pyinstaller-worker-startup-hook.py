"""Opt-in, bounded startup labels; never log input, arguments or environment."""
import os
import sys


if os.environ.get("OMNI_WORKER_STARTUP_DIAGNOSTICS") == "1":
    import atexit
    import threading

    _remaining = 63
    _milestone_lock = threading.Lock()

    def _milestone(label, final=False):
        global _remaining
        with _milestone_lock:
            if _remaining <= 0 and not final:
                return
            if not final:
                _remaining -= 1
            try:
                sys.stderr.write("omni-startup: " + label[:120] + "\n")
                sys.stderr.flush()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    def _import_milestone(event, values):
        if event == "import" and values and values[0] in {
            "torch", "worker_transport", "worker_lifecycle", "omni_core",
        }:
            _milestone("import-" + values[0])

    def _startup_trace(frame, event, value):
        filename = frame.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
        name = frame.f_code.co_name
        if filename in {"worker.py", "windows-stdio-probe.py"} and name in {"<module>", "main", "health"}:
            label = "entry" if name == "<module>" else name
        elif filename == "worker_transport.py" and name in {"serve_worker_stdio", "read_requests", "readline"}:
            label = name
        else:
            return None
        if event in {"call", "return"}:
            _milestone(label + "-" + event)
            if label == "readline" and event == "return" and value == b"":
                _milestone("stdio-eof")
        elif event == "exception":
            # Type names alone classify SystemExit versus Python failure;
            # exception values, frames, locals and RPC payloads stay private.
            _milestone(label + "-exception-" + value[0].__name__)
        return _startup_trace

    _milestone("hook-enter")
    sys.addaudithook(_import_milestone)
    sys.settrace(_startup_trace)
    threading.settrace(_startup_trace)
    atexit.register(_milestone, "python-exit", True)
