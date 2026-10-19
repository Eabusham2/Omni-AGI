"""Early, stderr-only fault diagnostics for the console worker; no model imports."""
import faulthandler
import sys


if sys.stderr is not None:
    try:
        # Python 3.11 emits the crashing thread's Python frames, not a core or
        # full native dump. The owning smoke process bounds what reaches CI.
        faulthandler.enable(file=sys.stderr, all_threads=False)
    except (AttributeError, OSError, RuntimeError, ValueError):
        # Diagnostics must not create a new startup failure on a closed or
        # unavailable stderr stream. Ordinary worker validation stays fatal.
        pass
