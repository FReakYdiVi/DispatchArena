"""Compatibility wrapper for legacy import path.

This module is executed by the Space container via:

    python -m dispatch_arena.server.app

So it must both re-export the real server symbols and forward module execution
to the actual ``main()`` entrypoint.
"""

from server.app import *  # noqa: F401,F403
from server.app import main as _real_main


if __name__ == "__main__":
    _real_main()
