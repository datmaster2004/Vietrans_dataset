"""Backward-compatible entry point for the generic VieTrans server runner."""

from run_component_server import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
