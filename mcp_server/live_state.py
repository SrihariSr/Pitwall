"""
Module-level holder for the active RaceState.

The application sets the active state once at startup using set_active_state().
MCP tools that need to read live data import get_active_state() and ask
the state directly.
"""
from race_state.state import RaceState

_active_state: RaceState | None = None

def set_active_state(state: RaceState) -> None:
    """
    Register the RaceState to use for live MCP tools.
    """
    global _active_state
    _active_state = state


def get_active_state() -> RaceState | None:
    """
    Get the registered RaceState, or None if no live state is set.
    """
    return _active_state