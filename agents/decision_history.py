"""
Pitwall's memory of its own recent decisions.

This module holds the last _MAX_HISTORY pit decisions made during the current run, in a
module-level deque. The orchestrator reads from this deque to inform each new
decision, and appends to it after producing one.
"""

from collections import deque
from dataclasses import dataclass

_MAX_HISTORY = 5

@dataclass(frozen=True)
class Decision:
    """Record of a past decision."""
    lap: int
    call: str

_history: deque[Decision] = deque(maxlen=_MAX_HISTORY)

def record(lap: int, call: str) -> None:
    """Record a new decision. Oldest entry is evicted automatically."""
    _history.append(Decision(lap=lap, call=call))

def recent() -> list[Decision]:
    """Return recent decisions, most recent first."""
    return list(reversed(_history))

def format_for_prompt() -> str:
    """Format the recent decisions as a block of text for the orchestrator prompt."""
    if not _history:
        return "RECENT DECISIONS: (none — this is the first cycle of the race)"

    lines = ["RECENT DECISIONS (most recent first):"]
    for memo in recent():
        lines.append(f"- L{memo.lap}: {memo.call}")
    return "\n".join(lines)


def clear() -> None:
    """Reset history. Used for development purposes only."""
    _history.clear()