"""
Pitwall MCP Server — Phase 1, Session 2: first real F1 tool.

Exposes get_session_state, which returns a high-level snapshot of an
F1 session at a given lap. Designed for replay mode: we operate on
historical sessions and pretend we're at a chosen lap.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
import pandas as pd

from mcp_server.sessions import load_session
from mcp_server.schemas import SessionState

mcp = FastMCP("pitwall")

# Mapping from FastF1's track-status numeric codes to our human-readable labels.
_TRACK_STATUS_MAP = {
    "1": "green",
    "2": "yellow",
    "4": "sc",    # safety car
    "5": "red",
    "6": "vsc",   # virtual safety car deployed
    "7": "vsc",   # virtual safety car ending
}

@mcp.tool()
def get_session_state(
    year: int,
    event: str,
    session_type: str,
    current_lap: int,
) -> SessionState:
    """
    Return a high-level snapshot of an F1 session at a chosen lap.

    Use this when you need orientation: what session, what lap, who is
    leading, what the track status is. This is the first call most agents
    should make to know where they are in the race.

    Parameters:
        year: Championship year, e.g. 2022
        event: Event name like "Monaco" or "Hungarian Grand Prix"
        session_type: "R" for race, "Q" for qualifying, "FP1"/"FP2"/"FP3", "S" for sprint
        current_lap: Lap number to evaluate state at. Must be >= 1.
    """
    session = load_session(year, event, session_type)

    laps = session.laps
    total_laps = int(laps["LapNumber"].max())

    # Clamp to the actual race length to avoid silent confusion.
    if current_lap < 1 or current_lap > total_laps:
        raise ValueError(f"current_lap {current_lap} out of range for this session!")

    # Find the leader at current_lap. The leader is the driver with Position == 1 on that lap. 
    # If multiple rows match (rare case), we take the first.
    lap_rows = laps[laps["LapNumber"] == current_lap]
    leader_rows = lap_rows[lap_rows["Position"] == 1]

    if len(leader_rows) == 0:
        #fall back to whoever has the fastest lap so far if data is missing for this lap
        leader_code = "UNK"
        leader_time = None
    else:
        leader = leader_rows.iloc[0]
        leader_code = str(leader["Driver"])
        leader_time = (
            float(leader["LapTime"].total_seconds())
            if pd.notna(leader["LapTime"])
            else None
        )

    # Track status at this lap. FastF1 stores per-lap status in the laps DataFrame.
    raw_status = str(lap_rows["TrackStatus"].iloc[0]) if len(lap_rows) > 0 else "1"
    
    # TrackStatus can be a multi-flag string like "14" (yellow + SC); we take the
    # most-severe single character for simplicity.
    status_char = max(raw_status, key=lambda c: int(c)) if raw_status else "1"
    track_status = _TRACK_STATUS_MAP.get(status_char, "green")

    return SessionState(
        year=year,
        event=session.event["EventName"],
        session_type=session_type,
        total_laps=total_laps,
        current_lap=current_lap,
        leader_driver_code=leader_code,
        leader_lap_time_seconds=leader_time,
        track_status=track_status,
    )


if __name__ == "__main__":
    mcp.run()