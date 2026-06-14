"""
Replay a race, maintain live state, and demonstrate querying it through
the live MCP tool functions (called directly, no protocol round-trip needed).
"""
import asyncio

from events.bus import EventBus
from feed_adapter.replay import replay_session
from race_state.state import RaceState
from mcp_server.live_state import set_active_state
from mcp_server.server import (
    get_current_race_state,
    get_live_driver_status,
    get_recent_pit_activity,
)


async def lap_decision_loop(state: RaceState, interval_seconds: float = 2.0):
    """
    Stand-in for the future orchestrator: every N real seconds, query
    the live MCP tools and print what they return.
    """
    seen_laps = set()
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            race = get_current_race_state()
        except RuntimeError:
            continue  # state not ready yet

        if race.current_lap == 0 or race.current_lap in seen_laps:
            continue
        seen_laps.add(race.current_lap)

        print(f"\n┌── Decision cycle @ lap {race.current_lap} ──")
        print(f"│ status={race.track_status}  rain={race.is_raining}  track_temp={race.track_temp_celsius}")
        print(f"│ leader={race.leader_driver_code}  SC active={race.is_safety_car_active}")

        # Query a specific driver's live status.
        try:
            lec = get_live_driver_status("LEC")
            print(f"│ LEC: lap {lec.current_lap} P{lec.position}  last {lec.last_lap_time_seconds}s  pits={lec.pit_stop_count}")
        except ValueError:
            print(f"│ LEC: no data yet")

        # Recent pit activity.
        pits = get_recent_pit_activity(last_n_laps=3)
        print(f"│ pits in last 3 laps: {pits.pit_count}")
        for p in pits.pits[-3:]:  # show up to 3
            print(f"│   - {p.driver_code} on lap {p.in_lap} off {p.compound_from}")
        print(f"└─────────────────────────────")


async def main():
    bus = EventBus()
    state = RaceState()

    # Register state as the active source for live MCP tools.
    set_active_state(state)
    await state.start(bus)

    decision_task = asyncio.create_task(lap_decision_loop(state, interval_seconds=2.0))

    await asyncio.sleep(0.1)

    await replay_session(
        bus,
        year=2022,
        event="Monaco",
        session_type="R",
        speed=30.0,
        start_lap=15,
        end_lap=25,
    )

    await asyncio.sleep(0.5)
    decision_task.cancel()
    await state.stop()


if __name__ == "__main__":
    asyncio.run(main())