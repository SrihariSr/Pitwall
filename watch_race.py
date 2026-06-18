"""
Watch Monaco 2022 with the full Orchestrator + subagents pipeline.

The orchestrator wakes every 3 laps (or on trigger events) and produces
a PitDecision by fusing Tyre Strategist, Gap Analyst, and Monte Carlo
outputs. Each decision is logged to decisions/decisions.jsonl.
"""
import asyncio

from events.bus import EventBus
from feed_adapter.replay import replay_session
from race_state.state import RaceState
from mcp_server.live_state import set_active_state
from mcp_server.server import get_current_race_state
from llm.client import LLMClient
from agents.orchestrator import decide


DECISION_CADENCE = 3   # consult every 3 laps unless a trigger fires


async def orchestrator_loop(
    client: LLMClient,
    state: RaceState,
    driver_code: str = "LEC",
    year: int = 2022,
    event: str = "Monaco",
    session_type: str = "R",
):
    seen_laps = set()
    last_decision_lap = -10
    last_track_status = None

    while True:
        await asyncio.sleep(1.0)
        try:
            race = get_current_race_state()
        except RuntimeError:
            continue

        if race.current_lap == 0 or race.current_lap in seen_laps:
            continue

        # Decide whether this is a decision lap.
        trigger = None
        if race.track_status != last_track_status and last_track_status is not None:
            trigger = "track_status"
        elif race.current_lap - last_decision_lap >= DECISION_CADENCE:
            trigger = "scheduled"

        last_track_status = race.track_status
        seen_laps.add(race.current_lap)

        if trigger is None:
            continue  # nothing fired, sleep through this lap

        try:
            decision = await decide(
                client, driver_code, year, event, session_type, trigger=trigger,
            )
        except Exception as e:
            print(f"[L{race.current_lap}] ORCHESTRATOR ERROR: {e}")
            continue

        last_decision_lap = race.current_lap

        print(f"\n┌── ORCHESTRATOR: Lap {race.current_lap} ({trigger}) ──")
        print(f"│ CALL: {decision.call}   confidence {decision.confidence:.2f}")
        print(f"│ Primary reason: {decision.primary_reason}")
        if decision.supporting_factors:
            print(f"│ Supporting:")
            for f in decision.supporting_factors:
                print(f"│   - {f}")
        if decision.risks:
            print(f"│ Risks:")
            for r in decision.risks:
                print(f"│   - {r}")
        print(f"└──────────────────────────")


async def main():
    bus = EventBus()
    state = RaceState()
    client = LLMClient()

    set_active_state(state)
    await state.start(bus)

    agent_task = asyncio.create_task(
        orchestrator_loop(client, state, driver_code="LEC")
    )

    await asyncio.sleep(0.1)

    await replay_session(
        bus,
        year=2022,
        event="Monaco",
        session_type="R",
        speed=2.0,
        start_lap=15,
        end_lap=25,
    )

    await asyncio.sleep(3.0)
    agent_task.cancel()
    await state.stop()


if __name__ == "__main__":
    asyncio.run(main())