"""
Watch Monaco 2022 with Tyre Strategist AND Gap Analyst running each lap.

Two subagents running concurrently via asyncio.gather. This is the pattern
the orchestrator will use to consult all 7 specialists at once.
"""
import asyncio

from events.bus import EventBus
from feed_adapter.replay import replay_session
from race_state.state import RaceState
from mcp_server.live_state import set_active_state
from mcp_server.server import get_current_race_state
from llm.client import LLMClient
from agents.tyre_strategist import assess_tyres
from agents.gap_analyst import assess_gaps


async def subagent_loop(
    client: LLMClient,
    state: RaceState,
    driver_code: str = "LEC",
    year: int = 2022,
    event: str = "Monaco",
    session_type: str = "R",
):
    """Each new lap, run Tyre Strategist and Gap Analyst concurrently."""
    seen_laps = set()
    while True:
        await asyncio.sleep(1.0)
        try:
            race = get_current_race_state()
        except RuntimeError:
            continue

        if race.current_lap == 0 or race.current_lap in seen_laps:
            continue
        seen_laps.add(race.current_lap)

        # Two LLM calls in parallel. Note: the rate limiter inside LLMClient
        # serialises them anyway (5/min limit), but the asyncio.gather
        # structure is correct for when we move to paid tier or more subagents.
        try:
            tyre, gap = await asyncio.gather(
                assess_tyres(client, driver_code, year, event, session_type),
                assess_gaps(client, driver_code, year, event, session_type),
            )
        except Exception as e:
            print(f"[L{race.current_lap}] SUBAGENT ERROR: {e}")
            continue

        print(f"\n══ LAP {race.current_lap} ══════════════════════")
        print(f"  Track status: {race.track_status}  |  {driver_code} P{tyre and gap.focal_position}")

        if tyre.has_sufficient_data:
            print(f"TYRE   cliff L{tyre.cliff_lap}  conf {tyre.confidence:.2f}")
        else:
            print(f"TYRE   (insufficient data, conf {tyre.confidence:.2f})")
        print(f"{tyre.reasoning}")

        print(f"GAP undercut={gap.undercut_threat}  overcut={gap.overcut_opportunity}  conf {gap.confidence:.2f}")
        print(f"{gap.reasoning}")
        if gap.closest_rivals:
            print(f"rivals: " + ", ".join(
                f"{r.driver_code}({r.gap_seconds:+.1f}s,{r.relationship})"
                for r in gap.closest_rivals
            ))


async def main():
    bus = EventBus()
    state = RaceState()
    client = LLMClient()

    set_active_state(state)
    await state.start(bus)

    agent_task = asyncio.create_task(
        subagent_loop(client, state, driver_code="LEC")
    )

    await asyncio.sleep(0.1)

    await replay_session(
        bus,
        year=2022,
        event="Monaco",
        session_type="R",
        speed=3.0,            # very slow — two LLM calls per lap on free tier
        start_lap=15,
        end_lap=22,
    )

    await asyncio.sleep(3.0)
    agent_task.cancel()
    await state.stop()


if __name__ == "__main__":
    asyncio.run(main())