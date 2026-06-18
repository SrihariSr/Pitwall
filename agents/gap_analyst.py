"""
The Gap Analyst subagent.

Reads the live gap picture through MCP and reports strategic threats and
opportunities relative to the focal driver. Like the Tyre Strategist, it
does NOT recommend pit stops — it only reports the strategic landscape.
"""
from mcp_server.live_state import get_active_state
from mcp_server.server import get_gaps_to_rivals, get_current_race_state
from llm.client import LLMClient
from agents.schemas import GapAssessment


SYSTEM_PROMPT = """
You are the Gap Analyst on an F1 team's pit wall.

Your only job: monitor time gaps to other drivers and surface strategic threats and opportunities.

You DO NOT care about:
- Tyre condition or stint age (the Tyre Strategist handles that)
- Weather conditions (the Weather Watcher handles that)
- Whether the driver should pit (the Orchestrator decides)

You DO care about:
- Undercut threats: rivals behind, close enough that if they pit first and we don't react, they emerge ahead
- Overcut opportunities: rivals ahead with the wrong strategic position, where staying out longer might leapfrog them
- Direct battles: rivals within 1.5s, in DRS/overtaking range
- The leader's gap, because it sets the context for everything else

How to think:
- The pit-stop time loss at most tracks is ~20-23 seconds. A rival behind by less than that window is in undercut range.
- At Monaco specifically, overtaking on track is nearly impossible — so a gap of 1-3s to a rival ahead is more strategic context than tactical opportunity.
- Undercut threats matter most when the rival has fresher tyres or younger stint, but you don't see that data — only the gap. Assume undercut viability from gap alone.
- Lapped traffic (gap_seconds=None) is not a strategic rival; ignore lapped drivers for relationship classification.

Categorisation:
- "high" threat/opportunity: within 5 seconds, gap closing or stable, no buffer
- "medium": within 10-15 seconds, situational
- "low": within pit-window but unlikely to act now
- "none": no rival in the relevant window

Biases:
- Don't overstate threats — most laps, the gap picture is stable
- Don't ignore lone leaders — "no threats, clear track" is a valuable signal
- Cite specific gaps in your reasoning, not generalities

Output: a GapAssessment. Pick 2-4 strategically relevant rivals for closest_rivals, not the whole field.
"""

async def assess_gaps(
    client: LLMClient,
    driver_code: str,
    year: int,
    event: str,
    session_type: str,
    model: str = "gemini-2.5-flash",
) -> GapAssessment:
    """
    Run the Gap Analyst on the focal driver.
    """
    state = get_active_state()
    if state is None:
        raise RuntimeError("No live RaceState active, cannot assess gaps.")

    current_lap = state.current_lap()
    if current_lap < 1:
        raise RuntimeError("Race has not started yet.")

    race = get_current_race_state()
    gaps = get_gaps_to_rivals(
        year=year, event=event,
        session_type=session_type,
        driver_code=driver_code,
        current_lap=current_lap,
    )

    user_prompt = _build_user_prompt(driver_code, current_lap, race, gaps)

    return await client.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=GapAssessment,
        temperature=0.2,
    )

def _build_user_prompt(driver_code, current_lap, race, gaps) -> str:
    """
    Format the variable per-call portion of the prompt.

    The rival list is shortened to the 8 most strategically relevant slots:
    everyone within ±25 seconds of the focal driver.
    """
    # Filter to rivals within pit-window in either direction.
    nearby = [
        r for r in gaps.rivals
        if r.gap_seconds is not None and abs(r.gap_seconds) <= 25.0
    ]
    # Sort by position so the prompt reads top-down.
    nearby.sort(key=lambda r: r.rival_position)

    rival_lines = "\n".join(
        f"  P{r.rival_position}  {r.rival_driver_code} ({r.rival_team})  "
        f"gap {r.gap_seconds:+.2f}s"
        for r in nearby
    )

    return f"""
    Driver {driver_code} at lap {current_lap}.
    Track status: {race.track_status}
    Focal driver position: P{gaps.focal_position}

    Gap to leader: {gaps.gap_to_leader_seconds:.2f}s
    Gap to car ahead: {gaps.gap_ahead_seconds if gaps.gap_ahead_seconds is not None else "—"}
    Gap to car behind: {gaps.gap_behind_seconds if gaps.gap_behind_seconds is not None else "—"}

    Nearby rivals (within ±25s, positive = ahead): {rival_lines}

    Assess undercut threats, overcut opportunities, and the strategic landscape.
    """