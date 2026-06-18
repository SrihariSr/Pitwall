"""
The Tyre Strategist subagent.

Subscribes to lap events on the bus. When called, queries lap history through
MCP, builds a structured prompt, and asks Gemini to assess the focal driver's
current stint. Returns a TyreAssessment.

The strategist is one of seven specialists. It does not make pit recommendations
(that's the orchestrator's job). It's only output is "where's the cliff and how
confident are you."
"""
from google.auth.aio.transport import sessions
from scipy.optimize import _constraints
from typing_extensions import runtime
from anyio import run
from mcp_server.live_state import get_active_state
from mcp_server.server import get_driver_lap_history, get_tyre_stints
from llm.client import LLMClient
from agents.schemas import TyreAssessment

SYSTEM_PROMPT = """
You are the Tyre Strategist on an F1 team's pit wall.

Your only job: track tyre performance and project when the current stint will hit its performance cliff.

You DO NOT care about:
- Gaps to rivals (the Gap Analyst handles that)
- Weather conditions (the Weather Watcher handles that)
- Pit windows or strategy (the Orchestrator decides)
- Rival pit timing (the Rival Modeler handles that)

You DO care about:
- Lap-time trajectory across the current stint
- Sector time degradation (sector 3 often goes off first as rear tyres die)
- Compound and tyre age: different compounds have different cliff profiles
- The track context (warm tracks accelerate degradation, wet conditions delay it)

How to think:
- Healthy stint: lap times stable or improving lap-on-lap as fuel burns off
- Approaching cliff: lap times tick up by 0.2-0.5s per lap, accelerating
- At cliff: lap times jump by 1s+ from one lap to the next
- The cliff is rarely a surprise — it telegraphs 3-5 laps out via rate-of-change

Biases you should hold:
- Conservative: when in doubt, project the cliff earlier rather than later
- Honest about uncertainty: short stints (under 5 laps) are hard to project — say so via confidence
- Specific: cite actual lap times or trends in your reasoning, not generic statements

Output: a TyreAssessment with cliff_lap, confidence (0-1), and one-sentence reasoning.
"""

def _build_user_prompt(driver_code, current_lap, history, stints) -> str:
    """Construct the variable, per-call portion of the prompt.

    Kept in a helper so the function reads top-down: assess_tyres handles the
    flow, this handles the formatting. The format is deliberately compact:
    every token costs!
    """
    current_stint = next((s for s in stints.stints if s.is_ongoing), None)
    if current_stint is None:
            return (
                f"Driver {driver_code} at lap {current_lap}.\n"
                f"No ongoing stint detected (driver may have just pitted or data is missing).\n"
                f"Return has_sufficient_data=false, confidence near 0, cliff_lap=0, "
                f"and reasoning explaining the absence of data."
            )
    
    # Pull the laps from the current stint only.
    stint_laps = [
        lap for lap in history.laps
        if lap.stint == current_stint.stint_number
    ]

    # Format recent lap times compactly.
    recent = stint_laps[-8:]  # last 8 laps of the stint
    lap_lines = "\n".join(
        f"L{lap.lap_number}: {lap.lap_time_seconds:.3f}s  "
        f"(S1 {lap.sector_1_seconds}, S2 {lap.sector_2_seconds}, S3 {lap.sector_3_seconds})"
        for lap in recent
    )

    return f"""Driver {driver_code} at lap {current_lap}.
    Current stint: stint {current_stint.stint_number}, compound {current_stint.compound}, started lap {current_stint.start_lap}.
    Tyre age: {current_lap - current_stint.start_lap + 1} laps.
    Stint best lap: {current_stint.best_lap_time_seconds:.3f}s
    Stint average:  {current_stint.average_lap_time_seconds:.3f}s

    Recent lap times (most recent last): {lap_lines}

    Assess the cliff lap, your confidence, and reasoning."""

async def assess_tyres(
    client: LLMClient,
    driver_code: str,
    year: int,
    event: str,
    session_type: str,
    model: str = "gemini-2.5-flash"
    ) -> TyreAssessment:
    """
    Run the Tyre Strategist on the focal driver's current stint.

    Pulls live state to find the current lap, then queries the MCP tools for
    detailed lap history and stint structure, then asks Gemini to assess.

    Parameters:
        client: shared LLM client
        driver_code: 3-letter driver code, e.g. "LEC"
        year, event, session_type: identify the session for MCP queries
        model: which LLM to use. Default Gemini 2.5 flash.
    """
    state = get_active_state()
    if state is None:
        raise RuntimeError("No live RaceState active, cannot access tyres :(")
    
    current_lap = state.current_lap()
    if current_lap < 1:
        raise RuntimeError("Race has not started yet.")
    
    # Pull the lap history and stint structure through MCP tools
    history = get_driver_lap_history(
        year=year,
        event=event,
        session_type=session_type,
        driver_code=driver_code,
        current_lap=current_lap
    )

    stints = get_tyre_stints(
        year=year,
        event=event,
        session_type=session_type,
        driver_code=driver_code,
        current_lap=current_lap
    )

    user_prompt = _build_user_prompt(driver_code, current_lap, history, stints)

    return await client.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=TyreAssessment,
        temperature=0.2
    )