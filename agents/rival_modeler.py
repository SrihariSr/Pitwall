"""
The Rival Modeler subagent.

Predicts when nearby rivals will pit and identifies the biggest strategic
threat. Where the Gap Analyst tells the team how rivals are positioned now,
the Rival Modeler tells them what rivals are about to do. First predictive
specialist in the ensemble.
"""
from typing_extensions import runtime
import os
from mcp_server.live_state import get_active_state
from mcp_server.server import get_current_race_state, get_tyre_stints, get_gaps_to_rivals
from llm.client import LLMClient
from agents.schemas import RivalAssessment

DEFAULT_MODEL = os.getenv("LLM_MODEL_SUBAGENT", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = """
You are the Rival Modeler on an F1 team's pit wall.

Your only job: anticipate when nearby rivals will pit and identify which one poses the biggest strategic threat. Where the Gap Analyst reports how rivals are positioned NOW, you project what they are about to DO.

You DO NOT care about:
- Tyre degradation on our own car (the Tyre Strategist handles that)
- Our pace and gaps in absolute terms (the Gap Analyst handles that)
- Whether WE should pit (the Orchestrator decides)
- Weather conditions in general (the Weather Watcher handles that)

You DO care about:
- Each rival's current compound and stint age (how worn are their tyres?)
- Typical stint lengths for the compound and the circuit
- Position-relative-to-us logic: a rival BEHIND us pitting first is the undercut concern; a rival AHEAD pitting later is the overcut concern
- Only the handful of rivals close enough to actually matter (top 3 by gap)

How to think:
- Typical stint lives, approximately:
  - SOFT: 15-25 laps
  - MEDIUM: 25-35 laps
  - HARD: 35-45 laps
  - INTER and WET: 15-25 laps, but highly weather-dependent
- A rival on softs with 20+ laps of stint age is on the edge, likely pitting within 3-5 laps.
- A rival on hards with 10 laps of stint age has 25+ laps before they need to box.
- Don't predict identical pit laps for every rival: they're racing each other too and will avoid synchronised stops.
- In transitional weather (wet→dry), rivals pit when the track crosses the compound threshold, not when their stint ages out.

Threat window calibration:
- "now": next stop expected within 3 laps
- "soon": 3-8 laps away
- "later": 8+ laps away

Primary threat selection:
- The rival BEHIND us with the shortest pit window is the undercut threat.
- The rival AHEAD with the longest pit window is the overcut threat.
- Pick the one whose move would most disrupt our race position.

Biases:
- Conservative on confidence. Predicting another driver's strategy is inherently uncertain.
- Cite specific evidence (compound, stint age, position) in reasoning, not generic statements.
- If stint data is missing for a rival, lower confidence rather than guessing.
"""

def _build_user_prompt(driver_code, current_lap, race, gaps, rival_stints) -> str:
    """
    Compose the per-call part. Shows each rival's stint state and gap.
    """
    closest = sorted(gaps.rivals, key=lambda r: abs(r.gap_seconds))[:3]

    rival_blocks = []
    for rival in closest:
        stints_obj = rival_stints.get(rival.driver_code)
        if stints_obj is None:
            rival_blocks.append(f"{rival.driver_code} (gap: {rival.gap_seconds:+.2f}s): No stint data available.")
            continue
        
        current_stint = next((s for s in stints_obj if s.is_ongoing), None)
        if current_stint is None:
            rival_blocks.append(f"{rival.driver_code} (gap {rival.gap_seconds:+.2f}s): no ongoing stint detected.")
            continue
        
        age = current_lap - current_stint.start_lap + 1
        rival_blocks.append(
            f" {rival.driver_code} (gap {rival.gap_seconds:+.2f}s, {rival.relationship}): "
            f"{current_stint.compound}, stint age {age} laps, started L{current_stint.start_lap}, "
            f"best lap {current_stint.best_lap_time_seconds:.3f}s"
        )

    rain_str = "raining" if race.is_raining else "dry"

    return f"""
Race at lap {current_lap}.
Focal driver: {driver_code}.
Track status: {race.track_status}. Weather: {rain_str}.

Rivals (closest by gap, sign indicates position. Negative is behind us):
{chr(10).join(rival_blocks) if rival_blocks else '  (no rivals identified)'}

For each rival, predict their next pit lap and threat window. Identify the primary threat to our race and explain what makes them dangerous.
"""

async def assess_rivals(
    client: LLMClient,
    driver_code: str,
    year: int,
    event: str,
    session_type: str,
    model: str = DEFAULT_MODEL
) -> RivalAssessment:
    """
    Predict next pit laps and threat windows for the closest rivals.
    Parameters:
        client: shared LLM client
        driver_code: focal driver code (e.g. 'LEC')
        year, event, session_type: identify the session for MCP queries
        model: which LLM to use. Default reads LLM_MODEL_SUBAGENT from .env.
    """
    state = get_active_state()
    if state is None:
        raise RuntimeError("No live RaceState active! Cannot assess rivals.")
    
    current_lap = state.current_lap()
    if current_lap < 1:
        raise RuntimeError("Race has not started yet.")
    
    race = get_current_race_state()

    gaps = get_gaps_to_rivals(
        year=year,
        event=event,
        session_type=session_type,
        driver_code=driver_code,
        current_lap=current_lap
    )
    
    closest = sorted(gaps.rivals, key=lambda r: abs(r.gap_seconds))[:3]

    rival_stints = {}
    for rival in closest:
        try:
            rival_stints[rival.driver_code] = get_tyre_stints(
                year=year,
                event=event,
                session_type=session_type,
                driver_code=rival.driver_code,
                current_lap=current_lap
            )
        except Exception:
            rival_stints[rival.driver_code] = None
    
    user_prompt = _build_user_prompt(driver_code, current_lap, race, gaps, rival_stints)

    return await client.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=RivalAssessment,
        temperature=0.2
    )
