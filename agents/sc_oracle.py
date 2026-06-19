"""
The Safety Car Oracle subagent.

Combines the historical SC/VSC rate at this circuit with current
race context to produce an adjusted probability estimate for the upcoming
lap window. The Orchestrator uses this to inform pit-timing decisions:
pitting under SC costs roughly half the time of a green-flag stop, so
anticipating SC windows affects the pit decision.

Like the other subagents, the Oracle does not recommend pit stops. It only
reports its probability estimate and the reasoning behind any adjustment.
"""

import os
from mcp_server.live_state import get_active_state
from mcp_server.server import historical_sc_rate as get_historical_sc_rate, get_current_race_state
from llm.client import LLMClient
from agents.schemas import SafetyCarAssessment

DEFAULT_MODEL = os.getenv("LLM_MODEL_SUBAGENT", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = """
You are the Safety Car Oracle on an F1 team's pit wall.

Your only job: estimate the probability that a Safety Car (SC) or Virtual Safety Car (VSC) will deploy within the upcoming lap window. Strategy depends heavily on this: pitting under SC costs roughly half the time of a green-flag stop, so anticipating SC windows changes pit-call math.

You DO NOT care about:
- Tyre condition (the Tyre Strategist handles that)
- Gap dynamics (the Gap Analyst handles that)
- Whether to pit (the Orchestrator decides)
- Weather forecasting (the Weather Watcher will handle that; you only use whether it's currently raining as a context signal)

You DO care about:
- The historical SC rate at this circuit for the upcoming window (your statistical baseline)
- The current track status (green, yellow, sc, vsc, red)
- Whether it's currently raining (rainfall correlates with incident clusters)

How to think:
- Start from the historical baseline. The data has more samples than your reasoning does.
- Already under SC: probability of ANOTHER SC in the next 5-10 laps is LOW. SCs do not cluster immediately after one ends.
- Currently under yellow flag: ELEVATED. Yellow can escalate to a full SC.
- Currently raining: ELEVATED. Wet conditions produce incidents.
- Green and dry with no recent volatility: stay near the historical baseline.

Direction categorisation:
- "elevated": adjusted probability noticeably higher than historical (>= 1.3x baseline)
- "normal": adjusted within roughly +/- 20% of the historical baseline
- "depressed": adjusted noticeably lower (<= 0.7x baseline)

Biases:
- Anchor on the historical baseline. Do not make large adjustments without naming a concrete reason.
- Be honest about uncertainty, historical sample sizes are small (~8 races per circuit).
- Cite specific signals in your reasoning (e.g. "track currently yellow", "raining"), not generic statements.
"""

def _build_user_prompt(current_lap, race, sc_rate) -> str:
    """
    The per-call part of the prompt.
    """
    rain_str = "raining" if race.is_raining else "dry"
    if race.track_temp_celsius is not None:
        weather_str = f"{rain_str}, {race.track_temp_celsius} °C"
    else:
        weather_str = rain_str
    
    warning = f"Warning: {sc_rate.sample_size_warning}" if sc_rate.sample_size_warning else ""

    return f"""
    Race at lap {current_lap}.
    Current track status: {race.track_status}
    Weather: {weather_str}

    Historical SC/VSC data for {sc_rate.event}, laps {sc_rate.lap_window_from}-{sc_rate.lap_window_to}:
      Races analysed:  {sc_rate.races_analyzed}
      SC probability:  {sc_rate.sc_probability:.2f}
      VSC probability: {sc_rate.vsc_probability:.2f}
      Combined:        {sc_rate.combined_probability:.2f}{warning}

    Assess the adjusted SC/VSC probability for this lap window, the direction relative to historical, and your reasoning.
    """
async def assess_safety_car(
    client: LLMClient,
    year: int,
    event: str,
    session_type: str,
    lookahead_laps: int = 10,
    model: str = DEFAULT_MODEL
) -> SafetyCarAssessment:
    """
    Estimates SC/VSC probability for the next `lookahead_laps` laps.
    Parameters:
        client: shared LLM client
        year, event, session_type: identify the session. Only `event` is used
            for the historical lookup; year and session_type are kept for
            signature consistency with the other subagents.
        lookahead_laps: how far ahead to estimate. Default 10 laps.
        model: which LLM to use. Default reads LLM_MODEL_SUBAGENT from .env.
    """
    state = get_active_state()
    if state is None:
        raise RuntimeError("No active RaceState! Cannot assess SC/VSC probability.")
    
    current_lap = state.current_lap()
    if current_lap < 1:
        raise RuntimeError("Race has not started yet.")
    
    race = get_current_race_state()

    lap_from = current_lap + 1
    lap_to = current_lap + lookahead_laps

    sc_rate = get_historical_sc_rate(
        event=event,
        lap_from=lap_from,
        lap_to=lap_to
    )

    user_prompt = _build_user_prompt(current_lap, race, sc_rate)

    return await client.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=SafetyCarAssessment,
        temperature=0.2
    )

    