"""
Pydantic schemas for subagent outputs.

Every subagent returns a small, tightly-scoped object. The orchestrator
reads all of them in one prompt to fuse into a single pit-stop call.

Schemas are intentionally minimal: fewer tokens means less hallucination surface,
easier for the orchestrator to reason about.
"""

from pydantic import BaseModel, Field

class TyreAssessment(BaseModel):
    """The Tyre Strategist's verdict on the focal driver's current stint."""

    has_sufficient_data: bool = Field(
        description=(
            "True if there's enough lap data to make a meaningful cliff "
            "projection. False if the stint is too young (under 3 laps) or "
            "data is missing. When False, cliff_lap is meaningless and "
            "should be ignored by consumers."
        )
    )
    cliff_lap: int = Field(
        description=(
            "The lap by which we expect the tyre to hit its performance "
            "cliff. Only meaningful when has_sufficient_data=True."
        )
    )
    confidence: float = Field(
        description=(
            "0.0-1.0 confidence in the cliff_lap estimate. Should be low "
            "(<0.4) when has_sufficient_data=False."
        ),
        ge=0.0, le=1.0,
    )
    reasoning: str = Field(
        description="One sentence justifying the call, citing specific evidence from the lap data."
    )

class RivalGapEntry(BaseModel):
    """A single rival's gap as the Gap Analyst sees it.

    Minimal: only the rivals the orchestrator actually needs to know about.
    The full per-rival data is in MCP; this is the curated summary.
    """

    driver_code: str = Field(description="3-letter rival code")
    position: int = Field(description="Rival's track position")
    gap_seconds: float = Field(
        description=(
            "Time gap in seconds. Positive = rival is AHEAD on track. "
            "Negative = rival is behind."
        )
    )
    relationship: str = Field(
        description=(
            "How this rival relates strategically: "
            "'undercut_threat' (within pit-window behind), "
            "'overcut_target' (within pit-window ahead, pitting first might be exploitable), "
            "'direct_battle' (within DRS/overtaking range), "
            "'context' (worth knowing but not strategically active)."
        )
    )


class GapAssessment(BaseModel):
    """
    The Gap Analyst's verdict on the focal driver's strategic gap situation.
    """

    focal_position: int = Field(description="Focal driver's current track position")
    undercut_threat: str = Field(
        description=(
            "Severity of undercut threat from drivers behind: "
            "'high' (clear and imminent risk), 'medium' (plausible), "
            "'low' (theoretically possible but unlikely), 'none' (no threat)."
        )
    )
    overcut_opportunity: str = Field(
        description=(
            "Whether an overcut on rivals ahead is on the table: "
            "'high' (clear opportunity), 'medium' (situational), "
            "'low' (unlikely to pay off), 'none' (no opportunity)."
        )
    )
    closest_rivals: list[RivalGapEntry] = Field(
        description=(
            "The 2-4 most strategically relevant rivals, in track-position order. "
            "Includes immediate threats and immediate opportunities, not the whole field."
        )
    )
    reasoning: str = Field(
        description="One sentence citing specific gaps and dynamics, not generalities."
    )
    confidence: float = Field(
        description=(
            "0.0-1.0 confidence in the assessment. Lower when the field is "
            "in flux (recent pit stops shuffling order, SC just ended)."
        ),
        ge=0.0,
        le=1.0,
    )