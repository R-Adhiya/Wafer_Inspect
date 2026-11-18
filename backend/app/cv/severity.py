"""Prototype defect and inspection severity classification.

IMPORTANT DISCLAIMER:
This module implements a documented, resolution-independent **prototype heuristic**
for ranking defects (LOW / MEDIUM / HIGH) and assigning an inspection-level severity
(NONE / LOW / MEDIUM / HIGH). It is NOT an industrial semiconductor fab standard
(such as SEMI M52, KLA excursion rules, or yield-impact models). Thresholds are
configurable in `SeverityConfig` and should be calibrated to domain requirements.

Default classification is resolution-independent: each defect's area is compared as
a fraction of the segmented wafer area (`area_px / wafer_area_px`). Optional
absolute-pixel thresholds (`medium_area_px`, `high_area_px`) are applied ONLY when
explicitly supplied by the user.
"""

from typing import List, Literal, Optional, Sequence, Tuple
from pydantic import BaseModel, Field, model_validator

DefectSeverity = Literal["LOW", "MEDIUM", "HIGH"]
InspectionSeverity = Literal["NONE", "LOW", "MEDIUM", "HIGH"]

_SEVERITY_LADDER: Tuple[DefectSeverity, ...] = ("LOW", "MEDIUM", "HIGH")


class SeverityConfig(BaseModel):
    """Configurable thresholds for the prototype defect and inspection severity heuristic.

    By default, per-defect severity uses resolution-independent fractions of total
    wafer area:
      - LOW:    area_fraction < medium_area_fraction (default < 0.08% of wafer area)
      - MEDIUM: medium_area_fraction <= area_fraction < high_area_fraction (0.08% - 0.30%)
      - HIGH:   area_fraction >= high_area_fraction (default >= 0.30% of wafer area)

    Users may optionally specify absolute pixel thresholds (`medium_area_px` and/or
    `high_area_px`), which override the corresponding fractional thresholds only when set.
    """

    medium_area_fraction: float = Field(
        default=0.0008,
        gt=0.0,
        lt=1.0,
        description="Minimum fraction of wafer area for a single defect to be classified MEDIUM (0.08%).",
    )
    high_area_fraction: float = Field(
        default=0.003,
        gt=0.0,
        lt=1.0,
        description="Minimum fraction of wafer area for a single defect to be classified HIGH (0.30%).",
    )
    medium_area_px: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional user-supplied absolute area threshold (in original px) for MEDIUM severity.",
    )
    high_area_px: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional user-supplied absolute area threshold (in original px) for HIGH severity.",
    )
    count_escalation: int = Field(
        default=5,
        ge=1,
        description="Escalate inspection severity by 1 level when defect_count >= count_escalation.",
    )
    area_escalation_fraction: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Escalate inspection severity by 1 level when total defect area / wafer area >= this fraction (1.0%).",
    )

    @model_validator(mode="after")
    def _validate_threshold_ordering(self) -> "SeverityConfig":
        if self.high_area_fraction <= self.medium_area_fraction:
            raise ValueError(
                f"high_area_fraction ({self.high_area_fraction}) must exceed "
                f"medium_area_fraction ({self.medium_area_fraction})."
            )
        if (
            self.medium_area_px is not None
            and self.high_area_px is not None
            and self.high_area_px <= self.medium_area_px
        ):
            raise ValueError(
                f"high_area_px ({self.high_area_px}) must exceed "
                f"medium_area_px ({self.medium_area_px})."
            )
        return self


def classify_defect_severity(
    area_px: float,
    wafer_area_px: float,
    config: Optional[SeverityConfig] = None,
) -> DefectSeverity:
    """Classify a single defect as LOW, MEDIUM, or HIGH using the prototype heuristic.

    Uses resolution-independent wafer-area fractions by default, or user-set absolute
    pixel thresholds if explicitly provided in `config`.
    """
    if config is None:
        config = SeverityConfig()

    safe_wafer_area = max(float(wafer_area_px), 1.0)
    eff_medium_px = (
        float(config.medium_area_px)
        if config.medium_area_px is not None
        else config.medium_area_fraction * safe_wafer_area
    )
    eff_high_px = (
        float(config.high_area_px)
        if config.high_area_px is not None
        else config.high_area_fraction * safe_wafer_area
    )

    area = float(area_px)
    if area >= eff_high_px:
        return "HIGH"
    if area >= eff_medium_px:
        return "MEDIUM"
    return "LOW"


def assess_inspection_severity(
    defect_areas_px: Sequence[float],
    defect_severities: Sequence[DefectSeverity],
    wafer_area_px: float,
    config: Optional[SeverityConfig] = None,
) -> Tuple[InspectionSeverity, str]:
    """Compute overall inspection severity and a data-backed human-readable reason.

    Rules:
      - 0 defects -> ("NONE", reason)
      - Otherwise -> max per-defect severity, escalated by 1 level (capped at HIGH)
        if `defect_count >= count_escalation` OR
        `total_defect_area / wafer_area >= area_escalation_fraction`.

    Returns:
        Tuple[InspectionSeverity, str]: (overall_severity, severity_reason)
    """
    if config is None:
        config = SeverityConfig()

    if len(defect_areas_px) != len(defect_severities):
        raise ValueError("defect_areas_px and defect_severities must have equal length.")

    defect_count = len(defect_severities)
    if defect_count == 0:
        return (
            "NONE",
            "No defects detected (0 regions; 0 px total defect area); inspection severity is NONE.",
        )

    safe_wafer_area = max(float(wafer_area_px), 1.0)
    total_area_px = float(sum(defect_areas_px))
    largest_area_px = float(max(defect_areas_px))
    total_area_fraction = total_area_px / safe_wafer_area
    total_area_pct = total_area_fraction * 100.0
    area_esc_pct = config.area_escalation_fraction * 100.0

    base_idx = max(_SEVERITY_LADDER.index(sev) for sev in defect_severities)
    base_severity = _SEVERITY_LADDER[base_idx]

    count_triggered = defect_count >= config.count_escalation
    area_triggered = total_area_fraction >= config.area_escalation_fraction

    triggers: List[str] = []
    if count_triggered:
        triggers.append(f"defect_count={defect_count} >= count_escalation={config.count_escalation}")
    if area_triggered:
        triggers.append(
            f"total_area_fraction={total_area_pct:.2f}% ({int(round(total_area_px))} px / "
            f"{int(round(safe_wafer_area))} px) >= area_escalation={area_esc_pct:.2f}%"
        )

    if triggers:
        escalated_idx = min(base_idx + 1, len(_SEVERITY_LADDER) - 1)
        final_severity: InspectionSeverity = _SEVERITY_LADDER[escalated_idx]
        trigger_text = " and ".join(triggers)
        if escalated_idx > base_idx:
            reason = (
                f"Inspection severity {final_severity} (escalated 1 level from max defect severity "
                f"{base_severity} because {trigger_text}; largest defect {int(round(largest_area_px))} px)."
            )
        else:
            reason = (
                f"Inspection severity {final_severity} (already at maximum level from max defect severity "
                f"{base_severity}; escalation condition also met: {trigger_text}; "
                f"largest defect {int(round(largest_area_px))} px)."
            )
        return final_severity, reason

    reason = (
        f"Inspection severity {base_severity} from max defect severity {base_severity} across "
        f"{defect_count} defect(s) (largest {int(round(largest_area_px))} px, total "
        f"{int(round(total_area_px))} px = {total_area_pct:.2f}% of wafer area; below "
        f"count_escalation={config.count_escalation} and area_escalation={area_esc_pct:.2f}%)."
    )
    return base_severity, reason
