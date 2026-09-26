"""
Validation for the soil analysis form.

Two separate ideas are kept apart:

  ERRORS   block the prediction. A value is rejected only when it is missing,
           not a number, or physically impossible (negative rainfall, pH of 20,
           humidity above 100 %).

  NOTICES  never block anything. When a value is valid but falls outside the
           range present in the training data, the user is told that the model
           is extrapolating there. Unusual field values are still legitimate
           field values, so they are accepted and flagged rather than refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config


@dataclass
class ValidationResult:
    values: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# Ranges actually present in the training CSV. Values outside these are
# accepted, but the user is told the model has not seen that territory.
_TRAINING_RANGE = {
    "N": (0, 140), "P": (5, 145), "K": (5, 205),
    "temperature": (8.8, 43.7), "humidity": (14.3, 100.0),
    "ph": (3.5, 10.0), "rainfall": (20.2, 298.6),
}


def validate_soil_inputs(form: dict) -> ValidationResult:
    """Parse and check the seven model inputs submitted by the form."""
    result = ValidationResult()

    for name, spec in config.INPUT_FIELDS.items():
        raw = (form.get(name) or "").strip()

        if not raw:
            result.errors[name] = f"Enter a value for {spec['label']}."
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            result.errors[name] = f"{spec['label']} must be a number."
            continue

        if value != value or value in (float("inf"), float("-inf")):
            result.errors[name] = f"{spec['label']} must be a real number."
            continue

        low, high = spec["min"], spec["max"]
        if value < low or value > high:
            unit = f" {spec['unit']}" if spec["unit"] else ""
            result.errors[name] = (
                f"{spec['label']} must be between {low}{unit} and {high}{unit}."
            )
            continue

        result.values[name] = value

        train_low, train_high = _TRAINING_RANGE[name]
        if value < train_low or value > train_high:
            unit = f" {spec['unit']}" if spec["unit"] else ""
            result.notices.append(
                f"{spec['label']} of {value:g}{unit} is outside the "
                f"{train_low:g}-{train_high:g}{unit} range covered by the training "
                "data. The result is still shown, but treat it with caution."
            )

    return result


def validate_district(name: str | None, known: list[str]) -> str | None:
    """Return the canonical district name, or None when it is not recognised."""
    if not name:
        return None
    target = str(name).strip().casefold()
    for candidate in known:
        if candidate.strip().casefold() == target:
            return candidate
    return None
