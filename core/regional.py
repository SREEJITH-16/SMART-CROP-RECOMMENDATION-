"""
Tamil Nadu regional crop profile.

This is a STATISTICS LOOKUP, not a machine learning prediction. It reads
district rows from Tamil_Nadu_Crop_Regional_Dataset.csv and ranks crops by the
cultivated area recorded for that district in 2021-22.

Cultivated area is not a suitability score and not a probability. A crop ranks
first because more hectares of it were recorded, which reflects the district's
history, irrigation, markets and policy - not a prediction that it will succeed
on any particular field. Every label this module produces says so.

Columns are read from the file at load time rather than assumed. The dataset
supplies, per district:
    District, Agro_Climatic_Zone_2021_22, Major_Soil_Types_2021_22, Data_Year,
    <Crop>_Area_ha_2021_22 for 14 principal crops,
    Top_1..5_Crop and their areas,
    Data_Use, Primary_Source, Source_Notes
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

import config
from core import crop_metadata

DISTRICT_COLUMN = "District"
# Column names use an underscore in the year, e.g. "Paddy_Area_ha_2021_22".
# Both separators are accepted so a future "2022-23" file also loads.
_AREA_SUFFIX_PATTERN = re.compile(r"_Area_ha_(\d{4}[-_]\d{2})$")
_NOT_AVAILABLE = "Data not available"


@dataclass
class RegionalCrop:
    """One crop recorded in a district, with its observed cultivated area."""
    name: str
    crop_key: str
    area_ha: int
    share_of_top: float          # bar width relative to the district's largest crop
    rank: int


@dataclass
class DistrictProfile:
    name: str
    agro_climatic_zone: str
    major_soil_types: list[str]
    data_year: str
    crops: list[RegionalCrop] = field(default_factory=list)
    source: str = _NOT_AVAILABLE
    source_notes: str = ""
    total_recorded_area_ha: int = 0

    @property
    def has_crop_data(self) -> bool:
        return bool(self.crops)


class RegionalDataUnavailableError(RuntimeError):
    """Raised when the Tamil Nadu dataset cannot be read."""


class RegionalDataService:
    def __init__(self) -> None:
        self.frame: pd.DataFrame | None = None
        self.area_columns: dict[str, str] = {}   # display name -> column name
        self.data_year: str = _NOT_AVAILABLE
        self.load_error: str | None = None
        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        try:
            if not config.TAMIL_NADU_CSV.exists():
                raise RegionalDataUnavailableError(
                    f"Regional dataset not found at {config.TAMIL_NADU_CSV.name}."
                )
            frame = pd.read_csv(config.TAMIL_NADU_CSV)
            if DISTRICT_COLUMN not in frame.columns:
                raise RegionalDataUnavailableError(
                    f"Regional dataset has no '{DISTRICT_COLUMN}' column."
                )

            # Discover the per-crop area columns instead of hard-coding them.
            for column in frame.columns:
                match = _AREA_SUFFIX_PATTERN.search(column)
                if not match or column.startswith("Top_"):
                    continue
                display = column[: match.start()].replace("_", " ").strip()
                self.area_columns[display] = column

            if not self.area_columns:
                raise RegionalDataUnavailableError(
                    "Regional dataset contains no per-crop area columns."
                )

            frame[DISTRICT_COLUMN] = frame[DISTRICT_COLUMN].astype(str).str.strip()
            self.frame = frame.sort_values(DISTRICT_COLUMN).reset_index(drop=True)

            if "Data_Year" in frame.columns and not frame["Data_Year"].isna().all():
                self.data_year = str(frame["Data_Year"].dropna().iloc[0])
        except Exception as exc:                        # noqa: BLE001 - shown in UI
            self.load_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self.frame is not None and self.load_error is None

    # ----------------------------------------------------------- public API

    def districts(self) -> list[str]:
        if not self.is_ready:
            return []
        return self.frame[DISTRICT_COLUMN].tolist()

    def district_count(self) -> int:
        return len(self.districts())

    def has_district(self, name: str) -> bool:
        return self._row_for(name) is not None

    def profile(self, district: str, limit: int = 8) -> DistrictProfile | None:
        """Build the full regional profile for one district, or None if unknown."""
        row = self._row_for(district)
        if row is None:
            return None

        crops = self._ranked_crops(row, limit=limit)
        return DistrictProfile(
            name=str(row[DISTRICT_COLUMN]),
            agro_climatic_zone=self._text(row, "Agro_Climatic_Zone_2021_22"),
            major_soil_types=self._soil_list(row),
            data_year=self._text(row, "Data_Year", default=self.data_year),
            crops=crops,
            source=self._text(row, "Primary_Source"),
            source_notes=self._text(row, "Source_Notes", default=""),
            total_recorded_area_ha=sum(c.area_ha for c in crops),
        )

    def source_citation(self) -> str:
        if not self.is_ready or "Primary_Source" not in self.frame.columns:
            return _NOT_AVAILABLE
        values = self.frame["Primary_Source"].dropna()
        return str(values.iloc[0]) if len(values) else _NOT_AVAILABLE

    # ------------------------------------------------------------- internals

    def _row_for(self, district: str) -> pd.Series | None:
        if not self.is_ready or not district:
            return None
        target = str(district).strip().casefold()
        matches = self.frame[
            self.frame[DISTRICT_COLUMN].str.strip().str.casefold() == target
        ]
        return None if matches.empty else matches.iloc[0]

    def _ranked_crops(self, row: pd.Series, limit: int) -> list[RegionalCrop]:
        """
        Rank the district's crops by recorded cultivated area, highest first.
        Crops with zero or missing area are excluded rather than shown as 0.
        """
        entries: list[tuple[str, int]] = []
        for display, column in self.area_columns.items():
            if column not in row.index:
                continue
            value = row[column]
            if pd.isna(value):
                continue
            area = int(value)
            if area > 0:
                entries.append((display, area))

        entries.sort(key=lambda item: item[1], reverse=True)
        entries = entries[:limit]
        if not entries:
            return []

        largest = entries[0][1]
        ranked = []
        for index, (display, area) in enumerate(entries, start=1):
            crop = crop_metadata.get(display)
            ranked.append(
                RegionalCrop(
                    name=crop.name if crop.key != "_unknown" else display,
                    crop_key=crop.key if crop.key != "_unknown" else display.lower(),
                    area_ha=area,
                    share_of_top=round(area / largest * 100, 1) if largest else 0.0,
                    rank=index,
                )
            )
        return ranked

    def _soil_list(self, row: pd.Series) -> list[str]:
        raw = self._text(row, "Major_Soil_Types_2021_22")
        if raw == _NOT_AVAILABLE:
            return []
        parts = [part.strip() for part in raw.split(";") if part.strip()]
        return parts or []

    @staticmethod
    def _text(row: pd.Series, column: str, default: str = _NOT_AVAILABLE) -> str:
        """Read a text cell, returning the default rather than inventing content."""
        if column not in row.index:
            return default
        value = row[column]
        if pd.isna(value) or not str(value).strip():
            return default
        text = str(value).strip()
        # The dataset uses explicit "not assigned / not specified" wording for
        # districts outside the zone table. Keep it - it is meaningful.
        return text


# Loaded once per process.
regional_service = RegionalDataService()
