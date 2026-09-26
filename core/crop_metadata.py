"""
Single source of truth for crop information.

Every crop appears exactly once, here. Templates read from this registry, so
crop facts are never duplicated across HTML files.

PROVENANCE RULES
----------------
Each crop carries two clearly separated kinds of information:

1. `observed` - computed at runtime from the ML training CSV by
   core.ml_model.CropRecommender. These are measured ranges from the dataset
   the model actually learned from, not agronomic recommendations.

2. `general` - indicative agronomic characteristics, shown in the UI under the
   heading "General information (indicative)". These are widely published
   characteristics, not measurements from the supplied datasets, and the
   interface labels them that way.

Where a value is not known, the field is set to None and the UI renders
"Data not available". Nothing is filled in with a plausible-looking guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeneralInfo:
    """Indicative characteristics. None means 'not available', never a guess."""
    soil: str | None = None
    temperature: str | None = None
    water_requirement: str | None = None   # Low / Moderate / High
    growing_period: str | None = None
    season: str | None = None


@dataclass(frozen=True)
class Crop:
    key: str                       # canonical lowercase id
    name: str                      # display name
    category: str
    image: str                     # filename in static/crops/
    description: str
    general: GeneralInfo = field(default_factory=GeneralInfo)
    aliases: tuple[str, ...] = ()  # other spellings that resolve to this crop
    in_ml_model: bool = False      # predictable by the soil-based model
    in_tn_dataset: bool = False    # appears in the Tamil Nadu statistics


# --------------------------------------------------------------------------
# Registry. Crops appearing in both systems are defined once with both flags.
# --------------------------------------------------------------------------

_CROPS: tuple[Crop, ...] = (
    Crop(
        key="rice", name="Rice", category="Cereal", image="rice.svg",
        description=(
            "A staple cereal grown across most of India. Rice is usually "
            "transplanted into puddled fields and needs standing water for much "
            "of its growth, which is why it dominates deltaic and canal-irrigated "
            "districts."
        ),
        general=GeneralInfo(
            soil="Clay and clay loam soils that hold water well",
            temperature="20 - 35 \u00b0C",
            water_requirement="High",
            growing_period="100 - 150 days",
            season="Kharif and Rabi (Samba, Kuruvai and Navarai in Tamil Nadu)",
        ),
        aliases=("paddy",), in_ml_model=True, in_tn_dataset=True,
    ),
    Crop(
        key="maize", name="Maize", category="Cereal", image="maize.svg",
        description=(
            "A versatile cereal used for food, cattle feed and starch. Maize is "
            "sensitive to waterlogging and performs best on well-drained land."
        ),
        general=GeneralInfo(
            soil="Well-drained loam and sandy loam",
            temperature="21 - 30 \u00b0C",
            water_requirement="Moderate",
            growing_period="90 - 120 days",
            season="Kharif and Rabi",
        ),
        in_ml_model=True, in_tn_dataset=True,
    ),
    Crop(
        key="cotton", name="Cotton", category="Fibre", image="cotton.svg",
        description=(
            "A long-duration fibre crop. Cotton needs a dry, bright spell during "
            "boll opening, so unseasonal rain at picking time is a common risk."
        ),
        general=GeneralInfo(
            soil="Black cotton soil and deep well-drained loam",
            temperature="21 - 35 \u00b0C",
            water_requirement="Moderate",
            growing_period="150 - 180 days",
            season="Kharif",
        ),
        in_ml_model=True, in_tn_dataset=True,
    ),
    Crop(
        key="jute", name="Jute", category="Fibre", image="jute.svg",
        description=(
            "A bast fibre crop grown mainly in the humid eastern states. Jute "
            "needs high humidity and warm, wet conditions through its growth."
        ),
        general=GeneralInfo(
            soil="Alluvial loam",
            temperature="24 - 37 \u00b0C",
            water_requirement="High",
            growing_period="100 - 150 days",
            season="Pre-monsoon to monsoon",
        ),
        in_ml_model=True,
    ),
    Crop(
        key="coconut", name="Coconut", category="Plantation", image="coconut.svg",
        description=(
            "A perennial palm grown largely along the coast. Once established a "
            "coconut garden yields for decades, but it takes several years to "
            "come into bearing."
        ),
        general=GeneralInfo(
            soil="Sandy loam, laterite and coastal alluvium with good drainage",
            temperature="25 - 32 \u00b0C",
            water_requirement="High",
            growing_period="Perennial; first harvest after about 5 - 7 years",
            season="Planted with the onset of the monsoon",
        ),
        in_ml_model=True,
    ),
    Crop(
        key="papaya", name="Papaya", category="Fruit", image="papaya.svg",
        description=(
            "A fast-bearing fruit crop that starts yielding within a year of "
            "planting. Papaya is highly sensitive to waterlogging and frost."
        ),
        general=GeneralInfo(
            soil="Well-drained sandy loam",
            temperature="22 - 35 \u00b0C",
            water_requirement="Moderate",
            growing_period="First harvest about 9 - 11 months after planting",
            season="Planted year-round under irrigation",
        ),
        in_ml_model=True,
    ),
    Crop(
        key="orange", name="Orange", category="Fruit", image="orange.svg",
        description=(
            "A citrus tree crop. Oranges need a distinct dry period to set fruit "
            "well and do poorly on heavy, poorly drained soils."
        ),
        general=GeneralInfo(
            soil="Well-drained light loam",
            temperature="15 - 30 \u00b0C",
            water_requirement="Moderate",
            growing_period="Perennial; bearing from about year 4",
            season=None,
        ),
        in_ml_model=True,
    ),
    Crop(
        key="apple", name="Apple", category="Fruit", image="apple.svg",
        description=(
            "A temperate fruit crop needing a sustained cold period each winter "
            "to flower properly. In India it is confined to the hills."
        ),
        general=GeneralInfo(
            soil="Well-drained loam",
            temperature="Temperate; requires winter chilling",
            water_requirement="Moderate",
            growing_period="Perennial; bearing from about year 4 - 5",
            season=None,
        ),
        in_ml_model=True,
    ),
    Crop(
        key="muskmelon", name="Muskmelon", category="Fruit", image="muskmelon.svg",
        description=(
            "A short-duration summer cucurbit. Muskmelon prefers hot, dry air, "
            "and humidity late in the season encourages disease."
        ),
        general=GeneralInfo(
            soil="Sandy loam with good drainage",
            temperature="24 - 35 \u00b0C",
            water_requirement="Moderate",
            growing_period="80 - 100 days",
            season="Summer",
        ),
        in_ml_model=True,
    ),
    Crop(
        key="watermelon", name="Watermelon", category="Fruit", image="watermelon.svg",
        description=(
            "A summer cucurbit grown on light soils, often on riverbeds. It "
            "needs warm days and steady irrigation while the fruit fills."
        ),
        general=GeneralInfo(
            soil="Sandy and sandy loam",
            temperature="24 - 35 \u00b0C",
            water_requirement="Moderate",
            growing_period="80 - 110 days",
            season="Summer",
        ),
        in_ml_model=True,
    ),
    Crop(
        key="grapes", name="Grapes", category="Fruit", image="grapes.svg",
        description=(
            "A perennial vine grown on trellises. Grapes need heavy potassium "
            "and phosphorus and a dry spell at ripening."
        ),
        general=GeneralInfo(
            soil="Well-drained loam, tolerant of moderately alkaline soil",
            temperature="15 - 35 \u00b0C",
            water_requirement="Moderate",
            growing_period="Perennial; pruned to a cycle of about 120 - 150 days",
            season=None,
        ),
        in_ml_model=True,
    ),
    Crop(
        key="mango", name="Mango", category="Fruit", image="mango.svg",
        description=(
            "A long-lived tree crop. Mango needs a dry spell before flowering, "
            "and rain at that stage can sharply reduce the crop."
        ),
        general=GeneralInfo(
            soil="Deep, well-drained loam",
            temperature="24 - 35 \u00b0C",
            water_requirement="Low to moderate once established",
            growing_period="Perennial; bearing from about year 4 - 5",
            season=None,
        ),
        in_ml_model=True,
    ),
    Crop(
        key="banana", name="Banana", category="Fruit", image="banana.svg",
        description=(
            "A heavy-feeding, heavy-drinking crop grown year-round under "
            "irrigation. Banana is vulnerable to wind damage and waterlogging."
        ),
        general=GeneralInfo(
            soil="Deep rich loam with good drainage",
            temperature="25 - 35 \u00b0C",
            water_requirement="High",
            growing_period="11 - 14 months",
            season="Planted year-round under irrigation",
        ),
        in_ml_model=True,
    ),
    Crop(
        key="pomegranate", name="Pomegranate", category="Fruit", image="pomegranate.svg",
        description=(
            "A hardy fruit crop suited to semi-arid areas. Pomegranate tolerates "
            "drought well but fruit splits if irrigation is irregular."
        ),
        general=GeneralInfo(
            soil="Well-drained loam, tolerant of light and slightly saline soil",
            temperature="25 - 35 \u00b0C",
            water_requirement="Low to moderate",
            growing_period="Perennial; bearing from about year 3",
            season=None,
        ),
        in_ml_model=True,
    ),
    Crop(
        key="lentil", name="Lentil", category="Pulse", image="lentil.svg",
        description=(
            "A cool-season pulse grown largely on residual soil moisture after "
            "the monsoon. Lentil fixes nitrogen and suits low-input rotations."
        ),
        general=GeneralInfo(
            soil="Loam and clay loam",
            temperature="18 - 30 \u00b0C",
            water_requirement="Low",
            growing_period="100 - 130 days",
            season="Rabi",
        ),
        in_ml_model=True,
    ),
    Crop(
        key="blackgram", name="Black Gram", category="Pulse", image="black_gram.svg",
        description=(
            "A short-duration pulse (urad) widely grown as a rice-fallow crop in "
            "the Cauvery delta, sown into residual moisture after the paddy "
            "harvest."
        ),
        general=GeneralInfo(
            soil="Clay loam and well-drained loam",
            temperature="25 - 35 \u00b0C",
            water_requirement="Low",
            growing_period="70 - 90 days",
            season="Rabi and rice-fallow",
        ),
        aliases=("black_gram", "black gram", "urad"),
        in_ml_model=True, in_tn_dataset=True,
    ),
    Crop(
        key="mungbean", name="Green Gram", category="Pulse", image="green_gram.svg",
        description=(
            "A very short-duration pulse (moong) that fits between two main "
            "crops. It fixes nitrogen and improves the soil for the next crop."
        ),
        general=GeneralInfo(
            soil="Well-drained loam and sandy loam",
            temperature="25 - 35 \u00b0C",
            water_requirement="Low",
            growing_period="60 - 90 days",
            season="Kharif, Rabi and summer",
        ),
        aliases=("green_gram", "green gram", "greengram", "moong"),
        in_ml_model=True, in_tn_dataset=True,
    ),
    Crop(
        key="mothbeans", name="Moth Beans", category="Pulse", image="mothbeans.svg",
        description=(
            "A drought-hardy pulse grown mainly in arid north-west India. It "
            "covers the ground quickly and is often used to check soil erosion."
        ),
        general=GeneralInfo(
            soil="Sandy and light soils",
            temperature="25 - 37 \u00b0C",
            water_requirement="Low",
            growing_period="70 - 90 days",
            season="Kharif",
        ),
        aliases=("moth beans", "matki"),
        in_ml_model=True,
    ),
    Crop(
        key="pigeonpeas", name="Red Gram", category="Pulse", image="red_gram.svg",
        description=(
            "A long-duration pulse (tur / arhar) often intercropped with cereals "
            "and oilseeds. Its deep roots let it survive dry spells."
        ),
        general=GeneralInfo(
            soil="Well-drained loam; does not tolerate waterlogging",
            temperature="20 - 35 \u00b0C",
            water_requirement="Low",
            growing_period="150 - 180 days",
            season="Kharif",
        ),
        aliases=("red_gram", "red gram", "redgram", "tur", "arhar"),
        in_ml_model=True, in_tn_dataset=True,
    ),
    Crop(
        key="kidneybeans", name="Kidney Beans", category="Pulse", image="kidneybeans.svg",
        description=(
            "A cool-season bean (rajma) grown in the hills and in northern "
            "plains during winter. It is sensitive to both heat and frost."
        ),
        general=GeneralInfo(
            soil="Well-drained loam",
            temperature="15 - 25 \u00b0C",
            water_requirement="Moderate",
            growing_period="90 - 120 days",
            season="Rabi in the plains, Kharif in the hills",
        ),
        aliases=("kidney beans", "rajma"),
        in_ml_model=True,
    ),
    Crop(
        key="chickpea", name="Chickpea", category="Pulse", image="chickpea.svg",
        description=(
            "A rabi pulse (gram) grown largely on stored soil moisture. Chickpea "
            "prefers dry air and suffers in humid conditions."
        ),
        general=GeneralInfo(
            soil="Loam and clay loam",
            temperature="15 - 30 \u00b0C",
            water_requirement="Low",
            growing_period="95 - 120 days",
            season="Rabi",
        ),
        aliases=("gram", "bengal gram"),
        in_ml_model=True,
    ),
    Crop(
        key="coffee", name="Coffee", category="Plantation", image="coffee.svg",
        description=(
            "A shade-grown perennial of the Western Ghats. Coffee needs cool "
            "humid hill conditions and well-distributed rain."
        ),
        general=GeneralInfo(
            soil="Deep, well-drained, slightly acidic loam rich in organic matter",
            temperature="15 - 28 \u00b0C",
            water_requirement="High",
            growing_period="Perennial; bearing from about year 3 - 4",
            season=None,
        ),
        in_ml_model=True,
    ),

    # ---- Crops present only in the Tamil Nadu regional dataset -------------
    Crop(
        key="cholam", name="Cholam (Sorghum)", category="Millet", image="cholam.svg",
        description=(
            "Sorghum, a hardy dryland cereal used for both grain and fodder. It "
            "withstands dry spells better than rice or maize."
        ),
        general=GeneralInfo(
            soil="Black soil and well-drained loam",
            temperature="26 - 33 \u00b0C",
            water_requirement="Low",
            growing_period="100 - 120 days",
            season="Kharif and Rabi",
        ),
        aliases=("sorghum", "jowar"), in_tn_dataset=True,
    ),
    Crop(
        key="cumbu", name="Cumbu (Pearl Millet)", category="Millet", image="cumbu.svg",
        description=(
            "Pearl millet, the most drought-tolerant of the common cereals. It "
            "is grown on light soils where rice and maize would fail."
        ),
        general=GeneralInfo(
            soil="Sandy and light soils",
            temperature="25 - 35 \u00b0C",
            water_requirement="Low",
            growing_period="80 - 95 days",
            season="Kharif and summer",
        ),
        aliases=("pearl millet", "pearlmillet", "bajra"), in_tn_dataset=True,
    ),
    Crop(
        key="ragi", name="Ragi (Finger Millet)", category="Millet", image="ragi.svg",
        description=(
            "Finger millet, valued for its calcium content and for storing well. "
            "It is grown both rainfed and under irrigation."
        ),
        general=GeneralInfo(
            soil="Red loam and well-drained soils",
            temperature="20 - 30 \u00b0C",
            water_requirement="Low to moderate",
            growing_period="100 - 120 days",
            season="Kharif and Rabi",
        ),
        aliases=("finger millet", "fingermillet"), in_tn_dataset=True,
    ),
    Crop(
        key="horse_gram", name="Horse Gram", category="Pulse", image="horse_gram.svg",
        description=(
            "A tough, low-input pulse grown on poor and shallow soils, mostly "
            "rainfed and often as the last crop of the season."
        ),
        general=GeneralInfo(
            soil="Poor, shallow and red soils",
            temperature="20 - 30 \u00b0C",
            water_requirement="Low",
            growing_period="90 - 120 days",
            season="Rabi",
        ),
        aliases=("horsegram", "kollu"), in_tn_dataset=True,
    ),
    Crop(
        key="sugarcane", name="Sugarcane", category="Cash crop", image="sugarcane.svg",
        description=(
            "A long-duration irrigated cash crop supplying sugar and jaggery "
            "mills. It occupies the field for a year or more."
        ),
        general=GeneralInfo(
            soil="Deep loam and clay loam with good drainage",
            temperature="20 - 35 \u00b0C",
            water_requirement="High",
            growing_period="10 - 12 months",
            season="Planted December - February or June - July",
        ),
        in_tn_dataset=True,
    ),
    Crop(
        key="groundnut", name="Groundnut", category="Oilseed", image="groundnut.svg",
        description=(
            "The main oilseed of Tamil Nadu, grown both rainfed and irrigated. "
            "Pods develop underground, so loose, well-drained soil matters."
        ),
        general=GeneralInfo(
            soil="Sandy loam and red loam, loose and well-drained",
            temperature="25 - 35 \u00b0C",
            water_requirement="Low to moderate",
            growing_period="100 - 130 days",
            season="Kharif and Rabi",
        ),
        aliases=("peanut",), in_tn_dataset=True,
    ),
    Crop(
        key="gingelly", name="Gingelly (Sesame)", category="Oilseed", image="gingelly.svg",
        description=(
            "Sesame, a short-duration oilseed grown on light soils. It tolerates "
            "drought but not waterlogging."
        ),
        general=GeneralInfo(
            soil="Well-drained sandy loam",
            temperature="25 - 35 \u00b0C",
            water_requirement="Low",
            growing_period="80 - 95 days",
            season="Kharif, Rabi and summer",
        ),
        aliases=("sesame", "til", "ellu"), in_tn_dataset=True,
    ),
    Crop(
        key="castor", name="Castor", category="Oilseed", image="castor.svg",
        description=(
            "A non-edible oilseed used industrially. Castor is drought-hardy and "
            "is often grown as a border or intercrop."
        ),
        general=GeneralInfo(
            soil="Red sandy loam and well-drained soils",
            temperature="20 - 35 \u00b0C",
            water_requirement="Low",
            growing_period="140 - 180 days",
            season="Kharif",
        ),
        in_tn_dataset=True,
    ),
)


# --------------------------------------------------------------------------
# Lookup index. Matching is case-insensitive and ignores spaces, hyphens and
# underscores, so "Black Gram", "black_gram", "BLACKGRAM" all resolve.
# --------------------------------------------------------------------------

def normalise(name: str) -> str:
    """Reduce any crop label to a comparable form."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_INDEX: dict[str, Crop] = {}
for _crop in _CROPS:
    for _token in (_crop.key, _crop.name, *_crop.aliases):
        _INDEX[normalise(_token)] = _crop

UNKNOWN_CROP = Crop(
    key="_unknown", name="Unknown crop", category="Not available",
    image="_unknown.svg",
    description="This crop is not in the project's crop information registry.",
)


def get(name: str) -> Crop:
    """Resolve any label to a Crop, falling back to a safe unknown placeholder."""
    return _INDEX.get(normalise(name), UNKNOWN_CROP)


def is_known(name: str) -> bool:
    return normalise(name) in _INDEX


def all_crops() -> list[Crop]:
    return sorted(_CROPS, key=lambda c: c.name)


def ml_crops() -> list[Crop]:
    return sorted((c for c in _CROPS if c.in_ml_model), key=lambda c: c.name)


def regional_crops() -> list[Crop]:
    return sorted((c for c in _CROPS if c.in_tn_dataset), key=lambda c: c.name)
