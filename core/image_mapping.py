"""
Crop -> image resolution.

Crop imagery uses a curated Wikimedia Commons source map with a local-asset
fallback. Real crop photographs are preferred when available; the source map
provides stable Wikimedia Commons file redirects for development/deployment.

Resolution order for any incoming label:
  1. the crop registry (handles case, spacing, underscores and aliases)
  2. a direct filename match in static/crops/
  3. the shared "_unknown.svg" fallback

IMPORTANT - the images shipped with this project are PLACEHOLDER SVG
illustrations supplied in Smart_Crop_Recommendation_Crop_Images.zip, plus
matching placeholders generated in the same style for crops the pack did not
cover. They are line drawings, not photographs of the crops. `is_placeholder`
reports this so the UI can say so honestly. Dropping real licensed photographs
into static/crops/ with the same base filename is all that is needed to
upgrade; no code changes are required.
"""

from __future__ import annotations

from functools import lru_cache

import config
from core import crop_metadata

FALLBACK_IMAGE = "_unknown.svg"

# Extensions checked when a photograph replaces a placeholder, in priority order.
_PHOTO_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png")


@lru_cache(maxsize=256)
def resolve(label: str) -> str:
    """
    Return the filename inside static/crops/ to use for this crop label.

    BUG FIX: this used to build one candidate list - [photo stems from
    crop.image] + [the placeholder SVG itself] + [photo stems from the
    incoming label] - and return the first name that existed on disk. That
    put the placeholder ahead of the incoming-label photo stems, so any crop
    whose downloaded photograph was not filed under the exact same stem as
    crop.image (e.g. "black_gram.svg" vs the downloaded "blackgram.jpg", or
    "cholam.svg" vs the downloaded "sorghum.jpg" for a Tamil Nadu alias) fell
    back to its placeholder illustration even though a real photograph was
    sitting right next to it. Every stem a photograph might be filed under -
    the registry's own image stem, the incoming label, and every alias the
    crop is known by - is now checked before any placeholder is considered.
    """
    crop = crop_metadata.get(label)
    slug = crop_metadata.normalise(label)

    stems = [slug]
    placeholder: str | None = None
    if crop.key != "_unknown":
        stems.append(crop.image.rsplit(".", 1)[0])
        stems.extend(crop_metadata.normalise(alias) for alias in crop.aliases)
        placeholder = crop.image

    seen: set[str] = set()
    photo_candidates: list[str] = []
    for stem in stems:
        if stem in seen:
            continue
        seen.add(stem)
        photo_candidates += [stem + ext for ext in _PHOTO_EXTENSIONS]

    # A real, downloaded photograph always wins over the placeholder drawing.
    for filename in photo_candidates:
        if (config.CROP_IMAGE_DIR / filename).is_file():
            return filename

    # Then the registry's own placeholder SVG...
    if placeholder and (config.CROP_IMAGE_DIR / placeholder).is_file():
        return placeholder

    # ...then a placeholder named directly after the incoming label, e.g. "paddy.svg".
    direct_svg = slug + ".svg"
    if (config.CROP_IMAGE_DIR / direct_svg).is_file():
        return direct_svg

    return FALLBACK_IMAGE


@lru_cache(maxsize=256)
def is_placeholder(label: str) -> bool:
    """True when the resolved asset is still a placeholder drawing."""
    return resolve(label).lower().endswith(".svg")


def alt_text(label: str) -> str:
    """Accessible alt text that does not claim to be a photograph."""
    crop = crop_metadata.get(label)
    name = crop.name if crop.key != "_unknown" else str(label).title()
    if is_placeholder(label):
        return f"Placeholder illustration representing {name}"
    return f"Photograph of {name}"


def audit() -> dict[str, list[str]]:
    """
    Report which registry crops have no asset at all. Used by tools/verify.py
    and surfaced on the About page.
    """
    missing, placeholders, photographs = [], [], []
    for crop in crop_metadata.all_crops():
        filename = resolve(crop.key)
        if filename == FALLBACK_IMAGE:
            missing.append(crop.name)
        elif filename.lower().endswith(".svg"):
            placeholders.append(crop.name)
        else:
            photographs.append(crop.name)
    return {
        "missing": missing,
        "placeholders": placeholders,
        "photographs": photographs,
    }


# NOTE: this module used to also carry a REAL_PHOTO_SOURCES table of external
# Wikimedia Commons hotlink URLs, used in preference to the photographs
# tools/download_crop_images.py already downloads into static/crops/. Several
# of those URLs were malformed (stray encoded punctuation, even an emoji, in
# the filename) and 404'd, so the results page loaded a broken image on every
# view before its onerror handler swapped in the local file. The local
# photograph is reliable, requires no network call at request time, and is
# already what resolve() picks when one exists - so the external table and
# the functions that read it were removed rather than fixed. Real photographs
# still upgrade a crop automatically: drop a file into static/crops/ with the
# crop's base filename (see the module docstring above).
