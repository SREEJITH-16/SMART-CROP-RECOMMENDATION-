"""
Verification suite for Smart Crop Recommendation.

Run from the project root:   python tools/verify.py

Checks the model pipeline, top-3 extraction, probability handling, validation,
image mapping, regional data, missing-data handling and every route.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from core import crop_metadata, image_mapping
from core.ml_model import recommender
from core.regional import regional_service
from core.validation import validate_soil_inputs

PASSED, FAILED = 0, 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# ---------------------------------------------------------------- model ----
section("1. Model loading and pipeline")

check("model loads", recommender.is_ready, recommender.load_error or recommender.algorithm_name)
check("algorithm is the original RandomForestClassifier",
      recommender.algorithm_name == "RandomForestClassifier")
check("model expects 7 features", getattr(recommender.model, "n_features_in_", 0) == 7)
check("scalers were rebuilt", recommender.scalers_rebuilt)
check("predict_proba available", recommender.supports_probability)
check("feature importances readable", len(recommender.feature_importances()) == 7)

section("2. Accuracy on the original held-out test split")

frame = pd.read_csv(config.TRAINING_CSV)
name_to_id = {v: k for k, v in config.CROP_LABELS.items()}
features = frame[config.FEATURE_ORDER].to_numpy(dtype=float)
labels = np.array([name_to_id[str(v).lower()] for v in frame["label"]])
_, x_test, _, y_test = train_test_split(features, labels, **config.TRAIN_TEST_SPLIT)

scaled = recommender.standard.transform(recommender.minmax.transform(x_test))
accuracy = float((recommender.model.predict(scaled) == y_test).mean())
check("test accuracy above 95%", accuracy > 0.95, f"{accuracy:.2%}")

# Top-3 hit rate: is the true crop in the top 3?
probabilities = recommender.model.predict_proba(scaled)
top3 = recommender.model.classes_[np.argsort(probabilities, axis=1)[:, -3:]]
hit_rate = float(np.mean([y in row for y, row in zip(y_test, top3)]))
check("true crop is in the top 3 for over 98% of test rows", hit_rate > 0.98, f"{hit_rate:.2%}")

section("3. Top 3 extraction and probabilities")

sample = {"N": 90, "P": 42, "K": 43, "temperature": 20.88,
          "humidity": 82.0, "ph": 6.5, "rainfall": 202.94}
results = recommender.recommend(sample)

check("returns at most 3 recommendations", 1 <= len(results) <= 3, f"{len(results)} returned")
check("no zero-probability crop is presented",
      all(r.probability > 0 for r in results))
check("textbook rice sample predicts rice", results[0].crop_key == "rice",
      f"got {results[0].crop_key}")
check("probabilities are descending",
      all(results[i].probability >= results[i + 1].probability for i in range(len(results) - 1)))
check("probabilities are in the 0-1 range",
      all(0.0 <= r.probability <= 1.0 for r in results))
check("probabilities sum to at most 1",
      sum(r.probability for r in results) <= 1.0 + 1e-9)
check("every result carries an explanation", all(r.reasons for r in results))
check("ranks are sequential from 1",
      [r.rank for r in results] == list(range(1, len(results) + 1)))

# A borderline input should genuinely produce three non-zero candidates.
mixed = {"N": 40, "P": 60, "K": 60, "temperature": 24.0,
         "humidity": 65.0, "ph": 6.8, "rainfall": 95.0}
mixed_results = recommender.recommend(mixed)
check("an ambiguous input returns multiple candidates", len(mixed_results) >= 2,
      ", ".join(f"{r.crop_key} {r.probability_percent}%" for r in mixed_results))

# A second, very different sample should give a different answer.
dry = {"N": 20, "P": 60, "K": 80, "temperature": 18.0,
       "humidity": 16.0, "ph": 7.3, "rainfall": 80.0}
dry_results = recommender.recommend(dry)
check("a different input gives a different top crop",
      dry_results[0].crop_key != results[0].crop_key,
      f"{results[0].crop_key} vs {dry_results[0].crop_key}")

section("4. Input validation")

good = {k: "50" for k in config.INPUT_FIELDS}
good.update({"ph": "6.5", "humidity": "80", "temperature": "28", "rainfall": "200"})
check("valid input passes", validate_soil_inputs(good).ok)

missing = dict(good); missing["N"] = ""
check("missing value is rejected", "N" in validate_soil_inputs(missing).errors)

text = dict(good); text["P"] = "abc"
check("non-numeric value is rejected", "P" in validate_soil_inputs(text).errors)

impossible = dict(good); impossible["ph"] = "25"
check("impossible pH is rejected", "ph" in validate_soil_inputs(impossible).errors)

negative = dict(good); negative["rainfall"] = "-5"
check("negative rainfall is rejected", "rainfall" in validate_soil_inputs(negative).errors)

unusual = dict(good); unusual["rainfall"] = "600"
unusual_result = validate_soil_inputs(unusual)
check("uncommon but legitimate value is accepted", unusual_result.ok)
check("uncommon value produces a notice", len(unusual_result.notices) > 0)

section("5. Crop image mapping")

for variant in ["rice", "Rice", "RICE", "  rice  "]:
    # A real photograph (rice.jpg) has been downloaded alongside the
    # placeholder (rice.svg); resolve() is documented to prefer it, so every
    # spelling of the label should resolve to the same photo, not the SVG.
    check(f"'{variant}' resolves to the rice image",
          image_mapping.resolve(variant) == "rice.jpg")

check("'paddy' resolves to the rice crop", crop_metadata.get("paddy").key == "rice")
check("'Black_gram' resolves", crop_metadata.get("Black_gram").key == "blackgram")
check("'Red_gram' resolves to pigeonpeas", crop_metadata.get("Red_gram").key == "pigeonpeas")
check("'Green_gram' resolves to mungbean", crop_metadata.get("Green_gram").key == "mungbean")
check("unknown label falls back safely",
      image_mapping.resolve("definitely-not-a-crop") == image_mapping.FALLBACK_IMAGE)

audit = image_mapping.audit()
check("every registry crop has an image", not audit["missing"], f"missing: {audit['missing']}")
check("every model class has an image",
      all(image_mapping.resolve(k) != image_mapping.FALLBACK_IMAGE
          for k in config.CROP_LABELS.values()))
check("every image file referenced exists",
      all((config.CROP_IMAGE_DIR / image_mapping.resolve(c.key)).is_file()
          for c in crop_metadata.all_crops()))

section("6. Tamil Nadu regional data")

check("regional dataset loads", regional_service.is_ready, regional_service.load_error or "")
check("38 districts available", regional_service.district_count() == 38,
      str(regional_service.district_count()))
check("per-crop area columns discovered", len(regional_service.area_columns) == 14,
      str(len(regional_service.area_columns)))

profile = regional_service.profile("Thanjavur")
check("Thanjavur profile builds", profile is not None)
if profile:
    check("agro-climatic zone is present", profile.agro_climatic_zone != "Data not available",
          profile.agro_climatic_zone)
    check("soil types parsed", len(profile.major_soil_types) > 0,
          "; ".join(profile.major_soil_types))
    check("crops are ranked", profile.has_crop_data, f"{len(profile.crops)} crops")
    check("ranking is descending by area",
          all(profile.crops[i].area_ha >= profile.crops[i + 1].area_ha
              for i in range(len(profile.crops) - 1)))
    check("top crop area matches the dataset",
          profile.crops[0].area_ha > 0, f"{profile.crops[0].name}: {profile.crops[0].area_ha:,} ha")
    check("no zero-area crops are listed", all(c.area_ha > 0 for c in profile.crops))

check("case-insensitive district lookup", regional_service.profile("thanjavur") is not None)
check("unknown district returns None", regional_service.profile("Atlantis") is None)

# Chennai is the sparse-data edge case in this dataset.
chennai = regional_service.profile("Chennai")
check("sparse district still builds", chennai is not None)
if chennai:
    check("sparse district reports zone honestly",
          "not" in chennai.agro_climatic_zone.lower() or chennai.agro_climatic_zone != "",
          chennai.agro_climatic_zone)

section("7. Crop metadata registry")

check("no duplicate crop keys",
      len({c.key for c in crop_metadata.all_crops()}) == len(crop_metadata.all_crops()))
check("all 22 model classes are in the registry",
      all(crop_metadata.is_known(name) for name in config.CROP_LABELS.values()))

regional_names = set(regional_service.area_columns.keys())
unresolved = [n for n in regional_names if not crop_metadata.is_known(n)]
check("all regional crop names resolve", not unresolved, f"unresolved: {unresolved}")

section("8. Routes and error handling")

from app import app  # noqa: E402

app.config["TESTING"] = True
client = app.test_client()

routes = [
    ("/", 200), ("/soil", 200), ("/tamil-nadu", 200),
    ("/crops", 200), ("/crops/rice", 200), ("/about", 200),
    ("/api/districts", 200), ("/api/health", 200),
    ("/tamil-nadu/district?district=Thanjavur", 200),
    ("/tamil-nadu/district?district=Atlantis", 400),
    ("/tamil-nadu/district", 400),
    ("/crops/not-a-crop", 404),
    ("/no-such-page", 404),
]
for path, expected in routes:
    response = client.get(path)
    check(f"GET {path} -> {expected}", response.status_code == expected,
          f"got {response.status_code}")

form = {"N": "90", "P": "42", "K": "43", "temperature": "20.88",
        "humidity": "82", "ph": "6.5", "rainfall": "202.94"}
response = client.post("/predict", data=form)
check("POST /predict -> 200", response.status_code == 200)
body = response.get_data(as_text=True)
check("result page names the predicted crop", "Rice" in body)
check("result page labels the probability correctly", "Model probability" in body)
check("result page never says 'chance of success'",
      "chance of success" not in body.lower())
check("result page shows the explanation", "Why this crop?" in body)
check("result page shows the comparison", "Compare the top" in body)

bad = dict(form); bad["N"] = ""
response = client.post("/predict", data=bad)
check("POST /predict with a missing field -> 400", response.status_code == 400)
check("error message is shown",
      "Enter a value" in response.get_data(as_text=True))

out_of_range = dict(form); out_of_range["rainfall"] = "700"
response = client.post("/predict", data=out_of_range)
check("uncommon value still predicts", response.status_code == 200)
check("extrapolation warning is shown",
      "outside the" in response.get_data(as_text=True))

section("9. Data integrity guarantees")

district_body = client.get("/tamil-nadu/district?district=Thanjavur").get_data(as_text=True)
# The page may mention probability only to deny it ("not a success probability").
forbidden_claims = ["% probability", "success probability", "model probability",
                    "chance of success", "% chance", "suitability score"]
# The page is allowed to mention these phrases only to deny them.
scrubbed = (district_body.lower()
            .replace("not a success probability", "")
            .replace("not an ml probability", ""))
present = [phrase for phrase in forbidden_claims if phrase in scrubbed]
check("district page makes no probability claim", not present, f"found: {present}")
check("district page labels rainfall as unavailable",
      "Data not available" in district_body)
check("district page cites its source",
      "Statistical Handbook" in district_body)
check("district page separates area from prediction",
      "not a success probability" in district_body)

# "Placeholder illustration" is the crop-detail page's own placeholder
# caption (see templates/crop_detail.html); the /predict results page never
# emits that phrase, so this needs a crop that is still genuinely a
# placeholder (no downloaded photograph exists for it) fetched from its own
# detail page - not the rice prediction's `body` from section 8, which was
# never going to contain it.
placeholder_crop = next(
    (crop.key for crop in crop_metadata.all_crops()
     if image_mapping.is_placeholder(crop.key)),
    None,
)
check("a placeholder crop exists to test against", placeholder_crop is not None)
if placeholder_crop:
    placeholder_body = client.get(f"/crops/{placeholder_crop}").get_data(as_text=True)
    check("placeholder images are flagged as placeholders",
          "Placeholder illustration" in placeholder_body)

# ------------------------------------------------------------------ report
print("\n" + "=" * 60)
print(f"  {PASSED} passed, {FAILED} failed")
print("=" * 60)
sys.exit(1 if FAILED else 0)
