# Smart Crop Recommendation

A web application that suggests crops in two clearly separated ways: a machine
learning prediction from soil and weather readings, and a lookup of published
district-level agricultural statistics for Tamil Nadu.

The two systems are never mixed, and every result on screen says where it came
from.

---

## Problem statement

Choosing what to sow is a decision made under uncertainty. Soil test reports
give farmers numbers (N, P, K, pH) but little guidance on what those numbers
imply. Separately, district agricultural statistics record what has actually
been grown, but sit in PDF handbooks that are hard to query.

This project makes both accessible, while being explicit that they answer
different questions and that neither is a substitute for agronomic advice.

## Objectives

1. Take the existing trained crop model and make it work correctly (see
   [Defects fixed](#defects-fixed-in-the-original-project) — it did not).
2. Show the top three crops with the model's own probabilities, not invented
   confidence figures.
3. Explain each recommendation against data that actually exists.
4. Add a Tamil Nadu regional profile driven by official statistics, labelled as
   statistics rather than prediction.
5. Never display a number the underlying data does not support.

## Features

- **Soil-based AI prediction** — seven inputs, top three crops, real model
  probabilities, per-crop explanation of how your readings compare with the
  training data.
- **Tamil Nadu regional profile** — 38 districts with agro-climatic zone, major
  soil types, and crops ranked by recorded cultivated area.
- **Crop guide** — one shared crop registry used by both features.
- **Methods and limitations page** — live system status, real feature
  importances, and a plain list of what the application cannot do.
- **Explicit missing-data handling** — "Data not available" instead of a guess.

## Technology stack

| Layer | Choice |
|---|---|
| Backend | Flask 3 |
| ML | scikit-learn (RandomForestClassifier, 100 trees) |
| Data | pandas, NumPy |
| Frontend | Jinja2 templates, hand-written CSS, vanilla JS |
| Serving | gunicorn |

No CSS framework and no external CDN, so the interface loads without network
access and does not have a generic Bootstrap appearance.

---

## The ML model

**Algorithm:** `RandomForestClassifier`, 100 trees — the original model from the
source project, loaded from `model/model.pkl`. It is **not** retrained here.

> Note for anyone reading the original brief: the model is a random forest, not
> a single decision tree. A random forest is an ensemble of decision trees, and
> the displayed probability is the share of those trees voting for a crop.

**Training data:** `data/Crop_recommendation_training_22crop.csv` — 2,200
samples, 22 crops, 100 samples each.

**Input features, in this exact order:**

| # | Feature | Unit |
|---|---|---|
| 1 | Nitrogen (N) | kg/ha |
| 2 | Phosphorus (P) | kg/ha |
| 3 | Potassium (K) | kg/ha |
| 4 | Temperature | °C |
| 5 | Humidity | % |
| 6 | Soil pH | — |
| 7 | Rainfall | mm |

The order was verified against the training notebook (`X = crop.drop('label',
axis=1)`) and the CSV column order, not assumed.

**Measured accuracy:** 99.3% on the notebook's own held-out test split
(`test_size=0.2, random_state=42`). The true crop appears in the top three for
100% of test rows. See [Limitations](#limitations) before quoting that figure.

### How a prediction is made

```
user input
   ↓  validation (type, range, physical plausibility)
   ↓  ordered as [N, P, K, temperature, humidity, ph, rainfall]
   ↓  MinMaxScaler
   ↓  StandardScaler
   ↓  RandomForestClassifier.predict_proba()
   ↓  sort 22 class probabilities, take the highest 3
top 3 recommendations
```

Both scalers are fitted on the same training rows the model learned from, using
the same split and seed.

### How the top 3 and the probabilities are calculated

`predict_proba()` returns one probability per class. Those values are sorted
descending and the first three are taken. Nothing is smoothed, rescaled or
padded. The displayed percentage is the model's raw output, rounded to one
decimal place.

If a model without `predict_proba` were substituted, the application detects
this, shows a ranking instead, and tells the user no probability is available.

### How "Why this crop?" works

For each of the seven inputs, the submitted value is compared against the
minimum and maximum recorded for that crop **in the training CSV**, computed at
startup. Each input is reported as inside, near (within 15% of the span) or
outside that range.

This is descriptive, not agronomic. It reports where your reading sits relative
to the data the model learned from. No causal claim is made, and no explanation
text is generated about soil chemistry the model has no knowledge of.

The results page and the About page additionally show the forest's real
`feature_importances_`, labelled as a global property of the model rather than
an explanation of one prediction.

---

## Tamil Nadu regional recommendation system

This is a **statistics lookup with no machine learning in it.**

```
district selection
   ↓  row lookup in Tamil_Nadu_Crop_Regional_Dataset.csv
   ↓  agro-climatic zone + major soil types (as recorded)
   ↓  14 crop-area columns, sorted by hectares, zero values dropped
regional crop profile
```

### Columns used

Discovered from the file at load time rather than hard-coded:

- `District`
- `Agro_Climatic_Zone_2021_22`
- `Major_Soil_Types_2021_22` (semicolon-separated)
- `Data_Year`
- 14 × `<Crop>_Area_ha_2021_22` — Paddy, Cholam, Cumbu, Ragi, Maize, Red gram,
  Green gram, Black gram, Horse gram, Sugarcane, Cotton, Groundnut, Gingelly,
  Castor
- `Primary_Source`, `Source_Notes`

The `Top_1..5_Crop` columns are present in the file but are **not** used — the
ranking is recomputed from the area columns so it stays correct if the crop list
changes.

### Why area is not a probability

A crop leads a district because more hectares were under it. That reflects
irrigation, markets, policy and history. It is not a prediction that the crop
will do well on any given field, and the interface says so on every district
page.

---

## Datasets

| File | Rows | Used for |
|---|---|---|
| `data/Crop_recommendation_training_22crop.csv` | 2,200 | The model's training data. Powers prediction and explanations. |
| `data/Crop_recommendation_extended_35crop.csv` | 3,700 | Shipped for reference only. **Not used** — see below. |
| `data/Tamil_Nadu_Crop_Regional_Dataset.csv` | 38 | The regional feature. |

Regional source: Government of Tamil Nadu, Statistical Handbook 2021-22,
Agriculture tables 4.1 and 4.4.

### Why the extended dataset is not used

The 3,700-row file adds 13 Tamil Nadu staples, which would be a genuine
improvement — except its rainfall column is on two incompatible scales:

- the original 22 crops: **20–299 mm** (per growing season)
- the added crops: **250–2,492 mm** (apparently annual)

A single threshold at 300 mm separates the two blocks with 93.9% accuracy. The
clearest evidence is `greengram` and `redgram`, which straddle both: they contain
the original mungbean/pigeonpeas rows *and* new rows at 400–1,000 mm. The same
crop, in two different units.

A model trained on that file would substantially learn which source a row came
from rather than agronomy. Reconciling rainfall to one definition is the
prerequisite for using it.

---

## Defects fixed in the original project

The inherited code had three faults. All are corrected at load time in
`core/ml_model.py`; the trained model's parameters are untouched.

### 1. The saved scalers were unusable

The training notebook called `fit_transform` on a single sample before pickling
the scalers, which re-fitted them on that one row:

```python
mx_features = mx.fit_transform(features)        # refits on ONE row
sc_mx_features = sc.fit_transform(mx_features)  # refits on ONE row
```

The shipped pickles therefore contained `data_min_ == data_max_` and an identity
StandardScaler.

| Pipeline | Accuracy on the held-out split |
|---|---|
| Original `app.py` as shipped | **9.6%** |
| Same model, correctly fitted scalers | **99.3%** |

The textbook rice sample returned *Mango*. The model was always fine; it was
receiving wrongly scaled input. Fixed by re-fitting both scalers on the original
training split. The originals are preserved in `model/legacy/` for comparison.

### 2. `monotonic_cst` missing

`model.pkl` was pickled with scikit-learn 1.3.2. Version 1.4 added
`monotonic_cst` to tree estimators, so `predict()` raises `AttributeError` on
newer versions. Restored to its 1.3.2 default of `None`.

### 3. Tree values stored as counts, not fractions

Up to 1.3, classifier trees stored raw class counts in `tree_.value` and
`predict_proba` normalised them at prediction time. From 1.4 the array holds
fractions and no normalisation happens.

Loading the old pickle into new scikit-learn skipped that step: probabilities for
one sample summed to **45.63 instead of 1.0**, and would have been displayed as a
percentage. The loader converts each node's values to fractions once at startup —
the same migration scikit-learn made internally. It detects the format rather
than assuming it, so it is safe against any version.

---

## Project architecture

```
smart_crop_recommendation/
├── app.py                  Flask routes only
├── config.py               paths, feature order, labels, validation bounds
├── wsgi.py                 production entry point
├── Procfile                gunicorn command
├── requirements.txt
├── README.md
│
├── core/
│   ├── ml_model.py         model loading, compat fixes, top-N prediction
│   ├── regional.py         Tamil Nadu statistics layer
│   ├── crop_metadata.py    single crop registry (names, aliases, info)
│   ├── image_mapping.py    crop → image resolution with fallback
│   └── validation.py       form validation
│
├── model/
│   ├── model.pkl           original trained forest, unmodified
│   └── legacy/             original broken scalers + training notebook
│
├── data/                   three CSVs (see Datasets)
│
├── static/
│   ├── css/styles.css
│   ├── js/main.js
│   ├── crops/              placeholder SVGs + manifest
│   └── images/
│
├── templates/              base + 8 pages
└── tools/verify.py         80-check verification suite
```

Crop facts live only in `core/crop_metadata.py`. Templates read them through
context processors, so nothing is duplicated across HTML files.

### Architecture for the professor

```
USER INPUT
   ↓
SOIL + ENVIRONMENTAL DATA (N, P, K, pH, temperature, humidity, rainfall)
   ↓
RANDOM FOREST ML MODEL (an ensemble of decision trees)
   ↓
TOP 3 CROP PREDICTIONS  (from predict_proba)
```

separately, and never combined:

```
DISTRICT
   ↓
TAMIL NADU AGRICULTURAL DATA (Statistical Handbook 2021-22)
   ↓
AGRO-CLIMATIC ZONE + SOIL TYPES + CROP AREA
   ↓
REGIONAL CROP PROFILE  (ranked by hectares)
```

---

## Installation

Requires Python 3.9 or newer.

```bash
git clone <your-repository-url>
cd smart_crop_recommendation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## How to run

```bash
python app.py
```

Open <http://localhost:5000>.

For development reloading:

```bash
FLASK_DEBUG=1 python app.py      # Windows: set FLASK_DEBUG=1 && python app.py
```

Debug mode is off unless `FLASK_DEBUG=1` is set, so a production host never runs
in debug.

## How to test

```bash
python tools/verify.py
```

80 checks across model loading, accuracy, top-3 extraction, probability
validity, input validation, image mapping, district lookup, regional ranking,
missing-data handling, every route, and the data-integrity guarantees (for
example, that the district page never claims a probability). Exits non-zero on
any failure.

## How to deploy

All paths are resolved relative to `config.py`, so no machine-specific paths
appear anywhere.

**Any host with gunicorn** (Render, Railway, Heroku, Fly):

```bash
gunicorn wsgi:application --bind 0.0.0.0:$PORT --workers 2
```

`Procfile` and `runtime.txt` are included. Commit `model/` and `data/` — the
application needs both at runtime. `PORT` is read from the environment.

**PythonAnywhere:** point the WSGI file at `wsgi.py` and set the static mapping
`/static/` → `<project>/static`.

Check `/api/health` after deploying; it reports whether the model and regional
dataset loaded.

---

## Limitations

These are stated in the application itself, on the Methods page.

- **99.3% is not field accuracy.** Each crop occupies a tight, well-separated
  band in the training data, which is why the score is so high. Real fields are
  messier and field accuracy will be lower.
- **Probability is not success.** It is the share of trees voting for a crop. It
  has not been calibrated against harvest outcomes.
- **The model cannot suggest Tamil Nadu staples.** Groundnut, cholam, cumbu,
  ragi, sugarcane and gingelly are not among its 22 classes, however suitable
  your conditions are.
- **The two features do not interact.** Selecting a district changes nothing
  about the prediction, and vice versa.
- **Cultivated area is history, not advice.**
- **No district rainfall.** Not supplied, so it is shown as "Data not
  available".
- **Crop images are placeholders.** They are illustrations from the supplied
  asset pack, plus matching ones generated in the same style for crops the pack
  did not cover. They are labelled "Placeholder illustration" wherever shown.
  Dropping licensed photographs into `static/crops/` with the same base filename
  replaces them — `core/image_mapping.py` prefers `.webp`, `.jpg`, `.jpeg` and
  `.png` over `.svg` automatically, with no code change.
- **No map.** No district boundary geometry was supplied, so a searchable
  dropdown is used rather than an approximate and misleading map.
- **Not agronomic advice.** Consult your agricultural extension officer.

## Future enhancements

The brief describes a combined mode using district, soil type, NPK, pH, weather,
season and irrigation together. **That mode does not exist**, and the application
never claims otherwise. It would need training data linking all of those to real
outcomes, and no such dataset was supplied.

The code is organised to accept it later:

1. Reconcile the rainfall units in the extended dataset, then retrain to cover
   Tamil Nadu staples.
2. Add a `core/combined.py` alongside the existing layers; `core/regional.py`
   already exposes district soil and zone, and `/api/districts` is live.
3. Calibrate probabilities against real outcome data before presenting them as
   anything more than model agreement.
4. Replace placeholder images with licensed photographs.
5. Add district boundary geometry for a real map.

## Credits

Built on the original *Crop_Recommendation* project: its trained
RandomForestClassifier and 22-crop dataset are preserved here unchanged.
Regional data from the Government of Tamil Nadu Statistical Handbook 2021-22.

## Real crop photographs

The recommendation cards now use a curated real-photo source map from Wikimedia Commons for all 22 ML crops and additional regional crops. The browser loads the actual crop photograph via a Wikimedia Commons file redirect, with the existing local SVG retained only as an offline fallback.

To make local JPG copies inside `static/crops/`, run:

```bash
python tools/download_crop_images.py
```

Before public redistribution, review the current license/attribution on each Wikimedia Commons source listed in `crop_photo_sources.csv`.
