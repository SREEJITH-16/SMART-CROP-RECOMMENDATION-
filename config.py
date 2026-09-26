"""
Central configuration for Smart Crop Recommendation.

All paths are resolved relative to this file so the project runs unchanged on
Windows, macOS, Linux and any PaaS host. No absolute machine paths anywhere.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"
CROP_IMAGE_DIR = STATIC_DIR / "crops"

# --- Model artefacts -------------------------------------------------------
# model.pkl is the ORIGINAL RandomForestClassifier from the source project.
# It is loaded as-is and never retrained. See core/ml_model.py for the two
# compatibility fixes applied at load time.
MODEL_PATH = MODEL_DIR / "model.pkl"

# The scalers shipped with the original project are unusable (see
# core/ml_model.py -> SCALER_BUG_NOTE). They are kept in model/legacy/ for
# reference only and are rebuilt from the training CSV at startup.
LEGACY_MINMAX_PATH = MODEL_DIR / "legacy" / "minmaxscaler.pkl"
LEGACY_STANDARD_PATH = MODEL_DIR / "legacy" / "standscaler.pkl"

# --- Datasets --------------------------------------------------------------
# The exact CSV the shipped model was trained on (2200 rows, 22 crops).
TRAINING_CSV = DATA_DIR / "Crop_recommendation_training_22crop.csv"

# A larger community/extended variant (3700 rows, 35 crops) supplied alongside
# the project. NOT used for prediction - see README "Data limitations".
EXTENDED_CSV = DATA_DIR / "Crop_recommendation_extended_35crop.csv"

# Tamil Nadu district-level agricultural statistics.
TAMIL_NADU_CSV = DATA_DIR / "Tamil_Nadu_Crop_Regional_Dataset.csv"

# --- Model contract --------------------------------------------------------
# Feature order verified against the training notebook (X = crop.drop('label'))
# and against the column order of the training CSV. Do not reorder.
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Train/test split used by the original notebook. Re-used so the rebuilt
# scalers are fitted on exactly the same rows the model was trained on.
TRAIN_TEST_SPLIT = {"test_size": 0.2, "random_state": 42}

# Integer label -> crop name, taken verbatim from the training notebook's
# crop_dict. The model predicts these integers.
CROP_LABELS = {
    1: "rice", 2: "maize", 3: "jute", 4: "cotton", 5: "coconut",
    6: "papaya", 7: "orange", 8: "apple", 9: "muskmelon", 10: "watermelon",
    11: "grapes", 12: "mango", 13: "banana", 14: "pomegranate", 15: "lentil",
    16: "blackgram", 17: "mungbean", 18: "mothbeans", 19: "pigeonpeas",
    20: "kidneybeans", 21: "chickpea", 22: "coffee",
}

TOP_N_RECOMMENDATIONS = 3

# --- Input validation bounds ----------------------------------------------
# Bounds are derived from the observed range of the training data, widened
# generously so that legitimate but uncommon field values are not rejected.
# See core/validation.py for how these are used.
INPUT_FIELDS = {
    "N":           {"label": "Nitrogen (N)",    "unit": "kg/ha", "min": 0,    "max": 300,  "step": "any", "placeholder": "e.g. 90"},
    "P":           {"label": "Phosphorus (P)",  "unit": "kg/ha", "min": 0,    "max": 300,  "step": "any", "placeholder": "e.g. 42"},
    "K":           {"label": "Potassium (K)",   "unit": "kg/ha", "min": 0,    "max": 400,  "step": "any", "placeholder": "e.g. 43"},
    "ph":          {"label": "Soil pH",         "unit": "",      "min": 0,    "max": 14,   "step": "any", "placeholder": "e.g. 6.5"},
    "temperature": {"label": "Temperature",     "unit": "\u00b0C", "min": -10, "max": 60,  "step": "any", "placeholder": "e.g. 28"},
    "humidity":    {"label": "Humidity",        "unit": "%",     "min": 0,    "max": 100,  "step": "any", "placeholder": "e.g. 80"},
    "rainfall":    {"label": "Rainfall",        "unit": "mm",    "min": 0,    "max": 1200, "step": "any", "placeholder": "e.g. 200"},
}
