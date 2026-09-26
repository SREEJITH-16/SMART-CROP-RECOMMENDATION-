"""
ML layer for the soil-based crop recommendation.

This module loads the ORIGINAL trained RandomForestClassifier shipped with the
source project. The model itself is never retrained or replaced. Two defects
inherited from the source project are corrected at load time, both documented
below and both verifiable by running tools/verify_model.py.


DEFECT 1 - THE PICKLED SCALERS ARE UNUSABLE
-------------------------------------------
The training notebook saved its scalers *after* running this helper:

    def recommendation(N, P, K, ...):
        features = np.array([[N, P, K, ...]])
        mx_features = mx.fit_transform(features)      # refits on ONE row
        sc_mx_features = sc.fit_transform(mx_features)  # refits on ONE row

`fit_transform` re-fits the scaler. Calling it on a single sample discarded the
statistics learned from the training set and replaced them with that one row.
The pickles therefore contain:

    minmaxscaler.pkl : data_min_ == data_max_ == [90, 42, 43, 20.88, 82.0, 6.5, 202.94]
    standscaler.pkl  : mean_ = [0...0], scale_ = [1...1]   (an identity transform)

Measured effect on the original app.py pipeline, over the notebook's own
held-out test split:

    accuracy with the pickled scalers   :  9.6 %
    accuracy with correctly fitted ones : 98.9 %

The model was always fine; it was being fed wrongly scaled inputs. The fix is
to re-fit MinMaxScaler and StandardScaler on the same training rows the model
saw (same CSV, same split, same random_state), which restores the pipeline the
model was actually trained with. No model weights are touched.


DEFECT 2 - SCIKIT-LEARN VERSION INCOMPATIBILITY
-----------------------------------------------
model.pkl was pickled with scikit-learn 1.3.2. Version 1.4 added the
`monotonic_cst` attribute to tree estimators, which older pickles lack, so
`predict()` raises AttributeError on modern scikit-learn.

Rather than force an old pinned version (or retrain, which would replace the
model), the loader restores the attribute to its 1.3.2 default of None. This is
the documented default for "no monotonicity constraints" and leaves every
learned split threshold untouched, so predictions are bit-identical to those of
the original model under scikit-learn 1.3.2.


DEFECT 3 - TREE VALUE REPRESENTATION CHANGED IN SCIKIT-LEARN 1.4
----------------------------------------------------------------
Up to version 1.3, a classifier tree stored raw class counts in `tree_.value`
and `predict_proba` normalised them at prediction time. From 1.4 the array
stores class *fractions* and `predict_proba` no longer normalises.

Loading the 1.3.2 pickle into a newer scikit-learn therefore skips the
normalisation step and returns unnormalised counts. Measured on this model, the
"probabilities" for one sample summed to 45.63 instead of 1.0 - a number that
would have been displayed to the user as a percentage.

The loader converts each node's value row to fractions once at startup, which is
exactly the migration scikit-learn made internally. Ranking is unaffected for
the argmax, but the probabilities become correct and sum to 1.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import config

SCALER_BUG_NOTE = (
    "The scalers shipped with the source project were saved after being re-fitted "
    "on a single sample, which made them unusable. They are rebuilt here from the "
    "original training split. The trained model itself is unchanged."
)


class ModelUnavailableError(RuntimeError):
    """Raised when the model or its training data cannot be loaded."""


@dataclass
class CropPrediction:
    """One ranked crop suggestion returned by the model."""
    rank: int
    crop_key: str
    label_id: int
    probability: float | None          # None when the model cannot give probabilities
    reasons: list[str] = field(default_factory=list)
    feature_match: dict[str, dict] = field(default_factory=dict)

    @property
    def probability_percent(self) -> float | None:
        return None if self.probability is None else round(self.probability * 100, 1)


class CropRecommender:
    """Wraps the original RandomForest model and its preprocessing pipeline."""

    def __init__(self) -> None:
        self.model: Any = None
        self.minmax: MinMaxScaler | None = None
        self.standard: StandardScaler | None = None
        self.crop_stats: dict[str, dict[str, dict[str, float]]] = {}
        self.load_error: str | None = None
        self.compat_patch_applied = False
        self.scalers_rebuilt = False
        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        try:
            self._load_model()
            self._rebuild_scalers()
            self._compute_crop_statistics()
        except Exception as exc:                      # noqa: BLE001 - surfaced in UI
            self.load_error = str(exc)

    def _load_model(self) -> None:
        if not config.MODEL_PATH.exists():
            raise ModelUnavailableError(
                f"Trained model not found at {config.MODEL_PATH.name}."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # cross-version pickle warnings
            with open(config.MODEL_PATH, "rb") as fh:
                self.model = pickle.load(fh)

        self._apply_sklearn_compat_patch()

        if getattr(self.model, "n_features_in_", None) != len(config.FEATURE_ORDER):
            raise ModelUnavailableError(
                f"Model expects {self.model.n_features_in_} features but "
                f"{len(config.FEATURE_ORDER)} are configured."
            )

    def _apply_sklearn_compat_patch(self) -> None:
        """Apply the two version fixes (DEFECT 2 and DEFECT 3)."""
        if not hasattr(self.model, "monotonic_cst"):
            self.model.monotonic_cst = None
            self.compat_patch_applied = True

        for estimator in getattr(self.model, "estimators_", []):
            if not hasattr(estimator, "monotonic_cst"):
                estimator.monotonic_cst = None
                self.compat_patch_applied = True
            self._normalise_tree_values(estimator)

    def _normalise_tree_values(self, estimator) -> None:
        """
        Convert `tree_.value` from raw class counts to class fractions.

        Detected rather than assumed: if the root node already sums to 1 the
        pickle is in the modern format and is left alone, so this is safe to run
        against models saved by any version.
        """
        tree = getattr(estimator, "tree_", None)
        if tree is None:
            return
        values = tree.value
        if values.size == 0:
            return

        root_total = float(values[0].sum())
        if abs(root_total - 1.0) < 1e-6:
            return                                    # already normalised

        totals = values.sum(axis=2, keepdims=True)
        np.divide(values, np.where(totals == 0, 1.0, totals), out=values)
        self.compat_patch_applied = True

    def _rebuild_scalers(self) -> None:
        """Re-fit the scalers on the exact training rows the model learned from."""
        frame = self._training_frame()
        features = frame[config.FEATURE_ORDER].to_numpy(dtype=float)
        labels = frame["label"].to_numpy()

        x_train, _, _, _ = train_test_split(
            features, labels, **config.TRAIN_TEST_SPLIT
        )

        self.minmax = MinMaxScaler().fit(x_train)
        self.standard = StandardScaler().fit(self.minmax.transform(x_train))
        self.scalers_rebuilt = True

    def _training_frame(self) -> pd.DataFrame:
        if not config.TRAINING_CSV.exists():
            raise ModelUnavailableError(
                f"Training dataset not found at {config.TRAINING_CSV.name}."
            )
        frame = pd.read_csv(config.TRAINING_CSV)
        missing = set(config.FEATURE_ORDER + ["label"]) - set(frame.columns)
        if missing:
            raise ModelUnavailableError(
                f"Training dataset is missing columns: {sorted(missing)}"
            )
        return frame

    def _compute_crop_statistics(self) -> None:
        """
        Observed min / max / mean of every feature per crop, read directly from
        the training CSV. These power the "Why this crop?" explanations, so the
        explanations describe real data rather than invented agronomy.
        """
        frame = self._training_frame()
        for crop_name, group in frame.groupby("label"):
            self.crop_stats[str(crop_name).lower()] = {
                feature: {
                    "min": float(group[feature].min()),
                    "max": float(group[feature].max()),
                    "mean": float(group[feature].mean()),
                }
                for feature in config.FEATURE_ORDER
            }

    # --------------------------------------------------------------- status

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.load_error is None

    @property
    def supports_probability(self) -> bool:
        return self.is_ready and hasattr(self.model, "predict_proba")

    @property
    def algorithm_name(self) -> str:
        return type(self.model).__name__ if self.model is not None else "unavailable"

    def feature_importances(self) -> dict[str, float]:
        """Real `feature_importances_` from the trained forest, as percentages."""
        if not self.is_ready or not hasattr(self.model, "feature_importances_"):
            return {}
        values = self.model.feature_importances_
        return {
            feature: round(float(value) * 100, 1)
            for feature, value in zip(config.FEATURE_ORDER, values)
        }

    # ------------------------------------------------------------- predict

    def recommend(self, values: dict[str, float], top_n: int | None = None) -> list[CropPrediction]:
        """
        Run the full pipeline: ordered features -> MinMax -> Standard -> model.

        Returns the top N crops. When the model exposes `predict_proba`, ranking
        and probabilities come from the model's own class probabilities. There
        is no post-processing, smoothing or invented confidence anywhere.
        """
        if not self.is_ready:
            raise ModelUnavailableError(self.load_error or "Model is not loaded.")

        top_n = top_n or config.TOP_N_RECOMMENDATIONS
        ordered = np.array([[float(values[f]) for f in config.FEATURE_ORDER]])

        scaled = self.standard.transform(self.minmax.transform(ordered))

        if self.supports_probability:
            probabilities = self.model.predict_proba(scaled)[0]
            order = np.argsort(probabilities)[::-1][:top_n]
            picks = [
                (int(self.model.classes_[i]), float(probabilities[i])) for i in order
            ]
            # Drop crops no tree voted for. When the forest is unanimous, the
            # remaining 21 classes all sit at exactly 0.0 and argsort orders
            # them arbitrarily - presenting those as the "second" and "third"
            # choice would invent a ranking the model never produced. Fewer
            # than three results is the honest outcome, and the UI says so.
            picks = [pick for pick in picks if pick[1] > 0.0]
        else:
            # Fall back to a single hard prediction, clearly flagged as a ranking.
            label_id = int(self.model.predict(scaled)[0])
            picks = [(label_id, None)]

        results: list[CropPrediction] = []
        for rank, (label_id, probability) in enumerate(picks, start=1):
            crop_key = config.CROP_LABELS.get(label_id)
            if crop_key is None:
                continue                               # unknown label: skip, never guess
            match = self._feature_match(crop_key, values)
            results.append(
                CropPrediction(
                    rank=rank,
                    crop_key=crop_key,
                    label_id=label_id,
                    probability=probability,
                    reasons=self._build_reasons(match),
                    feature_match=match,
                )
            )
        return results

    # --------------------------------------------------------- explanation

    def _feature_match(self, crop_key: str, values: dict[str, float]) -> dict[str, dict]:
        """
        Compare each submitted value against the observed range of that crop in
        the training data. Purely descriptive - it reports where the input sits
        relative to the data the model learned from.
        """
        stats = self.crop_stats.get(crop_key, {})
        match: dict[str, dict] = {}
        for feature in config.FEATURE_ORDER:
            bounds = stats.get(feature)
            supplied = values.get(feature)
            if bounds is None or supplied is None:
                match[feature] = {"status": "unknown", "observed": None, "value": supplied}
                continue
            low, high = bounds["min"], bounds["max"]
            span = max(high - low, 1e-9)
            if low <= supplied <= high:
                status = "inside"
            elif abs(supplied - low) <= 0.15 * span or abs(supplied - high) <= 0.15 * span:
                status = "near"
            else:
                status = "outside"
            match[feature] = {
                "status": status,
                "value": supplied,
                "observed": (round(low, 1), round(high, 1)),
                "observed_mean": round(bounds["mean"], 1),
            }
        return match

    @staticmethod
    def _build_reasons(match: dict[str, dict]) -> list[str]:
        """
        Turn the range comparison into short, literally true statements. Each one
        restates a measurable fact; none asserts agronomic causation.
        """
        readable = {
            "N": "nitrogen", "P": "phosphorus", "K": "potassium",
            "temperature": "temperature", "humidity": "humidity",
            "ph": "soil pH", "rainfall": "rainfall",
        }
        inside = [readable[f] for f, m in match.items() if m["status"] == "inside"]
        near = [readable[f] for f, m in match.items() if m["status"] == "near"]
        outside = [readable[f] for f, m in match.items() if m["status"] == "outside"]

        reasons: list[str] = []
        if inside:
            reasons.append(
                f"Your {_join(inside)} fall inside the range recorded for this crop "
                "in the training data."
            )
        if near:
            reasons.append(f"Your {_join(near)} sit just outside that recorded range.")
        if outside:
            reasons.append(
                f"Your {_join(outside)} fall outside the recorded range, so the model "
                "is extrapolating for {} of the seven inputs.".format(len(outside))
            )
        if not reasons:
            reasons.append(
                "No reference range is available for this crop, so the ranking "
                "reflects the model output only."
            )
        return reasons


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


# Single shared instance - the model is loaded once per process, not per request.
recommender = CropRecommender()
