"""
Smart Crop Recommendation - Flask application.

Two independent features are served side by side and are never mixed:

  1. Soil-based AI prediction
     user input -> validation -> MinMax + Standard scaling -> RandomForest
     -> class probabilities -> top 3 crops

  2. Tamil Nadu regional profile
     district -> official 2021-22 statistics -> agro-climatic zone, soils and
     crops ranked by recorded cultivated area

Feature 2 contains no machine learning. The interface labels each result with
where it came from.
"""

from __future__ import annotations

import os

from flask import (
    Flask, abort, jsonify, redirect, render_template, request, url_for,
)

import config
from core import crop_metadata, image_mapping
from core.ml_model import ModelUnavailableError, recommender
from core.regional import regional_service
from core.validation import validate_district, validate_soil_inputs

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Long cache for fingerprint-free static assets; harmless in development.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 60 * 60 * 24 * 7


# --------------------------------------------------------------------------
# Template helpers - every template resolves crop data through these, so crop
# facts live in one place only.
# --------------------------------------------------------------------------

@app.context_processor
def inject_helpers():
    return {
        "crop_info": crop_metadata.get,
        "crop_image": image_mapping.resolve,
        "crop_image_alt": image_mapping.alt_text,
        "crop_image_is_placeholder": image_mapping.is_placeholder,
        "model_ready": recommender.is_ready,
        "regional_ready": regional_service.is_ready,
        "current_year": 2026,
    }


@app.template_filter("hectares")
def format_hectares(value) -> str:
    """Format an area, or say plainly when it is missing."""
    try:
        return f"{int(value):,} ha"
    except (TypeError, ValueError):
        return "Data not available"


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


@app.template_global()
def crop_image_url(label):
    """
    Return the crop image to display.

    BUG FIX: this previously returned an external Wikimedia "Special:Redirect"
    hotlink whenever one was curated, even though a real photograph had
    already been downloaded to static/crops/ by tools/download_crop_images.py.
    Several curated URLs are also malformed (stray punctuation, even an emoji,
    baked into the encoded filename) and 404, so the results page - the app's
    main feature - was loading a broken image from a third-party site on
    every view before the onerror handler swapped it out. Worse, the branch
    that ran when a crop had NO curated source called an undefined name
    (`crop_image`, which only exists as a Jinja global, not a Python one) and
    would raise a NameError the moment that branch was actually reached.
    Local assets are now used directly: image_mapping.resolve() already
    prefers a real downloaded photograph over the placeholder SVG, so nothing
    is lost and there is no external dependency or crash risk left.
    """
    return url_for("static", filename="crops/" + image_mapping.resolve(label))

@app.template_global()
def crop_image_is_real_photo(label):
    """True when the resolved local asset is an actual photograph, not the placeholder SVG."""
    return not image_mapping.is_placeholder(label)

@app.route("/")
def home():
    return render_template(
        "index.html",
        district_count=regional_service.district_count(),
        crop_count=len(crop_metadata.ml_crops()),
        algorithm=recommender.algorithm_name,
    )


@app.route("/soil", methods=["GET"])
def soil_form():
    return render_template(
        "soil_input.html",
        fields=config.INPUT_FIELDS,
        submitted={},
        errors={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Validate the form, run the model, render the top 3."""
    if not recommender.is_ready:
        return render_template(
            "error.html",
            title="Prediction is unavailable",
            message=(
                "The trained model could not be loaded, so soil-based "
                "recommendations cannot be produced right now."
            ),
            detail=recommender.load_error,
        ), 503

    result = validate_soil_inputs(request.form)

    if not result.ok:
        return render_template(
            "soil_input.html",
            fields=config.INPUT_FIELDS,
            submitted=request.form,
            errors=result.errors,
        ), 400

    try:
        predictions = recommender.recommend(result.values)
    except ModelUnavailableError as exc:
        return render_template(
            "error.html",
            title="Prediction failed",
            message="The model could not score these values.",
            detail=str(exc),
        ), 503

    if not predictions:
        return render_template(
            "error.html",
            title="No recommendation could be produced",
            message=(
                "The model returned a crop that is not in the project's label "
                "list, so no result is shown rather than guessing one."
            ),
            detail=None,
        ), 500

    return render_template(
        "results.html",
        predictions=predictions,
        inputs=result.values,
        fields=config.INPUT_FIELDS,
        notices=result.notices,
        supports_probability=recommender.supports_probability,
        importances=recommender.feature_importances(),
        algorithm=recommender.algorithm_name,
    )


@app.route("/tamil-nadu")
def tamil_nadu():
    if not regional_service.is_ready:
        return render_template(
            "error.html",
            title="Regional data is unavailable",
            message="The Tamil Nadu dataset could not be loaded.",
            detail=regional_service.load_error,
        ), 503

    return render_template(
        "tamil_nadu.html",
        districts=regional_service.districts(),
        data_year=regional_service.data_year,
        source=regional_service.source_citation(),
    )


@app.route("/tamil-nadu/district")
def district_detail():
    if not regional_service.is_ready:
        return redirect(url_for("tamil_nadu"))

    known = regional_service.districts()
    name = validate_district(request.args.get("district"), known)

    if name is None:
        return render_template(
            "tamil_nadu.html",
            districts=known,
            data_year=regional_service.data_year,
            source=regional_service.source_citation(),
            error="Select a district from the list to see its crop profile.",
        ), 400

    profile = regional_service.profile(name)
    if profile is None:
        return render_template(
            "error.html",
            title="District not found",
            message=f"No record for '{name}' exists in the regional dataset.",
            detail=None,
        ), 404

    return render_template("district.html", profile=profile)


@app.route("/crops")
def crop_guide():
    return render_template("crop_guide.html", crops=crop_metadata.all_crops())


@app.route("/crops/<crop_key>")
def crop_detail(crop_key: str):
    crop = crop_metadata.get(crop_key)
    if crop.key == "_unknown":
        abort(404)
    return render_template(
        "crop_detail.html",
        crop=crop,
        observed=recommender.crop_stats.get(crop.key),
        fields=config.INPUT_FIELDS,
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        algorithm=recommender.algorithm_name,
        features=config.FEATURE_ORDER,
        importances=recommender.feature_importances(),
        supports_probability=recommender.supports_probability,
        model_ready=recommender.is_ready,
        model_error=recommender.load_error,
        scalers_rebuilt=recommender.scalers_rebuilt,
        compat_patch=recommender.compat_patch_applied,
        regional_ready=regional_service.is_ready,
        regional_error=regional_service.load_error,
        district_count=regional_service.district_count(),
        data_year=regional_service.data_year,
        source=regional_service.source_citation(),
        image_audit=image_mapping.audit(),
        crop_count=len(crop_metadata.ml_crops()),
    )


# --------------------------------------------------------------------------
# Small JSON endpoint, used by the district selector and available for the
# future combined mode described in the README.
# --------------------------------------------------------------------------

@app.route("/api/districts")
def api_districts():
    return jsonify({
        "available": regional_service.is_ready,
        "data_year": regional_service.data_year,
        "districts": regional_service.districts(),
    })


@app.route("/api/health")
def api_health():
    return jsonify({
        "model_loaded": recommender.is_ready,
        "model_algorithm": recommender.algorithm_name,
        "supports_probability": recommender.supports_probability,
        "regional_data_loaded": regional_service.is_ready,
        "districts": regional_service.district_count(),
    })


# --------------------------------------------------------------------------
# Error handlers - the app shows a page, never a stack trace.
# --------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(_error):
    return render_template(
        "error.html",
        title="Page not found",
        message="That page does not exist. Use the navigation above to continue.",
        detail=None,
    ), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template(
        "error.html",
        title="Something went wrong",
        message="The page could not be built. Try again from the home page.",
        detail=None,
    ), 500


if __name__ == "__main__":
    # debug follows FLASK_DEBUG so production hosts never run in debug mode.
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
