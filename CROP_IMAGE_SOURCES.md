# Crop image sources

The UI uses real crop photographs from Wikimedia Commons for the crops listed in `core/image_mapping.py`.
The browser loads the corresponding `Special:Redirect/file/...` URL, and `tools/download_crop_images.py` can download local JPG copies when network access is available.

Check each Wikimedia Commons file page for the current license/attribution requirements before public deployment.

Current sources include Rice Crops, MAIZE CROP, COTTON PLANT, Groundnut, Papaya Plant, Mango Tree, Watermelon plant, Apple plant, Black gram plant, Brinjal plants, Chilli plant, Okra plant, Pomegranate tree with Fruits, Sesame plant and Lentil plants.
