import base64

import cv2
import numpy as np
from flask import Flask, flash, render_template, request

from doc_scanner import scan_document_image

app = Flask(__name__)
app.secret_key = "doc-scanner-secret"


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_to_base64(image):
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files.get("document")

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Please choose an image to upload.")
            return render_template("index.html")

        if not allowed_file(uploaded_file.filename):
            flash("Unsupported file type. Please upload an image file.")
            return render_template("index.html")

        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            flash("The uploaded file could not be read as an image.")
            return render_template("index.html")

        try:
            original, scanned, _, _ = scan_document_image(image)
            original_b64 = encode_image_to_base64(original)
            scanned_b64 = encode_image_to_base64(scanned)
        except ValueError as error:
            flash(str(error))
            return render_template("index.html")

        return render_template(
            "index.html",
            original_image=original_b64,
            scanned_image=scanned_b64,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
