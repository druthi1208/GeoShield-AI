
import uuid
from pathlib import Path

import cv2
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ==================================================
# GEOSHIELD AI - ROAD DAMAGE DETECTION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = UPLOAD_FOLDER / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# Prefer the trained GeoShield models over generic YOLO.
MODEL_CANDIDATES = [
    PROJECT_DIR / "runs" / "detect" / "geoshield_v1" / "weights" / "best.pt",
    PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt",
    PROJECT_DIR / "models" / "best.pt",
    PROJECT_DIR / "yolo11n.pt",
]

# Road Damage Dataset class labels.
# Confirm these against the exact dataset version used for training.
DAMAGE_LABELS = {
    "D00": "Longitudinal crack",
    "D01": "Transverse crack",
    "D0w0": "Road damage - D0w0",
    "D10": "Longitudinal construction joint",
    "D11": "Transverse construction joint",
    "D20": "Alligator cracking",
    "D40": "Pothole",
    "D43": "Road damage - D43",
    "D44": "Road damage - D44",
    "D50": "Road damage - D50",
}

model = None
model_path = None
model_error = None

for candidate in MODEL_CANDIDATES:
    if not candidate.is_file():
        continue

    try:
        loaded_model = YOLO(str(candidate))
        model = loaded_model
        model_path = candidate
        model_error = None
        print(f"[GeoShield] Loaded trained model: {model_path}")
        break
    except Exception as exc:
        model_error = str(exc)
        print(f"[GeoShield] Failed to load {candidate}: {exc}")

if model is None:
    model_error = model_error or "No YOLO model file was found."
    print(f"[GeoShield] ERROR: {model_error}")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_model_class_name(class_id, names):
    """Get the actual class name reported by the loaded model."""
    if isinstance(names, dict):
        raw_name = names.get(class_id, str(class_id))
    else:
        raw_name = (
            names[class_id]
            if class_id < len(names)
            else str(class_id)
        )

    raw_name = str(raw_name)
    return DAMAGE_LABELS.get(raw_name, raw_name)


def image_url(filename):
    return url_for("uploaded_file", filename=f"results/{filename}")


# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "project": "GeoShield AI",
        "model_loaded": model is not None,
        "model_path": str(model_path) if model_path else None,
        "model_error": model_error,
        "model_classes": model.names if model is not None else {},
    })


# ==================================================
# SERVE ANNOTATED IMAGES
# ==================================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


# ==================================================
# ROAD DAMAGE ANALYSIS
# ==================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image was included in the request.",
        }), 400

    uploaded_image = request.files["image"]

    if not uploaded_image.filename:
        return jsonify({
            "success": False,
            "error": "Please select an image.",
        }), 400

    if not allowed_file(uploaded_image.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported image format. Use JPG, PNG, WEBP, or BMP.",
        }), 400

    if model is None:
        return jsonify({
            "success": False,
            "error": "The YOLO model could not be loaded.",
            "details": model_error,
        }), 503

    safe_name = secure_filename(uploaded_image.filename)
    extension = Path(safe_name).suffix.lower()

    if not extension:
        extension = ".jpg"

    filename = f"{uuid.uuid4().hex}{extension}"
    image_path = UPLOAD_FOLDER / filename

    try:
        uploaded_image.save(str(image_path))

        # Verify that the uploaded file is a readable image.
        image = cv2.imread(str(image_path))

        if image is None:
            return jsonify({
                "success": False,
                "error": "The uploaded file is not a readable image.",
            }), 400

        # Run inference using the selected trained model.
        predictions = model.predict(
            source=image,
            conf=0.25,
            imgsz=640,
            verbose=False,
        )

        prediction = predictions[0]
        detections = []
        names = prediction.names

        if prediction.boxes is not None:
            for box in prediction.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                coords = [
                    round(float(v), 2)
                    for v in box.xyxy[0].tolist()
                ]

                raw_class = (
                    names.get(class_id, str(class_id))
                    if isinstance(names, dict)
                    else names[class_id]
                )

                raw_class = str(raw_class)
                damage_name = get_model_class_name(class_id, names)

                detections.append({
                    "class_id": class_id,
                    "class_code": raw_class,
                    "class_name": damage_name,
                    "confidence": round(confidence, 4),
                    "confidence_percent": round(confidence * 100, 2),
                    "bbox": {
                        "x1": coords[0],
                        "y1": coords[1],
                        "x2": coords[2],
                        "y2": coords[3],
                    },
                })

        # Generate annotated image.
        annotated_filename = f"annotated_{uuid.uuid4().hex}.jpg"
        annotated_path = RESULT_FOLDER / annotated_filename

        annotated = prediction.plot()

        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError("Failed to save annotated image.")

        class_counts = {}
        for item in detections:
            label = item["class_name"]
            class_counts[label] = class_counts.get(label, 0) + 1

        # This is a preliminary count-based indicator, not a
        # validated engineering assessment of road safety.
        if len(detections) >= 5:
            risk_level = "HIGH"
        elif len(detections) >= 2:
            risk_level = "MEDIUM"
        elif len(detections) == 1:
            risk_level = "REVIEW"
        else:
            risk_level = "NO DETECTIONS"

        return jsonify({
            "success": True,
            "message": "Road image analysis completed.",
            "model": model_path.name,
            "total_detections": len(detections),
            "detections": detections,
            "class_counts": class_counts,
            "risk_level": risk_level,
            "annotated_image": image_url(annotated_filename),
        }), 200

    except Exception as exc:
        app.logger.exception("GeoShield analysis failed")

        return jsonify({
            "success": False,
            "error": "Road damage analysis failed.",
            "details": str(exc),
        }), 500

    finally:
        try:
            if image_path.exists():
                image_path.unlink()
        except OSError:
            app.logger.warning("Could not remove temporary uploaded image.")


# ==================================================
# ERRORS
# ==================================================

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "error": "Image exceeds the 10 MB upload limit.",
    }), 413


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
