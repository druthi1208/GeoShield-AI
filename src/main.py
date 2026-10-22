
from pathlib import Path
from ultralytics import YOLO


# ==================================================
# GEOSHIELD CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "geoshield_v1"
    / "weights"
    / "best.pt"
)


# These are the EXACT class names from data.yaml.
CLASS_NAMES = {
    0: "D00",
    1: "D01",
    2: "D0w0",
    3: "D10",
    4: "D11",
    5: "D20",
    6: "D40",
    7: "D43",
    8: "D44",
    9: "D50",
}


# ==================================================
# RISK CALCULATION
# ==================================================

def calculate_risk(detections):

    if not detections:
        return "LOW", "No road damage detected."

    severe_classes = {
        "D20",
        "D40",
        "D44",
    }

    severe_count = sum(
        1
        for detection in detections
        if detection["class_code"] in severe_classes
    )

    total_detections = len(detections)

    if severe_count >= 2:
        return "HIGH", "Multiple high-priority road-damage detections."

    if severe_count == 1:
        return "HIGH", "High-priority road damage detected."

    if total_detections >= 3:
        return "MODERATE", "Multiple road-damage detections."

    return "LOW", "Limited road damage detected."


# ==================================================
# IMAGE ANALYSIS
# ==================================================

def analyze_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GeoShield model not found:\n{MODEL_PATH}"
        )

    print("\nLoading GeoShield AI model...")

    model = YOLO(str(MODEL_PATH))

    print(f"Analyzing: {image_path.name}")
    print("Please wait...\n")

    results = model.predict(
        source=str(image_path),
        imgsz=320,
        conf=0.25,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "geoshield"),
        name="predictions",
        exist_ok=True,
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_label = result.names.get(
                class_id,
                CLASS_NAMES.get(class_id, f"Class {class_id}")
            )

            detections.append({
                "class_id": class_id,
                "class_code": class_label,
                "confidence": confidence,
            })

    return detections


# ==================================================
# REPORT
# ==================================================

def print_report(image_path, detections):

    risk_level, risk_message = calculate_risk(detections)

    print()
    print("=" * 60)
    print("                 GEOSHIELD AI")
    print("             ROAD DAMAGE REPORT")
    print("=" * 60)

    print(f"Image: {Path(image_path).name}")
    print(f"Total detections: {len(detections)}")
    print()

    if not detections:

        print("No road damage detected.")

    else:

        print("Detected road damage:")
        print("-" * 60)

        for number, detection in enumerate(detections, start=1):

            print(
                f"{number}. "
                f"{detection['class_code']} | "
                f"Confidence: {detection['confidence']:.1%}"
            )

    print()
    print("-" * 60)
    print(f"RISK LEVEL: {risk_level}")
    print(f"Assessment: {risk_message}")

    if risk_level == "HIGH":

        print(
            "Recommendation: Exercise extreme caution "
            "and inspect the affected road section."
        )

    elif risk_level == "MODERATE":

        print(
            "Recommendation: Reduce speed and "
            "exercise caution."
        )

    else:

        print(
            "Recommendation: Continue with normal caution."
        )

    print("-" * 60)
    print()
    print("Annotated result saved under:")
    print(
        PROJECT_ROOT
        / "runs"
        / "geoshield"
        / "predictions"
    )

    print("=" * 60)


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    print("=" * 60)
    print("                 GEOSHIELD AI")
    print("          ROAD DAMAGE DETECTION SYSTEM")
    print("=" * 60)

    image_path = input(
        "\nEnter the path to a road image: "
    ).strip().strip('"')

    try:

        detections = analyze_image(image_path)

        print_report(
            image_path,
            detections
        )

    except Exception as error:

        print("\nGeoShield error:")
        print(error)


if __name__ == "__main__":
    main()

