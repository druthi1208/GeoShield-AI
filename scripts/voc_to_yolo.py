import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = [
    "D00",
    "D01",
    "D0w0",
    "D10",
    "D11",
    "D20",
    "D40",
    "D43",
    "D44",
    "D50",
]

BASE_INPUT = Path("data/RDD2022-India.v5-rdd2022-india.voc")
BASE_OUTPUT = Path("data/RDD2022-India.yolo")

SPLITS = ["train", "valid", "test"]

for split in SPLITS:

    input_dir = BASE_INPUT / split
    output_dir = BASE_OUTPUT / split / "labels"

    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0

    for xml_file in input_dir.glob("*.xml"):
        root = ET.parse(xml_file).getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        labels = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()

            if class_name not in CLASSES:
                print(
                    f"WARNING: Unknown class '{class_name}' "
                    f"in {xml_file.name}"
                )
                continue

            class_id = CLASSES.index(class_name)

            box = obj.find("bndbox")

            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            labels.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

        output_file = output_dir / f"{xml_file.stem}.txt"
        output_file.write_text("\n".join(labels), encoding="utf-8")

        converted += 1

    print(f"{split}: converted {converted} XML files.")

print("Conversion complete.")
print(f"YOLO dataset saved to: {BASE_OUTPUT}")