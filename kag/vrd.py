import csv
from pathlib import Path
from relationship_detector.output import RelationshipDetectorOutput
from relationship_detector.inference import RelationshipDetector
from is_classification import config as is_config

VRD_TEST_FOLDER = "/mnt/renumics-research/datasets/vis-rel-data/test_img/test"


def run_vrd_test():
    relationship_detector = RelationshipDetector()

    truncate_output("/tmp/output.csv")

    image_paths = sorted(Path(VRD_TEST_FOLDER).glob("*.jpg"))
    for idx, image_path in enumerate(image_paths):
        print(f"Image {idx}/{len(image_paths)}")
        output = relationship_detector.detect(str(image_path))
        append_output("/tmp/output.csv", output)


def truncate_output(path):
    with open(path, "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["ImageId", "PredictionString"])


def append_output(path, relationship_detections: RelationshipDetectorOutput):
    with open(path, "a") as file:
        writer = csv.writer(file, delimiter=",")
        image_id = Path(relationship_detections.image_name).stem
        prediction_strings = []
        for rel in relationship_detections.relationships:
            confidence = rel.confidence
            relationship_label = rel.label
            label1 = rel.detection1.label
            xmin1 = rel.detection1.bbox.x0
            xmax1 = rel.detection1.bbox.x1
            ymin1 = rel.detection1.bbox.y0
            ymax1 = rel.detection1.bbox.y1

            if relationship_label == "is":
                label2 = rel.detection2.label
            else:
                label2 = rel.detection2.label

            if label2 == 'none':
                continue

            xmin2 = rel.detection2.bbox.x0
            xmax2 = rel.detection2.bbox.x1
            ymin2 = rel.detection2.bbox.y0
            ymax2 = rel.detection2.bbox.y1
            prediction_strings.append(f"{confidence:.6f} {label1} {xmin1:.6f} {ymin1:.6f} {xmax1:.6f} {ymax1:.6f} {label2} {xmin2:.6f} {ymin2:.6f} {xmax2:.6f} {ymax2:.6f} {relationship_label}")
        prediction_string = " ".join(prediction_strings)

        writer.writerow([image_id, prediction_string])


if __name__ == "__main__":
    run_vrd_test()

