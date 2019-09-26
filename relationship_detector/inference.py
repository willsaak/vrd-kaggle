from os import PathLike
import pickle

from object_detection.inference import ObjectDetector
from is_classification.inference import IsClassifier


class RelationshipDetector:
    def __init__(self):
        print("Initialize models...")
        self.object_detector = ObjectDetector()
        print("Initialized object detector")
        self.is_classifier = IsClassifier()
        print("Initialized 'is' classifier")

    def detect(self, image_path: PathLike):
        print("Detect objects...")
        detection_output = self.object_detector.detect(image_path)

        print("Classifiy 'is' relationships...")
        is_classifications = self.is_classifier.classify(detection_output)
        print(is_classifications)
        print('done')


if __name__ == "__main__":
    relationship_detector = RelationshipDetector()
    relationship_detector.detect(
        "/mnt/renumics-research/datasets/vis-rel-data/img/0000575f5a03db70.jpg"
    )
