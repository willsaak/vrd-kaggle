from os import PathLike

from object_detection.inference import ObjectDetector
from is_classification.inference import IsClassifier
from triplet_classification.inference import TripletClassifier
from .output import RelationshipDetectorOutput


class RelationshipDetector:
    def __init__(self):
        print("Initialize models...")
        self.object_detector = ObjectDetector(debug=True)
        print("Initialized object detector")
        self.is_classifier = IsClassifier()
        print("Initialized 'is' classifier")
        self.triplet_classifier = TripletClassifier()
        print("Initialized 'triplet' classifier")


    def detect(self, image_path: PathLike):
        print("Detect objects...")
        detection_output = self.object_detector.detect(image_path)

        print("Classify 'is' relationships...")
        is_out = self.is_classifier.classify(detection_output)
        print(f"'is' relationships: {len(is_out.relationships)}")

        print("Classify triplet relationships...")
        triplet_out = self.triplet_classifier.classify(detection_output)
        print(f"triplet relationships: {len(triplet_out.relationships)}")

        return RelationshipDetectorOutput(image_name=image_path,
                                          relationships=[*is_out.relationships, *triplet_out.relationships])


if __name__ == "__main__":
    relationship_detector = RelationshipDetector()
    relationship_detector.detect(
        "/mnt/renumics-research/datasets/vis-rel-data/img/0000575f5a03db70.jpg"
    )
