from os import PathLike

from object_detection.inference import ObjectDetector


class RelationshipDetector:
    def __init__(self):
        self.object_detector = ObjectDetector()

    def detect(self, image_path: PathLike):
        detection_output = self.object_detector(image_path)
        return "foo"


if __name__ == "__main__":
    relationship_detector = RelationshipDetector()
    relationship_detector.detect(
        "/mnt/renumics-research/datasets/vis-rel-data/img/0000575f5a03db70.jpg"
    )
