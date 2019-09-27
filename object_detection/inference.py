from os import PathLike

import numpy as np
import cv2
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from .output import DetectionOutput, BBox, Detection
from .visualization import visualize
from . import config


MODEL_PATH = "/home/william/resnet50_06_inference.h5"


class ObjectDetector:
    def __init__(self, debug=False):
        self.model = load_model(MODEL_PATH)
        self.debug = debug

    def detect(self, image_path: PathLike) -> DetectionOutput:
        image = read_image_bgr(image_path)
        if self.debug:
            print("debug")
            cv2.imwrite("debug/input_rgb.png", image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.debug:
            cv2.imwrite("debug/input_bgr.png", image)

        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, classes = self.model.predict_on_batch(
            np.expand_dims(image, axis=0)
        )

        width = image.shape[1]
        height = image.shape[0]
        bboxes = [BBox(x0=box[0]/width, x1=box[2]/width, y0=box[1]/height, y1=box[3]/height) for box in boxes[0]]
        detections = [
            Detection(bbox=bbox, label=config.labels[klass], confidence=score)
            for bbox, score, klass in zip(bboxes, scores[0], classes[0])
            if score > 0.25
        ]
        output = DetectionOutput(image_name=str(image_path), detections=detections)

        if self.debug:
            visualize(output, "debug/output.png")

        return output


if __name__ == "__main__":
    detector = ObjectDetector()
    detection_output = detector.detect(
        "/mnt/renumics-research/datasets/vis-rel-data/img/0000575f5a03db70.jpg"
    )

