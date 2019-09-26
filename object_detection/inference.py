from os import PathLike
from typing import List
import tensorflow as tf
import keras
import numpy as np
import cv2
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from .output import DetectionOutput, BBox, Detection
from .visualization import visualize


# MODEL_PATH = "weights/retinanet_resnet152_level_1_v1.2-inference.h5"
MODEL_PATH = "/home/william/resnet50_06_inference.h5"


class ObjectDetector:
    def __init__(self, debug=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

        self.model = load_model(MODEL_PATH)
        self.debug = debug

    def detect(self, image_path: PathLike) -> DetectionOutput:
        image = read_image_bgr(image_path)
        if self.debug:
            cv2.imwrite("debug/input_rgb.png", image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.debug:
            cv2.imwrite("debug/input_bgr.png", image)
        # cv2.imshow('sf', image)
        # cv2.waitKey()
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = self.model.predict_on_batch(
            np.expand_dims(image, axis=0)
        )

        width = image.shape[1]
        height = image.shape[0]
        bboxes = [BBox(x0=box[0]/width, x1=box[1]/width, y0=box[2]/height, y1=box[3]/height) for box in boxes[0]]
        detections = [
            Detection(bbox=bbox, klass=label, confidence=score)
            for bbox, score, label in zip(bboxes, scores[0], labels[0])
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

