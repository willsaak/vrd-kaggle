from output import DetectionOutput, BBox, Detection
from os import PathLike
from typing import List
from keras_retinanet.models import load_model
import PIL
import cv2
import tensorflow as tf
import keras
import numpy as np
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


MODEL_PATH = "weights/retinanet_resnet152_level_1_v1.2-inference.h5"


class ObjectDetector:
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

        self.model = load_model(MODEL_PATH)

    def detect(self, image_path: PathLike) -> DetectionOutput:
        image = read_image_bgr(image_path)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = self.model.predict_on_batch(
            np.expand_dims(image, axis=0)
        )
        boxes /= scale
        bboxes = [BBox(x0=box[0], x1=box[1], y0=box[2], y1=box[3]) for box in boxes]
        detections = [
            Detection(bbox=bbox, klass="unknown", confidence=0) for bbox in bboxes
        ]
        return DetectionOutput(image_name=image_path, detections=detections)


if __name__ == "__main__":
    detector = ObjectDetector()
    results = detector.detect(
        "/mnt/renumics-research/datasets/vis-rel-data/img/0000575f5a03db70.jpg"
    )
    print(results)
