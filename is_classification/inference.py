from typing import List

import keras
import numpy as np
import cv2
import copy

from object_detection.output import DetectionOutput, BBox, Detection
from object_detection.inference import ObjectDetector
import is_classification.config as config
from .output import IsClassifierOutput
from relationship_detector.output import Relationship
from object_detection.output import Detection

MODEL_PATH = "../weights/is_classifier_best_model.hdf5"


class IsClassifier:
    def __init__(self, debug=False):
        self.model = keras.models.load_model(MODEL_PATH)
        self.debug = debug

    def classify(self, detector_output: DetectionOutput) -> IsClassifierOutput:

        print(len(detector_output.detections))

        batch_detections = [d for d in detector_output.detections if self.can_be_is(d.label)]

        print(len(batch_detections))

        if not batch_detections:
            return IsClassifierOutput(relationships=[])

        # batch_images, batch_labels = [], []
        batch_logits = []

        for detection in batch_detections:
            preprocessed_img, label = self.preprocess_image(detector_output.image_name, detection)
            if preprocessed_img is None:
                continue
            if self.debug:
                cv2.imshow(str(label), preprocessed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            logits = self.model.predict([np.expand_dims(preprocessed_img, axis=0), np.expand_dims(label, axis=0)])[0]
            batch_logits.append(logits)
            # batch_images.append(preprocessed_img)
            # batch_labels.append(label)

        # batch_images = np.stack(batch_images)
        # batch_labels = np.stack(batch_labels)
        # logits = self.model.predict([batch_images, batch_labels], batch_size=16)
        logits = np.stack(batch_logits)
        probabilities = np.max(logits, axis=-1)
        class_ids = np.argmax(logits, axis=-1)

        return IsClassifierOutput(
            relationships=[
                Relationship(confidence=p, label="is", detection1=d,
                             detection2=Detection(confidence=d.confidence, bbox=d.bbox, label=config.classes_names[c]))
                for p, c, d in zip(probabilities, class_ids, batch_detections)
            ]
        )

    @staticmethod
    def can_be_is(label):
        return label in config.labels_names

    def preprocess_image(self, image_name, detection: Detection):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.debug:
            cv2.imshow('original', image)
        height, width = image.shape[:2]

        ymin = int(min(height - 1, max(0, round(detection.bbox.y0 * height))))
        ymax = int(min(height - 1, max(0, round(detection.bbox.y1 * height))))
        xmin = int(min(width - 1, max(0, round(detection.bbox.x0 * width))))
        xmax = int(min(width - 1, max(0, round(detection.bbox.x1 * width))))

        if ymin >= ymax or xmin >= xmax:
            return None, None

        cropped_image = image[ymin:ymax, xmin:xmax]
        cropped_image = cv2.resize(cropped_image, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        normalized_image = cropped_image.astype(np.float32) / 127.5 - 1.

        klass = config.labels_names.index(detection.label)

        label = keras.utils.to_categorical(klass, config.NUM_OF_LABELS)
        return normalized_image, label


if __name__ == "__main__":
    detector = ObjectDetector()
    detection_output = detector.detect(
        "/mnt/renumics-research/datasets/vis-rel-data/img/0000575f5a03db70.jpg"
    )

    classifier = IsClassifier()
    classifier_output = classifier.classify(detection_output)
    pass
