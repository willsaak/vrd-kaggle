from typing import List

import keras
import numpy as np
import cv2

from object_detection.output import DetectionOutput, BBox, Detection
from object_detection.inference import ObjectDetector
import is_classification.config as config

MODEL_PATH = "../weights/is_classifier_best_model.hdf5"


class IsClassifier:
    def __init__(self, debug=False):
        self.model = keras.models.load_model(MODEL_PATH)
        self.debug = debug

    def classify(self, detector_output: DetectionOutput) -> List[str]:

        batch_detections = [d for d in detector_output.detections if self.can_be_is(d.klass)]

        if not batch_detections:
            return []

        batch_images, batch_labels = [], []

        for detection in batch_detections:
            preprocessed_img, label = self.preprocess_image(detector_output.image_name, detection)
            cv2.imshow(str(label), preprocessed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            batch_images.append(preprocessed_img)
            batch_labels.append(label)

        batch_images = np.stack(batch_images)
        batch_labels = np.stack(batch_labels)
        probabilities = self.model.predict([batch_images, batch_labels])
        class_ids = np.argmax(probabilities, axis=-1)
        print(class_ids)

    @staticmethod
    def can_be_is(klass):
        return config.classes[klass] in config.labels_names

    @staticmethod
    def preprocess_image(image_name, detection: Detection):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('original', image)
        height, width = image.shape[:2]
        ymin = min(height-1, max(0, round(detection.bbox.y0 * height)))
        ymax = min(height - 1, max(0, round(detection.bbox.y1 * height)))
        xmin = min(width-1, max(0, round(detection.bbox.x0 * width)))
        xmax = min(width - 1, max(0, round(detection.bbox.x1 * width)))

        cropped_image = image[ymin:ymax, xmin:xmax]
        cropped_image = cv2.resize(cropped_image, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        normalized_image = cropped_image.astype(np.float32) / 127.5 - 1.
        label = keras.utils.to_categorical(detection.klass, config.NUM_OF_LABELS)
        return normalized_image, label


if __name__ == "__main__":
    detector = ObjectDetector()
    detection_output = detector.detect(
        "/mnt/renumics-research/datasets/vis-rel-data/img/0000575f5a03db70.jpg"
    )

    classifier = IsClassifier()
    classifier_output = classifier.classify(detection_output)
    pass
