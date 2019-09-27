from typing import List

import keras
import numpy as np
from typing import Tuple
import cv2
import copy

from object_detection.output import DetectionOutput, BBox, Detection
from object_detection.inference import ObjectDetector
from .output import TripletClassifierOutput
from relationship_detector.output import Relationship
from object_detection.output import Detection
from . import config

MODEL_PATH = "../weights/triplet_classifier_best_model.hdf5"


def batched(iterable, n):
    l = len(iterable)
    for ndx in range(0, len(iterable), n):
        yield iterable[ndx:min(ndx + n, l)]

def label_for_class(klass):
    label = config.class_names[klass]
    if label == 'none':
        label = None
    return label


class TripletClassifier:
    def __init__(self, debug=False):
        self.model = keras.models.load_model(MODEL_PATH)
        self.debug = debug

    def classify(self, detector_output: DetectionOutput) -> TripletClassifierOutput:

        print(len(detector_output.detections))

        detection_pairs = []
        for d1 in detector_output.detections:
            for d2 in detector_output.detections:
                klass = self.triplet_class(d1.label, d2.label)
                if klass:
                    detection_pairs.append((d1, d2, klass))

        if not detection_pairs:
            return TripletClassifierOutput(relationships=[])

        batch_logits = []

        for batch in batched(detection_pairs, 64):
            batch_images, batch_labels = [], []
            for d1, d2, klass in batch:
                preprocessed_img = self.preprocess_images(detector_output.image_name, d1, d2)
                if self.debug:
                    cv2.imshow(str(klass), preprocessed_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                batch_images.append(preprocessed_img)
                batch_labels.append(keras.utils.to_categorical(klass, config.NUM_OF_LABELS))
            batch_images = np.stack(batch_images)
            batch_labels = np.stack(batch_labels)
            logits = self.model.predict([batch_images, batch_labels])
            batch_logits.extend(logits)

        logits = np.stack(batch_logits)
        probabilities = np.max(logits, axis=-1)
        class_ids = np.argmax(logits, axis=-1)
        labels = [label_for_class(c) for c in class_ids]

        return TripletClassifierOutput(
            relationships=[
                Relationship(confidence=p, label=l, detection1=d[0], detection2=d[1])
                for p, l, d in zip(probabilities, labels, detection_pairs) if l
            ]
        )

    @staticmethod
    def triplet_class(label1, label2):
        triplet_label = f"{label1},{label2}"
        try:
            return config.label_names.index(triplet_label)
        except ValueError:
            return None

    def preprocess_images(self, image_name, detection1, detection2):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape

        ymin1 = int(min(height - 1, max(0, round(detection1.bbox.y0 * height))))
        ymax1 = int(min(height - 1, max(0, round(detection1.bbox.y1 * height))))
        xmin1 = int(min(width - 1, max(0, round(detection1.bbox.x0 * width))))
        xmax1 = int(min(width - 1, max(0, round(detection1.bbox.x1 * width))))
        ymin2 = int(min(height - 1, max(0, round(detection2.bbox.y0 * height))))
        ymax2 = int(min(height - 1, max(0, round(detection2.bbox.y1 * height))))
        xmin2 = int(min(width - 1, max(0, round(detection2.bbox.x0 * width))))
        xmax2 = int(min(width - 1, max(0, round(detection2.bbox.x1 * width))))

        ymin, ymax, xmin, xmax = min(ymin1, ymin2), max(ymax1, ymax2), min(xmin1, xmin2), max(xmax1, xmax2)

        if ymin >= ymax or xmin >= xmax:
            return None

        cropped_image = np.zeros((ymax - ymin, xmax - xmin, channels * 2), image.dtype)
        cropped_image[ymin1 - ymin:ymax1 - ymin, xmin1 - xmin:xmax1 - xmin, :channels] = image[ymin1:ymax1,
                                                                                         xmin1:xmax1]
        cropped_image[ymin2 - ymin:ymax2 - ymin, xmin2 - xmin:xmax2 - xmin, channels:] = image[ymin2:ymax2,
                                                                                         xmin2:xmax2]
        cropped_image = cv2.resize(cropped_image, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))

        normalized_image = cropped_image.astype(np.float32) / 127.5 - 1.

        return normalized_image
