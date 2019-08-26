from output import DetectionOutput
from os import PathLike
from typing import List
from keras_retinanet import load_model


def load_model():
    return None


def detect_objects(images: List[PathLike]) -> List[DetectionOutput]:
    model = load_model()
