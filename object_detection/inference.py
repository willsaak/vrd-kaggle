from output import DetectionOutput
from os import PathLike
from typing import List
from keras_retinanet.models import load_model


MODEL_PATH = "weights/retinanet_resnet152_level_1_v1.2-inference.h5"


def detect_objects(images: List[PathLike]) -> List[DetectionOutput]:
    model = load_model(MODEL_PATH)
