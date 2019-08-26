from dataclasses import dataclass
from typing import List


@dataclass
class BBox:
    x0: float
    x1: float
    y0: float
    y1: float


@dataclass
class Detection:
    bbox: BBox
    klass: str
    confidence: float


@dataclass
class DetectionOutput:
    image_name: str
    detections: List[Detection]
