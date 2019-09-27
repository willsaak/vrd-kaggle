from dataclasses import dataclass
from typing import List, Optional

from object_detection.output import BBox, Detection

@dataclass
class Relationship:
    confidence: float
    label: str
    detection1: Detection
    detection2: Detection

@dataclass
class RelationshipDetectorOutput:
    image_name: str
    relationships: List[Relationship]
