from dataclasses import dataclass
from typing import List
from relationship_detector.output import Relationship


@dataclass
class TripletClassifierOutput:
    relationships: List[Relationship]