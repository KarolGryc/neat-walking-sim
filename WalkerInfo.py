from dataclasses import dataclass
import numpy as np

@dataclass
class WalkerInfo:
    name: str
    headAltitude: float
    hDistance: float
    hSpeed: float
    torsoAngle: float
    lHipAngle: float
    rHipAngle: float
    lKneeAngle: float
    rKneeAngle: float
    energySpent: float
    stepsTaken: int

    def as_array(self):
        return np.array([
            self.headAltitude,
            self.hDistance,
            self.hSpeed,
            self.torsoAngle,
            self.lHipAngle,
            self.rHipAngle,
            self.lKneeAngle,
            self.rKneeAngle,
        ], dtype=np.float32)