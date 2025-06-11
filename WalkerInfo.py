from dataclasses import dataclass
import numpy as np

@dataclass
class WalkerInfo:
    headAltitude: float
    hDistance: float
    hSpeed: float
    torsoAngle: float
    lHipAngle: float
    rHipAngle: float
    lKneeAngle: float
    rKneeAngle: float
    energySpent: float
    lHipSpeed: float
    rHipSpeed: float
    lKneeSpeed: float
    rKneeSpeed: float
    # leftLegLead: float

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
            self.lHipSpeed,
            self.rHipSpeed,
            self.lKneeSpeed,
            self.rKneeSpeed,
            # self.leftLegLead
        ], dtype=np.float32)