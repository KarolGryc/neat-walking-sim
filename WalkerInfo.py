from dataclasses import dataclass

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

    def as_array(self):
        return (
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
        )