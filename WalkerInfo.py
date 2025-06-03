from dataclasses import dataclass

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