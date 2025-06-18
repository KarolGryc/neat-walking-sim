from Box2D import (
    b2World, b2PolygonShape
)
from Walker import Walker

TARGET_FPS = 100
TIME_STEP = 1.0 / TARGET_FPS
VELOCITY_ITERATIONS = 8
POSITION_ITERATIONS = 3

class SimulationForParallel:
    def __init__(self):
        self.world = b2World(gravity=(0, -9.81), doSleep=True)
        self.create_static_box((50, -0.25), (100, 0.5))

    def make_walker(self):
        self.walker = Walker((2, 1.5), self)

    def create_static_box(self, position, size, friction=0.5, restitution=0.85, angle=0):
        body = self.world.CreateStaticBody(
            position=position,
            angle=angle,
        )
        body.CreateFixture(
            shape=b2PolygonShape(box=(size[0] / 2, size[1] / 2)),
            friction=friction,
            restitution=restitution
        )
        return body

    def update(self, effort):    
        self.walker.update(TIME_STEP, effort)
        self.world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)
    
    def reset(self):
        self.world.ClearForces()
        self.walker.destroy()

    def run_step(self, efforts):
        self.update(efforts)