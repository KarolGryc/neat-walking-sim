from Box2D import b2World, b2PolygonShape
from Walker import Walker

class Simulation:
    TARGET_FPS = 100
    TIME_STEP = 1.0 / TARGET_FPS

    VELOCITY_ITERATIONS = 8
    POSITION_ITERATIONS = 3

    def __init__(self):
        self.world = b2World(gravity=(0, -9.81), doSleep=True)

        self.walkers = []

        self.cameraX, self.cameraY = 0, 4.5

        # ground
        self.create_static_box((50, -0.25), (100, 0.5), friction=0.4, restitution=0.1)

        self.reset()

    def make_walkers(self, num_walkers):
        new_walkers = [Walker((2, 1.5), self) for _ in range(num_walkers)]
        self.walkers.extend(new_walkers)

    def create_static_box(self, position, size, friction=0.8, restitution=0.4, angle=0):
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

    def update(self, efforts):    
        for walker, effort in zip(self.walkers, efforts):
            walker.update(self.TIME_STEP, effort)

        self.world.Step(self.TIME_STEP, self.VELOCITY_ITERATIONS, self.POSITION_ITERATIONS)

        self.cameraX = max(walker.torso.position[0] for walker in self.walkers)
    
    def reset(self):
        self.world.ClearForces()
        for walker in self.walkers:
            walker.destroy()

    def infos_array(self):
        return [walker.info().as_array() for walker in self.walkers]