import pygame
from Box2D import (
    b2World, b2PolygonShape, b2CircleShape, b2_staticBody, b2_dynamicBody
)
import math
from Walker import Walker
import numpy as np
from WalkerInfo import WalkerInfo


SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
VERTICAL_FOV = 10
TARGET_FPS = 100
TIME_STEP = 1.0 / TARGET_FPS
VELOCITY_ITERATIONS = 8
POSITION_ITERATIONS = 3

CAM_START_X = 0
CAM_START_Y = 4.5

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.world = b2World(gravity=(0, -9.81), doSleep=True)

        self.running = True

        self.PPM = SCREEN_HEIGHT / VERTICAL_FOV

        self.cameraY = CAM_START_Y
        self.cameraX = CAM_START_X

        self.walkers = []

        # ground
        self.create_static_box((50, -0.25), (100, 0.5), friction=0.4, restitution=0.1)

        self.reset()

    def make_walkers(self, num_walkers):
        self.walkers = [Walker((2, 1.5), self) for _ in range(num_walkers)]

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
    
    def world_to_screen(self, world_coords):
        x, y = world_coords
        screen_x = int( (x - self.cameraX) * self.PPM) + SCREEN_WIDTH // 2
        screen_y = int(-(y - self.cameraY) * self.PPM) + SCREEN_HEIGHT // 2
        return screen_x, screen_y

    def screen_to_world(self, screen_coords):
        x, y = screen_coords
        world_x = ( (x - SCREEN_WIDTH // 2) / self.PPM) + self.cameraX
        world_y = (-(y - SCREEN_HEIGHT // 2)) / self.PPM + self.cameraY
        return (world_x, world_y)
    
    def draw_polygon(self, fixture, color=(0, 150, 255), border_width=1):
        if not fixture.shape:
            return
        shape = fixture.shape
        vertices = [fixture.body.transform * v for v in shape.vertices]
        vertices = [self.world_to_screen(v) for v in vertices]
        pygame.draw.polygon(self.screen, color, vertices)
        pygame.draw.polygon(self.screen, (0, 0, 0), vertices, border_width)
    
    def draw_circle(self, fixture, color=(0, 150, 255), border_width=1):
        if not fixture.shape:
            return
        shape = fixture.shape
        center = self.world_to_screen(fixture.body.position)
        radius = int(shape.radius * self.PPM)
        pygame.draw.circle(self.screen, color, center, radius)
        pygame.draw.circle(self.screen, (0, 0, 0), center, radius, border_width)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                world_pos = self.screen_to_world(mouse_pos)
                self.create_dynamic_box(world_pos, (0.25, 0.25), density=1.0, friction=0.5)


    def update(self, efforts):    
        for walker, effort in zip(self.walkers, efforts):
            walker.update(TIME_STEP, effort)

        self.world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)

        self.cameraX = 1 + max(walker.torso.position[0] for walker in self.walkers)

    def draw(self):
        self.clock.tick(TARGET_FPS)

        self.screen.fill((255, 255, 255))

        for body in self.world.bodies:
            for fixture in body.fixtures:
                if not fixture.shape:
                    continue
                elif isinstance(fixture.shape, b2PolygonShape):
                    self.draw_polygon(fixture)
                elif isinstance(fixture.shape, b2CircleShape):
                    self.draw_circle(fixture)        # Find the walker that has traveled the furthest
        leading_walker_idx = max(range(len(self.walkers)), 
                                key=lambda i: self.walkers[i].info().hDistance)
        walker_info = self.walkers[leading_walker_idx].info()
        
        walker_info_texts = [
            f"Altitude: {walker_info.headAltitude:.2f}",
            f"Energy spent: {walker_info.energySpent:.2f}",
            f"Distance walked: {walker_info.hDistance:.2f}",
            f"Steps taken: {walker_info.stepsTaken}",
        ]
        font = pygame.font.Font(None, 24)
        y_offset = 10
        for text in walker_info_texts:
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += text_surface.get_height()

        pygame.display.flip()
    
    def reset(self):
        self.world.ClearForces()
        for walker in self.walkers:
            walker.destroy()
        self.cameraX = CAM_START_X
        self.cameraY = CAM_START_Y

    def run(self):
        self.reset()
        self.running = True
        while self.running:
            self.handle_events()
            self.update([0,0,0,0])
            self.draw()

    def run_step(self, efforts):
        self.update(efforts)