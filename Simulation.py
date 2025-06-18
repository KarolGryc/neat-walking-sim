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


class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.world = b2World(gravity=(0, -9.81), doSleep=True)

        self.running = True

        self.PPM = SCREEN_HEIGHT / VERTICAL_FOV

        self.cameraX, self.cameraY = 0, 4.5

        self.walkers = []

        self.create_static_box((50, -0.25), (100, 0.5))

        self.reset()

    def make_walkers(self, num_walkers):
        self.walkers = [Walker((2, 1.5), self) for _ in range(num_walkers)]

    def create_static_box(self, position, size, friction=0.5, restitution=0.8, angle=0):
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

    def update(self, efforts):    
        for walker, effort in zip(self.walkers, efforts):
            walker.update(TIME_STEP, effort)

        self.world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)

        self.cameraX = max(walker.torso.position[0] for walker in self.walkers)

    def draw(self, strings=[]):
        self.clock.tick(TARGET_FPS)

        self.screen.fill((255, 255, 255))

        for i in range(100):
            screenx = self.world_to_screen((i, 0))[0]
            pygame.draw.line(self.screen, (200, 200, 200), (screenx, 0), (screenx, SCREEN_HEIGHT))

        for body in self.world.bodies:
            color = body.userData['color'] if body.userData and 'color' in body.userData else (0, 150, 255)
            for fixture in body.fixtures:
                if not fixture.shape:
                    continue
                elif isinstance(fixture.shape, b2PolygonShape):
                    self.draw_polygon(fixture, color)
                elif isinstance(fixture.shape, b2CircleShape):
                    self.draw_circle(fixture, color)        # Find the walker that has traveled the furthest
        
        leader_idx = max(range(len(self.walkers)), key=lambda i: self.walkers[i].info().hDistance)
        walker_info = self.walkers[leader_idx].info()
        
        walker_info_texts = [
            f"Altitude: {walker_info.headAltitude:.2f}",
            f"Energy spent: {walker_info.energySpent:.2f}",
            f"Distance walked: {walker_info.hDistance:.2f}",
            # f"Time of left leg lead: {walker_info.leftLegLead:.2f}",
        ]
        font = pygame.font.Font(None, 24)
        y_offset = 10
        for text in strings:
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += text_surface.get_height()
            
        pygame.display.flip()
    
    def reset(self):
        self.world.ClearForces()
        for walker in self.walkers:
            walker.destroy()

    def infos_array(self):
        return [walker.info().as_array() for walker in self.walkers]