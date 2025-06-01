import pygame
from Box2D import (
    b2World, b2PolygonShape, b2CircleShape, b2_staticBody, b2_dynamicBody
)
import math
import Walker
import numpy as np


SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PPM = 50.0
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
VELOCITY_ITERATIONS = 8
POSITION_ITERATIONS = 3

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.world = b2World(gravity=(0, -10), doSleep=True)

        self.walker = Walker.Walker((2,3), self)

        self.running = True

        # ramps
        self.create_static_box((4, 6), (8, 0.25), angle=math.radians(-20), friction=0)
        self.create_static_box((8, 4.5), (1, 0.25), angle=math.radians(0), friction=0.9)

        # ground
        self.create_static_box((SCREEN_WIDTH / PPM / 2, 0), (SCREEN_WIDTH / PPM, 0.5), friction=0.5)


    def create_static_box(self, position, size, friction=0.8, angle=0):
        body = self.world.CreateStaticBody(
            position=position,
            angle=angle,
        )
        body.CreateFixture(
            shape=b2PolygonShape(box=(size[0] / 2, size[1] / 2)),
            friction=friction
        )
        return body

    def create_dynamic_box(self, position, size, density=1.0, friction=0.5):
        body = self.world.CreateDynamicBody(
            position=position,
            shapes=b2PolygonShape(box=(size[0] / 2, size[1] / 2))
        )
        fixture = body.fixtures[0]
        fixture.density = density
        fixture.friction = friction

        body.ResetMassData()
        return body
    
    def world_to_screen(self, world_coords):
        x, y = world_coords
        return int(x * PPM), int(SCREEN_HEIGHT - y * PPM)

    def screen_to_world(self, screen_coords):
        x, y = screen_coords
        return (x / PPM, (SCREEN_HEIGHT - y) / PPM)
    
    def draw_polygon(self, fixture, color=(0, 150, 255), border_color=(0, 0, 0), border_width=1):
        if not fixture.shape:
            return
        shape = fixture.shape
        vertices = [fixture.body.transform * v for v in shape.vertices]
        vertices = [self.world_to_screen(v) for v in vertices]
        # Fill the polygon
        pygame.draw.polygon(self.screen, color, vertices)
        # Draw the border
        pygame.draw.polygon(self.screen, border_color, vertices, border_width)
    
    def draw_circle(self, fixture, color=(0, 150, 255), border_color=(0, 0, 0), border_width=1):
        if not fixture.shape:
            return
        shape = fixture.shape
        center = self.world_to_screen(fixture.body.position)
        radius = int(shape.radius * PPM)
        # Fill the circle
        pygame.draw.circle(self.screen, color, center, radius)
        # Draw the border
        pygame.draw.circle(self.screen, border_color, center, radius, border_width)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                world_pos = self.screen_to_world(mouse_pos)
                self.create_dynamic_box(world_pos, (1, 1), density=1.0, friction=0.5)


    def update(self):
        keys = pygame.key.get_pressed()
        efforts = np.array([0, 0, 0, 0])
        if keys[pygame.K_w]:
            efforts += [1, -1, 1, -1]
        if keys[pygame.K_s]:
            efforts += [-1, 1, -1, 1]
        if keys[pygame.K_q]:
            efforts += [1, 0, 1, 0]
        if keys[pygame.K_a]:
            efforts += [-1, 0, -1, 0]
        if keys[pygame.K_e]:
            efforts += [0, -1, 0, -1]
        if keys[pygame.K_d]:
            efforts += [0, 1, 0, 1]
        

        
        self.walker.update(efforts)

        self.world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)
        self.clock.tick(TARGET_FPS)

    def draw(self):
        self.screen.fill((255, 255, 255))

        for body in self.world.bodies:
            for fixture in body.fixtures:
                if not fixture.shape:
                    continue
                elif isinstance(fixture.shape, b2PolygonShape):
                    self.draw_polygon(fixture)
                elif isinstance(fixture.shape, b2CircleShape):
                    self.draw_circle(fixture)

        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()

        pygame.quit()