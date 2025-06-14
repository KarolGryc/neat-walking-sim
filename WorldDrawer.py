import pygame
from Box2D import b2PolygonShape, b2CircleShape

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
VERTICAL_FOV = 10

class WorldDrawer:
    def __init__(self):
        pygame.init()

        self.sim = None
        
        self.PPM = SCREEN_HEIGHT / VERTICAL_FOV

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
    
    def world_to_screen(self, world_coords):
        x, y = world_coords
        screen_x = int( (x - self.sim.cameraX) * self.PPM) + SCREEN_WIDTH // 2
        screen_y = int(-(y - self.sim.cameraY) * self.PPM) + SCREEN_HEIGHT // 2
        return screen_x, screen_y

    def screen_to_world(self, screen_coords):
        x, y = screen_coords
        world_x = ( (x - SCREEN_WIDTH // 2) / self.PPM) + self.sim.cameraX
        world_y = (-(y - SCREEN_HEIGHT // 2)) / self.PPM + self.sim.cameraY
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
                exit(1)
    
    def draw(self, strings=[]):
        self.clock.tick(self.sim.TARGET_FPS)

        self.screen.fill((255, 255, 255))

        # meter lines
        for i in range(100):
            screenx = self.world_to_screen((i, 0))[0]
            pygame.draw.line(self.screen, (200, 200, 200), (screenx, 0), (screenx, SCREEN_HEIGHT))

        for body in self.sim.world.bodies:
            color = body.userData['color'] if body.userData and 'color' in body.userData else (0, 150, 255)
            for fixture in body.fixtures:
                if not fixture.shape:
                    continue
                elif isinstance(fixture.shape, b2PolygonShape):
                    self.draw_polygon(fixture, color)
                elif isinstance(fixture.shape, b2CircleShape):
                    self.draw_circle(fixture, color)        # Find the walker that has traveled the furthest

        font = pygame.font.Font(None, 24)
        y_offset = 10
        for text in strings:
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += text_surface.get_height()
            
        pygame.display.flip()