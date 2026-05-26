from .Renderings.Wireframe import wireframe
from .Renderings.Painter import painter
import pygame
import sys

class Renderer:
    def __init__(self, scene=None, background_color=(0,0,0), title='Soft_Motor', mode=0):
        self.scene = scene
        self.background_color = background_color
        self.title = title
        self.mode = mode
    
    def start(self, updater=None):
        pygame.init()
        screen = pygame.display.set_mode((self.scene.camera.width, self.scene.camera.height))
        pygame.display.set_caption(self.title)
        clock = pygame.time.Clock()

        run = True

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        run = False

            if updater:
                updater()

            screen.fill(self.background_color)

            if self.mode==0:
                painter(screen, self.scene)
            elif self.mode==1:
                wireframe(screen, self.scene)
            
            pygame.display.flip()
            clock.tick(60)
