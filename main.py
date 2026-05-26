from Core.Camera import Camera
from Core.Entity import Entity
from Core.Mesh import Mesh
from Core.Scene import Scene
from Graphics.Renderer import Renderer
import pygame
import math

Crash_mesh = Mesh('Assets\Crash Bandicoot\crashbandicoot.obj', 'Crash - Mesh')
Crash = Entity(Crash_mesh,(-350,0,0), (0,-90,0),color=(237, 33, 14))
Crash2 = Entity(Crash_mesh,(350,0,0), (0,-90,0),color=(237, 123, 0))
Crash3 = Entity(Crash_mesh,(0,0,350), (0,-90,0),color=(97, 223, 0))
Crash4 = Entity(Crash_mesh,(0,0,-350), (0,-90,0),color=(7, 33, 230))

View = Camera(Position=(0,100,0), width=1280, height=720)
World = Scene(View)
World.add_entity(Crash)
World.add_entity(Crash2)
World.add_entity(Crash3)
World.add_entity(Crash4)

Render = Renderer(World, mode=0)

primer_frame = True

def updater():
    global primer_frame

    speed = 2.0
    keys = pygame.key.get_pressed()

    dx, dy = pygame.mouse.get_rel()  # delta desde la última llamada
    pygame.mouse.set_visible(False)

    if primer_frame:
        primer_frame = False
    else:
        sensibilidad = 0.2
        View.Yaw   += dx * sensibilidad
        View.Pitch  = max(-89.0, min(89.0, View.Pitch - dy * sensibilidad))

    View.view_matrix()

    if keys[pygame.K_w]:
        fx, fz = View.Front[0], View.Front[2]
        mag = math.sqrt(fx*fx + fz*fz)
        if mag > 0:
            View.Position[0] += (fx/mag) * speed
            View.Position[2] += (fz/mag) * speed

    if keys[pygame.K_s]:
        fx, fz = View.Front[0], View.Front[2]
        mag = math.sqrt(fx*fx + fz*fz)
        if mag > 0:
            View.Position[0] -= (fx/mag) * speed
            View.Position[2] -= (fz/mag) * speed

    if keys[pygame.K_a]:
        View.Position[0] -= View.Right[0] * speed
        View.Position[2] -= View.Right[2] * speed

    if keys[pygame.K_d]:
        View.Position[0] += View.Right[0] * speed
        View.Position[2] += View.Right[2] * speed

    if keys[pygame.K_o]:
        View.Position[1] += speed
    if keys[pygame.K_u]:
        View.Position[1] -= speed

    pygame.mouse.set_pos((View.width // 2, View.height // 2))

Render.start(updater)