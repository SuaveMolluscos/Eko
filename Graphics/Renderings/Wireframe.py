from Graphics.Pipeline.VertexProcessor import *
import pygame
import numpy as np

def wireframe(screen, scene):
    camera = scene.camera
    width, height = camera.width, camera.height

    view_projection = camera.projection_matrix() @ camera.view_matrix()

    for entity in scene.entities:
        vertices = entity.entity_vertices
        if not vertices:
            continue

        verts   = np.array(vertices, dtype=float)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        clip    = (view_projection @ verts_h.T).T   # (N, 4)

        # NDC → píxeles
        vertices_2D = [None] * len(vertices)
        for i, c in enumerate(clip):
            w = c[3]
            if w <= 0:
                continue
            x_ndc = c[0] / w
            y_ndc = c[1] / w
            if abs(x_ndc) > 1.1 or abs(y_ndc) > 1.1:
                continue
            vertices_2D[i] = (
                int((x_ndc + 1) * width  / 2),
                int((1 - y_ndc) * height / 2)
            )

        # Dibujar caras
        for face in entity.mesh.faces:
            points = []
            for idx in face:
                p = vertices_2D[idx]
                if p is None:
                    break
                points.append(p)
            else:
                if len(points) >= 2:
                    pygame.draw.polygon(screen, entity.color, points, 1)