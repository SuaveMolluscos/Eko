from Graphics.Pipeline.VertexProcessor import *
import pygame
import numpy as np

def painter(screen, scene):
    camera = scene.camera
    width = camera.width
    height = camera.height
    
    view_matrix = camera.view_matrix()
    projection_matrix = camera.projection_matrix()
    
    view_projection = projection_matrix @ view_matrix
    
    faces_to_draw = []
    cache_2D = {}
    
    for entity in scene.entities:
        vertices = entity.entity_vertices
        
        if not vertices:
            continue
            
        verts = np.array(vertices, dtype=float)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        
        en_clip = (view_projection @ verts_h.T).T
        
        vertices_2D = []
        for clip in en_clip:
            w = clip[3]
            
            if w <= 0.001:
                vertices_2D.append(None)
                continue
                
            ndc = clip / w
            
            x_ndc = ndc[0]
            y_ndc = ndc[1]
            
            if abs(x_ndc) > 1.1 or abs(y_ndc) > 1.1:
                vertices_2D.append(None)
                continue
                
            X_2D = int((x_ndc + 1) * width / 2)
            Y_2D = int((1 - y_ndc) * height / 2)
            
            vertices_2D.append((X_2D, Y_2D))
        
        cache_2D[entity] = vertices_2D
    
    for entity in scene.entities:
        vertices_2D = cache_2D.get(entity)
        if not vertices_2D:
            continue
            
        verts = np.array(entity.entity_vertices, dtype=float)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        en_camara = (view_matrix @ verts_h.T).T
        
        for face in entity.mesh.faces:
            if len(face) < 3:
                continue
                
            puntos = []
            profundidad_total = 0
            valida = True
            
            for idx, vert_idx in enumerate(face):
                vert_2d = vertices_2D[vert_idx]
                
                if vert_2d is None:
                    valida = False
                    break
                    
                puntos.append(vert_2d)
                
                if idx < 3:
                    profundidad_total += en_camara[vert_idx][2]
            
            if valida and len(puntos) >= 3:
                profundidad_prom = profundidad_total / 3
                faces_to_draw.append((profundidad_prom, puntos, entity.color))
    faces_to_draw.sort(key=lambda x: x[0])
    
    for profundidad, puntos, color in faces_to_draw:
        pygame.draw.polygon(screen, color, puntos)
        # Opcional: wireframe
        #pygame.draw.polygon(screen, (0, 0, 0), puntos, 1)