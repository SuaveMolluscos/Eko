import math
import pygame
import sys
import numpy as np

class Malla:

    def __init__(self, ruta):
        self.ruta = ruta
        self.vertices = []
        self.caras = []

        try:
            with open(ruta, 'r') as archivo:
                lineas = archivo.readlines()

            vertices = []
            caras = []
            v_app = vertices.append
            c_app = caras.append

            for linea in lineas:
                if linea.startswith('v '):
                    p = linea.split()
                    v_app((float(p[1]), float(p[2]), float(p[3])))

                elif linea.startswith('f '):
                    indices = [int(p.split('/')[0]) - 1 for p in linea.split()[1:]]
                    i0 = indices[0]
                    for i in range(1, len(indices) - 1):
                        c_app((i0, indices[i], indices[i + 1]))

            self.vertices = vertices
            self.caras = caras

        except FileNotFoundError:
            print(f"ERROR: No se encontró el archivo '{ruta}'")
        except Exception as e:
            print(f"Error al cargar la malla '{ruta}': {e}")

class Transformar:
    def __init__(self, vertices, translacion=(0,0,0), rotacion=(0,0,0), escala=(1,1,1)):
        self._vertices    = vertices   # originales, NUNCA se modifican
        self._translacion = list(translacion)
        self._rotacion    = list(rotacion)
        self._escala      = list(escala)

    def Transformacion(self):
        ex, ey, ez = self._escala
        rx, ry, rz = self._rotacion
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        tx, ty, tz = self._translacion

        S  = np.array([[ex,0,0,0],[0,ey,0,0],[0,0,ez,0],[0,0,0,1]], dtype=float)
        Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype=float)
        Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=float)
        Rz = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
        T  = np.array([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]], dtype=float)

        M       = T @ Rz @ Ry @ Rx @ S
        verts   = np.array(self._vertices, dtype=float)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        return [tuple(r[:3]) for r in (M @ verts_h.T).T]


class Entidad:
    def __init__(self, malla, translacion=(0,0,0), rotacion=(0,0,0), escala=(1,1,1), color=(140,140,140)):
        self.malla  = malla
        self.color  = color
        self.translacion = list(translacion)
        self.rotacion    = list(rotacion)
        self.escala      = list(escala)

        # Transformar recibe los vértices originales una sola vez
        self._transformar = Transformar(self.malla.vertices, translacion, rotacion, escala)

        # Calcula la posición inicial
        self.vertices_transformados = self._transformar.Transformacion()

    def actualizar(self):
        # Sincroniza los parámetros y recalcula desde los originales
        self._transformar._translacion = self.translacion
        self._transformar._rotacion    = self.rotacion
        self._transformar._escala      = self.escala
        self.vertices_transformados    = self._transformar.Transformacion()

class Camara:
    def __init__(self, posicion=(0,0,0), yaw=0, pitch=0, fov=60, near=0.1, far=100, ancho=800, alto=600, foco=300):
        self.posicion = list(posicion)
        self.yaw      = yaw
        self.pitch    = pitch
        self.fov      = fov
        self.near     = near
        self.far      = far
        self.ancho    = ancho
        self.alto     = alto
        self.foco     = foco

    def get_matriz_vista(self):
        # Rotación inversa: ángulos negativos porque movemos el mundo, no la cámara
        yaw   = math.radians(-self.yaw)
        pitch = math.radians(-self.pitch)

        cy, sy = math.cos(yaw),   math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)

        # Traslación inversa
        tx, ty, tz = self.posicion

        T = np.array([[1, 0, 0, -tx],
                      [0, 1, 0, -ty],
                      [0, 0, 1, -tz],
                      [0, 0, 0,   1]], dtype=float)

        Ry = np.array([[ cy, 0, sy, 0],
                       [  0, 1,  0, 0],
                       [-sy, 0, cy, 0],
                       [  0, 0,  0, 1]], dtype=float)

        Rx = np.array([[1,  0,   0,  0],
                       [0,  cp, -sp, 0],
                       [0,  sp,  cp, 0],
                       [0,  0,   0,  1]], dtype=float)

        # Orden inverso al de entidades: primero traslada, luego rota
        return Rx @ Ry @ T
class Escena:
    def __init__(self, camara=None):
        self.camara = camara if camara is not None else Camara()
        self.entidades = []

    def agregar_entidad(self, entidad):
        self.entidades.append(entidad)

    def eliminar_entidad(self, entidad):
        self.entidades.remove(entidad)

class Render:
    def __init__(self, escena = None, COLOR_FONDO = (0, 0, 0), titulo="Motor 3D", modo=0):
        self.TITULO = titulo
        self.MODO = modo
        self.ESCENA = escena if escena is not None else Escena()
        self.COLOR_FONDO = COLOR_FONDO

    def proyectar_vertices(self, vertices_mundo):
        cam  = self.ESCENA.camara
        V    = cam.get_matriz_vista()

        # Transforma todos los vértices al espacio de cámara de una vez
        verts   = np.array(vertices_mundo, dtype=float)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        en_camara = (V @ verts_h.T).T          # (N, 4)

        resultado = []
        for r in en_camara:
            x, y, z = r[0], r[1], r[2]
            if z <= 0.1:
                resultado.append(None)
                continue
            factor = cam.foco / z
            x_2D = int(x * factor + cam.ancho / 2)
            y_2D = int(-y * factor + cam.alto / 2)
            resultado.append((x_2D, y_2D))

        return resultado

    def Pintor(self, pantalla):
        cam = self.ESCENA.camara
        V   = cam.get_matriz_vista()

        caras_Z = []

        for entidad in self.ESCENA.entidades:
            entidad.actualizar()

            # Transforma los vértices al espacio de cámara para el sorting
            verts   = np.array(entidad.vertices_transformados, dtype=float)
            verts_h = np.hstack([verts, np.ones((len(verts), 1))])
            en_camara = (V @ verts_h.T).T   # vértices en espacio cámara

            for i0, i1, i2 in entidad.malla.caras:
                # Z en espacio cámara — este sí refleja la profundidad real
                z_prom = (en_camara[i0][2] + en_camara[i1][2] + en_camara[i2][2]) / 3
                caras_Z.append((z_prom, entidad, i0, i1, i2))

        caras_Z.sort(key=lambda c: -c[0])

        # Cache de proyección 2D (igual que antes)
        cache_2D = {}
        for entidad in self.ESCENA.entidades:
            cache_2D[id(entidad)] = self.proyectar_vertices(entidad.vertices_transformados)

        for z, entidad, i0, i1, i2 in caras_Z:
            vertices_2D = cache_2D[id(entidad)]
            p0 = vertices_2D[i0]
            p1 = vertices_2D[i1]
            p2 = vertices_2D[i2]
            if p0 and p1 and p2:
                pygame.draw.polygon(pantalla, entidad.color, [p0, p1, p2])

    def Red(self, pantalla):
        for entidad in self.ESCENA.entidades:
            entidad.actualizar()
            vertices_2D = self.proyectar_vertices(entidad.vertices_transformados)

            for i0, i1, i2 in entidad.malla.caras:
                p0 = vertices_2D[i0]
                p1 = vertices_2D[i1]
                p2 = vertices_2D[i2]
                if p0 and p1 and p2:
                    pygame.draw.line(pantalla, entidad.color, p0, p1, 1)
                    pygame.draw.line(pantalla, entidad.color, p1, p2, 1)
                    pygame.draw.line(pantalla, entidad.color, p2, p0, 1)

    def Iniciar(self, actualizar=None):
        pygame.init()
        cam = self.ESCENA.camara
        pantalla = pygame.display.set_mode((cam.ancho, cam.alto))
        pygame.display.set_caption(self.TITULO)
        reloj = pygame.time.Clock()

        while True:
            # Eventos
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if actualizar:
                actualizar()

            # Limpiar pantalla cada frame
            pantalla.fill(self.COLOR_FONDO)

            if self.MODO == 0:
                self.Pintor(pantalla)
            elif self.MODO == 1:
                self.Red(pantalla)
            else:
                self.Pintor(pantalla)
            
            pygame.display.flip()
            reloj.tick(60)