from .Mesh import Mesh
from .ObservableList import ObservableList
from Mathematics.Transformation import Transform

class Entity:

    def __init__(self, mesh=None, translation=(0,0,0), rotation=(0,0,0), scale=(1,1,1), color=(140,140,140)):

        if mesh is None:
            mesh = Mesh(r'Assets\cube.obj')

        self.mesh = mesh
        self.color = color
        self._translation = ObservableList(list(translation), self.update_vertices)
        self._rotation = ObservableList(list(rotation), self.update_vertices)
        self._scale = ObservableList(list(scale), self.update_vertices)
        self._transform = Transform(mesh.vertices, self._translation, self._rotation, self._scale)
        self.entity_vertices = (self._transform.get_transformed_vertices())

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation[:] = value

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation[:] = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale[:] = value

    def update_vertices(self):
        self._transform.translation = self._translation
        self._transform.rotation = self._rotation
        self._transform.scale = self._scale
        self.entity_vertices = (self._transform.get_transformed_vertices())

    def __repr__(self):
        return f"Entity: mesh: '{self.mesh.name}', translation:{self._translation}, rotation:{self._rotation}, scale:{self.scale}"