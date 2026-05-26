from .Camera import Camera

class Scene:
    def __init__(self, camera=None):
        if camera == None:
            camera = Camera()
        self.camera = camera
        self.entities = []

    def add_entity(self, entity):
        self.entities.append(entity)
    
    def delete_entity(self, entity):
        self.entities.remove(entity)