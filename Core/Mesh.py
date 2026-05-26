class Mesh:

    def __init__(self, path, name=None):
        if name == None:
            name = f'Mesh: {path}'
        self.name = name
        self.vertices = []
        self.faces = []

        try:
            with open(path, 'r') as file:
                lines = file.readlines()

            vertices = []
            faces = []
            v_app = vertices.append
            f_app = faces.append

            for line in lines:
                if line.startswith('v '):
                    p = line.split()
                    v_app((float(p[1]), float(p[2]), float(p[3])))

                elif line.startswith('f '):
                    index = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
                    i0 = index[0]
                    for i in range(1, len(index) - 1):
                        f_app((i0, index[i], index[i + 1]))

            self.vertices = vertices
            self.faces = faces

        except FileNotFoundError:
            print(f"ERROR: File not found '{path}'")
        except Exception as e:
            print(f"Error loading mesh '{path}': {e}")