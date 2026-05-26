class ObservableList(list):

    def __init__(self, data, callback):
        super().__init__(data)
        self.callback = callback

    def _changed(self):
        self.callback()

    # lista[i] = valor
    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        self._changed()

    # lista.append(valor)
    def append(self, value):
        super().append(value)
        self._changed()

    # lista.extend([...])
    def extend(self, values):
        super().extend(values)
        self._changed()

    # lista.insert(i, valor)
    def insert(self, index, value):
        super().insert(index, value)
        self._changed()

    # del lista[i]
    def __delitem__(self, index):
        super().__delitem__(index)
        self._changed()

    # lista.clear()
    def clear(self):
        super().clear()
        self._changed()

    # lista.pop()
    def pop(self, index=-1):
        value = super().pop(index)
        self._changed()
        return value

    # lista.remove(valor)
    def remove(self, value):
        super().remove(value)
        self._changed()