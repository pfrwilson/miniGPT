class Registry: 
    def __init__(self): 
        self._objects = {}

    def register(self, name=None):
        def wrapper(obj): 
            key = name or obj.__name__
            self._objects[key] = obj 
            return obj 

        return wrapper 

    def get(self, key): 
        return self._objects[key]

    def list(self): 
        return list(self._objects.keys())

    def dict(self): 
        return self._objects.copy()