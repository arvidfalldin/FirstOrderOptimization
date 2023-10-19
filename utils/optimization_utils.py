

class ObjectiveFunction():
    def __init__(self,
                 function_handle,
                 gradient_handle):
        self._eval = function_handle
        self._grad = gradient_handle

    def __call__(self, x):
        return self._eval(x)

    def grad(self, x):
        return self._grad(x)
