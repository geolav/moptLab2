class CallCounter:
    """Оборачивает функцию и считает число вызовов."""
    def __init__(self, func):
        self.func  = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

    def reset(self):
        self.count = 0