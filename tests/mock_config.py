class MockConfig:
    def __init__(self, **kwargs):
        self.d = kwargs
    def __getattr__(self, name):
        return self.d.get(name)
    def __getitem__(self, name):
        return self.d[name]
    def get(self, name, default=None):
        return self.d.get(name, default)
    def update(self, d):
        self.d.update(d)
    def copy(self):
        return MockConfig(**self.d.copy())
