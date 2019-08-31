import numpy as np

class sigmoid:
    def calc(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.calc(x) * (1 - self.calc(x))
