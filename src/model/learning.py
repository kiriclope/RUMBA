import numpy as np

class HebbianLearningModels:

    def __init__(self):
        self.eta = 0
        self.alpha = 0
        
    def hebbian_rule(self, wij, ri, rj):

        Dwij = self.eta * ri * rj

    def oja_rule(self, wij, ri, rj):  # sum wij2 = 1/alpha
        Dwij = self.eta * (ri * rj - self.alpha * ri * rj * wij)

    # def bcm_rule(self, wij, ri, rj):
