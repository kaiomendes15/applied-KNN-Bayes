import numpy as np


class ZScoreScaler:
    
    def __init__(self):
        self.mean = None
        self.std = None


    def fit(self, X):
        
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std[self.std == 0] = 1.0


    def transform(self, X):

        if self.mean is None:
            raise RuntimeError("Chame fit() antes do transform()")

        return (X - self.mean) / self.std
