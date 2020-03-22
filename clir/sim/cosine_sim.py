from clir.sim.similarity import Similarity
import numpy as np


class CosineSimilarity(Similarity):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def compute(self, x_en, x_multi):
        x_en = np.array(x_en)
        x_multi = np.array(x_multi)
        sims = np.dot(x_en, x_multi.T) / (np.linalg.norm(x_en) * np.linalg.norm(x_multi, axis=1))
        return sims.tolist()
