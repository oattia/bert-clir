from clir.cca.CCA import CCA as CCABase

from sklearn.cross_decomposition import CCA


class LinearCCAModel(CCABase):

    def __init__(self, num_components):
        super(LinearCCAModel, self).__init__(num_components)
        self.model = None

    def train(self, x_en, x_multi):
        self.model = CCA(n_components=self.num_components)
        self.model.fit(x_en, x_multi)

    def predict(self, x_en, x_multi):
        X_c, Y_c = self.model.transform(x_en, x_multi)
        return X_c, Y_c
