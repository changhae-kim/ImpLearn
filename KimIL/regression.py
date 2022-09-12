import numpy

from metric_learn import MLKR

from scipy.special import logsumexp
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class KimMLKR(MLKR):

    def __init__(self, regularizer='L0', alpha=0.0,
            n_components=None, init='random', tol=None, max_iter=1000, verbose=False, preprocessor=None, random_state=None):
        self.regularizer = regularizer
        self.alpha = alpha
        super().__init__(n_components, init, tol, max_iter, verbose, preprocessor, random_state)
        return

    def _loss(self, flatA, X, y):
        cost, grad = super()._loss(flatA, X, y)
        if self.regularizer in ['l1', 'L1', 'lasso', 'Lasso', 'LASSO']:
            n = len(y)
            reg_cost = cost / n + self.alpha * numpy.sum(numpy.abs(flatA))
            reg_grad = grad / n + self.alpha * numpy.sign(flatA)
        elif self.regularizer in ['l2', 'L2', 'ridge', 'Ridge']:
            n = len(y)
            reg_cost = cost / n + self.alpha * numpy.sum(flatA * flatA)
            reg_grad = grad / n + 2.0 * self.alpha * flatA
        else:
            reg_cost = cost
            reg_grad = grad
        return reg_cost, reg_grad


class Kernel():

    def __init__(self, norm='minmax', X_norm=None, y_norm=None, regularizer='L0', alpha=0.0, random_state=None):
        if X_norm is None:
            X_norm = norm
        if y_norm is None:
            y_norm = norm
        self.X_norm = X_norm
        self.y_norm = y_norm
        self.regularizer = regularizer
        self.alpha = alpha
        self.random_state = random_state
        return

    def fit(self, X, y, verbose=False):

        self.X_train = X
        self.y_train = y

        if self.X_norm:
            if self.X_norm in ['minmax', 'MinMax']:
                self.X_scaler = MinMaxScaler()
            elif self.X_norm in ['standard', 'Standard']:
                self.X_scaler = StandardScaler()
            self.X_scaler.fit(self.X_train)
            self.X_train = self.X_scaler.transform(self.X_train)

        if self.y_norm:
            if self.y_norm in ['minmax', 'MinMax']:
                self.y_scaler = MinMaxScaler()
            elif self.y_norm in ['standard', 'Standard']:
                self.tearget_scaler = StandardScaler()
            self.y_scaler.fit(self.y_train[:, numpy.newaxis])
            self.y_train = self.y_scaler.transform(self.y_train[:, numpy.newaxis]).ravel()

        model = KimMLKR(regularizer=self.regularizer, alpha=self.alpha, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)

        self.matrix = model.get_mahalanobis_matrix()

        return

    def predict(self, X):
        if self.X_norm:
            X = self.X_scaler.transform(X)
        dX = X[:, numpy.newaxis, :] - self.X_train[numpy.newaxis, :, :]
        dX2 = numpy.einsum('ijk,ijl,kl->ij', dX, dX, self.matrix)
        softmax = numpy.exp( -dX2 - logsumexp(-dX2, axis=1)[:, numpy.newaxis] )
        y = numpy.einsum('ij,j->i', softmax, self.y_train)
        if self.y_norm:
            y = self.y_scaler.inverse_transform(y[:, numpy.newaxis]).ravel()
        return y


if __name__ == '__main__':

    rng = numpy.random.RandomState(0)

    
    n_samples, n_features = 10, 5
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    print(X)
    print(y)
    kernel = Kernel(norm=None)
    kernel.fit(X, y)
    print(kernel.matrix)
    z = kernel.predict(X)
    print(z)
    
    '''
    n_samples, n_features = 200, 2
    X = 5.0 * rng.randn(n_samples, n_features)
    y = numpy.cos((X[:, 0]**2.0 + X[:, 1]**2.0)**0.5)
    print(X)
    print(y)
    kernel = Kernel(norm=None)
    kernel.fit(X, y)
    print(kernel.matrix)
    n_tests = 20
    Z = 5.0 * rng.randn(n_tests, n_features)
    z = numpy.cos((Z[:, 0]**2.0 + Z[:, 1]**2.0)**0.5)
    print(z)
    z = kernel.predict(Z)
    print(z)
    '''

