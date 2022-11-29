import numpy
import time

from metric_learn import MLKR
from sklearn.metrics import pairwise_distances

from scipy.special import logsumexp
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MyMLKR(MLKR):

    def __init__(self, weights=None, regularizer='L0', alpha=0.0,
            n_components=None, init='random', tol=None, max_iter=1000, verbose=False, preprocessor=None, random_state=None):
        self.weights = weights
        self.regularizer = regularizer
        self.alpha = alpha
        super().__init__(n_components, init, tol, max_iter, verbose, preprocessor, random_state)
        return

    def _weighted_loss(self, flatA, X, y, w=None):

        if w is None:
            w = numpy.ones_like(y)

        if self.n_iter_ == 0 and self.verbose:
            header_fields = ['Iteration', 'Objective Value', 'Time(s)']
            header_fmt = '{:>10} {:>20} {:>10}'
            header = header_fmt.format(*header_fields)
            cls_name = self.__class__.__name__
            print('[{cls}]'.format(cls=cls_name))
            print('[{cls}] {header}\n[{cls}] {sep}'.format(cls=cls_name, header=header, sep='-' * len(header)))

        start_time = time.time()

        A = flatA.reshape((-1, X.shape[1]))
        X_embedded = numpy.dot(X, A.T)
        dist = pairwise_distances(X_embedded, squared=True)
        numpy.fill_diagonal(dist, numpy.inf)
        softmax = numpy.exp(- dist - logsumexp(- dist, axis=1)[:, numpy.newaxis])
        yhat = softmax.dot(y)
        ydiff = yhat - y
        cost = (w * ydiff ** 2).sum()

        W = softmax * w[:, numpy.newaxis] * ydiff[:, numpy.newaxis] * (y - yhat[:, numpy.newaxis])
        W_sym = W + W.T
        numpy.fill_diagonal(W_sym, - W.sum(axis=0))
        grad = 4 * (X_embedded.T.dot(W_sym)).dot(X)

        if self.verbose:
            start_time = time.time() - start_time
            values_fmt = '[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}'
            print(values_fmt.format(cls=self.__class__.__name__, n_iter=self.n_iter_, loss=cost, start_time=start_time))
            sys.stdout.flush()

        self.n_iter_ += 1

        return cost, grad.ravel()

    def _loss(self, flatA, X, y):
        cost, grad = self._weighted_loss(flatA, X, y, self.weights)
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
        self.X_train = None
        self.y_train = None
        self.matrix = None
        return

    def fit(self, X, y, w=None, verbose=False):

        if w is None:
            w = numpy.ones_like(y)

        self.X_train = X
        self.y_train = y
        self.weights = w

        if self.X_norm is not None:
            if self.X_norm in ['minmax', 'MinMax']:
                self.X_scaler = MinMaxScaler()
            elif self.X_norm in ['standard', 'Standard']:
                self.X_scaler = StandardScaler()
            self.X_scaler.fit(self.X_train)
            self.X_train = self.X_scaler.transform(self.X_train)

        if self.y_norm is not None:
            if self.y_norm in ['minmax', 'MinMax']:
                self.y_scaler = MinMaxScaler()
            elif self.y_norm in ['standard', 'Standard']:
                self.y_scaler = StandardScaler()
            self.y_scaler.fit(self.y_train[:, numpy.newaxis])
            self.y_train = self.y_scaler.transform(self.y_train[:, numpy.newaxis]).ravel()

        model = MyMLKR(weights=self.weights, regularizer=self.regularizer, alpha=self.alpha, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)

        self.matrix = model.get_mahalanobis_matrix()

        return

    def predict(self, X):
        if self.X_norm is not None:
            X = self.X_scaler.transform(X)
        dX = X[:, numpy.newaxis, :] - self.X_train[numpy.newaxis, :, :]
        dX2 = numpy.einsum('ijk,ijl,kl->ij', dX, dX, self.matrix)
        softmax = numpy.exp( -dX2 - logsumexp(-dX2, axis=1)[:, numpy.newaxis] )
        y = numpy.einsum('ij,j->i', softmax, self.y_train)
        if self.y_norm is not None:
            y = self.y_scaler.inverse_transform(y[:, numpy.newaxis]).ravel()
        return y

    def errors(self):
        X = self.X_train
        y = self.y_train
        dX = X[:, numpy.newaxis, :] - self.X_train[numpy.newaxis, :, :]
        dX2 = numpy.einsum('ijk,ijl,kl->ij', dX, dX, self.matrix)
        numpy.fill_diagonal(dX2, numpy.inf)
        softmax = numpy.exp( -dX2 - logsumexp(-dX2, axis=1)[:, numpy.newaxis] )
        y_pred = numpy.einsum('ij,j->i', softmax, self.y_train)
        if self.y_norm is not None:
            y = self.y_scaler.inverse_transform(y[:, numpy.newaxis]).ravel()
            y_pred = self.y_scaler.inverse_transform(y_pred[:, numpy.newaxis]).ravel()
        y_err = y - y_pred
        return y_err

    def loss(self):
        y_err = self.errors()
        err = numpy.sqrt(numpy.mean(self.weights*y_err*y_err))
        return err


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

