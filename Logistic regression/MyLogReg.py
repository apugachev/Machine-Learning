import time
import numpy as np
from scipy.special import expit
from numpy import logaddexp
from sklearn.metrics import log_loss
from scipy.spatial.distance import euclidean
from numpy import linalg as LA
from sklearn.base import BaseEstimator


class LogRegBonus(BaseEstimator):
    def __init__(self, lambda_1=0.001, lambda_2=0, gd_type='full',
                 tolerance=1e-6, max_iter=1000, w0=None, alpha=1e-3, hist=True):
        """
        lambda_1: L1 regularization param
        lambda_2: L2 regularization param
        gd_type: 'full' or 'stochastic'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        alpha: learning rate
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.alpha = alpha
        self.w = None
        self.time_history = []
        self.loss_history = []
        self.hist = hist

    def fit(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: self
        """

        if self.w0 is None:
            self.w = np.zeros(X[0].size)
        else:
            self.w = self.w0

        fulltime = 0
        i = 0

        X = np.array(X)
        y = np.array(y)

        while True:

            start = time.clock()

            if (self.gd_type == 'full'):
                answ = self.calc_gradient(X, y)
            else:
                rand_index = np.random.randint(0, len(X))
                rand_x = X[rand_index]
                rand_y = y[rand_index]
                answ = self.calc_gradient(rand_x, rand_y)

            fulltime += time.clock() - start

            w_new = self.w - self.alpha * answ

            dist = euclidean(w_new, self.w)

            if self.hist:
                self.time_history.append(fulltime)
                self.loss_history.append(self.calc_loss(X, y))

            i = i + 1
            self.w = w_new

            if (abs(dist) < self.tolerance or i > self.max_iter):
                break

        return self

    def predict_proba(self, X):
        """
        X: np.array of shape (l, d)
        ---
        output: np.array of shape (l, 2) where
        first column has probabilities of -1
        second column has probabilities of +1
        """
        if self.w is None:
            raise Exception('Not trained yet')

        pred = np.empty((2, len(X)))
        prob = expit(X.dot(self.w))
        pred[0] = np.ones(len(X)) - prob
        pred[1] = prob
        pred = pred.transpose()

        return pred

    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        # gradient = np.zeros(X[0].size)
        
        L1 = self.lambda_1 * np.sign(self.w)
        L1[abs(self.w) < self.tolerance] = 0 # зануляем близкие к нулю значения весов
        L2 = self.lambda_2 * self.w
        
        regularization = L1 + L2

        if (isinstance(X[0], np.float64)):
            return -y * X / (1 + np.exp(y * np.dot(self.w, X))) + regularization
        else:
            return np.dot((-y * expit(-y * np.dot(X, self.w))), X) / len(X) + regularization

        # gradient += regularization
        # return gradient

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float
        """

        loss_fun = -1 * np.log(expit(y * np.dot(X, self.w))).mean() + \
            self.lambda_2 * LA.norm(self.w)**2 / 2 + \
            self.lambda_1 * LA.norm(self.w)

        return loss_fun
