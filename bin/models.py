import numpy as np
from scipy.stats import linregress, pearsonr
from utils import take2d
from tqdm import tqdm
import h5py
import scipy

class FastCVRidge(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        Kinv = np.linalg.inv(X.T.dot(X) + self.alpha*np.identity(X.shape[0]))
        self.S = X.dot(Kinv).dot(X.T)
        self.y_ = self.S.dot(y)
        self.e_ = self.y_ - y

    def leave_one_out(self):
        return np.mean((np.ravel(self.e_) / (1.0 - np.diag(self.S))) ** 2)

    def kfold(self, test_indices, return_mean=True, approx=False):
        k = len(test_indices)
        mse_cv = np.zeros(k)
        S_diag = np.diag(self.S)
        for i, indices, in enumerate(test_indices):
            if approx:
                mse_cv[i] = np.mean((np.ravel(self.e_[indices]) / (1.0 - S_diag[indices])) ** 2)
            else:
                Si = take2d(self.S, indices, indices)
                Ii = np.identity(Si.shape[0])
                Bi = Si.dot(np.linalg.inv(Ii - Si)) + Ii
                ei_ = self.e_[indices]
                mse_cv[i] = np.squeeze(ei_.T.dot(Bi.T).dot(Bi).dot(ei_)) / indices.shape[0]
        if return_mean:
            return mse_cv.mean()
        else:
            return mse_cv

class FastLMMSimple(object):
    def __init__(self, G0=None, alpha=1.0, Smin=0.1):
        self.alpha = alpha
        self.Smin = Smin
        self.G0 = G0
        self.K0 = G0.dot(G0.T)
        U, S, V = np.linalg.svd(G0)
        S **= 2
        S_nonzero = (S >= Smin)
        self.U = U[:, S_nonzero]
        self.S = S[S_nonzero]

    def standardize_K(self, K):
        return K * K.shape[0] / np.trace(K)

    def log_likelihood(self, X, y, h2, UX=None, Uy=None):
        y = y.reshape((-1, 1))
        N = X.shape[0]
        k = self.U.shape[1]

        S = self.S
        U = self.U
        LL = 0.0
        h2 = np.clip(h2, 1e-10, 1.0 - 1e-10)
        delta = 1.0 / h2 - 1
        Sd = self.S + delta

        # transformed X and y for fixed effects
        X_ = np.empty((k + N, X.shape[1]))
        y_ = np.empty((k + N, 1))
        if UX is None:
            UX = U.T.dot(X)
        if Uy is None:
            Uy = U.T.dot(y)
        Sd_sqrt = np.sqrt(Sd)
        X_[:k] = UX / Sd_sqrt[:, np.newaxis]
        y_[:k] = Uy / Sd_sqrt[:, np.newaxis]
        X_[k:] = (X - U.dot(UX)) / np.sqrt(delta)
        y_[k:] = (y - U.dot(Uy)) / np.sqrt(delta)
        U_, S_, V_ = np.linalg.svd(X_, full_matrices=False)
        V_ = V_.T
        D_ = S_ / (S_ * S_ + self.alpha)
        beta = (V_ * D_[np.newaxis, :]).dot(U_.T).dot(y_)
        var_g = np.mean((y_ - X_.dot(beta)) ** 2)

        logdetK = np.log(Sd).sum()
        logdelta = (N - k) * np.log(delta)
        logvar_g = N * np.log(var_g)
        LL = -0.5 * (N * np.log(2 * np.pi) + logdetK + logdelta + N + logvar_g)
        result = {'LL': LL,
                  'beta': beta,
                  'var_g': var_g,
                  'h2': h2}
        return result

    def optimize(self):
        UX = self.U.T.dot(self.X)
        Uy = self.U.T.dot(self.y)

        h2grid = np.linspace(0.0, 1.0, 11, endpoint=True)
        h2grid[0] = 0.01
        h2grid[-1] = 0.99
        LL = np.zeros((len(h2grid)))

        def f(h2):
            result = self.log_likelihood(self.X, self.y, h2, UX, Uy)
            print('h2=%f, LL=%f' % (h2, result['LL']))
            return -result['LL']

        h2 = scipy.optimize.fminbound(f, x1=0.0, x2=0.01)
        result = self.log_likelihood(self.X, self.y, h2, UX, Uy)

        self.beta_ = result['beta']
        self.LL_ = result['LL']

    def fit(self, X, y, G0, h2=None):
        y = y.reshape((-1, 1))
        self.X = X
        self.y = y
        self.K0 = G0.dot(G0.T)
        S, U = np.linalg.eigh(self.K0)
        S_nonzero = (S >= self.Smin)
        self.U = U[:, S_nonzero]
        self.S = S[S_nonzero]

        result = self.log_likelihood(X, y, h2)
        self.beta_ = result['beta']
        self.LL_ = result['LL']
        self.var_g_ = result['var_g']
        self.h2_ = result['h2']

    def predict(self, X, y, K0_star):
        var_g = self.var_g_
        h2 = self.h2_
        var_e = (1.0 / h2 - 1.0) * var_g
        C = var_g * self.K0 + var_e * np.identity(self.K0.shape[0])
        k_star = self.X.dot(X.T)
        a = var_g * k_star.T.dot(np.linalg.inv(C))
        c = X.dot(X.T) + var_g * X.dot(X.T) + var_e * np.identity(X.shape[0])
        y_mean = X.dot(self.beta_) + a.dot(self.y)
        y_var = c - a.dot(k_star)
        return y_mean, y_var

class MixedRidge(object):
    def __init__(self, alphas=None):
        if alphas is not None:
            self.alphas = np.asarray(alphas)
        else:
            self.alphas = alphas

    def regress_beta(self, X, y, alphas=None, return_S=False):
        U, D, V = np.linalg.svd(X, full_matrices=False)
        D2 = D ** 2
        V = V.T
        if alphas is not None:
            if len(alphas.shape) > 0 and (alphas.shape[0] != X.shape[1]):
                raise ValueError('number of alphas is not equal to the number of columns in X')
            Kinv = (D2 / (D2 + alphas))[np.newaxis, :]
        else:
            Kinv = 1.0 / D2
        if self.alphas is not None:
            beta = (V * (D / (D2 + alphas))[np.newaxis, :]).dot(U.T).dot(y)
        else:
            beta = (V / D[np.newaxis, :]).dot(U.T).dot(y)
        if return_S:
            if self.alphas is not None:
                S = (U * Kinv).dot(U.T)
            else:
                S = U.dot(U.T)
            return beta, S
        else:
            return beta

    def optimize(self, X1, X2, y, max_iter=100, tol=1e-5):
        p1, p2 = X1.shape[1], X2.shape[1]
        N = X1.shape[0]
        X = np.zeros((N, p1 + p2 + 1))
        X[:, -1] = 1.0

        mse = np.mean(y ** 2)
        delta_mse = mse
        gamma = np.random.uniform(0.0, 1.0)
        i_iter = 0
        gammas = []
        mses = []
        while np.abs(delta_mse) > tol:
            X[:, :p1] = X1 * (1.0 - gamma)
            X[:, p1:-1] = X2 * gamma
            beta = self.regress_beta(X, y, alphas=self.alphas)
            mse_cur = np.mean((y - X.dot(beta)) ** 2)
            delta_mse = mse_cur - mse
            mse = mse_cur
            y1 = X1.dot(beta[:p1])
            y2 = X2.dot(beta[p1:-1])
            res = beta[-1]
            y_gamma = np.ravel(y - y1 - res)
            x_gamma = np.ravel(y2 - y1)
            gamma, intercept, _, _, _ = linregress(x_gamma, y_gamma)

            gammas.append(gamma)
            mses.append(mse)
            i_iter += 1
            if i_iter > max_iter:
                break
        gammas = np.asarray(gammas)
        mses = np.asarray(mses)

        self.gamma_ = gammas[np.argmin(mses)]

    def optimize_grid(self, X1, X2, y, gamma_min=0.0, gamma_max=1.0, n_gamma=21, cv=False):
        p1, p2 = X1.shape[1], X2.shape[1]
        N = X1.shape[0]
        X = np.zeros((N, p1 + p2 + 1))
        X[:, -1] = 1.0

        mses = np.zeros(n_gamma)
        gammas = np.linspace(gamma_min, gamma_max, n_gamma, endpoint=True)
        for i, gamma in tqdm(enumerate(gammas), total=n_gamma):
            X[:, :p1] = X1 * (1.0 - gamma)
            X[:, p1:-1] = X2 * gamma
            beta = self.regress_beta(X, y, alphas=self.alphas)
            mses[i] = np.mean((y - X.dot(beta)) ** 2)
        self.gamma_ = gammas[np.argmin(mses)]
        self.mse_ = mses[np.argmin(mses)]
        self.mses_ = mses

        beta, S = self.regress_beta(X, y, alphas=self.alphas, return_S=True)
        self.beta_ = beta
        self.beta_ = beta
        self.beta1_ = beta[:p1]
        self.beta2_ = beta[p1:-1]
        self.intercept_ = beta[-1]
        if cv:
            self.S = S
            self.y_ = self.S.dot(y)
            self.e_ = self.y_ - y
            self.y = y

    def fit(self, X1, X2, y, gamma=0.5, cv=False):
        # mix factor
        if X1.shape[0] != X2.shape[0]:
            raise ValueError('number of rows in X1 and X2 is not equal')
        N = X1.shape[0]
        if y.shape[0] != N:
            raise ValueError('number of rows in y is not equal to X1 and X2')

        p1, p2 = X1.shape[1], X2.shape[1]
        X = np.zeros((N, p1 + p2 + 1))
        X[:, -1] = 1.0

        X[:, :p1] = X1 * (1.0 - gamma)
        X[:, p1:-1] = X2 * gamma
        beta, S = self.regress_beta(X, y, alphas=self.alphas, return_S=True)
        y1 = X1.dot(beta[:p1])
        y2 = X2.dot(beta[p1:-1])
        self.y1_ = y1
        self.y2_ = y2

        if cv:
            self.S = S
            self.y_ = self.S.dot(y)
            self.e_ = self.y_ - y
            self.y = y

        self.gamma_ = gamma
        self.beta_ = beta
        self.beta1_ = beta[:p1]
        self.beta2_ = beta[p1:-1]
        self.intercept_ = beta[-1]

    def predict(self, X1, X2):
        X = np.concatenate([X1 * (1.0 - self.gamma_),
                            X2 * self.gamma_,
                            np.ones((X1.shape[0], 1), dtype=X1.dtype)], axis=1)
        return X.dot(self.beta_)

    def kfold(self, test_indices, subset_indices=None, return_mean=True, approx=False):
        k = len(test_indices)
        S_diag = np.diag(self.S)
        #err_n = np.zeros(self.S.shape[0])
        #err_sum = np.zeros(self.S.shape[0])
        #err_ss = np.zeros(self.S.shape[0])

        pcc_cv = np.full(k, np.nan)
        mse_cv = np.full(k, np.nan)

        if subset_indices is None:
            subset_indices = test_indices
        for i, indices, in tqdm(enumerate(test_indices), total=len(test_indices)):
            if approx:
                mse_cv[i] = np.mean((np.ravel(self.e_[subset_indices]) / (1.0 - S_diag[subset_indices])) ** 2)
            else:
                Si = take2d(self.S, indices, indices)
                Ii = np.identity(Si.shape[0])
                Bi = Si.dot(np.linalg.inv(Ii - Si)) + Ii
                ei_ = self.e_[indices]
                err = Bi.dot(ei_)
                #err_n[indices] += 1
                #err_sum[indices] += np.ravel(err)
                #err_ss[indices] += np.ravel(err ** 2)
                if subset_indices[i].shape[0] > 0:
                    err_all = np.full(self.S.shape[0], np.nan)
                    err_all[indices] = err
                    err_subset = err_all[subset_indices[i]]
                    mse_cv[i] = np.squeeze(err_subset.T.dot(err_subset)) / subset_indices[i].shape[0]
                    pcc_cv[i] = pearsonr(self.y_[subset_indices[i]], self.y_[subset_indices[i]] + err_subset)[0]

        #self.err_sum = err_sum
        #self.err_ss = err_ss
        #self.err_n = err_n
        self.pcc_cv = pcc_cv
        if return_mean:
            return mse_cv.mean()
        else:
            return mse_cv

    def leave_one_out(self):
        return np.mean((np.ravel(self.e_) / (1.0 - np.diag(self.S))) ** 2)






