import numpy as np
import numba
import h5py
from metric_regressor_grad import compute_mse_grad_linear, compute_mse_grad_linear_ard

@numba.jit
def compute_K(X):
    '''Compute X.dot(X.T)
    '''
    N, p = X.shape
    K = np.zeros((N, N), dtype=X.dtype)
    for i in range(N):
        for j in range(N):
            if i <= j:
                for k in range(p):
                    K[i, j] += X[i, k] * X[j, k]
            else:
                K[i, j] = K[j, i]
    return K

@numba.jit
def compute_K_grad(A, X, k, l, g):
    '''
    Compute gradients of K (X.dot(X.T)) wrt. to A[k, l]
    :param A: [q, p]
    :param X: [N, p]
    :param k: 0 <= k < q
    :param l: 0 <= l < p
    :param g: output array for gradients. [q, p]
    :return: None
    '''
    N, p = X.shape

    for i in range(N):
        for j in range(N):
            if i <= j:
                for m in range(p):
                    g[i, j] = A[k, m] * (X[i, m] * X[j, l] + X[j, m] * X[i, l])
            else:
                g[i, j] = g[j, i]


@numba.jit
def compute_K12_grad(A, X1, X2, k, l, g):
    '''
    Compute gradients of K12 (X1.dot(X2.T)) wrt. to A[k, l]
    :param A: [q, p]
    :param X1: [N, p]
    :param X2: [M, p]
    :param k: 0 <= k < q
    :param l: 0 <= l < p
    :param g: output array for gradients. [q, p]
    :return: None
    '''
    N, p = X1.shape
    M = X2.shape[0]
    # g = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            for m in range(p):
                g[i, j] = A[k, m] * (X1[i, m] * X2[j, l] + X2[j, m] * X1[i, l])

'''
@numba.jit
def compute_mse_grad(A, X1, X2, Kinv1, K2, a, err):
    N1, p = X1.shape
    N2 = X2.shape[0]
    q = A.shape[0]

    mse_grad = np.zeros_like(A)
    K2Kinv1 = K2.dot(Kinv1)
    K1_grad = np.zeros((N1, N1))
    K2_grad = np.zeros((N2, N1))
    for k in range(q):
        for l in range(p):
            compute_K_grad(A, X1, k, l, K1_grad)
            compute_K12_grad(A, X2, X1, k, l, K2_grad)
            mse_grad[k, l] = err.T.dot((K2_grad + K2Kinv1.dot(K1_grad)).dot(a))
    return mse_grad
'''

class MetricRegressor(object):
    def __init__(self, input_dim=1, hidden_dim=1, alpha=0.0, sparse_rate=1.0,
                 kernel='linear_ard'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.kernel = kernel
        if kernel == 'linear_ard':
            self.theta = np.ones(input_dim)
        elif kernel == 'projection':
            self.theta = np.random.normal(loc=2.0 / (input_dim + hidden_dim),
                                  size=(hidden_dim, input_dim))
        else:
            raise ValueError('unknown kernel: %s'%kernel)
        if sparse_rate < 1.0:
            self.mask = np.random.choice([0, 1], size=self.theta.shape,
                                     p=[1.0 - sparse_rate, sparse_rate]).astype('int32')
            self.theta[self.mask] = 0.0
        else:
            self.mask = None

    def optimize(self, X1, X2, y1, y2, theta, alpha):
        N1, p = X1.shape
        # forward pass
        X1_ = self.kernel_transform(X1, theta)
        X2_ = self.kernel_transform(X2, theta)
        K1 = X1_.dot(X1_.T)
        K1[np.r_[:N1], np.r_[:N1]] += alpha
        Kinv1 = np.linalg.inv(K1)
        K2 = X2_.dot(X1_.T)
        a = Kinv1.dot(y1)
        y2_pred = K2.dot(a)

        err = y2_pred - y2
        mse = np.mean(err ** 2)
        # backward pass
        if self.kernel == 'linear_ard':
            mse_grad, K_grad = compute_mse_grad_linear_ard(theta, X1, X2, Kinv1, K2, a, err, self.mask)
        elif self.kernel == 'projection':
            mse_grad = compute_mse_grad_linear(theta, X1, X2, Kinv1, K2, a, err, self.mask)
        else:
            raise ValueError('unknown kernel: %s'%self.kernel)
        return mse, mse_grad

    def kfold_generator(self, X, y, n_splits):
        from sklearn.model_selection import KFold
        while True:
            kfold = KFold(n_splits, shuffle=True)
            for train_index, test_index in kfold.split(X, y):
                yield X[train_index], X[test_index], y[train_index], y[test_index]

    def kernel_transform(self, X, theta):
        if self.kernel == 'linear_ard':
            return X*theta.reshape((1, -1))
        elif self.kernel == 'projection':
            return X.dot(theta.T)
        else:
            raise ValueError('unknown kernel: %s'%self.kernel)

    def fit(self, X, y, data_generator=None,
            lr=0.001, max_iter=100, tol=1e-5, n_batches=3, momentum=0.9):
        y = y.reshape((-1, 1))
        N, p = X.shape

        mse = 0.0
        mses = []
        velocities = []
        velocity = np.zeros_like(self.theta)
        velocity_mask = (np.random.randint(10, size=self.theta.shape) == 0)
        mse_grads = []

        if data_generator is None:
            data_generator = self.kfold_generator(X, y, n_batches)
        pbar = tqdm(total=max_iter*n_batches)
        for i in range(max_iter):
            mses_epoch = np.zeros(n_batches)
            #mse_grad_epoch = np.zeros([n_batches] + list(self.theta.shape))
            for j in range(n_batches):
                X1, X2, y1, y2 = data_generator.next()
                mse_cur_j, mse_grad = self.optimize(X1, X2, y1, y2, self.theta, self.alpha)
                velocity = velocity*momentum + lr*mse_grad
                self.theta -= velocity
                mses_epoch[j] = mse_cur_j
                pbar.set_postfix(mse=mse_cur_j)
                pbar.update(1)
                mses.append(mse_cur_j)
                #mse_grad_epoch[j] = mse_grad
                velocities.append(velocity[velocity_mask].reshape((1, -1)))
                mse_grads.append(mse_grad[velocity_mask].reshape((1, -1)))

            #mse_grad_epoch = mse_grad_epoch.mean(axis=0)
            #velocity = velocity * momentum + lr * mse_grad
            #self.theta -= velocity

            mse_epoch = np.mean(mses_epoch)
            if i > 0:
                if np.abs(mse_epoch - mse) < tol:
                    break
            mse = mse_epoch

        pbar.close()

        mses = np.asarray(mses)
        self.velocities_ = np.concatenate(velocities, axis=0)
        self.mse_grads_ = np.concatenate(mse_grads, axis=0)
        self.mses_ = mses
        self.X_ = self.kernel_transform(X, self.theta)
        self.y = y
        K = compute_K(self.X_)
        K[np.r_[:N], np.r_[:N]] += self.alpha
        self.a = np.linalg.inv(K).dot(y)

    def fit2(self, X, y):
        N, p = X.shape
        y = y.reshape((-1, 1))
        X_ = X.dot(self.A.T)
        K = X_.dot(X_.T)
        K[np.r_[:N], np.r_[:N]] += self.alpha
        a = np.linalg.inv(K).dot(y)
        self.a = a
        self.X = X
        self.y = y

    def predict(self, X):
        X2_ = self.kernel_transform(X, self.theta)
        return X2_.dot(self.X_.T).dot(self.a)

    def save(self, filename):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('theta', data=self.theta)
            f.create_dataset('X_', data=self.X_)
            f.create_dataset('a', data=self.a)
