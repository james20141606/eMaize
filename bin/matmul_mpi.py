#! /usr/bin/env python
import numpy as np
from mpi4py import MPI

def matmul_mpi(m=1000, r=1000, k=200):
    """A(m, r) x B(r, n) = C(m, n)
    """
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    n = comm_size * k

    if comm_rank == 0:
        A = np.random.randint(5, size=(m, r), dtype='int64')
        BT = np.random.randint(5, size=(comm_size, r * k), dtype='int64')
        CT = np.empty((comm_size, m * k), dtype='int64')
    else:
        A = np.empty((m, r), dtype='int64')
        BT = None
        CT = None
    bT = np.empty(r * k)
    comm.Bcast(A, root=0)
    comm.Scatter(BT, bT, root=0)
    bT = bT.reshape((k, r))
    cT = np.dot(A, bT.T).T
    cT = np.ravel(cT)
    comm.Gather(cT, CT, root=0)
    if comm_rank == 0:
        pass
        #CT = CT.reshape((k * comm_size, m))
        #BT = BT.reshape((k * comm_size, r))
        #C = CT.T
        #C_correct = np.dot(A, BT.T)
        # np.savetxt('tmp/C.txt', C, fmt='%d')
        # np.savetxt('tmp/C_correct.txt', C_correct, fmt='%d')
        #assert np.allclose(np.ravel(C_correct), np.ravel(C))

def matmul(m, r, k):
    A = np.random.randint(5, size=(m, r), dtype='int64')
    B = np.random.randint(5, size=(r, k), dtype='int64')
    C = np.dot(A, B)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    if comm.Get_size() > 1:
        matmul_mpi(1000, 1000, 20000)
    else:
        matmul(2500, 2500, 200*10)





