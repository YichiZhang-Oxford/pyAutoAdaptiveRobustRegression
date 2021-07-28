from ctypes import *
import numpy as np
import os
from sys import platform

basepath = os.path.dirname(os.path.abspath(__file__))

if platform == 'win32':
    dllpath = os.path.join(basepath, "./bin/win32/pyAutoAdaptiveRobustRegression.dll")
elif platform == 'linux' or platform == 'linux2':
    dllpath = os.path.join(basepath, "./bin/linux/pyAutoAdaptiveRobustRegression.so")
elif platform == "darwin":
    dllpath = os.path.join(basepath, "./bin/macos/pyAutoAdaptiveRobustRegression.dylib")
else:
    print("invalid platform")
    exit(1)

# print(dllpath)

lib = CDLL(dllpath)

# -----
# MyVec
# -----

_free_my_vec = lib.freeMyVec
_free_my_vec.restype = None

class MyVec(Structure):
    _fields_ = [("data", POINTER(c_double)),
                ("n", c_size_t)]

    def make(self, np_array):
        self.data = np_array.ctypes.data_as(POINTER(c_double))
        self.n = len(np_array)

    def free(self):
        _free_my_vec(self)

    def to_numpy(self):
        return np.ctypeslib.as_array(self.data, (self.n,))

_free_my_vec.argtypes = (POINTER(MyVec),)

# -----
# MyMat
# -----

_free_my_mat = lib.freeMyMat
_free_my_mat.restype = None

class MyMat(Structure):
    _fields_ = [("data", POINTER(c_double)),
                ("num_rows", c_size_t),
                ("num_cols", c_size_t),]

    def make(self, np_array):
        self.data = np_array.ctypes.data_as(POINTER(c_double))
        self.num_rows, self.num_cols = np_array.shape

    def free(self):
        _free_my_mat(self)

    def to_numpy(self):
        return np.ctypeslib.as_array(self.data, (self.num_rows, self.num_cols))

_free_my_mat.argtypes = (POINTER(MyMat),)

# ----------
# Huber Mean
# ----------

_huberMean = lib.huberMean
_huberMean.argtypes = (MyVec, c_double, c_int)
_huberMean.restype = c_double

def huber_mean(X, tol=0.001, ite_max=500):
    _X = MyVec()
    _X.make(X)
    result = _huberMean(_X, tol, ite_max)
    return result

# ---------
# Huber Cov
# ---------

class HuberCovResult(Structure):
    _fields_ = [("mu", MyVec),
                ("cov", MyMat)]

_huberCov = lib.huberCov
_huberCov.argtypes = (POINTER(MyMat),)
_huberCov.restype = HuberCovResult

_freeHuberCovResult = lib.freeHuberCovResult
_freeHuberCovResult.argtypes = (POINTER(HuberCovResult),)
_freeHuberCovResult.restype = None

def huber_cov(X):
    _X = MyMat()
    _X.make(X)
    result_cpp = _huberCov(_X)
    result = {
        "mu": result_cpp.mu.to_numpy(),
        "cov": result_cpp.cov.to_numpy()
    }
    _freeHuberCovResult(result_cpp)
    return result

# ----------------
# Huber Regression
# ----------------

_huber_reg = lib.huberReg
_huber_reg.argtypes = (MyMat, MyVec, c_double, c_double, c_int)
_huber_reg.restype = MyVec

def huber_reg(X, Y, tol = 0.0001, constTau = 1.345, ite_max = 5000):
    _X = MyMat()
    _Y = MyVec()
    _X.make(X)
    _Y.make(Y)
    result_cpp = _huber_reg(_X, _Y, tol, constTau, ite_max)
    result = result_cpp.to_numpy()
    result_cpp.free()
    return result

# -------------------------
# Adaptive Huber Regression
# -------------------------

_ada_huber_reg = lib.adaHuberReg
_ada_huber_reg.argtypes = (MyMat, MyVec, c_double, c_int)
_ada_huber_reg.restype = MyVec

def ada_huber_reg(X, Y, tol = 0.0001, ite_max = 5000):
    _X = MyMat()
    _Y = MyVec()
    _X.make(X)
    _Y.make(Y)
    result_cpp = _ada_huber_reg(_X, _Y, tol, ite_max)
    result = result_cpp.to_numpy()
    result_cpp.free()
    return result

# -------------------------
# Adaptive Gradient Descent
# -------------------------

_agd = lib.agd
_agd.argtypes = (MyVec, c_double, c_int)
_agd.restype = (c_double)

def agd(Y, epsilon = 1e-5, ite_max = 5000):

    _Y = MyVec()
    _Y.data = Y.ctypes.data_as(POINTER(c_double))
    _Y.n = len(Y)
    result = _agd(_Y, epsilon, ite_max)
    return result

# ------------------------------------------------------
# Adaptive Gradient Descent with Barzilai-Borwein Method
# ------------------------------------------------------

_agdBB = lib.agdBB
_agdBB.argtypes = (MyVec, c_double, c_int)
_agdBB.restype = (c_double)

def agd_bb(Y, epsilon = 1e-5, ite_max = 5000):

    _Y = MyVec()
    _Y.data = Y.ctypes.data_as(POINTER(c_double))
    _Y.n = len(Y)
    result = _agdBB(_Y, epsilon, ite_max)
    return result

# --------------------------------------------------
# Adaptive Gradient Descent with Backtracking Method
# --------------------------------------------------

_agdBacktracking = lib.agdBacktracking
_agdBacktracking.argtypes = (MyVec, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_int)
_agdBacktracking.restype = (c_double)

def agd_backtracking(Y, s1 = 1.0, gamma1 = 0.5, beta1 = 0.8, s2 = 1.0, gamma2 = 0.5, beta2 = 0.8, epsilon = 1e-5, ite_max = 5000):

    _Y = MyVec()
    _Y.data = Y.ctypes.data_as(POINTER(c_double))
    _Y.n = len(Y)
    result = _agdBacktracking(_Y, s1, gamma1, beta1, s2, gamma2, beta2, epsilon, ite_max)
    return result