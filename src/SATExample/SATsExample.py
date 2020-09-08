import os
import sys
import numpy as np
import cvxopt
from cvxopt import matrix,spmatrix
import pandas as pd
import time
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize
print(os.getcwd())
np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
from genTestDat import genTestData2D, prodMats2D
from est2d import *
from npMatrix2d import *

def recodeFactor(factor):

    # New copy of factor vector
    factor = np.array(factor)

    # Work out unique levels of the factor
    uniqueValues = np.unique(factor)
    
    # Loop through the levels replacing them with
    # a 0:l coding where l is the number of levels
    for i in np.arange(len(uniqueValues)):

        factor[factor==uniqueValues[i]]=i

    return(factor)

def testSchoolExample():

    # Read in csv
    dataReduced = pd.read_csv('./schools_reduced.csv')

    # Number of subjects in reduced model
    nr = len(dataReduced)

    # Work out factors for reduced model
    studfac = recodeFactor(np.array(dataReduced['studid'].values))
    tchrfac = recodeFactor(np.array(dataReduced['tchrid'].values))

    # Work out math and year for reduced model
    math = np.array(dataReduced['math'].values).reshape(len(dataReduced),1)
    year = np.array(dataReduced['year'].values).reshape(len(dataReduced),1)

    # Construct X for reduced model
    X = np.concatenate((np.ones((nr,1)),year),axis=1)
    Y = math

    # Construct Z for reduced model
    Z_f1 = np.zeros((nr,len(np.unique(studfac))))
    Z_f1[np.arange(nr),studfac] = 1
    Z_f2 = np.zeros((nr,len(np.unique(tchrfac))))
    Z_f2[np.arange(nr),tchrfac] = 1
    Z = np.concatenate((Z_f1,Z_f2),axis=1)

    # Convergence tolerance
    tol = 1e-6

    # nlevels for reduced
    nlevels = np.array([len(np.unique(studfac)),len(np.unique(tchrfac))])

    # nraneffs for reduced
    nraneffs = np.array([1,1])

    # Obtain the product matrices and start recording the computation time
    t1 = time.time()
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Run Fisher Scoring
    paramVector_FS,_,nit,llh = FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, nr, init_paramVector=None)
    t2 = time.time()

    print('FS')
    print(t2-t1)
    print(nit)
    print(paramVector_FS)
    print(paramVector_FS[3:,]*paramVector_FS[2])
    print(llh-234/2*np.log(2*np.pi))

    sigma2 = paramVector_FS[2]
    d_sigma2t = np.diag(np.repeat(paramVector_FS[3], nlevels[0]))
    d_sigma2s = np.diag(np.repeat(paramVector_FS[4], nlevels[1]))

    Dest = scipy.linalg.block_diag(d_sigma2t,d_sigma2s)

    varb = sigma2*np.linalg.inv(XtX - XtZ @ forceSym2D(np.linalg.solve(np.eye(Dest.shape[0]) + Dest @ ZtZ, Dest)) @ ZtX)
    print(np.sqrt(np.diagonal(varb)))


    
    # Obtain the product matrices and start recording the computation time
    t1 = time.time()
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Run Fisher Scoring
    paramVector_FS,_,nit,llh = pFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, nr, init_paramVector=None)
    t2 = time.time()

    print('pFS')
    print(t2-t1)
    print(nit)
    print(paramVector_FS)
    print(paramVector_FS[3:,]*paramVector_FS[2])
    print(llh-234/2*np.log(2*np.pi))

    sigma2 = paramVector_FS[2]
    d_sigma2t = np.diag(np.repeat(paramVector_FS[3], nlevels[0]))
    d_sigma2s = np.diag(np.repeat(paramVector_FS[4], nlevels[1]))

    Dest = scipy.linalg.block_diag(d_sigma2t,d_sigma2s)

    varb = sigma2*np.linalg.inv(XtX - XtZ @ forceSym2D(np.linalg.solve(np.eye(Dest.shape[0]) + Dest @ ZtZ, Dest)) @ ZtX)
    print(np.sqrt(np.diagonal(varb)))

    


    # Obtain the product matrices and start recording the computation time
    t1 = time.time()
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Run Fisher Scoring
    paramVector_FS,_,nit,llh = SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, nr, init_paramVector=None)
    t2 = time.time()

    print(XtX.shape)
    print(ZtZ.shape)


    print('SFS')
    print(t2-t1)
    print(nit)
    print(paramVector_FS)
    print(paramVector_FS[3:,]*paramVector_FS[2])
    print(llh-234/2*np.log(2*np.pi))

    sigma2 = paramVector_FS[2]
    d_sigma2t = np.diag(np.repeat(paramVector_FS[3], nlevels[0]))
    d_sigma2s = np.diag(np.repeat(paramVector_FS[4], nlevels[1]))

    Dest = scipy.linalg.block_diag(d_sigma2t,d_sigma2s)

    varb = sigma2*np.linalg.inv(XtX - XtZ @ forceSym2D(np.linalg.solve(np.eye(Dest.shape[0]) + Dest @ ZtZ, Dest)) @ ZtX)
    print(np.sqrt(np.diagonal(varb)))


    
    # Obtain the product matrices and start recording the computation time
    t1 = time.time()
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Run Fisher Scoring
    paramVector_FS,_,nit,llh = pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, nr, init_paramVector=None)
    t2 = time.time()


    print('pSFS')
    print(t2-t1)
    print(nit)
    print(paramVector_FS)
    print(paramVector_FS[3:,]*paramVector_FS[2])
    print(llh-234/2*np.log(2*np.pi))

    sigma2 = paramVector_FS[2]
    d_sigma2t = np.diag(np.repeat(paramVector_FS[3], nlevels[0]))
    d_sigma2s = np.diag(np.repeat(paramVector_FS[4], nlevels[1]))

    Dest = scipy.linalg.block_diag(d_sigma2t,d_sigma2s)

    varb = sigma2*np.linalg.inv(XtX - XtZ @ forceSym2D(np.linalg.solve(np.eye(Dest.shape[0]) + Dest @ ZtZ, Dest)) @ ZtX)
    print(np.sqrt(np.diagonal(varb)))


    
    t1 = time.time()
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Run Fisher Scoring
    paramVector_FS,_,nit,llh = cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, nr, init_paramVector=None)
    t2 = time.time()


    print('cSFS')
    print(t2-t1)
    print(nit)
    print(paramVector_FS)
    print(paramVector_FS[3:,]*paramVector_FS[2])
    print(llh-234/2*np.log(2*np.pi))

    sigma2 = paramVector_FS[2]
    d_sigma2t = np.diag(np.repeat(paramVector_FS[3], nlevels[0]))
    d_sigma2s = np.diag(np.repeat(paramVector_FS[4], nlevels[1]))

    Dest = scipy.linalg.block_diag(d_sigma2t,d_sigma2s)

    varb = sigma2*np.linalg.inv(XtX - XtZ @ forceSym2D(np.linalg.solve(np.eye(Dest.shape[0]) + Dest @ ZtZ, Dest)) @ ZtX)
    print(np.sqrt(np.diagonal(varb)))

    beta = np.array([[597.71405], [28.55724]])
    

    sigma2 = np.array([[237.9489]])

    d_sigma2s = np.diag(np.repeat(340.6898/237.9489, nlevels[0]))
    d_sigma2t = np.diag(np.repeat(604.9726/237.9489, nlevels[1]))
    
    D = scipy.linalg.block_diag(d_sigma2s,d_sigma2t)


    ete = YtY - 2*YtX @ beta + beta.transpose() @ XtX @ beta
    Zte = ZtY - ZtX @ beta
    DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(ZtZ.shape[0]) + D @ ZtZ, D))

    print(llh2D(X.shape[0], ZtZ, Zte, ete, sigma2, DinvIplusZtZD,Dest)-234/2*np.log(2*np.pi))
