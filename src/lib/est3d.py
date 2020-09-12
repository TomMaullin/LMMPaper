import os
import numpy as np
import scipy
from npMatrix3d import *
from npMatrix2d import *

# ============================================================================
#
# This file contains all parameter estimation methods used by the LMM code for
# estimating multiple models (as oppose to `est2d.py`, which is written for
# parameter estimation of only one model). The methods* given here are:
#
# - `FS`: Fisher Scoring
# - `pFS`: Pseudo-Fisher Scoring
# - `SFS`: Simplified Fisher Scoring
# - `pFS`: Pseudo-Simplified Fisher Scoring
#
# *Note: cSFS (cholesky Simplified Fisher Scoring), which is available in 
# `est2d.py` is not available here as it was slower than the above and 
# including it wasn't of much practical use.
#
# This code is only employed in this repository to speed up the computation of
# the Satterthwaithe degrees of freedom baseline truth simulations. All other
# code and reported performance is based on the code from the `est2d.py` file.
#
# ----------------------------------------------------------------------------
#
# Author: Tom Maullin (Last edited 07/04/2020)
#
# ============================================================================


# ============================================================================
# 
# This below function performs Fisher Scoring for the multiple Linear
# Mixed Models. It is based on the update rule:
#
#     \theta_h = \theta_h + lam*I(\theta_h)^(-1) (dl/d\theta_h)
#
# Where \theta_h is the vector (beta, sigma2, vech(D1),...vech(Dr)), lam is a
# scalar stepsize, I(\theta_h) is the Fisher Information matrix of \theta_h 
# and dl/d\theta_h is the derivative of the log likelihood of \theta_h with
# respect to \theta_h.
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X (can vary across models or can be
#           fixed across models). 
#  - `XtY`: X transpose multiplied by Y (varies across models).
#  - `XtZ`: X transpose multiplied by Z (can vary across models or can be
#           fixed across models). 
#  - `YtX`: Y transpose multiplied by X (varies across models).
#  - `YtY`: Y transpose multiplied by Y (varies across models).
#  - `YtZ`: Y transpose multiplied by Z (varies across models).
#  - `ZtX`: Z transpose multiplied by X (can vary across models or can be
#           fixed across models). 
#  - `ZtY`: Z transpose multiplied by Y (varies across models).
#  - `ZtZ`: Z transpose multiplied by Z (can vary across models or can be
#           fixed across models). If we are looking at a random intercept
#           design the variable ZtZ only holds the diagonal elements of the
#           matrix Z'Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#              factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#              random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations (can vary across models or can be
#         fixed across models). 
#  - `reml`: Restricted maximum likelihood estimation (currently not implemented)
#            Default: False. 
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `savedparams`: \theta_h in the previous notation; the vector (beta, 
#                   sigma2, vech(D1),...vech(Dr)) for every model.
#
# ============================================================================
def FS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol,n, reml=False):
    
    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[1]

    # Number of models, v
    v = XtY.shape[0]
    
    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------

    # Inital beta
    beta = initBeta3D(XtX, XtY)
    
    # Work out e'e, X'e and Z'e
    Xte = XtY - (XtX @ beta)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr3D(YtX, YtY, XtX, beta)
    
    # Initial sigma2
    sigma2 = initSigma23D(ete, n)
    
    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()
        
    # ------------------------------------------------------------------------------
    # Inital D
    # ------------------------------------------------------------------------------
    # Dictionary version
    Ddict = dict()
    for k in np.arange(len(nraneffs)):

        Ddict[k] = makeDnnd3D(initDk3D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))
    
    # Full version of D (not needed in the 1 random effect 1 random factor case as there
    # is only one unique element in D)
    if r == 1 and nraneffs[0] == 1:
        D = None
    else:
        D = getDfromDict3D(Ddict, nraneffs, nlevels)
    
    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of parameters
    tnp = np.int32(p + 1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1)
    # ------------------------------------------------------------------------------
    # Inverse of (I+Z'ZD) multiplied by D. If we are looking at a single random 
    # factor single random effect model DinvIplusZtZD will only hold the diagonal 
    # elements of D(I+Z'ZD)^(-1)
    DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs) 
    
    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = np.ones(v)

    # Initial log likelihoods
    llhprev = -10*np.ones(XtY.shape[0])
    llhcurr = 10*np.ones(XtY.shape[0])
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k
    permdict = dict()
    for k1 in np.arange(len(nraneffs)):
        for k2 in np.arange(len(nraneffs)):
            permdict[str(k1)+str(k2)] = None

    # ------------------------------------------------------------------------------
    # Converged models and parameter saving
    # ------------------------------------------------------------------------------
    # Vector checking if all models converged
    converged_global = np.zeros(v)
    
    # Vector of saved parameters which have converged
    savedparams = np.zeros((v, np.int32(np.sum(nraneffs*(nraneffs+1)/2) + p + 1),1))
    
    # Number of iterations
    nit=0

    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    while np.any(np.abs(llhprev-llhcurr)>tol):

        # Update number of iterations
        nit = nit + 1

        # Change current likelihood to previous
        llhprev = llhcurr
        
        # Work out how many models are left
        v_iter = XtY.shape[0]
        
        # --------------------------------------------------------------------------
        # Derivatives
        # --------------------------------------------------------------------------

        # Derivative wrt beta
        dldB = get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte, nraneffs)  
        
        # Derivative wrt sigma^2
        dldsigma2 = get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD, nraneffs, reml, p)
        
        # For each factor, factor k, work out dl/dD_k
        dldDdict = dict()
        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Calculate derivative with respect to D_k
            #-----------------------------------------------------------------------
            # Work out derivative
            if ZtZmatdict[k] is None:
                dldDdict[k],ZtZmatdict[k] = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,None,reml, ZtX, XtX)
            else:
                dldDdict[k],_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmatdict[k],reml, ZtX, XtX)
        
        # --------------------------------------------------------------------------
        # Covariances
        # --------------------------------------------------------------------------
        # Construct the Fisher Information matrix
        FisherInfoMat = np.zeros((v_iter,tnp,tnp))
        
        # Covariance of dl/dsigma2
        covdldsigma2 = n/(2*(sigma2**2))
        
        # Add dl/dsigma2 covariance
        FisherInfoMat[:,p,p] = covdldsigma2
        
        # Add dl/dbeta covariance
        covdldB = get_covdldbeta3D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2, nraneffs)
        FisherInfoMat[np.ix_(np.arange(v_iter), np.arange(p),np.arange(p))] = covdldB
        
        # Add dl/dsigma2 dl/dD covariance
        for k in np.arange(len(nraneffs)):

            # Calculate covariance between sigma2 and D
            if ZtZmatdict[k] is None:
                covdldsigma2dD,ZtZmatdict[k] = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, ZtZmat=None)
            else:
                covdldsigma2dD,_ = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, ZtZmat=ZtZmatdict[k])

            # Assign to the relevant block
            FisherInfoMat[:,p, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigma2dD.reshape(FisherInfoMat[:,p, FishIndsDk[k]:FishIndsDk[k+1]].shape)
            FisherInfoMat[:,FishIndsDk[k]:FishIndsDk[k+1],p:(p+1)] = FisherInfoMat[:,p:(p+1), FishIndsDk[k]:FishIndsDk[k+1]].transpose((0,2,1))
            
        # Add dl/dD covariance for each pair (k1,k2) of random factors
        for k1 in np.arange(len(nraneffs)):
            for k2 in np.arange(k1+1):

                # Work out indices corresponding to random factor k1 and random factor k2
                IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
                IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

                #-----------------------------------------------------------------------
                # Calculate covariance of derivative with respect to D_k
                #-----------------------------------------------------------------------
                if permdict[str(k1)+str(k2)] is None:
                    covdldDk1dDk2,permdict[str(k1)+str(k2)] = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=None)
                else:
                    covdldDk1dDk2,_ = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=permdict[str(k1)+str(k2)])

                # Add to FImat
                FisherInfoMat[np.ix_(np.arange(v_iter), IndsDk1, IndsDk2)] = covdldDk1dDk2
                FisherInfoMat[np.ix_(np.arange(v_iter), IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(np.arange(v_iter), IndsDk1, IndsDk2)].transpose((0,2,1))

        # Check Fisher Information matrix is symmetric
        FisherInfoMat = forceSym3D(FisherInfoMat)

        # --------------------------------------------------------------------------
        # Concatenate paramaters and derivatives together
        # --------------------------------------------------------------------------
        paramVector = np.concatenate((beta, sigma2.reshape(v_iter,1,1)),axis=1)
        derivVector = np.concatenate((dldB, dldsigma2.reshape(v_iter,1,1)),axis=1)

        for k in np.arange(len(nraneffs)):

            paramVector = np.concatenate((paramVector, mat2vech3D(Ddict[k])),axis=1)
            derivVector = np.concatenate((derivVector, dupMatTdict[k] @ mat2vec3D(dldDdict[k])),axis=1)
        
        # --------------------------------------------------------------------------
        # Update step
        # --------------------------------------------------------------------------
        paramVector = paramVector + np.einsum('i,ijk->ijk',lam,(np.linalg.solve(FisherInfoMat, derivVector)))
        
        # --------------------------------------------------------------------------
        # Get the new parameters
        # --------------------------------------------------------------------------
        beta = paramVector[:,0:p,:]
        sigma2 = paramVector[:,p:(p+1)][:,0,0]
        
        # D as a dictionary
        for k in np.arange(len(nraneffs)):

            Ddict[k] = makeDnnd3D(vech2mat3D(paramVector[:,FishIndsDk[k]:FishIndsDk[k+1],:]))
            
        # Full version of D (not needed in the 1 random effect 1 random factor case as there
        # is only one unique element in D)
        if r == 1 and nraneffs[0] == 1:
            D = None
        else:
            D = getDfromDict3D(Ddict, nraneffs, nlevels)
        
        # --------------------------------------------------------------------------
        # Matrices for next iteration
        # --------------------------------------------------------------------------
        # Get e'e
        ete = ssr3D(YtX, YtY, XtX, beta)

        # Get Z'e
        Zte = ZtY - (ZtX @ beta)
        
        # Inverse of (I+Z'ZD) multiplied by D. If we are looking at a single random 
        # factor single random effect model DinvIplusZtZD will only hold the diagonal 
        # elements of D(I+Z'ZD)^(-1)
        DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs)  
        
        # Check sigma2 hasn't hit a boundary
        sigma2[sigma2<0]=1e-10

        # --------------------------------------------------------------------------
        # Update the step size
        # --------------------------------------------------------------------------
        llhcurr = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D, Ddict, nlevels, nraneffs, reml, XtX, XtZ, ZtX)
        lam[llhprev>llhcurr] = lam[llhprev>llhcurr]/2
                
        # --------------------------------------------------------------------------
        # Work out which models converged and reduce the set of models we look at
        # next iteration
        # --------------------------------------------------------------------------
        # Get model indices in various formats
        indices_ConAfterIt, indices_notConAfterIt, indices_ConDuringIt, localconverged, localnotconverged = getConvergedIndices(converged_global, (np.abs(llhprev-llhcurr)<tol))

        # Update the record of which models have converged.
        converged_global[indices_ConDuringIt] = 1

        # Save parameters from this run
        savedparams[indices_ConDuringIt,:,:]=paramVector[localconverged,:,:]

        # --------------------------------------------------------------------------
        # Update matrices
        # --------------------------------------------------------------------------
        XtY = XtY[localnotconverged, :, :]
        YtX = YtX[localnotconverged, :, :]
        YtY = YtY[localnotconverged, :, :]
        ZtY = ZtY[localnotconverged, :, :]
        YtZ = YtZ[localnotconverged, :, :]
        Zte = Zte[localnotconverged, :, :]
        ete = ete[localnotconverged, :, :]
        # In the one factor, one random effect case we only have the diagonal 
        # D(I+Z'Z)^(-1) elements because D(I+Z'Z)^(-1) is diagonal in this use
        # case.
        if r==1 and nraneffs[0]==1:
            DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :]
        else:
            DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :, :]

        # Design varies across models
        if XtX.shape[0] > 1:

            XtX = XtX[localnotconverged, :, :]
            ZtX = ZtX[localnotconverged, :, :]
            XtZ = XtZ[localnotconverged, :, :]
                
            # In the one factor, one random effect case we only have the diagonal 
            # Z'Z elements because Z'Z is diagonal in this use case.
            if r==1 and nraneffs[0]==1:
                ZtZ = ZtZ[localnotconverged, :]
            else:
                ZtZ = ZtZ[localnotconverged, :, :]

            # ----------------------------------------------------------------------
            # Update ZtZmat
            # ----------------------------------------------------------------------
            # ZtZmat
            for k in np.arange(len(nraneffs)):
                if ZtZmatdict[k] is not None:
                    ZtZmatdict[k]=ZtZmatdict[k][localnotconverged, :, :]

        # Update n
        if hasattr(n, "ndim"):
            # Check if n varies with model
            if n.shape[0] > 1:
                if n.ndim == 1:
                    n = n[localnotconverged]
                if n.ndim == 2:
                    n = n[localnotconverged,:]
                if n.ndim == 3:
                    n = n[localnotconverged,:,:]

        # --------------------------------------------------------------------------
        # Update step size and likelihoods
        # --------------------------------------------------------------------------
        # Update step size
        lam = lam[localnotconverged]

        # Update log likelihoods
        llhprev = llhprev[localnotconverged]
        llhcurr = llhcurr[localnotconverged]

        # --------------------------------------------------------------------------
        # Update paramaters
        # --------------------------------------------------------------------------
        beta = beta[localnotconverged, :, :]
        sigma2 = sigma2[localnotconverged]
        # We don't need this representation of D in the simple case of 1 random 
        # factor, 1 random effect
        if not (r == 1 and nraneffs[0] == 1):
            D = D[localnotconverged, :, :]

        for k in np.arange(len(nraneffs)):
            Ddict[k] = Ddict[k][localnotconverged, :, :]
            
        # --------------------------------------------------------------------------
        # Matrices needed later:
        # --------------------------------------------------------------------------
        # X transpose e and Z transpose e
        Xte = XtY - (XtX @ beta)
        Zte = ZtY - (ZtX @ beta)
    
    return(savedparams)


# ============================================================================
# 
# This below function performs pesudo Fisher Scoring for multiple Linear Mixed
# Models. It is based on the update rule:
#
#     \theta_f = \theta_f + lam*I(\theta_f)^+ (dl/d\theta_f)
#
# Where \theta_f is the vector (beta, sigma2, vec(D1),...vec(Dr)), lam is a
# scalar stepsize, I(\theta_f) is the Fisher Information matrix of \theta_f 
# and dl/d\theta_f is the derivative of the log likelihood of \theta_f with
# respect to \theta_f. 
#
# Note that, as \theta_f is written in terms of 'vec', rather than 'vech',
# (full  vector, 'f', rather than half-vector, 'h'), the information matrix
# will have repeated rows (due to \theta_f having repeated entries). Because
# of this, this method is based on the "pseudo-Inverse" (represented by the 
# + above), hence the name.
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X (can vary across models or can be
#           fixed across models). 
#  - `XtY`: X transpose multiplied by Y (varies across models).
#  - `XtZ`: X transpose multiplied by Z (can vary across models or can be
#           fixed across models). 
#  - `YtX`: Y transpose multiplied by X (varies across models).
#  - `YtY`: Y transpose multiplied by Y (varies across models).
#  - `YtZ`: Y transpose multiplied by Z (varies across models).
#  - `ZtX`: Z transpose multiplied by X (can vary across models or can be
#           fixed across models). 
#  - `ZtY`: Z transpose multiplied by Y (varies across models).
#  - `ZtZ`: Z transpose multiplied by Z (can vary across models or can be
#           fixed across models). If we are looking at a random intercept
#           design the variable ZtZ only holds the diagonal elements of the
#           matrix Z'Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#              factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#              random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations (can vary across models or can be
#         fixed across models). 
#  - `reml`: Restricted maximum likelihood estimation (currently not implemented)
#            Default: False. 
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `savedparams`: \theta_h in the previous notation; the vector (beta, 
#                   sigma2, vech(D1),...vech(Dr)) for every model.
#
# ============================================================================
def pFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol,n, reml=False):

    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[1]

    # Number of models, v
    v = XtY.shape[0]
    
    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------

    # Inital beta
    beta = initBeta3D(XtX, XtY)
    
    # Work out e'e, X'e and Z'e
    Xte = XtY - (XtX @ beta)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr3D(YtX, YtY, XtX, beta)
    
    # Initial sigma2
    sigma2 = initSigma23D(ete, n)
    
    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()

    # ------------------------------------------------------------------------------
    # Inital D
    # ------------------------------------------------------------------------------
    # Dictionary version
    Ddict = dict()
    for k in np.arange(len(nraneffs)):

        Ddict[k] = makeDnnd3D(initDk3D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))
    
    # Full version of D (not needed in the 1 random effect 1 random factor case as there
    # is only one unique element in D)
    if r == 1 and nraneffs[0] == 1:
        D = None
    else:
        D = getDfromDict3D(Ddict, nraneffs, nlevels)
    
    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(p + 1 + np.sum(nraneffs**2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs**2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)
    
    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1) 
    # ------------------------------------------------------------------------------
    # Inverse of (I+Z'ZD) multiplied by D. If we are looking at a single random 
    # factor single random effect model DinvIplusZtZD will only hold the diagonal 
    # elements of D(I+Z'ZD)^(-1)
    DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs) 
    
    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = np.ones(v)

    # Initial log likelihoods
    llhprev = -10*np.ones(XtY.shape[0])
    llhcurr = 10*np.ones(XtY.shape[0])
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k
    permdict = dict()
    for k1 in np.arange(len(nraneffs)):
        for k2 in np.arange(len(nraneffs)):
            permdict[str(k1)+str(k2)] = None

    # ------------------------------------------------------------------------------
    # Converged models and parameter saving
    # ------------------------------------------------------------------------------
    # Vector checking if all models converged
    converged_global = np.zeros(v)
    
    # Vector of saved parameters which have converged
    savedparams = np.zeros((v, np.int32(np.sum(nraneffs**2) + p + 1),1))
    
    # Number of iterations
    nit=0

    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    while np.any(np.abs(llhprev-llhcurr)>tol):

        # Update number of iterations
        nit = nit + 1
            
        # --------------------------------------------------------------------------
        # Update loglikelihood and number of models
        # --------------------------------------------------------------------------
        # Change current likelihood to previous
        llhprev = llhcurr
        
        # Work out how many models are left
        v_iter = XtY.shape[0]
        
        # --------------------------------------------------------------------------
        # Derivatives
        # --------------------------------------------------------------------------
        # Derivative wrt beta
        dldB = get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte, nraneffs)  
        
        # Derivative wrt sigma^2
        dldsigma2 = get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD, nraneffs, reml, p)
        
        # For each factor, factor k, work out dl/dD_k
        dldDdict = dict()
        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Calculate derivative with respect to D_k
            #-----------------------------------------------------------------------
            # Work out derivative
            if ZtZmatdict[k] is None:
                dldDdict[k],ZtZmatdict[k] = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,None,reml, ZtX, XtX)
            else:
                dldDdict[k],_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmatdict[k],reml, ZtX, XtX)
        
        # --------------------------------------------------------------------------
        # Covariances
        # --------------------------------------------------------------------------
        # Construct the Fisher Information matrix
        FisherInfoMat = np.zeros((v_iter,tnp,tnp))
        
        # Covariance of dl/dsigma2
        covdldsigma2 = n/(2*(sigma2**2))
        
        # Add dl/dsigma2 covariance
        FisherInfoMat[:,p,p] = covdldsigma2
        
        # Add dl/dbeta covariance
        covdldB = get_covdldbeta3D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2, nraneffs)
        FisherInfoMat[np.ix_(np.arange(v_iter), np.arange(p),np.arange(p))] = covdldB
        
        # Add dl/dsigma2 dl/dD covariance
        for k in np.arange(len(nraneffs)):

            # Calculate covariance between sigma2 and D
            if ZtZmatdict[k] is None:
                covdldsigma2dD,ZtZmatdict[k] = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, ZtZmat=None)
            else:
                covdldsigma2dD,_ = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, ZtZmat=ZtZmatdict[k])

            # Assign to the relevant block
            FisherInfoMat[:,p, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigma2dD.reshape(FisherInfoMat[:,p, FishIndsDk[k]:FishIndsDk[k+1]].shape)
            FisherInfoMat[:,FishIndsDk[k]:FishIndsDk[k+1],p:(p+1)] = FisherInfoMat[:,p:(p+1), FishIndsDk[k]:FishIndsDk[k+1]].transpose((0,2,1))
            
        # Add dl/dD covariance for each pair (k1,k2) of random factors k1 and k2.
        for k1 in np.arange(len(nraneffs)):
            for k2 in np.arange(k1+1):

                # Work out indices of random factor k1 and random factor k2
                IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
                IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

                #-----------------------------------------------------------------------
                # Calculate covariance of derivative with respect to D_k
                #-----------------------------------------------------------------------
                if permdict[str(k1)+str(k2)] is None:
                    covdldDk1dDk2,permdict[str(k1)+str(k2)] = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, perm=None)
                else:
                    covdldDk1dDk2,_ = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, perm=permdict[str(k1)+str(k2)])

                # Add to FImat
                FisherInfoMat[np.ix_(np.arange(v_iter), IndsDk1, IndsDk2)] = covdldDk1dDk2
                FisherInfoMat[np.ix_(np.arange(v_iter), IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(np.arange(v_iter), IndsDk1, IndsDk2)].transpose((0,2,1))
                     

        # --------------------------------------------------------------------------
        # Concatenate paramaters and derivatives together
        # --------------------------------------------------------------------------
        paramVector = np.concatenate((beta, sigma2.reshape(v_iter,1,1)),axis=1)
        derivVector = np.concatenate((dldB, dldsigma2.reshape(v_iter,1,1)),axis=1)

        for k in np.arange(len(nraneffs)):
            paramVector = np.concatenate((paramVector, mat2vec3D(Ddict[k])),axis=1)
            derivVector = np.concatenate((derivVector, mat2vec3D(dldDdict[k])),axis=1)
        
        # --------------------------------------------------------------------------
        # Update step
        # --------------------------------------------------------------------------
        paramVector = paramVector + np.einsum('i,ijk->ijk',lam,np.linalg.solve(forceSym3D(FisherInfoMat), derivVector))
        
        # --------------------------------------------------------------------------
        # Get the new parameters
        # --------------------------------------------------------------------------
        beta = paramVector[:,0:p,:]
        sigma2 = paramVector[:,p:(p+1)][:,0,0]
        
        # D as a dictionary
        for k in np.arange(len(nraneffs)):

            Ddict[k] = makeDnnd3D(vec2mat3D(paramVector[:,FishIndsDk[k]:FishIndsDk[k+1],:]))
            
        # Full version of D (not needed in the 1 random effect 1 random factor case as there
        # is only one unique element in D)
        if r == 1 and nraneffs[0] == 1:
            D = None
        else:
            D = getDfromDict3D(Ddict, nraneffs, nlevels)

        # --------------------------------------------------------------------------
        # Recalculate matrices
        # --------------------------------------------------------------------------
        ete = ssr3D(YtX, YtY, XtX, beta)
        Zte = ZtY - (ZtX @ beta)
        
        # --------------------------------------------------------------------------
        # Inverse of (I+Z'ZD) multiplied by D
        # --------------------------------------------------------------------------
        # Inverse of (I+Z'ZD) multiplied by D. If we are looking at a single random 
        # factor single random effect model DinvIplusZtZD will only hold the diagonal 
        # elements of D(I+Z'ZD)^(-1)
        DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs)  
        
        # Check sigma2 hasn't hit a boundary
        sigma2[sigma2<0]=1e-10

        # --------------------------------------------------------------------------
        # Update the step size and likelihoods
        # --------------------------------------------------------------------------
        llhcurr = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D, Ddict, nlevels, nraneffs, reml, XtX, XtZ, ZtX)
        lam[llhprev>llhcurr] = lam[llhprev>llhcurr]/2
                
        # --------------------------------------------------------------------------
        # Work out which models converged
        # --------------------------------------------------------------------------
        # Get indices of converged models
        indices_ConAfterIt, indices_notConAfterIt, indices_ConDuringIt, localconverged, localnotconverged = getConvergedIndices(converged_global, (np.abs(llhprev-llhcurr)<tol))
        # Update record of converged models
        converged_global[indices_ConDuringIt] = 1

        # --------------------------------------------------------------------------
        # Save parameters from this run
        # --------------------------------------------------------------------------
        savedparams[indices_ConDuringIt,:,:]=paramVector[localconverged,:,:]

        # --------------------------------------------------------------------------
        # Update matrices
        # --------------------------------------------------------------------------
        XtY = XtY[localnotconverged, :, :]
        YtX = YtX[localnotconverged, :, :]
        YtY = YtY[localnotconverged, :, :]
        ZtY = ZtY[localnotconverged, :, :]
        YtZ = YtZ[localnotconverged, :, :]
        Zte = Zte[localnotconverged, :, :]
        ete = ete[localnotconverged, :, :]
        # In the one factor, one random effect case we only have the diagonal 
        # D(I+Z'Z)^(-1) elements because D(I+Z'Z)^(-1) is diagonal in this use
        # case.
        if r==1 and nraneffs[0]==1:
            DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :]
        else:
            DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :, :]

        # Design varies across models
        if XtX.shape[0] > 1:

            XtX = XtX[localnotconverged, :, :]
            ZtX = ZtX[localnotconverged, :, :]
            XtZ = XtZ[localnotconverged, :, :]
                
            # In the one factor, one random effect case we only have the diagonal 
            # Z'Z elements because Z'Z is diagonal in this use case.
            if r==1 and nraneffs[0]==1:
                ZtZ = ZtZ[localnotconverged, :]
            else:
                ZtZ = ZtZ[localnotconverged, :, :]
                
            # ----------------------------------------------------------------------
            # Update ZtZmat
            # ----------------------------------------------------------------------
            # ZtZmat
            for k in np.arange(len(nraneffs)):
                if ZtZmatdict[k] is not None:
                    ZtZmatdict[k]=ZtZmatdict[k][localnotconverged, :, :]

        # Update n
        if hasattr(n, "ndim"):
            # Check if n varies with model
            if n.shape[0] > 1:
                if n.ndim == 1:
                    n = n[localnotconverged]
                if n.ndim == 2:
                    n = n[localnotconverged,:]
                if n.ndim == 3:
                    n = n[localnotconverged,:,:]

        # --------------------------------------------------------------------------
        # Update step size and log likelihoods
        # --------------------------------------------------------------------------
        lam = lam[localnotconverged]
        llhprev = llhprev[localnotconverged]
        llhcurr = llhcurr[localnotconverged]

        # --------------------------------------------------------------------------
        # Update parameter estimates
        # --------------------------------------------------------------------------
        beta = beta[localnotconverged, :, :]
        sigma2 = sigma2[localnotconverged]
        # We don't need this representation of D in the simple case of 1 random 
        # factor, 1 random effect
        if not (r == 1 and nraneffs[0] == 1):
            D = D[localnotconverged, :, :]

        for k in np.arange(len(nraneffs)):
            Ddict[k] = Ddict[k][localnotconverged, :, :]
            
        # --------------------------------------------------------------------------
        # Matrices needed later by many calculations
        # --------------------------------------------------------------------------
        # X transpose e and Z transpose e
        Xte = XtY - (XtX @ beta)
        Zte = ZtY - (ZtX @ beta)
    
    return(savedparams)


# ============================================================================
# 
# This below function performs Simplified Fisher Scoring for multiple Linear
# Mixed Models. It is based on the update rules:
#
#                    beta = (X'V^(-1)X)^(-1)(X'V^(-1)Y)
#
#                           sigma2 = e'V^(-1)e/n
#
#                            for k in {1,...,r};
#           vech(D_k) = \theta_f + lam*I(vech(D_k))^(-1) (dl/dvech(D_k))
#
# Where:
#  - lam is a scalar stepsize.
#  - I(vech(D_k)) is the Fisher Information matrix of vech(D_k).
#  - dl/dvech(D_k) is the derivative of the log likelihood of vech(D_k) with
#    respect to vech(D_k). 
#  - e is the residual vector (e=Y-X\beta)
#  - V is the matrix (I+ZDZ')
#
# The name "Simplified" here comes from a convention adopted in (Demidenko 
# 2014).
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X (can vary across models or can be
#           fixed across models). 
#  - `XtY`: X transpose multiplied by Y (varies across models).
#  - `XtZ`: X transpose multiplied by Z (can vary across models or can be
#           fixed across models). 
#  - `YtX`: Y transpose multiplied by X (varies across models).
#  - `YtY`: Y transpose multiplied by Y (varies across models).
#  - `YtZ`: Y transpose multiplied by Z (varies across models).
#  - `ZtX`: Z transpose multiplied by X (can vary across models or can be
#           fixed across models). 
#  - `ZtY`: Z transpose multiplied by Y (varies across models).
#  - `ZtZ`: Z transpose multiplied by Z (can vary across models or can be
#           fixed across models). If we are looking at a random intercept
#           design the variable ZtZ only holds the diagonal elements of the
#           matrix Z'Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#              factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#              random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations (can vary across models or can be
#         fixed across models). 
#  - `reml`: Restricted maximum likelihood estimation (currently not implemented)
#            Default: False. 
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `savedparams`: \theta_h in the previous notation; the vector (beta, 
#                   sigma2, vech(D1),...vech(Dr)) for every model.
#
# ============================================================================
def SFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol,n, reml=False):

    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[1]

    # Number of models, v
    v = XtY.shape[0]
    
    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------

    # Inital beta
    beta = initBeta3D(XtX, XtY)
    
    # Work out e'e, X'e and Z'e
    Xte = XtY - (XtX @ beta)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr3D(YtX, YtY, XtX, beta)
    
    # Initial sigma2
    sigma2 = initSigma23D(ete, n)
    
    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()
        
    # ------------------------------------------------------------------------------
    # Inital D
    # ------------------------------------------------------------------------------
    # Dictionary version
    Ddict = dict()
    for k in np.arange(len(nraneffs)):

        Ddict[k] = makeDnnd3D(initDk3D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))
    
    # Full version of D (not needed in the 1 random effect 1 random factor case as there
    # is only one unique element in D)
    if r == 1 and nraneffs[0] == 1:
        D = None
    else:
        D = getDfromDict3D(Ddict, nraneffs, nlevels)
    
    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(p + 1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1)
    # ------------------------------------------------------------------------------
    # Inverse of (I+Z'ZD) multiplied by D. If we are looking at a single random 
    # factor single random effect model DinvIplusZtZD will only hold the diagonal 
    # elements of D(I+Z'ZD)^(-1)
    DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs) 
    
    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = np.ones(v)

    # Initial log likelihoods
    llhprev = -10*np.ones(XtY.shape[0])
    llhcurr = 10*np.ones(XtY.shape[0])
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k
    permdict = dict()
    for k in np.arange(len(nraneffs)):
        permdict[str(k)] = None

    # ------------------------------------------------------------------------------
    # Converged models and parameter saving
    # ------------------------------------------------------------------------------
    # Vector checking if all models converged
    converged_global = np.zeros(v)
    
    # Vector of saved parameters which have converged
    savedparams = np.zeros((v, np.int32(np.sum(nraneffs*(nraneffs+1)/2) + p + 1),1))
    
    # ------------------------------------------------------------------------------
    # Work out D indices (there is one block of D per level)
    # ------------------------------------------------------------------------------
    Dinds = np.zeros(np.sum(nlevels)+1)

    # Loop through and add each index
    counter = 0
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):
            Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1
            
    # Last index will be missing so add it
    Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nraneffs[-1]
    
    # Make sure indices are ints
    Dinds = np.int64(Dinds)
    
    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    nit=0
    while np.any(np.abs(llhprev-llhcurr)>tol):

        # Update number of iterations
        nit = nit + 1
            
        # --------------------------------------------------------------------------
        # Update loglikelihood and number of models
        # --------------------------------------------------------------------------
        # Change current likelihood to previous
        llhprev = llhcurr
        
        # Work out how many models are left
        v_iter = XtY.shape[0]
        
        # --------------------------------------------------------------------------
        # Update beta
        # --------------------------------------------------------------------------
        # This can be performed faster in the one factor, one random effect case by
        # using only the diagonal elements of DinvIplusZtZD 
        if r == 1 and nraneffs[0] == 1:

            # Multiply by Z'X
            DinvIplusZtZDZtX = np.einsum('ij,ijk->ijk', DinvIplusZtZD, ZtX)

        else:

            # Multiply by Z'X
            DinvIplusZtZDZtX = DinvIplusZtZD @ ZtX

        # Work out X'V^(-1)X and X'V^(-1)Y by dimension reduction formulae
        XtiVX = XtX - DinvIplusZtZDZtX.transpose((0,2,1)) @ ZtX
        XtiVY = XtY - DinvIplusZtZDZtX.transpose((0,2,1)) @ ZtY

        # Calculate beta
        beta = np.linalg.solve(XtiVX, XtiVY)
        
        # Update sigma^2
        ete = ssr3D(YtX, YtY, XtX, beta)
        Zte = ZtY - (ZtX @ beta)

        # Make sure n is correct shape
        if hasattr(n, "ndim"):
            if np.prod(n.shape) > 1:
                n = n.reshape(ete.shape)

        # --------------------------------------------------------------------------
        # Update sigma2
        # --------------------------------------------------------------------------
        # This can be performed faster in the one factor, one random effect case by
        # using only the diagonal elements of DinvIplusZtZD. In the reml version we
        # use n-p instead of n.
        if r == 1 and nraneffs[0] == 1:
            if reml == False:
                sigma2 = (1/n*(ete - Zte.transpose((0,2,1)) @ np.einsum('ij,ijk->ijk',DinvIplusZtZD, Zte))).reshape(v_iter)
            else:
                sigma2 = (1/(n-p)*(ete - Zte.transpose((0,2,1)) @ np.einsum('ij,ijk->ijk',DinvIplusZtZD, Zte))).reshape(v_iter)
        else:
            if reml == False:
                sigma2 = (1/n*(ete - Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte)).reshape(v_iter)
            else:
                sigma2 = (1/(n-p)*(ete - Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte)).reshape(v_iter)

        # --------------------------------------------------------------------------
        # Update D
        # --------------------------------------------------------------------------
        counter = 0
        # Loop though unique blocks of D updating one at a time
        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Calculate derivative with respect to D_k
            #-----------------------------------------------------------------------
            # Work out derivative
            if ZtZmatdict[k] is None:
                dldDk,ZtZmatdict[k] = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,None,reml, ZtX, XtX)
            else:
                dldDk,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmatdict[k],reml, ZtX, XtX)
    
            #-----------------------------------------------------------------------
            # Calculate covariance of derivative with respect to D_k
            #-----------------------------------------------------------------------
            if permdict[str(k)] is None:
                covdldDk1dDk2,permdict[str(k)] = get_covdldDk1Dk23D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=None)
            else:
                covdldDk1dDk2,_ = get_covdldDk1Dk23D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=permdict[str(k)])

            #-----------------------------------------------------------------------
            # Work out update amount
            #-----------------------------------------------------------------------
            update = np.linalg.solve(forceSym3D(covdldDk1dDk2), dupMatTdict[k] @ mat2vec3D(dldDk))
            
            # Multiply by stepsize
            update = np.einsum('i,ijk->ijk',lam, update)
            
            # Update D_k
            Ddict[k] = makeDnnd3D(vech2mat3D(mat2vech3D(Ddict[k]) + update))
            
            # Add D_k back into D and recompute DinvIplusZtZD (This isn't necessary for the
            # one random effect, one random factor use case as D only contains one unique 
            # element)
            if not (r == 1 and nraneffs[0] == 1):

                for j in np.arange(nlevels[k]):
                    D[:, Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                    counter = counter + 1
                
            # Inverse of (I+Z'ZD) multiplied by D
            DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs)  
        
        # --------------------------------------------------------------------------
        # Recalculate matrices
        # --------------------------------------------------------------------------
        ete = ssr3D(YtX, YtY, XtX, beta)
        Zte = ZtY - (ZtX @ beta)
        
        # Check sigma2 hasn't hit a boundary
        sigma2[sigma2<0]=1e-10
        
        # --------------------------------------------------------------------------
        # Update the step size and log likelihoods
        # --------------------------------------------------------------------------
        llhcurr = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D, Ddict, nlevels, nraneffs, reml, XtX, XtZ, ZtX)
        lam[llhprev>llhcurr] = lam[llhprev>llhcurr]/2
        
        # --------------------------------------------------------------------------
        # Work out which models converged
        # --------------------------------------------------------------------------

        # Get indices of converged models
        indices_ConAfterIt, indices_notConAfterIt, indices_ConDuringIt, localconverged, localnotconverged = getConvergedIndices(converged_global, (np.abs(llhprev-llhcurr)<tol))

        # Record which models converged this iteration
        converged_global[indices_ConDuringIt] = 1

        # --------------------------------------------------------------------------
        # Save parameters from this run
        # --------------------------------------------------------------------------
        savedparams[indices_ConDuringIt,0:p,:]=beta[localconverged,:,:]
        savedparams[indices_ConDuringIt,p:(p+1),:]=sigma2[localconverged].reshape(sigma2[localconverged].shape[0],1,1)
        
        for k in np.arange(len(nraneffs)):
            
            # Get vech form of D_k
            vech_Dk = mat2vech3D(Ddict[k][localconverged,:,:])
            
            # Make sure it has correct shape (i.e. shape (num models converged, num
            # random effectss for factor k squared, 1))
            vech_Dk = vech_Dk.reshape(len(localconverged),nraneffs[k]*(nraneffs[k]+1)//2,1)
            savedparams[indices_ConDuringIt,FishIndsDk[k]:FishIndsDk[k+1],:]=vech_Dk
            
        # --------------------------------------------------------------------------
        # Update matrices
        # --------------------------------------------------------------------------
        XtY = XtY[localnotconverged, :, :]
        YtX = YtX[localnotconverged, :, :]
        YtY = YtY[localnotconverged, :, :]
        ZtY = ZtY[localnotconverged, :, :]
        YtZ = YtZ[localnotconverged, :, :]
        Zte = Zte[localnotconverged, :, :]
        ete = ete[localnotconverged, :, :]

        # In the one factor, one random effect case we only have the diagonal 
        # D(I+Z'Z)^(-1) elements because D(I+Z'Z)^(-1) is diagonal in this use
        # case.
        if r==1 and nraneffs[0]==1:
            DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :]
        else:
            DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :, :]

        # Design varies across models
        if XtX.shape[0] > 1:

            XtX = XtX[localnotconverged, :, :]
            ZtX = ZtX[localnotconverged, :, :]
            XtZ = XtZ[localnotconverged, :, :]
                
            # In the one factor, one random effect case we only have the diagonal 
            # Z'Z elements because Z'Z is diagonal in this use case.
            if r==1 and nraneffs[0]==1:
                ZtZ = ZtZ[localnotconverged, :]
            else:
                ZtZ = ZtZ[localnotconverged, :, :]
                
            # ----------------------------------------------------------------------
            # Update ZtZmat
            # ----------------------------------------------------------------------
            # ZtZmat
            for k in np.arange(len(nraneffs)):
                if ZtZmatdict[k] is not None:
                    ZtZmatdict[k]=ZtZmatdict[k][localnotconverged, :, :]

        # Update n
        if hasattr(n, "ndim"):
            # Check if n varies with model
            if n.shape[0] > 1:
                if n.ndim == 1:
                    n = n[localnotconverged]
                if n.ndim == 2:
                    n = n[localnotconverged,:]
                if n.ndim == 3:
                    n = n[localnotconverged,:,:]

        # --------------------------------------------------------------------------
        # Update step size and log-likelihoods
        # --------------------------------------------------------------------------
        lam = lam[localnotconverged]
        llhprev = llhprev[localnotconverged]
        llhcurr = llhcurr[localnotconverged]

        # --------------------------------------------------------------------------
        # Update parameters
        # --------------------------------------------------------------------------
        beta = beta[localnotconverged, :, :]
        sigma2 = sigma2[localnotconverged]
        # We don't need this representation of D in the simple case of 1 random 
        # factor, 1 random effect
        if not (r == 1 and nraneffs[0] == 1):
            D = D[localnotconverged, :, :]

        for k in np.arange(len(nraneffs)):
            Ddict[k] = Ddict[k][localnotconverged, :, :]
    
    return(savedparams)


# ============================================================================
# 
# This below function performs pseudo-Simplified Fisher Scoring for multiple
# Linear Mixed Models. It is based on the update rules:
#
#                       beta = (X'V^(-1)X)^(-1)(X'V^(-1)Y)
#
#                             sigma2 = e'V^(-1)e/n
#
#                              for k in {1,...,r};
#              vec(D_k) = \theta_f + lam*I(vec(D_k))^+ (dl/dvec(D_k))
#
# Where:
#  - lam is a scalar stepsize.
#  - I(vec(D_k)) is the Fisher Information matrix of vec(D_k).
#  - dl/dvec(D_k) is the derivative of the log likelihood of vec(D_k) with
#    respect to vec(D_k). 
#  - e is the residual vector (e=Y-X\beta)
#  - V is the matrix (I+ZDZ')
#
# Note that, as vf(D) is written in terms of 'vec', rather than 'vech',
# (full  vector, 'f', rather than half-vector, 'h'), the information matrix
# will have repeated rows (due to vf(D) having repeated entries). Because
# of this, this method is based on the "pseudo-Inverse" (represented by the 
# + above), hence the name.
#
# The name "Simplified" here comes from a convention adopted in (Demidenko 
# 2014).
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X (can vary across models or can be
#           fixed across models). 
#  - `XtY`: X transpose multiplied by Y (varies across models).
#  - `XtZ`: X transpose multiplied by Z (can vary across models or can be
#           fixed across models). 
#  - `YtX`: Y transpose multiplied by X (varies across models).
#  - `YtY`: Y transpose multiplied by Y (varies across models).
#  - `YtZ`: Y transpose multiplied by Z (varies across models).
#  - `ZtX`: Z transpose multiplied by X (can vary across models or can be
#           fixed across models). 
#  - `ZtY`: Z transpose multiplied by Y (varies across models).
#  - `ZtZ`: Z transpose multiplied by Z (can vary across models or can be
#           fixed across models). If we are looking at a random intercept
#           design the variable ZtZ only holds the diagonal elements of the
#           matrix Z'Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#              factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#              random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations (can vary across models or can be
#         fixed across models). 
#  - `reml`: Restricted maximum likelihood estimation. Default: False.
# 
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `savedparams`: \theta_h in the previous notation; the vector (beta, 
#                   sigma2, vech(D1),...vech(Dr)) for every model.
#
# ============================================================================
def pSFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, reml=False):

    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[1]

    # Number of models, v
    v = XtY.shape[0]
    
    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------

    # Inital beta
    beta = initBeta3D(XtX, XtY)
    
    # Work out e'e, X'e and Z'e
    Xte = XtY - (XtX @ beta)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr3D(YtX, YtY, XtX, beta)
    
    # Initial sigma2
    sigma2 = initSigma23D(ete, n)
    
    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()
        
    # ------------------------------------------------------------------------------
    # Inital D
    # ------------------------------------------------------------------------------
    # Dictionary version
    Ddict = dict()
    for k in np.arange(len(nraneffs)):

        Ddict[k] = makeDnnd3D(initDk3D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))

    # Full version of D (not needed in the 1 random effect 1 random factor case as there
    # is only one unique element in D)
    if r == 1 and nraneffs[0] == 1:
        D = None
    else:
        D = getDfromDict3D(Ddict, nraneffs, nlevels)
    
    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(p + 1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)
    
    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1)
    # ------------------------------------------------------------------------------
    # Inverse of (I+Z'ZD) multiplied by D. If we are looking at a single random 
    # factor single random effect model DinvIplusZtZD will only hold the diagonal 
    # elements of D(I+Z'ZD)^(-1)
    DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs) 

    # ------------------------------------------------------------------------------
    # Step size and log likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = np.ones(v)

    # Initial log likelihoods
    llhprev = -10*np.ones(XtY.shape[0])
    llhcurr = 10*np.ones(XtY.shape[0])
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k
    permdict = dict()
    for k in np.arange(len(nraneffs)):
        permdict[str(k)] = None

    # ------------------------------------------------------------------------------
    # Converged models and parameter saving
    # ------------------------------------------------------------------------------
    # Vector checking if all models converged
    converged_global = np.zeros(v)
    
    # Vector of saved parameters which have converged
    savedparams = np.zeros((v, np.int32(np.sum(nraneffs*(nraneffs+1)/2) + p + 1),1))
    
    # ------------------------------------------------------------------------------
    # Work out D indices (there is one block of D per level)
    # ------------------------------------------------------------------------------
    Dinds = np.zeros(np.sum(nlevels)+1)
    counter = 0

    # Loop through and add an index for each block of D.
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):
            Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1
            
    # Last index will be missing so add it
    Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nraneffs[-1]
    
    # Make sure indices are ints
    Dinds = np.int64(Dinds)
    
    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    nit=0
    while np.any(np.abs(llhprev-llhcurr)>tol):

        # Update number of iterations
        nit = nit + 1
            
        # --------------------------------------------------------------------------
        # Update loglikelihood and number of models
        # --------------------------------------------------------------------------
        # Change current likelihood to previous
        llhprev = llhcurr
        
        # Work out how many models are left
        v_iter = XtY.shape[0]
        
        # --------------------------------------------------------------------------
        # Update beta
        # --------------------------------------------------------------------------
        # This can be performed faster in the one factor, one random effect case by
        # using only the diagonal elements of DinvIplusZtZD 
        if r == 1 and nraneffs[0] == 1:

            # Multiply by Z'X
            DinvIplusZtZDZtX = np.einsum('ij,ijk->ijk', DinvIplusZtZD, ZtX)

        else:

            # Multiply by Z'X
            DinvIplusZtZDZtX = DinvIplusZtZD @ ZtX

        # Work out X'V^(-1)X and X'V^(-1)Y by dimension reduction formulae
        XtiVX = XtX - DinvIplusZtZDZtX.transpose((0,2,1)) @ ZtX
        XtiVY = XtY - DinvIplusZtZDZtX.transpose((0,2,1)) @ ZtY

        # Calculate beta
        beta = np.linalg.solve(XtiVX, XtiVY)

        # Update sigma^2
        ete = ssr3D(YtX, YtY, XtX, beta)
        Zte = ZtY - (ZtX @ beta)

        # Make sure n is correct shape
        if hasattr(n, "ndim"):
            if np.prod(n.shape) > 1:
                n = n.reshape(ete.shape)

        # --------------------------------------------------------------------------
        # Update sigma2
        # --------------------------------------------------------------------------
        # This can be performed faster in the one factor, one random effect case by
        # using only the diagonal elements of DinvIplusZtZD. In the reml version we
        # use n-p instead of n.
        if r == 1 and nraneffs[0] == 1:
            if reml == False:
                sigma2 = (1/n*(ete - Zte.transpose((0,2,1)) @ np.einsum('ij,ijk->ijk',DinvIplusZtZD, Zte))).reshape(v_iter)
            else:
                sigma2 = (1/(n-p)*(ete - Zte.transpose((0,2,1)) @ np.einsum('ij,ijk->ijk',DinvIplusZtZD, Zte))).reshape(v_iter)
        else:
            if reml == False:
                sigma2 = (1/n*(ete - Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte)).reshape(v_iter)
            else:
                sigma2 = (1/(n-p)*(ete - Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte)).reshape(v_iter)
        
        # --------------------------------------------------------------------------
        # Update D
        # --------------------------------------------------------------------------
        counter = 0
        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Calculate derivative with respect to D_k
            #-----------------------------------------------------------------------
            # Work out derivative
            if ZtZmatdict[k] is None:
                dldDk,ZtZmatdict[k] = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,None,reml, ZtX, XtX)
            else:
                dldDk,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmatdict[k],reml, ZtX, XtX)
        
            #-----------------------------------------------------------------------
            # Calculate covariance of derivative with respect to D_k
            #-----------------------------------------------------------------------
            if permdict[str(k)] is None:
                covdldDk1dDk2,permdict[str(k)] = get_covdldDk1Dk23D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, perm=None)
            else:
                covdldDk1dDk2,_ = get_covdldDk1Dk23D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, perm=permdict[str(k)])

            #-----------------------------------------------------------------------
            # Work out update amount
            #-----------------------------------------------------------------------
            update_p = np.linalg.solve(forceSym3D(covdldDk1dDk2), mat2vec3D(dldDk))
            
            # Multiply by stepsize
            update_p = np.einsum('i,ijk->ijk',lam, update_p)

            # Update D_k
            Ddict[k] = makeDnnd3D(vec2mat3D(mat2vec3D(Ddict[k]) + update_p))

            # Add D_k back into D and recompute DinvIplusZtZD (This isn't necessary for the
            # one random effect, one random factor use case as D only contains one unique 
            # element)
            if not (r == 1 and nraneffs[0] == 1):

                for j in np.arange(nlevels[k]):

                    D[:, Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                    counter = counter + 1
            
        # --------------------------------------------------------------------------
        # Obtain D(I+Z'ZD)^(-1)
        # --------------------------------------------------------------------------
        # Inverse of (I+Z'ZD) multiplied by D. If we are looking at a single random 
        # factor single random effect model DinvIplusZtZD will only hold the diagonal 
        # elements of D(I+Z'ZD)^(-1)
        DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs)  

        # --------------------------------------------------------------------------
        # Recalculate matrices
        # --------------------------------------------------------------------------
        ete = ssr3D(YtX, YtY, XtX, beta)
        Zte = ZtY - (ZtX @ beta)
        
        # Check sigma2 hasn't hit a boundary
        sigma2[sigma2<0]=1e-10

        # --------------------------------------------------------------------------
        # Update the step size and log likelihood
        # --------------------------------------------------------------------------
        llhcurr = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D, Ddict, nlevels, nraneffs, reml, XtX, XtZ, ZtX)

        lam[llhprev>llhcurr] = lam[llhprev>llhcurr]/2
        
        # --------------------------------------------------------------------------
        # Work out which models converged
        # --------------------------------------------------------------------------
        # Obatin indices of converged models
        indices_ConAfterIt, indices_notConAfterIt, indices_ConDuringIt, localconverged, localnotconverged = getConvergedIndices(converged_global, (np.abs(llhprev-llhcurr)<tol))

        # Record which models converged.
        converged_global[indices_ConDuringIt] = 1

        # --------------------------------------------------------------------------
        # Save parameters from this run
        # --------------------------------------------------------------------------
        savedparams[indices_ConDuringIt,0:p,:]=beta[localconverged,:,:]
        savedparams[indices_ConDuringIt,p:(p+1),:]=sigma2[localconverged].reshape(sigma2[localconverged].shape[0],1,1)
        
        for k in np.arange(len(nraneffs)):
            
            # Get vech form of D_k
            vech_Dk = mat2vech3D(Ddict[k][localconverged,:,:])
            
            # Make sure it has correct shape (i.e. shape (num models converged, num 
            # random effects for factor k squared, 1))
            vech_Dk = vech_Dk.reshape(len(localconverged),nraneffs[k]*(nraneffs[k]+1)//2,1)
            savedparams[indices_ConDuringIt,FishIndsDk[k]:FishIndsDk[k+1],:]=vech_Dk
            
        # --------------------------------------------------------------------------
        # Update matrices
        # --------------------------------------------------------------------------
        XtY = XtY[localnotconverged, :, :]
        YtX = YtX[localnotconverged, :, :]
        YtY = YtY[localnotconverged, :, :]
        ZtY = ZtY[localnotconverged, :, :]
        YtZ = YtZ[localnotconverged, :, :]
        ete = ete[localnotconverged, :, :]

        # Design varies across models
        if XtX.shape[0] > 1:

            XtX = XtX[localnotconverged, :, :]
            ZtX = ZtX[localnotconverged, :, :]
            XtZ = XtZ[localnotconverged, :, :]
                
            # In the one factor, one random effect case we only have the diagonal 
            # Z'Z elements because Z'Z is diagonal in this use case.
            if r==1 and nraneffs[0]==1:
                ZtZ = ZtZ[localnotconverged, :]
            else:
                ZtZ = ZtZ[localnotconverged, :, :]


            # ----------------------------------------------------------------------
            # Update ZtZmat
            # ----------------------------------------------------------------------
            # ZtZmat
            for k in np.arange(len(nraneffs)):
                if ZtZmatdict[k] is not None:
                    ZtZmatdict[k]=ZtZmatdict[k][localnotconverged, :, :]

        # Update n
        if hasattr(n, "ndim"):
            # Check if n varies with model
            if n.shape[0] > 1:
                if n.ndim == 1:
                    n = n[localnotconverged]
                if n.ndim == 2:
                    n = n[localnotconverged,:]
                if n.ndim == 3:
                    n = n[localnotconverged,:,:]
                
        # In the one factor, one random effect case we only have the diagonal 
        # D(I+Z'Z)^(-1) elements because D(I+Z'Z)^(-1) is diagonal in this use
        # case.
        if r==1 and nraneffs[0]==1:
            DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :]
        else:
            DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :, :]

        # --------------------------------------------------------------------------
        # Update step size and log likelihoods
        # --------------------------------------------------------------------------
        lam = lam[localnotconverged]
        llhprev = llhprev[localnotconverged]
        llhcurr = llhcurr[localnotconverged]

        # --------------------------------------------------------------------------
        # Update parameters
        # --------------------------------------------------------------------------
        beta = beta[localnotconverged, :, :]
        sigma2 = sigma2[localnotconverged]

        # We don't need this representation of D in the simple case of 1 random 
        # factor, 1 random effect
        if not (r == 1 and nraneffs[0] == 1):
            D = D[localnotconverged, :, :]

        for k in np.arange(len(nraneffs)):
            Ddict[k] = Ddict[k][localnotconverged, :, :]
            
        # --------------------------------------------------------------------------
        # Matrices needed later by many calculations
        # ----------------------------------------------------------------------------
        # X transpose e and Z transpose e
        Xte = XtY - (XtX @ beta)
        Zte = ZtY - (ZtX @ beta)
    
    return(savedparams)

