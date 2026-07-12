import numpy as np
import numba as nb
import ot

from lib.fused_pgw import init_mass


def gopt_cost(gamma,C,mu,nu,Lambda1=0.0,Lambda2=0.0):
    """Return the linear transport cost and marginal-violation penalty for a plan."""
    trans_cost=np.sum(C*gamma)
    penalty_d=np.sum(Lambda1*(np.abs(mu-gamma.sum(1))))
    penalty_c=np.sum(Lambda2*(np.abs(nu-gamma.sum(0))))
    return trans_cost,penalty_d+penalty_c

@nb.njit(cache=True)
def kl_div(p,q,mass=False,eps=1e-16):
    """Compute KL divergence, optionally including the mass correction term."""
    value = np.sum(p * np.log(p / q + eps))
    if mass:
        value = value + np.sum(q - p)
    return value

@nb.njit(cache=True)
def define_K(p, q, M, reg, mass=None):
    # Robustly handle 1D and 2D p
    """Build the Gibbs kernel used by Sinkhorn scaling."""
    if p.ndim > 1:
        n_hists = p.shape[1]
    else:
        n_hists = 1
    if n_hists > 1:
        K = np.exp(-M / reg)
    else:
        K = np.exp(-M / reg) * p[:, np.newaxis] * q[np.newaxis, :]
    if mass is not None:
        K = K * mass / np.sum(K)  # make the total mass of K to be mass
    return K


@nb.njit(cache=True)
def define_Lambda(n,m,Lambda1=None,Lambda2=None):
    """Normalize scalar or vector marginal penalties to source and target lengths."""
    if Lambda1 is None:
        Lambda1=np.full(n, 0.0)
    elif isinstance(Lambda1, (float, int)):
        Lambda1 = np.full(n, Lambda1)
    elif Lambda1.shape[0] == 1:
        Lambda1 = np.full(n, Lambda1[0])
    
    # Handle scalar input for Lambda2
    if Lambda2 is None:
        Lambda2 = np.full(m,Lambda1[0])
    elif isinstance(Lambda2, (float, int)):
        Lambda2 = np.full(m, Lambda2)
    elif Lambda2.shape[0] == 1:
        Lambda2 = np.full(m, Lambda2[0])

    if Lambda1.shape[0] != n:
        raise ValueError("Lambda1 should have the same number of rows as n.")
    if Lambda2.shape[0] != m:
        raise ValueError("Lambda2 should have the same number of rows as m.")
    return Lambda1,Lambda2


@nb.njit(cache=True)
def sinkhorn_knopp_gopt_numba_old(mu, nu, M, Lambda1=None,Lambda2=None, reg=0.1, numItermax=1000,F1='TV',F2='TV'):
    r"""
    Solve the entropic GOPT problem and return the OT matrix

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    mu : array-like, shape (mu,) float64
        samples weights in the source domain
    nu : array-like, shape (nu,) float64
    M : array-like, shape (dia_mu, dia_nu) float64
        loss matrix
    lambda1: shape (mu,) positive (can be zero) array, float 64
    lambda2: shape (nu,) positive (can be zero) array. float 64 
    
    reg : float
        Entropic Regularization term >0
    
    numItermax : int64, optional
        Max number of iterations
    stopThr : float64, optional
        Stop threshold on error (>0)
    F1 (F2): penalty type for the first (second) term: 
        TV: total variation 
        PTV: partial total variation
        inf: (lambda_1)_i\to \infty, for each $i$. In this case, PTV, TV converge to hard constraint, $\iota_{\gamma 1_m\leq p}$
    Returns
    -------
    gamma : array-like, shape (dim_mu, dim_nu)
        Optimal transportation matrix for the given parameters

    Examples
    --------


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    


    n,m = M.shape[0], M.shape[1]
    stopThr=1e-10
    
    #initialize u,v 
    u,v = np.ones(n, dtype=float),np.ones(m, dtype=float)
    
    Lambda1,Lambda2=define_Lambda(n,m,Lambda1,Lambda2)
    
    # initialize K 
    K = np.exp(-M/reg)    
    Lambda1_exp_u,Lambda1_exp_l=np.exp(Lambda1/reg),np.exp(-Lambda1/reg)
    Lambda2_exp_u,Lambda2_exp_l=np.exp(Lambda2/reg),np.exp(-Lambda2/reg)
    

    
    # initilize proximal-divider operator 


    if F1=='TV':
        def proxdiv_F1(s,mu,Lambda1_exp_l,Lambda1_exp_u):          
            return np.clip(mu/s,Lambda1_exp_l,Lambda1_exp_u)
    elif F1=='PTV':
        def proxdiv_F1(s,mu,Lambda1_exp_l,Lambda1_exp_u):
            return np.minimum(mu/s,Lambda1_exp_u)
    elif F1=='inf':
        def proxdiv_F1(s,mu,Lambda1_exp_l,Lambda1_exp_u):             
            return mu/s
        

    if F2=='TV':
        def proxdiv_F2(s,nu,Lambda2_exp_l,Lambda2_exp_u):
            return np.clip(nu/s,Lambda2_exp_l,Lambda2_exp_u)
    elif F2=='PTV':
        def proxdiv_F2(s,nu,Lambda2_exp_l,Lambda2_exp_u):
            return np.minimum(nu/s,Lambda2_exp_u)
    elif F2=='inf':
        def proxdiv_F2(s,nu,Lambda2_exp_l,Lambda2_exp_u):             
            return nu/s
        
    if F1 not in ['TV','PTV','inf']:
        raise ValueError("F1 should be either TV or PTV.")
    if F2 not in ['TV','PTV','inf']:
        raise ValueError("F2 should be either TV or PTV.")
                           
    # main loop
    for ii in range(numItermax):
        u_pre=u.copy()
        v_pre=v.copy()
        v = proxdiv_F2(np.dot(K.T,u),nu,Lambda2_exp_l,Lambda2_exp_u)
        u = proxdiv_F1(np.dot(K,v)  ,mu,Lambda1_exp_l,Lambda1_exp_u)
        if ii % 10 == 0:
            err = np.linalg.norm(u_pre - u)+np.linalg.norm(v_pre - v)  # violation of marginal
            if err < stopThr:
                break
    gamma=np.expand_dims(u,1)*(K*v.T)
    
    return gamma





def gopt_lp(mu,nu,M,Lambda1=None,Lambda2=None,numItermax=100000,numThreads=1):
    """
    linear programming solvers for the GOPT problem. In this case, we chose PTV for penalties 
    and returns the OT plan by linear programming in PythonOT 
    
    Parameters
    ----------
    mu : np.ndarray (dim_mu,) float64 
        Unnormalized histogram of dimension `dia_mu`
    nu : np.ndarray (dim_nu,) float64
        Unnormalized histograms of dimension `dia_nu`
    M : np.ndarray (dim_mu, dim_nu) float64
        cost matrix
    reg : float
        Regularization term > 0
    numItermax : int64, optional
        Max number of iterations
    
    lambda1: shape (mu,) positive (can be zero) array, float 64
    lambda2: shape (nu,) positive (can be zero) array. float 64 
    
    Returns
    -------
    gamma : (dim_mu, dim_nu) ndarray
        Optimal transportation matrix for the given parameters
        
    """
    n,m=M.shape
    # check Lambda1,Lambda2
    Lambda1,Lambda2=define_Lambda(n,m,Lambda1,Lambda2)


    mu1,nu1=np.zeros(n+1),np.zeros(m+1)
    mu1[0:n],nu1[0:m]=mu,nu
    mu1[-1],nu1[-1]=np.sum(nu),np.sum(mu)
    M1=np.zeros((n+1,m+1),dtype=np.float64)
    Lambda_matrix=np.expand_dims(Lambda1,1)+Lambda2
    M1[0:n,0:m]=M-Lambda_matrix
    gamma1=ot.lp.emd(mu1,nu1,M1,numItermax=numItermax,numThreads=numThreads)
    gamma=gamma1[0:n,0:m]
    return gamma


@nb.njit(cache=True)
def proxdiv(s,mu,Lambda1_exp_l,Lambda1_exp_u,F='TV'):
    """Apply the proximal update associated with a selected marginal divergence."""
    if F=='TV':
        return np.clip(mu/s,Lambda1_exp_l,Lambda1_exp_u)
    elif F=='PTV':
        return np.minimum(mu/s,Lambda1_exp_u)
    elif F=='inf':
        return mu/s
    

@nb.njit(cache=True)
def proxdiv_shift(s,mu,Lambda,reg,u=0.0,F='TV'):
    """Apply the shifted proximal update used by stabilized Sinkhorn iterations."""
    Lambda1_exp_l=np.exp((-u-Lambda)/reg)
    Lambda1_exp_u=np.exp((-u+Lambda)/reg)
    if F=='TV':
        return np.clip(mu/s,Lambda1_exp_l,Lambda1_exp_u)
    elif F=='PTV':
        return np.minimum(mu/s,Lambda1_exp_u)
    elif F=='inf':
        return mu/s
    
    

def logsumexp(M, axis):
    r"""Log-sum-exp reduction compatible with autograd (no numpy implementation)"""
    amax = np.amax(M, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(M - amax), axis=axis)) + np.squeeze(amax, axis=axis)

    

@nb.njit(cache=True)
def logsumexp_numba(M, axis=0):
    """Numba-compatible log-sum-exp helper for stabilized updates."""
    n0, n1 = M.shape

    if axis == 0:
        out = np.empty(n1, dtype=M.dtype)
        for j in range(n1):
            col = M[:, j]
            amax = np.max(col)  # works on 1D vector
            s = 0.0
            for i in range(n0):
                s += np.exp(col[i] - amax)
            out[j] = np.log(s) + amax
        return out

    elif axis == 1:
        out = np.empty(n0, dtype=M.dtype)
        for i in range(n0):
            row = M[i, :]
            amax = np.max(row)  # works on 1D vector
            s = 0.0
            for j in range(n1):
                s += np.exp(row[j] - amax)
            out[i] = np.log(s) + amax
        return out
    


@nb.njit(cache=True)
def proxdiv_log(s,mu_log,Lambda_bound,axis=0,F='TV'):
    """Apply a marginal-divergence proximal update in log space."""
    if F=='TV':
        return np.clip(mu_log-logsumexp_numba(s,axis=axis),-Lambda_bound,Lambda_bound)
    elif F=='PTV':

        return np.minimum(mu_log-logsumexp_numba(s,axis=axis),Lambda_bound)
    elif F=='inf':
        return mu_log-logsumexp_numba(s,axis=axis)


@nb.njit(cache=True)
def sinkhorn_knopp_gopt_numba(mu, nu, M, Lambda1=None,Lambda2=None, reg=0.1, numItermax=1000,F1='TV',F2='TV'):
    r"""
    Solve the entropic GOPT problem and return the OT matrix

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    mu : array-like, shape (mu,) float64
        samples weights in the source domain
    nu : array-like, shape (nu,) float64
    M : array-like, shape (dia_mu, dia_nu) float64
        loss matrix
    lambda1: shape (mu,) positive (can be zero) array, float 64
    lambda2: shape (nu,) positive (can be zero) array. float 64 
    
    reg : float
        Entropic Regularization term >0
    
    numItermax : int64, optional
        Max number of iterations
    stopThr : float64, optional
        Stop threshold on error (>0)
    F1 (F2): penalty type for the first (second) term: 
        TV: total variation 
        PTV: partial total variation
        inf: (lambda_1)_i\to \infty, for each $i$. In this case, PTV, TV converge to hard constraint, $\iota_{\gamma 1_m\leq p}$
    Returns
    -------
    gamma : array-like, shape (dim_mu, dim_nu)
        Optimal transportation matrix for the given parameters

    Examples
    --------


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    


    n,m = M.shape[0], M.shape[1]
    stopThr=1e-10
    
    #initialize u,v 
    u,v = np.ones(n, dtype=float)/n,np.ones(m, dtype=float)/m
    
    Lambda1,Lambda2=define_Lambda(n,m,Lambda1,Lambda2)

    
    # initialize K 
    
    K = define_K(mu, nu, M, reg)    
    Lambda1_exp_u,Lambda1_exp_l=np.exp(Lambda1/reg),np.exp(-Lambda1/reg)
    Lambda2_exp_u,Lambda2_exp_l=np.exp(Lambda2/reg),np.exp(-Lambda2/reg)
    

    
    # initilize proximal-divider operator 
        
    if F1 not in ['TV','PTV','inf']:
        raise ValueError("F1 should be either TV or PTV.")
    if F2 not in ['TV','PTV','inf']:
        raise ValueError("F2 should be either TV or PTV.")
                           
    # main loop
    for ii in range(numItermax):
        u_pre=u.copy()
        v_pre=v.copy()
        v = proxdiv(np.dot(K.T,u),nu,Lambda2_exp_l,Lambda2_exp_u,F=F2)
        u = proxdiv(np.dot(K,v)  ,mu,Lambda1_exp_l,Lambda1_exp_u,F=F1)
        if ii % 10 == 0:
            err = np.linalg.norm(u_pre - u)+np.linalg.norm(v_pre - v)  # violation of marginal
            if err < stopThr:
                break
    gamma=u[:, np.newaxis] * (K * v.T)
    #np.expand_dims(u,1)*(K*v.T)
    
    return gamma


@nb.njit(cache=True)
def sinkhorn_knopp_gopt_numba_stable(mu, nu, M, Lambda1=None,Lambda2=None, reg=0.1, upper_bound=1e3,numItermax=1000,F1='TV',F2='TV'):
    r"""
    Solve the entropic GOPT problem and return the OT matrix

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    mu : array-like, shape (mu,) float64
        samples weights in the source domain
    nu : array-like, shape (nu,) float64
    M : array-like, shape (dia_mu, dia_nu) float64
        loss matrix
    lambda1: shape (mu,) positive (can be zero) array, float 64
    lambda2: shape (nu,) positive (can be zero) array. float 64 
    
    reg : float
        Entropic Regularization term >0
    
    numItermax : int64, optional
        Max number of iterations
    stopThr : float64, optional
        Stop threshold on error (>0)
    F1 (F2): penalty type for the first (second) term: 
        TV: total variation 
        PTV: partial total variation
        inf: (lambda_1)_i\to \infty, for each $i$. In this case, PTV, TV converge to hard constraint, $\iota_{\gamma 1_m\leq p}$
    Returns
    -------
    gamma : array-like, shape (dim_mu, dim_nu)
        Optimal transportation matrix for the given parameters

    Examples
    --------


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    


    n,m = M.shape[0], M.shape[1]
    stopThr=1e-10
    
    #initialize u,v 
    u,v = np.ones(n, dtype=float)/n,np.ones(m, dtype=float)/m
    log_u_shift,log_v_shift=np.zeros(n, dtype=float),np.zeros(m, dtype=float)
    Lambda1,Lambda2=define_Lambda(n,m,Lambda1,Lambda2)
    # initialize K 
    K = define_K(mu, nu, M, reg)   
   
    # initilize proximal-divider operator 
        
    if F1 not in ['TV','PTV','inf']:
        raise ValueError("F1 should be either TV or PTV.")
    if F2 not in ['TV','PTV','inf']:
        raise ValueError("F2 should be either TV or PTV.")
                           
    # main loop

    for ii in range(numItermax):
        u_pre=u.copy()
        v_pre=v.copy()
        v = proxdiv_shift(np.dot(K.T,u),nu,Lambda2,reg,log_v_shift,F=F2)
        u = proxdiv_shift(np.dot(K,v),  mu,Lambda1,reg,log_u_shift,F=F1)
        if np.max(u)>=upper_bound or np.max(v)>=upper_bound:
            log_u_shift,log_v_shift= log_u_shift+reg*np.log(u),log_v_shift+reg*np.log(v)  # shift to avoid numerical issues
            K= np.exp((log_u_shift[:, np.newaxis] + log_v_shift[np.newaxis,:] - M)/reg)
            u,v=np.ones(n, dtype=float)/n,np.ones(m, dtype=float)/m  # reset u,v
        

        if ii % 10 == 0:
            err = np.linalg.norm(u_pre - u)+np.linalg.norm(v_pre - v)  # violation of marginal
            if err < stopThr:
                break
    gamma=u[:, np.newaxis] * (K * v.T)
    #np.expand_dims(u,1)*(K*v.T)
    
    return gamma


@nb.njit(cache=True)
def sinkhorn_knopp_gopt_numba2_log(mu, nu, M, Lambda1=None,Lambda2=None, reg=0.1, numItermax=1000,F1='TV',F2='TV'):
    r"""
    Solve the entropic GOPT problem and return the OT matrix

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    mu : array-like, shape (mu,) float64
        samples weights in the source domain
    nu : array-like, shape (nu,) float64
    M : array-like, shape (dia_mu, dia_nu) float64
        loss matrix
    lambda1: shape (mu,) positive (can be zero) array, float 64
    lambda2: shape (nu,) positive (can be zero) array. float 64 
    
    reg : float
        Entropic Regularization term >0
    
    numItermax : int64, optional
        Max number of iterations
    stopThr : float64, optional
        Stop threshold on error (>0)
    F1 (F2): penalty type for the first (second) term: 
        TV: total variation 
        PTV: partial total variation
        inf: (lambda_1)_i\to \infty, for each $i$. In this case, PTV, TV converge to hard constraint, $\iota_{\gamma 1_m\leq p}$
    Returns
    -------
    gamma : array-like, shape (dim_mu, dim_nu)
        Optimal transportation matrix for the given parameters

    Examples
    --------


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    


    n,m = M.shape[0], M.shape[1]
    stopThr=1e-10
    
    #initialize u,v 
    log_u,log_v = np.zeros(n, dtype=float),np.zeros(m, dtype=float)

    Lambda1,Lambda2=define_Lambda(n,m,Lambda1,Lambda2)

    
    # initialize K 
    log_K = -M/reg    
    Lambda1_bound=Lambda1/reg 
    Lambda2_bound=Lambda2/reg 

    log_mu,log_nu=np.log(mu),np.log(nu)

    
    # initilize proximal-divider operator 
        
    if F1 not in ['TV','PTV','inf']:
        raise ValueError("F1 should be either TV or PTV.")
    if F2 not in ['TV','PTV','inf']:
        raise ValueError("F2 should be either TV or PTV.")
                           
    # main loop
    for ii in range(numItermax):
        log_u_pre=log_u.copy()
        log_v_pre=log_v.copy()
        log_v = proxdiv_log(log_K+log_u[:, np.newaxis],log_nu,Lambda2_bound,F=F2,axis=0)
        log_u = proxdiv_log(log_K+log_v[np.newaxis,:],log_mu,Lambda1_bound,F=F1,axis=1)
        if ii % 10 == 0:
            err = np.linalg.norm(log_u_pre - log_u)+np.linalg.norm(log_v_pre - log_v)  # violation of marginal
            if np.exp(err) < stopThr:
                break
    gamma=np.exp(log_u[:, np.newaxis] + log_K + log_v[np.newaxis,:])
    #np.expand_dims(u,1)*(K*v.T)
    
    return gamma


@nb.njit(cache=True)
def tensor_dot_param(C1, C2, Lambda=0, loss="square_loss"):
    """Prepare loss-transformed structure matrices for the GW tensor product."""
    if loss == "square_loss":

        def f1(r1):
            return r1**2 - 2 * Lambda

        def f2(r2):
            return r2**2

        def h1(r1):
            return r1

        def h2(r2):
            return 2 * r2

    # else:
    #     warnings.warn("loss function error")

    fC1 = f1(C1)
    fC2 = f2(C2)
    hC1 = h1(C1)
    hC2 = h2(C2)

    return fC1, fC2, hC1, hC2

@nb.njit(cache=True)
def tensor_dot_func(fC1, fC2, hC1, hC2, Gamma):
    # Gamma=np.ascontiguousarray(Gamma)
    """Evaluate the GW tensor product for a current transport plan."""
    n, m = Gamma.shape
    C1 = fC1.dot(Gamma.sum(1).reshape((-1, 1)))  # .dot(np.ones((1,m)))
    C2 = Gamma.sum(0).dot(fC2.T)
    tensor_dot = (C1 + C2) - hC1.dot(Gamma).dot(hC2.T)
    return tensor_dot




#@nb.njit(cache=True)
def tensor_product(C1,C2,Lambda=0,loss='square_loss'):
    """Return the structural quadratic term used in Sinkhorn FPGW updates."""
    if loss == 'square_loss':
        fC1, fC2, hC1, hC2 = tensor_dot_param(C1, C2, Lambda, loss="square_loss")
        fC1, fC2, hC1, hC2 = (
        np.ascontiguousarray(fC1),
        np.ascontiguousarray(fC2),
        np.ascontiguousarray(hC1),
        np.ascontiguousarray(hC2),)
        fC1t, fC2t, hC1t, hC2t = tensor_dot_param(C1.T, C2.T, Lambda, loss="square_loss")
        fC1t, fC2t, hC1t, hC2t = (
            np.ascontiguousarray(fC1t),
            np.ascontiguousarray(fC2t),
            np.ascontiguousarray(hC1t),
            np.ascontiguousarray(hC2t),)
        
        def M_circ_gamma(gamma):
            return tensor_dot_func(fC1, fC2, hC1, hC2, gamma)
        def Mt_circ_gamma(gamma):
            return tensor_dot_func(fC1t, fC2t, hC1t, hC2t, gamma)
            
    elif loss=='sub_graph_loss':
        def M_circ_gamma(gamma):
            return tensor_dot_edge(C1, C2, gamma)
        def Mt_circ_gamma(gamma):
            return tensor_dot_edge(C1.T, C2.T, gamma)
            
            
    return M_circ_gamma,Mt_circ_gamma

def entropic_semirelaxed_partial_fused_gromov_wasserstein(
    M,
    C1,
    C2,
    p=None,
    q=None,
    loss_fun="square_loss",
    symmetric=True,
    epsilon=0.01,
    omega2=0.5,
    Lambda1=0.5,
    Lambda2=None,
    mass=None,
    G0=None,
    max_iter=1e4,
    reg=0.01,
    tol=1e-9,
    log=False,
    verbose=False,
    random_state=0,
    F1='PTV',
    F2='PTV'
):
    r"""
    Computes the entropic-regularized semi-relaxed FGW transport between two graphs (see :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`)
    estimated using a Mirror Descent algorithm following the KL geometry.

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix between features
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}` source weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices


    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like of shape (ns,nt) or string, optional
        If `G0=None` the initial transport plan of the solver is :math:`\mathbf{p} \frac{\mathbf{1}_{nt}}{nt}^\top`.
        If G0 is a tensor it must satisfy marginal constraints and will be
        used as initial transport of the solver.
        if G0 is a string it will be interpreted as a method for
        :func:`ot.gromov.semirelaxed_init_plan` taking values in "product",
        "random_product", "random", "fluid", "fluid_soft", "spectral",
        "spectral_soft", "kmeans", "kmeans_soft".
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error computed on transport plans
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
    random_state: int, optional
        Random seed used in stochastic initialization methods.

    Returns
    -------
    G : array-like, shape (`ns`, `nt`)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-semirelaxed-fused-gromov-wasserstein:
    References
    ----------
    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    #arr = [M, C1, C2]
    n,m=M.shape 
    
    if p is None:
        p=np.ones(n)/n 
    if q is None:
        q=np.ones(m)/m
    if G0 is None:
        G0= np.outer(p, q)
    if omega2>1 or omega2<0:
        raise ValueError("Problem infeasible. Parameter omega should be in [0,1].")
    if mass is None:
        mass=init_mass(mass,p,q)
    
    
    cpt = 0
    err = 1e10
    G = G0

    #
    fC1, fC2, hC1, hC2 = tensor_dot_param(C1, C2, Lambda=0, loss="square_loss")
    fC1, fC2, hC1, hC2 = (np.ascontiguousarray(fC1),np.ascontiguousarray(fC2),np.ascontiguousarray(hC1),np.ascontiguousarray(hC2))
    
    omega1=1-omega2
    def c_pi(pi):
        return 1/2**omega1*M+omega2*tensor_dot_func(fC1, fC2, hC1, hC2, pi)

    if log:
        log = {"err": []}

    while err > tol and cpt < max_iter:
        pi=G.copy()
        C_pi=c_pi(pi)
        G=sinkhorn_knopp_gopt_numba(p, q, C_pi, Lambda1=Lambda1,Lambda2=Lambda2, reg=reg, numItermax=1000,F1=F1,F2=F2)


        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(G - pi)

            if log:
                log["err"].append(err)
                log["pi_sum"].append(G.sum())

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

        cpt += 1
    if log:
        return G, log

    return G

        
    

@nb.njit(cache=True)
def sinkhorn_knopp_mopt_numba(mu, nu, M, mass=None, reg=0.1, numItermax=100000):
    r"""
    (we modify the code in PythonOT) 
    Solves the partial optimal transport problem
    and returns the OT plan vis Sinkhorn algorithm (we modify the code in PythonOT)

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma,
                 \mathbf{M} \rangle_F + \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \gamma \mathbf{1} &\leq \mathbf{a} \\
             \gamma^T \mathbf{1} &\leq \mathbf{b} \\
             \gamma &\geq 0 \\
             \mathbf{1}^T \gamma^T \mathbf{1} = m
             &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\} \\

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega`  is the entropic regularization term,
      :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[3] <references-entropic-partial-wasserstein>` (prop. 5)


    Parameters
    ----------
    mu : np.ndarray (dia_mu,) float64
        Unnormalized histogram of dimension `dia_mu`
    b : np.ndarray (dia_nu,) float64
        Unnormalized histograms of dimension `dia_nu`
    M : np.ndarray (dia_mu, dia_nu)
        cost matrix
    reg : float
        Regularization term > 0
    mass : float64, optional
        Amount of mass to be transported
    numItermax : int64, optional
        Max number of iterations
    stopThr : float64, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (dia_mu, dia_nu) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> mu = [.1, .2]
    >>> nu = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(entropic_partial_wasserstein(a, b, M, 1, 0.1), 2)
    array([[0.06, 0.02],
           [0.01, 0.  ]])


    .. _references-entropic-partial-wasserstein:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
       (2015). Iterative Bregman projections for regularized transportation
       problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    See Also
    --------
    ot.partial.partial_wasserstein: exact Partial Wasserstein
    """


    n, m = M.shape
    dx = np.ones(n, dtype=np.float64)
    dy = np.ones(m, dtype=np.float64)
    stopThr=1e-13
    if mass is None:
        mass=min(mu.sum(),nu.sum())

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute

    K=define_K(mu,nu,M,reg,mass)
    err, cpt = 1, 0
    q1 = np.ones(K.shape)
    q2 = np.ones(K.shape)
    q3 = np.ones(K.shape)
    

    if mass < 0:
        raise ValueError("Problem infeasible. Parameter mass should be greater"
                         " than 0.")
    elif mass > min(np.sum(mu), np.sum(nu)):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|mu|_1, |nu|_1).")
        

    while (err > stopThr and cpt < numItermax):
        Kprev = K
        K = K * q1
        K1 = np.dot(np.diag(np.minimum(mu / np.sum(K, axis=1), dx)), K)
        q1 = q1 * Kprev / K1
        K1prev = K1
        K1 = K1 * q2
        K2 = np.dot(K1, np.diag(np.minimum(nu / np.sum(K1, axis=0), dy)))
        q2 = q2 * K1prev / K2
        K2prev = K2
        K2 = K2 * q3
        K = K2 * (mass / np.sum(K2))
        q3 = q3 * K2prev / K


        if cpt % 10 == 0:
            err = np.linalg.norm(Kprev - K)

        cpt = cpt + 1
    if cpt==numItermax-1:
        print('warning, maximum iteration reached')
    return K


def mopt_lp(mu, nu, M, mass=None, numItermax=100000,numThreads=1):
    r"""
    (we modify the code in PythonOT) 
    Solves the partial optimal transport problem
    and returns the OT plan vis Sinkhorn algorithm (we modify the code in PythonOT)

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma,
                 \mathbf{M} \rangle_F + \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \gamma \mathbf{1} &\leq \mathbf{a} \\
             \gamma^T \mathbf{1} &\leq \mathbf{b} \\
             \gamma &\geq 0 \\
             \mathbf{1}^T \gamma^T \mathbf{1} = m
             &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\} \\

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega`  is the entropic regularization term,
      :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[3] <references-entropic-partial-wasserstein>` (prop. 5)


    Parameters
    ----------
    mu : np.ndarray (dia_mu,) float64
        Unnormalized histogram of dimension `dia_mu`
    b : np.ndarray (dia_nu,) float64
        Unnormalized histograms of dimension `dia_nu`
    M : np.ndarray (dia_mu, dia_nu)
        cost matrix
    reg : float
        Regularization term > 0
    m : float64, optional
        Amount of mass to be transported
    numItermax : int64, optional
        Max number of iterations
    stopThr : float64, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (dia_mu, dia_nu) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> mu = [.1, .2]
    >>> nu = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(entropic_partial_wasserstein(a, b, M, 1, 0.1), 2)
    array([[0.06, 0.02],
           [0.01, 0.  ]])


    .. _references-entropic-partial-wasserstein:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
       (2015). Iterative Bregman projections for regularized transportation
       problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    See Also
    --------
    ot.partial.partial_wasserstein: exact Partial Wasserstein
    """

    mu = np.asarray(mu, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    n, m = M.shape
    if mass is None:
        mass=min(np.sum(mu),np.sum(nu))
    if mass < 0:
        raise ValueError("Problem infeasible. Parameter mass should be greater"
                         " than 0.")
    elif mass > np.min(np.stack((np.sum(mu), np.sum(nu)))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|mu|_1, |nu|_1).")
        
          

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute
    mu_extended  = np.concatenate((mu, np.ones(1)*(np.sum(nu)-mass)))
    nu_extended  = np.concatenate((nu, np.ones(1)*(np.sum(mu)-mass)))
    
    alpha,beta,A=0.0,1.0,np.max(M)+1
    M_extended=np.zeros((n+1,m+1))
    M_extended[0:n,0:m]=M
#    M_extended[0:n,m],M_extended[m,0:m]=alpha,alpha
    M_extended[n,m]=A+2*alpha+beta

    gamma_extended=ot.lp.emd(mu_extended,nu_extended,M_extended,numItermax=numItermax,numThreads=numThreads)
    gamma=gamma_extended[0:n,0:m]
    
    return gamma





@nb.njit(cache=True)
def entropic_semirelaxed_partial_fused_gromov_wasserstein_numba(
    M,C1,C2,p=None,q=None,loss_fun="square_loss",symmetric=True,omega2=0.5,mass=None,Lambda1=0.5,Lambda2=None,G0=None,max_iter=1e4,
    reg=0.01,tol=1e-9,verbose=False,random_state=0,F1='PTV',F2='PTV',stable=False,upperbound=15.0,Type='pot', two_step = True):
    r"""
    Computes the entropic-regularized semi-relaxed FGW transport between two graphs (see :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`)
    estimated using a Mirror Descent algorithm following the KL geometry.

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix between features
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}` source weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices


    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like of shape (ns,nt) or string, optional
        If `G0=None` the initial transport plan of the solver is :math:`\mathbf{p} \frac{\mathbf{1}_{nt}}{nt}^\top`.
        If G0 is a tensor it must satisfy marginal constraints and will be
        used as initial transport of the solver.
        if G0 is a string it will be interpreted as a method for
        :func:`ot.gromov.semirelaxed_init_plan` taking values in "product",
        "random_product", "random", "fluid", "fluid_soft", "spectral",
        "spectral_soft", "kmeans", "kmeans_soft".
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error computed on transport plans
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
    random_state: int, optional
        Random seed used in stochastic initialization methods.

    Returns
    -------
    G : array-like, shape (`ns`, `nt`)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-semirelaxed-fused-gromov-wasserstein:
    References
    ----------
    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    #arr = [M, C1, C2]
    n,m=M.shape 
    
    if p is None:
        p=np.ones(n)/n 
    if q is None:
        q=np.ones(m)/m
    if G0 is None:
        G0= np.outer(p, q)
    if Type =='mpot':
        mass=init_mass(mass,p,q)
    elif Type not in ['pot','mpot']:
        raise ValueError("Type should be either 'pot' or 'mpot'.")

    if omega2>1 or omega2<0:
        raise ValueError("Problem infeasible. Parameter omega should be in [0,1].")

    Lambda1,Lambda2=define_Lambda(n,m,Lambda1=Lambda1,Lambda2=Lambda2)
    
    
    cpt = 0
    err = 1e10
    G = G0
    G_gamma = G0

    #
    fC1, fC2, hC1, hC2 = tensor_dot_param(C1, C2, Lambda=0, loss="square_loss")
    fC1, fC2, hC1, hC2 = (np.ascontiguousarray(fC1),np.ascontiguousarray(fC2),np.ascontiguousarray(hC1),np.ascontiguousarray(hC2))
    omega1=1-omega2



    while err > tol and cpt < max_iter:

        if Type=='pot' and not stable:
            if two_step == False:
                pi=G.copy()
                C_pi=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, pi)+reg*kl_div(pi,np.outer(p,q))
                Lambda1_new,Lambda2_new,reg_new=Lambda1*np.sum(pi),Lambda2*np.sum(pi),reg*np.sum(pi)

                G=sinkhorn_knopp_gopt_numba(p, q, C_pi, Lambda1=Lambda1_new,Lambda2=Lambda2_new, reg=reg_new, numItermax=1000,F1=F1,F2=F2)    
                G=np.sqrt(np.sum(pi)/np.sum(G))*G
                
            
            if two_step:
                gamma=G_gamma.copy()            
                C_gamma=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, gamma)+reg*kl_div(gamma,np.outer(p,q))            
                Lambda1_new,Lambda2_new,reg_new = Lambda1*np.sum(gamma),Lambda2*np.sum(gamma),reg*np.sum(gamma) 
                G=sinkhorn_knopp_gopt_numba(p, q, C_gamma, Lambda1=Lambda1_new,Lambda2=Lambda2_new, reg=reg_new, numItermax=1000,F1=F1,F2=F2)
                G=np.sqrt(np.sum(gamma)/np.sum(G))*G
                
                pi=G.copy()
                C_pi=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, pi)+reg*kl_div(pi,np.outer(p,q))
                Lambda1_new,Lambda2_new,reg_new = Lambda1*np.sum(pi),Lambda2*np.sum(pi),reg*np.sum(pi)
                G_gamma=sinkhorn_knopp_gopt_numba(p, q, C_pi, Lambda1=Lambda1_new,Lambda2=Lambda2_new, reg=reg_new, numItermax=1000,F1=F1,F2=F2)
                G_gamma=np.sqrt(np.sum(pi)/np.sum(G_gamma))*G_gamma
                if cpt % 10 == 0:
                    # we can speed up the process by checking for the error only all
                    # the 10th iterations
                    err = np.linalg.norm(G_gamma - gamma)
                    if err <= tol:
                        break
                
        elif Type=='pot' and stable:
            if two_step == False:
                pi=G.copy()
                C_pi=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, pi)+reg*kl_div(pi,np.outer(p,q))
                Lambda1_new,Lambda2_new,reg_new=Lambda1*np.sum(pi),Lambda2*np.sum(pi),reg*np.sum(pi)

                G=sinkhorn_knopp_gopt_numba_stable(p, q, C_pi, Lambda1=Lambda1_new,Lambda2=Lambda2_new, reg=reg_new, numItermax=1000,F1=F1,F2=F2, upper_bound=upperbound)    
                G=np.sqrt(np.sum(pi)/np.sum(G))*G
                
                if cpt % 10 == 0:
                    err = np.linalg.norm(pi - G)
                    if err <= tol:
                        break
            if two_step:
                gamma=G_gamma.copy()            
                C_gamma=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, gamma)+reg*kl_div(gamma,np.outer(p,q))            
                Lambda1_new,Lambda2_new,reg_new = Lambda1*np.sum(gamma),Lambda2*np.sum(gamma),reg*np.sum(gamma) 
                G=sinkhorn_knopp_gopt_numba_stable(p, q, C_gamma, Lambda1=Lambda1_new,Lambda2=Lambda2_new, reg=reg_new, numItermax=1000,F1=F1,F2=F2, upper_bound=upperbound)
                G=np.sqrt(np.sum(gamma)/np.sum(G))*G
                
                pi=G.copy()
                C_pi=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, pi)+reg*kl_div(pi,np.outer(p,q))
                Lambda1_new,Lambda2_new,reg_new = Lambda1*np.sum(pi),Lambda2*np.sum(pi),reg*np.sum(pi)
                G_gamma=sinkhorn_knopp_gopt_numba_stable(p, q, C_pi, Lambda1=Lambda1_new,Lambda2=Lambda2_new, reg=reg_new, numItermax=1000,F1=F1,F2=F2, upper_bound=upperbound)
                G_gamma=np.sqrt(np.sum(pi)/np.sum(G_gamma))*G_gamma
                if cpt % 10 == 0:
                    # we can speed up the process by checking for the error only all
                    # the 10th iterations
                    err = np.linalg.norm(G_gamma - gamma)
                    if err <= tol:
                        break
                
        elif Type=='mpot':
            if two_step:
                gamma=G_gamma.copy()            
                C_gamma=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, gamma)+reg*kl_div(gamma,np.outer(p,q))            
                Lambda1_new,Lambda2_new,reg_new = Lambda1*np.sum(gamma),Lambda2*np.sum(gamma),reg*np.sum(gamma) 
                G=sinkhorn_knopp_mopt_numba(p, q, C_gamma, mass=mass, reg=reg_new, numItermax=1000)
                G=np.sqrt(np.sum(gamma)/np.sum(G))*G
                
                pi=G.copy()
                C_pi=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, pi)+reg*kl_div(pi,np.outer(p,q))
                Lambda1_new,Lambda2_new,reg_new = Lambda1*np.sum(pi),Lambda2*np.sum(pi),reg*np.sum(pi)
                G_gamma=sinkhorn_knopp_mopt_numba(p, q, C_pi, mass=mass, reg=reg_new, numItermax=1000)
                G_gamma=np.sqrt(np.sum(pi)/np.sum(G_gamma))*G_gamma
                if cpt % 10 == 0:
                    err = np.linalg.norm(G_gamma - gamma)
                    if err <= tol:
                        break
            
            if two_step == False:
                pi=G.copy()
                C_pi=1/2*omega1*M+(omega2)*tensor_dot_func(fC1, fC2, hC1, hC2, pi)+reg*kl_div(pi,np.outer(p,q))
                Lambda1_new,Lambda2_new,reg_new=Lambda1*np.sum(pi),Lambda2*np.sum(pi),reg*np.sum(pi)
    
                G=sinkhorn_knopp_mopt_numba(p, q, C_pi, mass=mass, reg=reg_new, numItermax=1000)        
                G=np.sqrt(np.sum(pi)/np.sum(G))*G

                if cpt % 10 == 0:
                    # we can speed up the process by checking for the error only all
                    # the 10th iterations
                    err = np.linalg.norm(pi - G)
                    if err <= tol:
                        break


        cpt += 1
        # if log:
        #     return G, log

    return G
