# Frank-Wolfe solvers for Fused Partial Gromov-Wasserstein / Partial Gromov-Wasserstein.
#
# Imported and adapted from the FPGW (Fused Partial Gromov-Wasserstein) repository:
#   https://github.com/yikun-baio/fused-pgw
# The algorithms in this file originate there; SpaOT is a downstream user.
# Some functions have been modified for SpaOT analyses.

import numpy as np
import numba as nb
import ot
from scipy.optimize._linesearch import scalar_search_armijo

from .gromov_utils import (
    def_tensor_product,
    gwloss_partial,
    gwloss_partial_numba,
    init_plan,
)
from .opt import emd_lp


@nb.njit(cache=True)
def solve_quadratic(a,b):
    """Return the best Frank-Wolfe step for a one-dimensional quadratic line search."""
    if a > 0:  # due to numerical precision
        if b > 0:
            gamma = 0
        else:
            gamma = min(1, np.divide(-b, 2.0 * a))
    else:
        if (a + b) < 0:
            gamma = 1
        else:
            gamma = 0
    return gamma


@nb.njit(cache=True)
def tensor_dot_edge(C1,C2,gamma):
    """Compute the edge-wise tensor product term used in the structural GW gradient."""
    n,m=C1.shape[0],C2.shape[0]
    dot=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            M_ij=np.zeros((n,m))
            for i1 in range(n):
                for j1 in range(m):
                    M_ij[i1,j1]=min(C1[i,i1]-C2[j,j1],0)
            dot[i,j]=np.sum(M_ij*gamma)
    return dot

def line_search_armijo(f,xk,pk,gfk,old_fval,args=(),c1=1e-4,alpha0=0.99,alpha_min=0.0,alpha_max=None,nx=None,**kwargs):
    r"""
    Armijo linesearch function that works with matrices

    Find an approximate minimum of :math:`f(x_k + \alpha \cdot p_k)` that satisfies the
    armijo conditions.

    .. note:: If the loss function f returns a float (resp. a 1d array) then
        the returned alpha and fa are float (resp. 1d arrays).

    Parameters
    ----------
    f : callable
        loss function
    xk : array-like
        initial position
    pk : array-like
        descent direction
    gfk : array-like
        gradient of `f` at :math:`x_k`
    old_fval : float or 1d array
        loss value at :math:`x_k`
    args : tuple, optional
        arguments given to `f`
    c1 : float, optional
        :math:`c_1` const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)
    alpha_min : float, default=0.
        minimum value for alpha
    alpha_max : float, optional
        maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    alpha : float or 1d array
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float or 1d array
        loss value at step alpha

    """

    xk0, pk0 = xk, pk



    fc = [0]

    def phi(alpha1):
        # it's necessary to check boundary condition here for the coefficient
        # as the callback could be evaluated for negative value of alpha by
        # `scalar_search_armijo` function here:
        #
        # https://github.com/scipy/scipy/blob/11509c4a98edded6c59423ac44ca1b7f28fba1fd/scipy/optimize/linesearch.py#L686
        #
        # see more details https://github.com/PythonOT/POT/issues/502
        alpha1 = np.clip(alpha1, alpha_min, alpha_max)
        # The callable function operates on nx backend
        fc[0] += 1
        fval = f(xk0 + alpha1 * pk0, *args)
        return fval
        

    if old_fval is None:
        phi0 = phi(0.0)
    else:
        phi0 = old_fval


    
    
    derphi0 = np.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0, amin=alpha_min
    )

    if alpha is None:
        return 0.0, fc[0], phi0
    else:
        alpha = np.clip(alpha, alpha_min, alpha_max)
        return alpha,fc[0],phi1



def do_linesearch(cost,G,deltaG,Grad,f_val,amijo=True,M_circ_gamma=None):
    #Gc= st
    #G=xt
    #deltaG=st-xt
    #Gc+alpha*deltaG=st+alpha(st-xt)
    """
    Solve the linesearch in the FW iterations
    Parameters
    ----------
    cost : method
        The FGW cost
    G : ndarray, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : ndarray (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : ndarray (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val :  float
        Value of the cost at G
    amijo : bool, optionnal
            If True the steps of the line-search is found via an amijo research. Else closed form is used.
            If there is convergence issues use False.
    C1 : ndarray (ns,ns), optionnal
        Structure matrix in the source domain. Only used when amijo=False
    C2 : ndarray (nt,nt), optionnal
        Structure matrix in the target domain. Only used when amijo=False
    reg : float, optionnal
          Regularization parameter. Corresponds to the alpha parameter of FGW. Only used when amijo=False
    Gc : ndarray (ns,nt)
        Optimal map found by linearization in the FW algorithm. Only used when amijo=False
    constC : ndarray (ns,nt)
             Constant for the gromov cost. See [3]. Only used when amijo=False
    M : ndarray (ns,nt), optionnal
        Cost matrix between the features. Only used when amijo=False
    Returns
    -------
    alpha : float
            The optimal step size of the FW
    fc : useless here
    f_val :  float
             The value of the cost for the next iteration
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if amijo:
        alpha, fc, f_val = line_search_armijo(cost, G, deltaG, Grad, f_val)
    else:
        a = np.sum(M_circ_gamma(deltaG)*deltaG)
        b = np.sum(Grad * deltaG)
        alpha=solve_quadratic(a,b)
        
        #c=cost(G) #f(xt)

        #alpha=solve_1d_linesearch_quad_funct(a,b,c)
        fc=None
        f_val=cost(G+alpha*deltaG)
        
    return alpha,fc,f_val
        
@nb.njit()
def init_mass(mass,p,q):
    """Validate or infer the amount of mass transported in a partial OT/GW problem."""
    if mass is None:
        mass = min((np.sum(p), np.sum(q)))
    if mass < 0:
        raise ValueError("Problem infeasible. Parameter mass should be greater" " than 0.")
    if mass > min((np.sum(p), np.sum(q))):
        raise ValueError(
            "Problem infeasible. Parameter mass should lower or"
            " equal than min(|a|_1, |b|_1)."
        )
    return mass

@nb.njit()
def check_pq(n1,n2,p, q):
    """Create default source and target masses when the caller does not provide them."""
    if p is None:
        p = np.ones(n1)/min(n1,n2)
    if q is None:
        q = np.ones(n2)/min(n1,n2)
    return p,q

@nb.njit()
def pot_extention(p,q,mass=0,nb_dummies=1,Type='mpot'):
    """Append dummy mass used to solve partial OT subproblems as balanced OT problems."""
    n1,n2=p.shape[0],q.shape[0]
    if Type=='mpot':
        q_extended = np.append(q, [(np.sum(p) - mass) / nb_dummies] * nb_dummies)
        p_extended = np.append(p, [(np.sum(q) - mass) / nb_dummies] * nb_dummies)
        M_extended = np.zeros((n1+nb_dummies,n2+nb_dummies))
    elif Type=='pot':
        p_extended, q_extended, M_extended = (np.zeros(n1 + 1),np.zeros(n2 + 1),np.full((n1 + 1, n2 + 1),-1e-15),
    )
        p_extended[0:n1], p_extended[-1] = p, q.sum()
        q_extended[0:n2], q_extended[-1] = q, p.sum()
    else:
        raise ValueError("Type should be either 'mpot' or 'pot'.")
    return p_extended, q_extended, M_extended

@nb.njit()
def init_ot_param(numItermax,n1,n2,Type='emd'):
    """Prepare the extended cost matrix and masses for the linear OT update."""
    if Type =='emd':
        numItermax=max(1e6,(n1+n2)*100)
    if Type=='sinkhorn':
        numItermax= max(200,(n1+n2)*100)
    return numItermax

# This function is adapted from PythonOT package
def fused_partial_gromov_wasserstein(
        C,C1,C2,p=None,q=None,omega2=1,mass=None,Lambda=0,nb_dummies=1,G0=None,numThreads=5,symmetric=True,
        numItermax=None,numItermax_gw=1000,tol=1e-7,log=False,verbose=False,loss_fun='square_loss',
        line_search=True,amijo=False,Type='mpot',**kwargs,
        ):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    Lambda : float, optional. This is the Lambda / omega2 in the equation 8 in the paper.
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """
    n1,n2=C1.shape[0],C2.shape[0]
    numItermax=init_ot_param(numItermax,n1,n2,Type='emd')
    p,q=check_pq(n1,n2,p,q)
    mass=init_mass(mass,p,q)


    if omega2>1 or omega2<0:
        raise ValueError("Problem infeasible. Parameter omega should be in [0,1].")
    omega1=1-omega2
    if G0 is None:
        G0=init_plan(p,q,mass=mass,Type=Type)
    if Type not in ['pot','mpot','ot']:
        raise ValueError("Type should be either 'pot', 'mpot' or 'ot'.")

    
    p_extended, q_extended, M_extended = pot_extention(p,q,mass=mass,nb_dummies=nb_dummies,Type=Type)
    M_circ_gamma,Mt_circ_gamma=def_tensor_product(C1,C2,Lambda=0,loss=loss_fun)
    def cost(G):
        return np.sum(omega1*C * G) +omega2 * np.sum(M_circ_gamma(G)*G)

    cpt = 0
    err = 1

    if log:
        log = {"err": []}
    iter_num = 0
    f_val=cost(G0)
    G=G0
    while err > tol and cpt < numItermax_gw:
        iter_num += 1
        Gprev = np.copy(G)
        if symmetric:
            grad = omega1*C+omega2*2*M_circ_gamma(G) - 2 * Lambda * G.sum()
        else:
            grad = omega1*C+omega2*(M_circ_gamma(G)+Mt_circ_gamma(G)) - 2 * Lambda * G.sum()
        M_extended[0:n1, 0:n2] = grad
        if Type=='mpot':
            M_extended[-nb_dummies:, -nb_dummies:] = np.max(grad) * 2

        G_extend = ot.emd(p_extended,q_extended,M_extended,numItermax=numItermax,
            numThreads=numThreads,log=False)
        

        G = G_extend[0:n1,0:n2]
        deltaG = G - Gprev
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(deltaG)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial_numba(C1, C2, G)))

        if line_search:
            alpha,fc,f_val= do_linesearch(cost,Gprev,deltaG,grad,f_val,amijo=amijo,M_circ_gamma=M_circ_gamma)
        else:
            alpha = 1

        G0 = Gprev + alpha * deltaG
        cpt += 1
    #print('done computing fpgw, cpt ',cpt)

    if log:
        log["partial_gw_dist"] = gwloss_partial_numba(C1, C2, G0)
        return G, log
    else:
        return G  # ,iter_num




def fused_partial_gromov_wasserstein_mass(
    C,
    C1,
    C2,
    p,
    q,
    omega2=1,
    mass=None,
    nb_dummies=1,
    G0=None,
    thres=1,
    symmetric=True,
    numItermax=None,
    numItermax_gw=1000,
    tol=1e-7,
    log=False,
    verbose=False,
    loss_fun='square_loss',
    line_search=True,
    amijo=False,
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """

    if mass is None:
        mass = np.min((np.sum(p), np.sum(q)))
    elif mass < 0:
        raise ValueError("Problem infeasible. Parameter mass should be greater" " than 0.")
    elif mass > np.min((np.sum(p), np.sum(q))):
        raise ValueError(
            "Problem infeasible. Parameter mass should lower or"
            " equal than min(|a|_1, |b|_1)."
        )
    #omega1=1-omega2
    if omega2>1 or omega2<0:
        raise ValueError("Problem infeasible. Parameter omega should be in [0,1].")
    
    C=C*(1-omega2)

    
    if G0 is None:
        G0 = np.outer(p, q) * mass / (np.sum(p) * np.sum(q))
    def cost(G):
        return np.sum(C * G) +omega2 * np.sum(M_circ_gamma(G)*G)
        
        
    n1,n2=p.shape[0],q.shape[0]
    if numItermax is None:
        numItermax=(n1+n2)*100
            
    #dim_G_extended = (n1 + nb_dummies, n2 + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - mass) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - mass) / nb_dummies] * nb_dummies)
    M_emd = np.zeros((n1+nb_dummies,n2+nb_dummies))
    M_circ_gamma,Mt_circ_gamma=def_tensor_product(C1,C2,Lambda=0,loss=loss_fun)
    cpt = 0
    err = 1

    if log:
        log = {"err": []}
    iter_num = 0
    f_val=cost(G0)
    while err > tol and cpt < numItermax_gw:
        iter_num += 1
        Gprev = np.copy(G0)
        if symmetric:
            grad = C+omega2*omega2*2*M_circ_gamma(G0)
        else:
            grad = C+omega2*(M_circ_gamma(G0)+Mt_circ_gamma(G0))
        M_emd[:n1, :n2] = grad
        M_emd[-nb_dummies:, -nb_dummies:] = np.max(grad) * 2


        Gc, logemd = emd_lp(
            p_extended,
            q_extended,
            M_emd,
            numItermax=numItermax,
            numThreads=thres,
            log=True,
            **kwargs
        )


        G0 = Gc[: len(p), : len(q)]
        
              

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0)))

        deltaG = G0 - Gprev
        if line_search:
            alpha,fc,f_val= do_linesearch(cost,Gprev,deltaG,grad,f_val,amijo=amijo,M_circ_gamma=M_circ_gamma)
        else:
            alpha = 1

        G0 = Gprev + alpha * deltaG
        cpt += 1

    if log:
        log["partial_gw_dist"] = gwloss_partial(C1, C2, G0)
        return G0[: len(p), : len(q)], log
    else:
        return G0[: len(p), : len(q)]  # ,iter_num
    



def fused_partial_gromov_lambda(
    C,
    C1,
    C2,
    p,
    q,
    Lambda,
    omega2=1,
    G0=None,
    nb_dummies=1,
    thres=1,
    numItermax_gw=1000,
    numItermax=None,
    tol=1e-7,
    log=False,
    verbose=False,
    line_search=True,
    amijo=False,
    seed=0,
    truncate=True,
    symmetric=True,
    loss_fun='square_loss',
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """

    # if m is None:
    #     m = np.min((np.sum(p), np.sum(q)))
    # elif m < 0:
    #     raise ValueError("Problem infeasible. Parameter m should be greater"
    #                      " than 0.")
    # elif m > np.min((np.sum(p), np.sum(q))):
    #     raise ValueError("Problem infeasible. Parameter m should lower or"
    #                      " equal than min(|a|_1, |b|_1).")

    if G0 is None:
        G0 = np.outer(p, q)*np.min((np.sum(p),np.sum(q)))/(np.sum(p)*np.sum(q))
    if omega2>1 or omega2<0:
        raise ValueError("Problem infeasible. Parameter omega should be in [0,1].")

    C=(1-omega2)*C

    #print('G0 sum is',np.sum(G0))

    cpt = 0
    err = 1

    if log:
        log_dict = {"err": [], "G0_mass": [], "Gprev_mass": []}

 
    C1, C2 = np.ascontiguousarray(C1), np.ascontiguousarray(C2)
    M_circ_gamma,Mt_circ_gamma=def_tensor_product(C1,C2,Lambda,loss=loss_fun)

    iter_num = 0
    n, m = C1.shape[0], C2.shape[0]
    if numItermax is None:
        numItermax = n * 100
    p_sum, q_sum = p.sum(), q.sum()
    G0_orig = np.zeros((n, m))

    mu_extended, nu_extended, M_extended = (
        np.zeros(n + 1),
        np.zeros(m + 1),
        np.full((n + 1, m + 1),-1e-15),
    )
    mu_extended[0:n], mu_extended[-1] = p, q_sum
    nu_extended[0:m], nu_extended[-1] = q, p_sum
    def cost(G):
        return np.sum(C * G) +omega2 * np.sum(M_circ_gamma(G)*G)
    f_val=cost(G0)
    while err > tol and cpt < numItermax_gw:
        # iter_num+=1
        Gprev = G0.copy()

        if symmetric:
            grad=C+2*omega2*M_circ_gamma(Gprev)
        else:
            grad=C+omega2*(M_circ_gamma(Gprev)+Mt_circ_gamma(Gprev))
        
        
        M_extended[0:n, 0:m] = grad

        gamma_extended, log_dict = emd_lp(
            mu_extended,
            nu_extended,
            M_extended,
            numItermax=numItermax,
            log=log,
            **kwargs
        )

        G0 = G0_orig.copy()
        G0[0:n, 0:m] = gamma_extended[:-1, :-1]
        # G0[np.ix_(idx_x, idx_y)] = gamma_extended[:-nb_dummies, :-nb_dummies]
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0)))

        # line search
        deltaG = G0 - Gprev

        # line search
        if line_search:            
            alpha, fc, f_val=do_linesearch(cost=cost,G=Gprev,deltaG=deltaG,Grad=grad,f_val=f_val,amijo=amijo,M_circ_gamma=M_circ_gamma)
            #Mt_circ_deltaG = M_circ_gamma(deltaG)
            #a = np.sum(Mt_circ_deltaG * deltaG)
            #b = np.sum(grad * deltaG)
            #alpha=solve_quadratic(a,b)
            #if alpha==0:
            #    cpt= numItermax_gw
        else:
            alpha = 1

        G0 = Gprev + alpha * deltaG
        cpt += 1
    if log:
        log_dict.update(innerlog_)
        return G0, log_dict  # ,iter_num
    else:
        return G0  # ,iter_num


def solve_gromov_linesearch(
    G,
    deltaG,
    cost_G,
    C1,
    C2,
    M,
    reg,
    alpha_min=None,
    alpha_max=None,
    symmetric=False,
    **kwargs,
):
    """
    Solve the linesearch in the FW iterations for any inner loss that decomposes as in Proposition 1 in :ref:`[12] <references-solve-linesearch>`.

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    C1 : array-like (ns,ns), optional
        Transformed Structure matrix in the source domain.
        For the 'square_loss' and 'kloss_fun', we provide hC1 from ot.gromov.init_matrix
    C2 : array-like (nt,nt), optional
        Transformed Structure matrix in the source domain.
        For the 'square_loss' and 'kloss_fun', we provide hC2 from ot.gromov.init_matrix
    M : array-like (ns,nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha
    np : backend, optional
        If let to its default value None, a backend test will be conducted.
    symmetric : bool, optional
        Either structures are to be assumed symmetric or not. Default value is False.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).

    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:

    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    dot = np.dot(np.dot(C1, deltaG), C2.T)
    a = -2*reg * np.sum(dot * deltaG)
    if symmetric:
        b = np.sum(M * deltaG) - 4 * reg * np.sum(dot * G)
    else:
        b = np.sum(M * deltaG) - reg * (
            np.sum(dot * G) + np.sum(np.dot(np.dot(C1, G), C2.T) * deltaG)
        )


    return a,b

def pgw_cost(C1,C2,gamma,loss_fun):
    """Compute the structural partial GW cost for a fixed transport plan."""
    M_circ_gamma,Mt_circ_gamma=def_tensor_product(C1,C2,loss=loss_fun,Lambda=0)
    M_circ_G=M_circ_gamma(gamma)
    cost=np.sum(M_circ_G*gamma)
    return cost
def pgw_penalty(p,q,gamma,Lambda):
    """Compute the unmatched-mass penalty for a partial GW transport plan."""
    return Lambda*(p.sum()**2+q.sum()**2-2*gamma.sum()**2)
    
def fused_pgw_cost(C,C1,C2,gamma,omega2,loss_fun='square_loss'):
    """Compute the SpaOT/FPGW objective value for a fixed transport plan."""
    C=(1-omega2)*C
    quadratic_cost=omega2*pgw_cost(C1,C2,gamma,loss_fun)
    linear_cost=np.sum(C*gamma)
    return linear_cost+quadratic_cost

def fused_pgw_cost_penalty(C,C1,C2,p,q,gamma,Lambda,omega2,loss_fun='square_loss'):
    """Compute the SpaOT/FPGW cost plus unmatched-mass penalty."""
    trans_cost=fused_pgw_cost(C,C1,C2,gamma,omega2,loss_fun='square_loss')
    penalty=Lambda*(p.sum()**2+q.sum()**2-2*gamma.sum()**2)
    return trans_cost,penalty
    


