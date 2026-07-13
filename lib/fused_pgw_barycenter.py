# FPGW / PGW / GW barycenter routines.
#
# Imported and adapted from the FPGW (Fused Partial Gromov-Wasserstein) repository:
#   https://github.com/yikun-baio/fused-pgw
# The algorithms in this file originate there; SpaOT is a downstream user.
# Some functions have been modified for SpaOT analyses.

import numpy as np
import numba as nb
import ot
from scipy import stats
from scipy.sparse import random
from sklearn import manifold
from sklearn.manifold import MDS

from ot.backend import get_backend
from ot.utils import check_random_state, dist, list_to_array, unif

from .fgw.bregman import sinkhorn_scaling
from .fused_pgw import fused_partial_gromov_wasserstein
from .gromov_utils import cost_matrix_d

try:
    from ot.gromov import gromov_wasserstein
except ImportError:
    gromov_wasserstein = None

try:
    from ot.partial import partial_gromov_wasserstein
except ImportError:
    partial_gromov_wasserstein = None

# Legacy partial-GW barycenter code used this helper name. Keep the alias so
# older callers fail only if the installed POT version lacks partial GW.
partial_gromov_ver1 = partial_gromov_wasserstein


def _require_solver(solver, name):
    """Raise a clear error when an optional POT solver is unavailable."""
    if solver is None:
        raise ImportError(
            f"{name} is required for this legacy barycenter routine but is not "
            "available in the installed POT version."
        )
    return solver


def ensure_2d(x):
    """
    Ensures that x is a 2D NumPy array.
    
    - If x has shape (n,), it converts it to (n,1).
    - If x has shape (n, d), it remains unchanged.
    
    Parameters:
    x (np.ndarray): Input array.
    
    Returns:
    np.ndarray: A 2D array with shape (n, d).
    """
    x = np.asarray(x)  # Ensure input is a NumPy array
    if x.ndim == 1:
        return x.reshape(-1, 1)  # Convert (n,) to (n,1)
    return x  # Keep (n,d) unchanged




def update_cross_feature_matrix(X,Y):

    """
    Updates M the distance matrix between the features
    calculated at each iteration    
    ----------
    X : ndarray, shape (N,d)
        First features matrix, N: number of samples, d: dimension of the features
    Y : ndarray, shape (M,d)
        Second features matrix, N: number of samples, d: dimension of the features
    Returns
    ----------
    M : ndarray, shape (N,M)
    
    """

    return ot.dist(ensure_2d(np.array(X)),ensure_2d(np.array(Y)))
    #return ot.dist(ensure_2d(np.array(X)),ensure_2d(np.array(Y)), metric='hamming')

def update_Ms(X, Ys):
    """Compute feature-cost matrices from barycenter features to each source."""
    l = [np.asarray(update_cross_feature_matrix(X, Ys[s]), dtype=np.float64) for s in range(len(Ys))]
    return l


def random_gamma_init(p, q, **kwargs):
    """Create a random feasible coupling by Sinkhorn-scaling a dense seed."""
    rvs = stats.beta(1e-1, 1e-1).rvs
    S = random(len(p), len(q), density=1, data_rvs=rvs, random_state=0)
    return sinkhorn_scaling(p, q, S.A, **kwargs)
    



# It is referenced from PythonOT (https://pythonot.github.io/)
def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

# It is adapted from PythonOT (https://pythonot.github.io/)
def smacof_mds(C, dim, max_iter=3000, eps=1e-9,seed=3):
    """
    It is imported from PythonOT
    Returns an interpolated point cloud following the dissimilarity matrix C
    using SMACOF multidimensional scaling (MDS) in specific dimensioned
    target space

    Parameters
    ----------
    C : ndarray, shape (ns, ns)
        dissimilarity matrix
    dim : int
          dimension of the targeted space
    max_iter :  int
        Maximum number of iterations of the SMACOF algorithm for a single run
    eps : float
        relative tolerance w.r.t stress to declare converge

    Returns
    -------
    npos : ndarray, shape (R, dim)
           Embedded coordinates of the interpolated point cloud (defined with
           one isometry)
    """

    rng = np.random.RandomState(seed=seed)

    mds = manifold.MDS(
        dim,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity='precomputed',
        n_init=1,
        normalized_stress='auto')
    pos = mds.fit(C).embedding_

    nmds = manifold.MDS(
        2,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity="precomputed",
        random_state=rng,
        normalized_stress='auto',
        n_init=1)
    npos = nmds.fit_transform(C, init=pos)


    return npos


def MDS_test(dist_mat, d, seed=0):
    """Embed a distance matrix into `d` dimensions with metric MDS."""
    mds = MDS(n_components=d, dissimilarity='precomputed', random_state=seed, normalized_stress='auto')
    lower_dimensional_points = mds.fit_transform(dist_mat)
    return lower_dimensional_points


def rotation_2d(theta=0):
    """Return a two-dimensional rotation matrix."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
    ])




# This function is referenced from PythonOT
def update_square_loss(p, lambdas, T, Cs, nx=None):
    r"""
    Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of the GW
    barycenter problem in :ref:`[12]`:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        nx = get_backend(p, *T, *Cs)
    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s], Cs[s]),
            T[s].T
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)

    return tmpsum / ppt


def update_square_loss_pgw( lambdas, T, Cs):
    r"""
    Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of the GW
    barycenter problem in :ref:`[12]`:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([
        lambdas[s] * np.dot(
            np.dot(T[s], Cs[s]),
            T[s].T
        ) for s in range(len(T))
    ])
    
    ppt =sum([
        lambdas[s] * np.outer(T[s].sum(1), T[s].sum(1))
        for s in range(len(T))
    ])
    
    return tmpsum / ppt



def pgw_barycenters(
        N, Cs, ps=None, p=None, Lambda_list=None,lambdas=None, loss_fun='square_loss', max_iter=1000, tol=1e-9, verbose=False,
        log=False, init_C=None, random_state=None,stop_criterion='barycenter', **kwargs):
    r"""
    Returns the Gromov-Wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.
        If let to its default value None, uniform distributions are taken.
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        fused gromov-wasserstein transport problems.s
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape(N,N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - values used in convergence evaluation.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if stop_criterion not in ['barycenter']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")

    Cs = list_to_array(*Cs)
    arr = [*Cs]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    nx = get_backend(*arr)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = cost_matrix_d(xalea,xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    cpt = 0
    err = 1e15  # either the error on 'barycenter' or 'loss'

        
    if Lambda_list is None:
        c_max=[np.max(C) for C in Cs]
        Lambda_list=np.array(c_max)
        

    if stop_criterion == 'barycenter':
        inner_log = False
    else:
        inner_log = True
        curr_loss = 1e15

    if log:
        log_ = {}
        log_['err'] = []
        if stop_criterion == 'loss':
            log_['loss'] = []

    while (err > tol and cpt < max_iter):
        if stop_criterion == 'barycenter':
            Cprev=C.copy()
        
                              

        # get transport plans
        partial_solver = _require_solver(partial_gromov_ver1, "ot.partial.partial_gromov_wasserstein")
        res = [partial_solver(
                C, Cs[s], p, ps[s], Lambda_list[s], G0=None, tol=1e-5, log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]
   
       
        T = res


        # update barycenters
        if loss_fun == 'square_loss':
            C = update_square_loss_nb_pgw(p, lambdas, T, Cs)

        # elif loss_fun == 'kl_loss':
        #     C = update_kl_loss(p, lambdas, T, Cs, nx)
        
        if stop_criterion == 'barycenter':
            err = nx.norm(C - Cprev)
        if log:
            log_['err'].append(err)


        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log_['T'] = Ts
        log_['p'] = p

        return C, log_
    else:
        return C
    

    
def mpgw_barycenters(
        N, Cs, ps=None, p=None, lambdas=None,mass_list=None, loss_fun='square_loss', max_iter=1000, tol=1e-9, verbose=False,
        log=False, init_C=None, stop_criterion = 'barycenter', random_state=None, **kwargs):
    r"""
    Returns the Gromov-Wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.
        If let to its default value None, uniform distributions are taken.
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        fused gromov-wasserstein transport problems.s
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape(N,N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - values used in convergence evaluation.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if stop_criterion not in ['barycenter']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter'}.")

    Cs = list_to_array(*Cs)
    arr = [*Cs]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    nx = get_backend(*arr)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = cost_matrix_d(xalea,xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    cpt = 0
    err = 1e15  # either the error on 'barycenter' or 'loss'

    # if warmstartT:
    #     T = [None] * S
        
    if mass_list is None:
        mass_list = np.zeros(S)
        for id_K in range(S):
            mass_list[id_K] = min(p.sum(), ps[id_K].sum())
            

    if stop_criterion == 'barycenter':
        inner_log = False
    else:
        inner_log = True
        curr_loss = 1e15

    if log:
        log_ = {}
        log_['err'] = []
        if stop_criterion == 'loss':
            log_['loss'] = []

    while (err > tol and cpt < max_iter):
        #print('cpt is',cpt)
        Cprev=C.copy()
        # get transport plans
        partial_solver = _require_solver(partial_gromov_wasserstein, "ot.partial.partial_gromov_wasserstein")
        res = [partial_solver(
                C, Cs[s], p, ps[s], mass_list[s], G0=None,
                tol=1e-5, log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]
        
       
        Ts = res
        # else:
        #     T = [output[0] for output in res]
        #     curr_loss = np.sum([output[1]['gw_dist'] for output in res])

        # update barycenters
        if loss_fun == 'square_loss':
            C = update_square_loss_nb_pgw(lambdas, Ts, Cs)

        # elif loss_fun == 'kl_loss':
        #     C = update_kl_loss(p, lambdas, T, Cs, nx)
        if stop_criterion == 'barycenter':
            err = nx.norm(C - Cprev)
            if log:
                log_['err'].append(err)


        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log_['T'] = Ts
        log_['p'] = p

        return C, log_
    else:
        return C



def gw_barycenters(
        N, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
        symmetric=True,max_iter=400, numItermax=None, tol=1e-7,
        stop_criterion='barycenter', warmstartT=False, verbose=False,
        log=False, init_C=None, random_state=None, **kwargs):
    r"""
    Returns the Gromov-Wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.
        If let to its default value None, uniform distributions are taken.
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        fused gromov-wasserstein transport problems.s
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape(N,N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - values used in convergence evaluation.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if stop_criterion not in ['barycenter', 'loss']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")

    #Cs = list_to_array(*Cs)
    arr = [*Cs]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    nx = get_backend(*arr)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    cpt = 0
    err = 1e15  # either the error on 'barycenter' or 'loss'

    if warmstartT:
        T = [None] * S

    if stop_criterion == 'barycenter':
        inner_log = False
    else:
        inner_log = True
        curr_loss = 1e15

    if log:
        log_ = {}
        log_['err'] = []
        if stop_criterion == 'loss':
            log_['loss'] = []
    if numItermax is None:
        numItermax=100*N
    while (err > tol and cpt < max_iter):
        print('loop in barycenter is %i/%i'%(cpt,max_iter), end='\r')
        if stop_criterion == 'barycenter':
            Cprev = C
        else:
            prev_loss = curr_loss
        
        if warmstartT:
            gw_solver = _require_solver(gromov_wasserstein, "ot.gromov.gromov_wasserstein")
            res = [gw_solver(
                C, Cs[s], p, ps[s], G0=T[s],
                numItermax=numItermax, tol=1e-7, log=inner_log, verbose=verbose)
                for s in range(S)]
        else:
            gw_solver = _require_solver(gromov_wasserstein, "ot.gromov.gromov_wasserstein")
            res = [gw_solver(
                C, Cs[s], p, ps[s], G0=None,
                numItermax=numItermax, tol=1e-7, log=inner_log, verbose=verbose)
                for s in range(S)]

        T = res

        # update barycenters
        if loss_fun == 'square_loss':
            C = update_square_loss_nb_pgw(p, lambdas, T, Cs)        

        # update convergence criterion
        #if stop_criterion == 'barycenter':
        err = nx.norm(C - Cprev)
        if log:
            log_['err'].append(err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log_['T'] = Ts
        log_['p'] = p

        return C, log_
    else:
        return C



@nb.njit(cache=True)
def update_square_loss_nb_pgw(lambdas, Ts, Cs):
    r"""
    Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of the GW
    barycenter problem in :ref:`[12]`:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    Ts : list of S array-like of shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    # Correct order mistake in Equation 14 in [12]
    N,n=len(Ts),Ts[0].shape[0]
    tmpsum,ppt=np.zeros((n,n)),np.zeros((n,n))
    for s in range(N):
        tmpsum+=lambdas[s] * np.dot(np.dot(Ts[s], Cs[s]),Ts[s].T)
        T1=Ts[s].sum(1)
        ppt+=lambdas[s] * np.outer(T1, T1)

    # Safely compute the result with element-wise handling of division by zero
    #result = np.zeros_like(tmpsum)
    result = np.where(ppt != 0, tmpsum / ppt, 0.0)
    
    # non_zero_mask = ppt != 0  # Identify non-zero entries in ppt
    # result[non_zero_mask] = tmpsum[non_zero_mask] / ppt[non_zero_mask]
    
    return result
    
@nb.njit(cache=True)
def barycentric_projection(X0, X1, gamma, threshold=1e-10):
    """Project source features onto barycenter nodes with a transport plan."""
    domain=np.sum(gamma,1)>threshold
    p1_hat=np.sum(gamma,1) # martial of plan  
    X1_hat=X0.copy() 
    X1_hat[domain]=gamma.dot(X1)[domain]/np.expand_dims(p1_hat,1)[domain]
    return X1_hat
    
@nb.njit(cache=True)
def update_X(Xhat_list, p_list, beta_list, threshold=1e-10):
    """Average projected feature matrices using source weights and masses."""
    K=len(Xhat_list)
    X=Xhat_list[0]
    n,d=X.shape
    num=np.zeros((n,d))
    denum=np.zeros(n)
    for i in range(K):
        X,beta,p=Xhat_list[i],beta_list[i],p_list[i]
        num+=beta*X * p[:, np.newaxis]
        denum+=beta*p
    denum1=denum[:, np.newaxis]
    # Create a mask where p is non-zero
    non_zero_mask = denum >=1e-12
    result= np.zeros_like(num)
    result[non_zero_mask]=num[non_zero_mask]/denum1[non_zero_mask]
    return result

@nb.njit(cache=True)
def update_feature_matrix_pot(lambdas, Xs, Ts, X0):
    """Update continuous barycenter features from partial transport plans."""
    Xhat_list=[barycentric_projection(X0,X1,gamma,threshold=1e-10) for (X1,gamma) in zip(Xs,Ts)]
    p_list=[gamma.sum(1) for gamma in Ts]
    X0=update_X(Xhat_list,p_list,lambdas)
    return X0

###############discrete version
def barycentric_projection_discrete(X0, X1, gamma, threshold=1e-10):
    """
    Barycentric projection for discrete labels.
    X0: (n0,) array of initial barycenter labels
    X1: (n1,) array of source labels (discrete integers)
    gamma: (n0, n1) transport plan
    """
    domain = np.sum(gamma, 1) > threshold
    X0_new = X0.copy()
    
    for i in np.where(domain)[0]:
        weights = gamma[i]
        labels = X1
        # Compute the weighted vote for each label
        unique_labels = np.unique(labels)
        vote = np.array([weights[labels == l].sum() for l in unique_labels])
        X0_new[i] = unique_labels[np.argmax(vote)]
        
    return X0_new


def update_feature_matrix_pot_discrete(lambdas, Xs, Ts, X0):
    """
    Update barycenter discrete labels using multiple source label sets and transport plans.

    lambdas: list or array of weights for each source
    Xs: list of source label arrays (discrete integers)
    Ts: list of transport matrices corresponding to X0 -> Xs[i]
    X0: initial barycenter label array
    """
    n_sources = len(Xs)
    assert n_sources == len(Ts) == len(lambdas)

    # Weighted projection of each source
    Xhat_list = [barycentric_projection_discrete(X0, Xs[i], Ts[i]) for i in range(n_sources)]
    
    # Combine projections using lambda weights (weighted voting across sources)
    X0_new = X0.copy()
    n_nodes = len(X0)
    
    for i in range(n_nodes):
        votes = {}
        for j in range(n_sources):
            label = Xhat_list[j][i]
            votes[label] = votes.get(label, 0) + lambdas[j]
        # Choose label with maximum weighted vote
        X0_new[i] = max(votes, key=votes.get)
    
    return X0_new




def fmpgw_barycenters(N,Ys,Cs,ps,lambdas,alpha,mass_list,fixed_structure=False,fixed_features=False,p=None,loss_fun='square_loss',
                    max_iter=100, tol=1e-9,verbose=False,log=True,init_C=None,init_X=None):
 
    """
    Compute the fgw barycenter as presented eq (5) in [3].
    ----------
    N : integer 
        Desired number of samples of the target barycenter
    Ys: list of ndarray, each element has shape (ns,d)
        Features of all samples
    Cs : list of ndarray, each element has shape (ns,ns)
         Structure matrices of all samples
    ps : list of ndarray, each element has shape (ns,)
        masses of all samples
    lambdas : list of float
              list of the S spaces' weights
    alpha : float
            Alpha parameter for the fgw distance
    fixed_structure :  bool
                       Wether to fix the structure of the barycenter during the updates
    fixed_features :  bool
                       Wether to fix the feature of the barycenter during the updates
    init_C :  ndarray, shape (N,N), optional 
              initialization for the barycenters' structure matrix. If not set random init
    init_X :  ndarray, shape (N,d), optional 
              initialization for the barycenters' features. If not set random init
    Returns
    ----------
    X : ndarray, shape (N,d)
        Barycenters' features
    C : ndarray, shape (N,N)
        Barycenters' structure matrix
    log_:
        T : list of (N,ns) transport matrices
        Ms : all distance matrices between the feature of the barycenter and the other features dist(X,Ys) shape (N,ns)
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    np.random.seed(2)
    S = len(Cs)
    d= ensure_2d(Ys[0]).shape[1] #dimension on the node features
    if p is None:
        p=np.ones(N)/N
    
    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    Ys = [np.asarray(Ys[s], dtype=np.float64) for s in range(S)]
    
    
    lambdas = np.asarray(lambdas, dtype=np.float64)

    if fixed_structure:
        if init_C is None:
            C=Cs[0]
        else:
            C=init_C
    else:
        if init_C is None:
            xalea = np.random.randn(N, 2)
            C = dist(xalea, xalea)
            C /= C.max()
        else:
            C = init_C
    C=np.asarray(C)

    if fixed_features:
        if init_X is None:
            X=Ys[0]
        else :
            X= init_X
    else:
        if init_X is None: 
            X=np.zeros((N,d))
        else:
            X = init_X
    X=np.asarray(X)
    if X.ndim == 1: 
        X=ensure_2d(X)
    if Ys[0].ndim==1:
        Ys2=[ensure_2d(Y) for Y in Ys]
        Ys=Ys2
    # all features much be n*d shape
    

    #T=[np.outer(p,q) for q in ps]
    Ts=[np.outer(p,q) for q in ps]
    #[mass*random_gamma_init(p,q)/(p.sum()*q.sum()) for (q,mass) in zip(ps,mass_list)]
    
   


    # X is N,d
    # Ys is ns,d
    Ms = update_Ms(X,Ys)
    # Ms is N,ns

    cpt = 0
    err_feature = 1
    err_structure = 1

    if log:
        log_={}
        log_['err_feature']=[]
        log_['err_structure']=[]
        log_['Ts_iter']=[]

    while((err_feature > tol or err_structure > tol) and cpt < max_iter):
        Cprev = C
        Xprev = X

        if not fixed_features:
            #Ys_temp=[ensure_2d(y).T for y in Ys] 
             #Ys1=[Y.T for Y in Ys]
             #Ts1=[T.T for T in Ts]
             #X=update_feature_matrix(lambdas,Ys1,Ts,p)
             #X=X.T
    
             X=update_feature_matrix_pot(lambdas,Ys,Ts,X) 
             #print('error in feature update', np.linalg.norm(X1-X))
    
             #X=update_feature_matrix_pot(lambdas,Ys,Ts,Xprev) 

        # X must be N,d
        # Ys must be ns,d
        Ms=update_Ms(X,Ys)

        if not fixed_structure:
            if loss_fun == 'square_loss':      
                C = update_square_loss_nb_pgw(lambdas, Ts, Cs)
        
        Ts = [fused_partial_gromov_wasserstein(Ms[s],C,Cs[s],p,ps[s],alpha,None,numItermax=None, tol=tol, symmetric=True,  loss_fun=loss_fun, verbose=verbose, Lambda = 1, numItermax_gw = 51) for s in range(S)]

        
        
            # T is N,ns
        
        err_feature = np.linalg.norm(X - Xprev)
        err_structure = np.linalg.norm(C - Cprev)

        if log:
            log_['Ts_iter'].append(Ts)
            log_['err_feature'].append(err_feature)
            log_['err_structure'].append(err_structure)
            
            print('iter: ',cpt)
            print('err_feature ',err_feature)
            print('err_structure', err_structure)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err_structure))
            print('{:5d}|{:8e}|'.format(cpt, err_feature))

        cpt += 1
    # log_['T']=T # ce sont les matrices du barycentre de la target vers les Ys
    # log_['p']=p
    # log_['Ms']=Ms #Ms sont de tailles N,ns

    return X,C #,log_




def fugw_barycenters(N,Ys,Cs,ps,lambdas,alpha,mass_list,fixed_structure=False,fixed_features=False,p=None,loss_fun='square_loss',
                    max_iter=100, tol=1e-9,verbose=False,log=True,init_C=None,init_X=None):
 
    """
    Compute the fgw barycenter as presented eq (5) in [3].
    ----------
    N : integer 
        Desired number of samples of the target barycenter
    Ys: list of ndarray, each element has shape (ns,d)
        Features of all samples
    Cs : list of ndarray, each element has shape (ns,ns)
         Structure matrices of all samples
    ps : list of ndarray, each element has shape (ns,)
        masses of all samples
    lambdas : list of float
              list of the S spaces' weights
    alpha : float
            Alpha parameter for the fgw distance
    fixed_structure :  bool
                       Wether to fix the structure of the barycenter during the updates
    fixed_features :  bool
                       Wether to fix the feature of the barycenter during the updates
    init_C :  ndarray, shape (N,N), optional 
              initialization for the barycenters' structure matrix. If not set random init
    init_X :  ndarray, shape (N,d), optional 
              initialization for the barycenters' features. If not set random init
    Returns
    ----------
    X : ndarray, shape (N,d)
        Barycenters' features
    C : ndarray, shape (N,N)
        Barycenters' structure matrix
    log_:
        T : list of (N,ns) transport matrices
        Ms : all distance matrices between the feature of the barycenter and the other features dist(X,Ys) shape (N,ns)
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    np.random.seed(2)
    S = len(Cs)
    d= ensure_2d(Ys[0]).shape[1] #dimension on the node features
    if p is None:
        p=np.ones(N)/N
    
    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    Ys = [np.asarray(Ys[s], dtype=np.float64) for s in range(S)]
    
    
    lambdas = np.asarray(lambdas, dtype=np.float64)

    if fixed_structure:
        if init_C is None:
            C=Cs[0]
        else:
            C=init_C
    else:
        if init_C is None:
            xalea = np.random.randn(N, 2)
            C = dist(xalea, xalea)
            C /= C.max()
        else:
            C = init_C
    C=np.asarray(C)

    if fixed_features:
        if init_X is None:
            X=Ys[0]
        else :
            X= init_X
    else:
        if init_X is None: 
            X=np.zeros((N,d))
        else:
            X = init_X
    X=np.asarray(X)
    if X.ndim == 1: 
        X=ensure_2d(X)
    if Ys[0].ndim==1:
        Ys2=[ensure_2d(Y) for Y in Ys]
        Ys=Ys2
    # all features much be n*d shape
    

    #T=[np.outer(p,q) for q in ps]
    Ts=[np.outer(p,q) for q in ps]
    
    
    Ms = update_Ms(X,Ys)    
    Ts = [ot.gromov.fused_unbalanced_gromov_wasserstein(C,Cs[s],p,ps[s],M=Ms[s],alpha=0.03, epsilon = 0.01)[0] for s in range(len(Cs))]
  
    #[mass*random_gamma_init(p,q)/(p.sum()*q.sum()) for (q,mass) in zip(ps,mass_list)]
    

    # X is N,d
    # Ys is ns,d
    #Ms = update_Ms(X,Ys)
    # Ms is N,ns

    cpt = 0
    err_feature = 1
    err_structure = 1

    if log:
        log_={}
        log_['err_feature']=[]
        log_['err_structure']=[]
        log_['Ts_iter']=[]

    while((err_feature > tol or err_structure > tol) and cpt < max_iter):
        Cprev = C
        Xprev = X

        if not fixed_features:
            #Ys_temp=[ensure_2d(y).T for y in Ys] 
             #Ys1=[Y.T for Y in Ys]
             #Ts1=[T.T for T in Ts]
             #X=update_feature_matrix(lambdas,Ys1,Ts,p)
             #X=X.T
    
             X=update_feature_matrix_pot(lambdas,Ys,Ts,X) 
             #print('error in feature update', np.linalg.norm(X1-X))
    
             #X=update_feature_matrix_pot(lambdas,Ys,Ts,Xprev) 

        # X must be N,d
        # Ys must be ns,d
        Ms=update_Ms(X,Ys)

        if not fixed_structure:
            if loss_fun == 'square_loss':      
                C = update_square_loss_nb_pgw(lambdas, Ts, Cs)
        
        #Ts = [fused_partial_gromov_wasserstein(Ms[s],C,Cs[s],p,ps[s],alpha,None,numItermax=None, tol=tol, symmetric=True,  loss_fun=loss_fun, verbose=verbose) for s in range(S)]
        Ts = [ot.gromov.fused_unbalanced_gromov_wasserstein(C,Cs[s],p,ps[s],M=Ms[s],alpha=0.03, epsilon = 0.01)[0] for s in range(S)]
               
        
            # T is N,ns
        
        err_feature = np.linalg.norm(X - Xprev)
        err_structure = np.linalg.norm(C - Cprev)

        if log:
            log_['Ts_iter'].append(Ts)
            log_['err_feature'].append(err_feature)
            log_['err_structure'].append(err_structure)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err_structure))
            print('{:5d}|{:8e}|'.format(cpt, err_feature))

        cpt += 1
    # log_['T']=T # ce sont les matrices du barycentre de la target vers les Ys
    # log_['p']=p
    # log_['Ms']=Ms #Ms sont de tailles N,ns

    return X,C #,log_




