# Gromov-Wasserstein tensor-product and loss utilities.
#
# Imported and adapted from the FPGW (Fused Partial Gromov-Wasserstein) repository:
#   https://github.com/yikun-baio/fused-pgw
# The algorithms in this file originate there; SpaOT is a downstream user.
# Some functions have been modified for SpaOT analyses.

import numpy as np
import numba as nb
from ot.partial import gwloss_partial


@nb.njit(cache=True)
def init_plan(p, q, mass=None, Type="pot"):
    """Initialize a feasible transport plan for OT, partial OT, or partial GW.

    Parameters
    ----------
    p, q : ndarray
        Source and target masses.
    mass : float or None
        Amount of mass to transport. If omitted for partial problems, the
        maximum feasible common mass is used.
    Type : {"pot", "mpot", "ot"}
        Problem type. `pot`/`mpot` use partial mass scaling; `ot` creates a
        balanced-style outer-product initialization.
    """
    if Type in ["mpot", "pot"] and mass is None:
        G0 = np.outer(p, q) * min((np.sum(p), np.sum(q))) / (np.sum(p) * np.sum(q))
    elif Type in ["mpot", "pot"] and mass is not None:
        G0 = np.outer(p, q) * mass / (np.sum(p) * np.sum(q))
    elif Type in ["ot"]:
        G0 = np.outer(p, q) / np.sqrt(np.sum(p) * np.sum(q))
    else:
        raise ValueError("Type should be either 'mpot', 'pot' or 'ot'.")
    return G0


@nb.njit(cache=True)
def tensor_dot_param(C1, C2, Lambda=0, loss="square_loss"):
    """Precompute transformed structure matrices for the GW tensor product.

    For square loss, the structural term can be written as
    `f(C1) * Gamma * 1 + 1 * Gamma * f(C2)^T - h(C1) * Gamma * h(C2)^T`.
    This helper returns the `f` and `h` transforms used by `tensor_dot_func`.
    `Lambda` shifts the source-side square term for penalized partial GW.
    """
    if loss == "square_loss":

        def f1(r1):
            return r1**2 - 2 * Lambda

        def f2(r2):
            return r2**2

        def h1(r1):
            return r1

        def h2(r2):
            return 2 * r2
    else:
        raise ValueError("Only square_loss is supported.")

    fC1 = f1(C1)
    fC2 = f2(C2)
    hC1 = h1(C1)
    hC2 = h2(C2)

    return fC1, fC2, hC1, hC2


@nb.njit(cache=True)
def tensor_dot_func(fC1, fC2, hC1, hC2, Gamma):
    """Evaluate the GW tensor product for a fixed transport plan."""
    C1 = fC1.dot(Gamma.sum(1).reshape((-1, 1)))
    C2 = Gamma.sum(0).dot(fC2.T)
    tensor_dot = (C1 + C2) - hC1.dot(Gamma).dot(hC2.T)
    return tensor_dot


def def_tensor_product(C1, C2, Lambda=0, loss="square_loss"):
    """Build callable tensor-product operators for a pair of structures.

    Returns two functions: one for `(C1, C2)` and one for the transposed
    structures. The Frank-Wolfe solver uses both when source/target structures
    may be asymmetric.
    """
    if loss == "square_loss":
        fC1, fC2, hC1, hC2 = tensor_dot_param(C1, C2, Lambda, loss="square_loss")
        fC1, fC2, hC1, hC2 = (
            np.ascontiguousarray(fC1),
            np.ascontiguousarray(fC2),
            np.ascontiguousarray(hC1),
            np.ascontiguousarray(hC2),
        )
        fC1t, fC2t, hC1t, hC2t = tensor_dot_param(C1.T, C2.T, Lambda, loss="square_loss")
        fC1t, fC2t, hC1t, hC2t = (
            np.ascontiguousarray(fC1t),
            np.ascontiguousarray(fC2t),
            np.ascontiguousarray(hC1t),
            np.ascontiguousarray(hC2t),
        )

        def M_circ_gamma(gamma):
            return tensor_dot_func(fC1, fC2, hC1, hC2, gamma)

        def Mt_circ_gamma(gamma):
            return tensor_dot_func(fC1t, fC2t, hC1t, hC2t, gamma)

    elif loss == "sub_graph_loss":

        def M_circ_gamma(gamma):
            return tensor_dot_edge(C1, C2, gamma)

        def Mt_circ_gamma(gamma):
            return tensor_dot_edge(C1.T, C2.T, gamma)
    else:
        raise ValueError("loss should be either 'square_loss' or 'sub_graph_loss'.")

    return M_circ_gamma, Mt_circ_gamma


@nb.njit(cache=True)
def gwgrad_partial_numba(C1, C2, T, loss="square"):
    """Compute the GW gradient for a possibly partial transport plan.

    This is adapted from PythonOT's partial GW utilities. The usual normalized
    GW shortcut is avoided because partial plans may transport less than unit
    mass.
    """
    if loss == "square":
        cC1 = np.dot(C1**2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
        cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2**2)
        constC = cC1 + cC2
        A = -2 * np.dot(C1, T).dot(C2.T)
        tens = constC + A
    elif loss == "dot":
        A = -2 * np.dot(C1, T).dot(C2.T)
        tens = A
    else:
        raise ValueError("loss should be either 'square' or 'dot'.")
    return tens


@nb.njit(cache=True)
def gwloss_partial_numba(C1, C2, T):
    """Compute the partial GW structural loss for a fixed plan."""
    g = gwgrad_partial_numba(C1, C2, T) * 0.5
    return np.sum(g * T)


@nb.njit()
def cost_matrix_d(X, Y, loss="square"):
    """Compute a pairwise feature-cost matrix.

    Parameters
    ----------
    X, Y : ndarray, shape (n, d) and (m, d)
        Feature matrices for source and target nodes.
    loss : {"square", "dot"}
        `square` returns squared Euclidean distances. `dot` returns the
        bilinear term `-2 * <x, y>` used inside square-loss expansions.
    """
    X1 = np.expand_dims(X, 1)
    Y1 = np.expand_dims(Y, 0)
    if loss == "square":
        M = np.sum((X1 - Y1) ** 2, 2)
    elif loss == "dot":
        M = np.sum(-2 * (X1 * Y1), 2)
    else:
        raise ValueError("loss should be either 'square' or 'dot'.")
    return M


@nb.njit()
def tensor_dot_ori(M, Gamma):
    """Evaluate a dense four-dimensional GW tensor against a transport plan.

    This direct reference implementation is useful for debugging but is much
    slower and more memory-intensive than `tensor_dot_func`.
    """
    n, m = Gamma.shape
    gradient = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for i1 in range(n):
                for j1 in range(m):
                    gradient[i, j] += M[i, j, i1, j1] * Gamma[i1, j1]
    return gradient


@nb.njit(cache=True)
def construct_M(C1, C2):
    """Construct the full four-dimensional square-loss GW tensor.

    `M[i, j, i1, j1]` stores `(C1[i, i1] - C2[j, j1]) ** 2`. Prefer the
    factorized tensor helpers for real analyses.
    """
    n, m = C1.shape[0], C2.shape[0]
    M = np.zeros((n, m, n, m))
    for i in range(n):
        for j in range(m):
            for i1 in range(n):
                for j1 in range(m):
                    M[i, j, i1, j1] = (C1[i, i1] - C2[j, j1]) ** 2
    return M


@nb.njit(cache=True)
def tensor_dot_edge(C1, C2, gamma):
    """Tensor product for sub-graph loss based on negative edge differences."""
    n, m = C1.shape[0], C2.shape[0]
    dot = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M_ij = np.zeros((n, m))
            for i1 in range(n):
                for j1 in range(m):
                    M_ij[i1, j1] = min(C1[i, i1] - C2[j, j1], 0)
            dot[i, j] = np.sum(M_ij * gamma)
    return dot


def GW_dist(C1, C2, gamma):
    """Compute the GW structural cost for a fixed transport plan."""
    M_gamma = gwgrad_partial_numba(C1, C2, gamma)
    dist = np.sum(M_gamma * gamma)
    return dist


def MPGW_dist(C1, C2, gamma):
    """Compute the mass-partial GW structural cost for a fixed plan."""
    M_gamma = gwgrad_partial_numba(C1, C2, gamma)
    dist = np.sum(M_gamma * gamma)
    return dist


def PGW_dist_with_penalty(C1, C2, gamma, p1, p2, Lambda):
    """Compute partial GW structural cost and unmatched-mass penalty."""
    M_gamma = gwgrad_partial_numba(C1, C2, gamma)
    dist = np.sum(M_gamma * gamma)
    penalty = Lambda * (p1.sum() ** 2 + p2.sum() ** 2 - 2 * gamma.sum() ** 2)
    return dist, penalty
