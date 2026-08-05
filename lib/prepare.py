"""Utilities for preparing AnnData inputs for SpaOT.

This module converts two AnnData objects into the matrices required by
``fused_partial_gromov_wasserstein`` without depending on moscot. By default it
mirrors ``moscot.problems.space.AlignmentProblem.prepare`` defaults for spatial
alignment: local PCA for the linear term, scalar-normalized spatial coordinates
for the quadratic terms, and uniform probability marginals.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist


MassMode = Literal["probability", "balanced-min", "count"]


@dataclass
class SpaOTProblem:
    """Prepared SpaOT inputs.

    Attributes
    ----------
    C
        Cross-domain feature cost matrix with shape ``(n_source, n_target)``.
    C1
        Within-source structure matrix with shape ``(n_source, n_source)``.
    C2
        Within-target structure matrix with shape ``(n_target, n_target)``.
    p
        Source mass vector with shape ``(n_source,)``.
    q
        Target mass vector with shape ``(n_target,)``.
    source_features, target_features
        Feature matrices used to build ``C``. With default settings, these are
        moscot-compatible local PCA coordinates.
    source_spatial, target_spatial
        Spatial coordinate matrices used to build ``C1`` and ``C2``. With
        default settings, these are moscot-compatible normalized coordinates.
    """

    C: np.ndarray
    C1: np.ndarray
    C2: np.ndarray
    p: np.ndarray
    q: np.ndarray
    source_features: np.ndarray
    target_features: np.ndarray
    source_spatial: np.ndarray
    target_spatial: np.ndarray


def _as_dense_array(x, name: str) -> np.ndarray:
    """Return ``x`` as a dense floating-point 2D NumPy array."""
    if sparse.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _get_feature_matrix(adata, layer: Optional[str], obsm_key: Optional[str]) -> np.ndarray:
    if layer is not None and obsm_key is not None:
        raise ValueError("Specify at most one of `layer` and `obsm_key`.")
    if obsm_key is not None:
        if obsm_key not in adata.obsm:
            raise KeyError(f"AnnData object is missing .obsm[{obsm_key!r}].")
        return _as_dense_array(adata.obsm[obsm_key], f"adata.obsm[{obsm_key!r}]")
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"AnnData object is missing .layers[{layer!r}].")
        return _as_dense_array(adata.layers[layer], f"adata.layers[{layer!r}]")
    return _as_dense_array(adata.X, "adata.X")


def _get_spatial_matrix(adata, spatial_key: str) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"AnnData object is missing .obsm[{spatial_key!r}].")
    return _as_dense_array(adata.obsm[spatial_key], f"adata.obsm[{spatial_key!r}]")


def _concat_matrices(x, y):
    """Concatenate dense or sparse matrices like moscot's local PCA callback."""
    if sparse.issparse(x):
        return sparse.vstack([x, sparse.csr_matrix(y)])
    if sparse.issparse(y):
        return sparse.vstack([sparse.csr_matrix(x), y])
    return np.vstack([x, y])


def _local_pca_features(source, target, layer: Optional[str], n_comps: int, scale: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce moscot's ``local-pca`` callback for the ``xy`` term."""
    if layer is None:
        x = source.X
        y = target.X
    else:
        if layer not in source.layers:
            raise KeyError(f"Source AnnData object is missing .layers[{layer!r}].")
        if layer not in target.layers:
            raise KeyError(f"Target AnnData object is missing .layers[{layer!r}].")
        x = source.layers[layer]
        y = target.layers[layer]

    n_source = x.shape[0]
    data = _concat_matrices(x, y)
    if data.shape[1] <= n_comps:
        data = _as_dense_array(data, "local-pca input")
    else:
        try:
            import scanpy as sc
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("scanpy is required to reproduce moscot's local-pca preprocessing.") from exc
        data = sc.pp.pca(data, n_comps=n_comps)

    data = _as_dense_array(data, "local-pca output")
    if scale:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1
        data = (data - mean) / std
    return data[:n_source], data[n_source:]


def _spatial_norm(spatial: np.ndarray) -> np.ndarray:
    """Reproduce moscot's ``spatial-norm`` callback."""
    spatial = np.asarray(spatial, dtype=float)
    std = spatial.std()
    if std == 0:
        raise ValueError("Cannot normalize spatial coordinates with zero standard deviation.")
    return (spatial - spatial.mean()) / std


def _normalize_matrix(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    min_value = np.min(M)
    max_value = np.max(M)
    if np.isclose(max_value, min_value):
        return np.zeros_like(M, dtype=float)
    return (M - min_value) / (max_value - min_value)


def _make_masses(n_source: int, n_target: int, mode: MassMode) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "probability":
        return np.ones(n_source) / n_source, np.ones(n_target) / n_target
    if mode == "balanced-min":
        scale = min(n_source, n_target)
        return np.ones(n_source) / scale, np.ones(n_target) / scale
    if mode == "count":
        return np.ones(n_source), np.ones(n_target)
    raise ValueError("`mass_mode` must be one of 'probability', 'balanced-min', or 'count'.")


def _intersect_vars(source, target):
    common = source.var_names.intersection(target.var_names)
    if len(common) == 0:
        raise ValueError("Source and target AnnData objects do not share any variables.")
    return source[:, common].copy(), target[:, common].copy()


def prepare_spaot_problem(
    source,
    target,
    *,
    spatial_key: str = "spatial",
    layer: Optional[str] = None,
    obsm_key: Optional[str] = None,
    use_moscot_defaults: bool = True,
    n_comps: int = 30,
    pca_scale: bool = False,
    normalize_spatial: bool = True,
    feature_metric: str = "euclidean",
    structure_metric: str = "euclidean",
    normalize_feature_cost: bool = True,
    normalize_structure_cost: bool = False,
    intersect_vars: bool = True,
    mass_mode: MassMode = "probability",
) -> SpaOTProblem:
    """Prepare feature and structure costs from two AnnData objects.

    Default behavior reproduces the moscot preparation used by
    ``AlignmentProblem.prepare(batch_key=..., policy="sequential")`` for a single
    pair of already selected source and target AnnData objects:

    - ``joint_attr=None`` -> ``local-pca`` on concatenated ``source.X`` and
      ``target.X`` with ``n_comps=30`` and ``scale=False``.
    - ``normalize_spatial=True`` -> scalar standardization of each section's
      spatial coordinates: ``(spatial - spatial.mean()) / spatial.std()``.
    - ``a=None`` and ``b=None`` -> uniform probability marginals.

    Parameters
    ----------
    source, target
        AnnData-like objects. The returned transport plan orientation is
        source-by-target.
    spatial_key
        Key in ``.obsm`` containing spatial coordinates.
    layer
        Optional layer to use for local PCA or raw features.
    obsm_key
        Optional ``.obsm`` representation to use as the feature matrix. Setting
        this bypasses moscot's default local PCA behavior.
    use_moscot_defaults
        If true, reproduce moscot's default callbacks. Set false to use raw
        ``.X``/``layer`` features and raw spatial coordinates.
    n_comps
        Number of principal components used by moscot's ``local-pca`` callback.
    pca_scale
        Whether to standardize PCA coordinates after PCA. Moscot defaults to
        false.
    normalize_spatial
        Whether to reproduce moscot's scalar spatial normalization.
    feature_metric, structure_metric
        Metrics passed to ``scipy.spatial.distance.cdist``. The CRC scripts use
        Euclidean distances on the prepared moscot arrays.
    normalize_feature_cost
        If true, scale ``C`` to ``[0, 1]``, matching the CRC script.
    normalize_structure_cost
        If true, scale ``C1`` and ``C2`` separately to ``[0, 1]``.
    intersect_vars
        If true, subset source and target to shared variables before PCA/raw
        feature extraction.
    mass_mode
        Mass convention for ``p`` and ``q``. ``"probability"`` matches moscot's
        default marginals.
    """
    if intersect_vars and obsm_key is None:
        source, target = _intersect_vars(source, target)

    if use_moscot_defaults and obsm_key is None:
        source_features, target_features = _local_pca_features(
            source,
            target,
            layer=layer,
            n_comps=n_comps,
            scale=pca_scale,
        )
    else:
        source_features = _get_feature_matrix(source, layer=layer, obsm_key=obsm_key)
        target_features = _get_feature_matrix(target, layer=layer, obsm_key=obsm_key)

    source_spatial = _get_spatial_matrix(source, spatial_key=spatial_key)
    target_spatial = _get_spatial_matrix(target, spatial_key=spatial_key)
    if use_moscot_defaults and normalize_spatial:
        source_spatial = _spatial_norm(source_spatial)
        target_spatial = _spatial_norm(target_spatial)

    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            "Source and target feature dimensions differ: "
            f"{source_features.shape[1]} vs {target_features.shape[1]}."
        )
    if source_features.shape[0] != source_spatial.shape[0]:
        raise ValueError("Source feature rows do not match source spatial rows.")
    if target_features.shape[0] != target_spatial.shape[0]:
        raise ValueError("Target feature rows do not match target spatial rows.")

    C = cdist(source_features, target_features, metric=feature_metric)
    C1 = cdist(source_spatial, source_spatial, metric=structure_metric)
    C2 = cdist(target_spatial, target_spatial, metric=structure_metric)

    if normalize_feature_cost:
        C = _normalize_matrix(C)
    if normalize_structure_cost:
        C1 = _normalize_matrix(C1)
        C2 = _normalize_matrix(C2)

    p, q = _make_masses(source_features.shape[0], target_features.shape[0], mass_mode)

    return SpaOTProblem(
        C=C,
        C1=C1,
        C2=C2,
        p=p,
        q=q,
        source_features=source_features,
        target_features=target_features,
        source_spatial=source_spatial,
        target_spatial=target_spatial,
    )


def prepare_spaot_matrices(source, target, **kwargs):
    """Return ``(C, C1, C2, p, q)`` for direct SpaOT solver calls."""
    problem = prepare_spaot_problem(source, target, **kwargs)
    return problem.C, problem.C1, problem.C2, problem.p, problem.q
