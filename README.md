# SpaOT

SpaOT (Spatial Optimal Transport) is a spatial multi-omics integration framework for aligning heterogeneous spatial omics samples. It implements a total variation-regularized Fused Partial Gromov-Wasserstein (FPGW) formulation that combines molecular or learned feature similarity with spatial structure while allowing unmatched mass. This makes the algorithm suitable for samples with different cellular composition, measurement throughput, resolution, modality, and noise.

![SpaOT schematic](./paper_fig/fig1.jpg)

## Algorithm

SpaOT represents each spatial omics sample as a metric-measure space:

- nodes are cells, spots, pixels, image tiles, or other spatial units;
- node features encode molecular profiles, image embeddings, or another shared representation;
- structural matrices encode within-sample spatial distances;
- node masses encode the amount of source and target mass available for transport.

For a source sample and a target sample, SpaOT optimizes a transport plan by combining:

- an OT term, which compares source and target features through a cross-sample cost matrix;
- a GW term, which compares source and target spatial structures through pairwise distance matrices;
- a partial/unbalanced mass penalty, which permits noisy, missing, or modality-specific regions to remain unmatched.

The resulting transport plan can be used for spatial alignment, label transfer, feature imputation, cross-resolution mapping, and cross-modality integration. The transport cost can be used as a sample-level distance or as an input to barycenter computation.

## Software Requirements

Most workflows require the following Python ecosystem:

- Python ≥ 3.10
- NumPy
- SciPy
- pandas
- Scanpy
- AnnData
- POT
- moscot
- PyTorch
- matplotlib
- scikit-learn

Several analyses additionally require:

- R (for pathway enrichment)
- COMMOT
- DINO image encoder dependencies

Please refer to the main repository installation guide for the complete software environment.



## Installation

```bash
conda create -n SpaOT python
conda activate SpaOT
git clone https://github.com/ShuangWang1/SpaOT.git
cd SpaOT
pip install .
```

## Basic Usage

### Toy Data

```python
import numpy as np

from lib.fused_pgw import fused_partial_gromov_wasserstein, fused_pgw_cost

data = np.load("./toy_data/graph_data_input_matrices.npz")

M = data["M"] # M: source-by-target feature cost matrix
C1 = data["C1"] # C1: source structural distance matrix
C2 = data["C2"] # C2: target structural distance matrix

n_source, n_target = M.shape
p = np.ones(n_source) / max(n_source, n_target)
q = np.ones(n_target) / max(n_source, n_target)

transport_plan = fused_partial_gromov_wasserstein(
    M,
    C1,
    C2,
    p=p,
    q=q,
    omega2=0.5,
    Lambda=1.0,
    Type="pot",
)

cost = fused_pgw_cost(
    M,
    C1,
    C2,
    transport_plan,
    omega2=0.5,
    loss_fun="square_loss",
)
```

Important parameters:

- `M`: source-by-target feature cost matrix.
- `C1`, `C2`: source and target structural distance matrices.
- `p`, `q`: source and target node mass vectors.
- `omega2`: structure-feature tradeoff. `0.0` reduces to partial OT, `1.0` emphasizes partial GW, and intermediate values run the fused SpaOT objective.
- `Lambda`: penalty controlling unmatched mass in the partial formulation.
- `Type`: linear OT backend used by the solver.

## Preparing Inputs

For a typical SpaOT run:

1. Normalize features for each modality.
2. Construct a shared feature space or feature-cost matrix across samples.
3. Compute `M` with a distance such as Euclidean or cosine distance.
4. Compute `C1` and `C2` from spatial coordinates.
5. Scale feature and structure costs to comparable ranges.
6. Choose node masses, commonly uniform masses for a first run.
7. Run `fused_partial_gromov_wasserstein`.
8. Use the transport plan for mapping or the transport cost for downstream distance-based analysis.

## Barycenter Utilities

`lib/fused_pgw_barycenter.py` implements FPGW barycenter routines for summarizing a collection of spatial samples. Barycenters can be used to estimate a representative spatial organization for a group of samples and to compare groups through SpaOT-derived distances.

## Citation and Provenance

SpaOT is an application of the **Fused Partial Gromov-Wasserstein (FPGW)** framework to
spatial multi-omics integration. The FPGW formulation and all optimal-transport solvers used
here — Fused PGW, PGW, partial OT, the entropic (Sinkhorn) PGW solvers, and the FPGW
barycenter routines — are introduced in the FPGW paper and imported from its repository:

> **FPGW** — https://github.com/yikun-baio/fused-pgw

SpaOT does not re-derive these algorithms. If you use SpaOT, please cite the FPGW paper in
addition to the SpaOT paper.

```bibtex
% TODO: FPGW paper bibtex
% TODO: SpaOT paper bibtex
```

The `lib/` folder further vendors third-party optimal-transport code, inherited through the
FPGW repository:

| Folder | Upstream repository | Reference |
|---|---|---|
| `lib/fgw/` | [tvayer/FGW](https://github.com/tvayer/FGW) | Fused Gromov-Wasserstein (Vayer et al.) |
| `lib/fugw/` | [alexisthual/fugw](https://github.com/alexisthual/fugw) | Fused Unbalanced GW (Thual et al., NeurIPS 2022) |
| `lib/fugw/solvers/utils.py` | [thibsej/unbalanced_gromov_wasserstein](https://github.com/thibsej/unbalanced_gromov_wasserstein) | Unbalanced GW (Séjourné et al., NeurIPS 2021), reached indirectly via FUGW |

See [lib/README.md](./lib/README.md) for the per-module breakdown. The solvers also build on
[POT (Python Optimal Transport)](https://pythonot.github.io/).


