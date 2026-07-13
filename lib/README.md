# References

The `lib/` folder does not contain original optimal-transport solvers. Every solver used by
SpaOT is imported from the FPGW repository,

  **FPGW — Fused Partial Gromov-Wasserstein**, https://github.com/yikun-baio/fused-pgw

and adapted for spatial multi-omics analyses. The FPGW paper is the origin of the
Fused Partial Gromov-Wasserstein formulation, of the Frank-Wolfe and entropic
(Sinkhorn) FPGW solvers, of the partial OT routines, and of the FPGW/PGW barycenter
routines. SpaOT applies these algorithms; it does not introduce them.

## Modules imported from FPGW (https://github.com/yikun-baio/fused-pgw)

- `fused_pgw.py`: Frank-Wolfe style Fused Partial GW / Partial GW solvers.
- `sinkhorn_pgw.py`: entropic, TV-regularized Sinkhorn solvers for Fused Partial GW.
  These are derived and introduced in the FPGW paper; SpaOT is a downstream user.
- `fused_pgw_barycenter.py`: FPGW / PGW / GW barycenter routines.
- `gromov_utils.py`: Gromov-Wasserstein tensor-product and loss utilities.
- `opt.py`: partial OT and linear / linearized OT solvers.
- `graph_clustering.py`: graph benchmark and clustering helpers.

## Third-party components, vendored via the FPGW repository

- `fgw/`: Fused Gromov-Wasserstein graph utilities from FGW,
  https://github.com/tvayer/FGW (Vayer et al.).
- `fugw/`: dense Fused Unbalanced Gromov-Wasserstein solver utilities from FUGW,
  https://github.com/alexisthual/fugw (Thual et al., NeurIPS 2022).
  `fugw/solvers/utils.py` in turn contains routines (`solver_sinkhorn`,
  `solver_sinkhorn_sparse`) adapted from UGW, the Unbalanced Gromov-Wasserstein
  divergence, https://github.com/thibsej/unbalanced_gromov_wasserstein
  (Séjourné, Vialard and Peyré, NeurIPS 2021). SpaOT's dependence on UGW is therefore
  indirect, through FUGW.

Some imported functions have been modified for SpaOT analyses.
