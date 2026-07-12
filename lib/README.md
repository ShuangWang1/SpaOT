# References

The `lib/` folder contains the SpaOT/FPGW implementation and a small set of adapted utilities required by the demo and manuscript reproducibility scripts.

Retained adapted components:

- `fgw`: selected Fused Gromov-Wasserstein graph utilities adapted from FGW, https://github.com/tvayer/FGW
- `fugw`: selected dense FUGW solver utilities adapted from FUGW, https://github.com/alexisthual/fugw

Core SpaOT modules:

- `fused_pgw.py`: Frank-Wolfe style SpaOT/FPGW solver used by the demo and most reproducibility scripts.
- `sinkhorn_pgw.py`: entropic Sinkhorn-style SpaOT/FPGW solver retained as a manuscript contribution.
- `fused_pgw_barycenter.py`: FPGW barycenter routines used by the diagnostic stratification analyses.
- `graph_clustering.py`: graph benchmark and clustering helpers used by Fig. 2.

Some imported functions have been modified for SpaOT analyses.
