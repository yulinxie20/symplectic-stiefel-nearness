"""论文采用的 Gao 非单调 RSD 与 Jensen--Zimmermann R-TR2 基线。"""
from .common import SolveResult, SolverOptions
from .gao_nonmonotone import solve_gao_nonmonotone
from .jensen_trustregions import JensenTrustRegionOptions, solve_jensen_trustregions

__all__ = ["SolveResult", "SolverOptions", "JensenTrustRegionOptions",
           "solve_gao_nonmonotone", "solve_jensen_trustregions"]
