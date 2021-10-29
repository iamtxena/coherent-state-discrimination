# ideal_probabilities.py

from abc import ABC
from typing import List, Tuple
import math
from scipy.optimize import minimize


class IdealProbabilities(ABC):
    """ Class to compute the ideal probability

    """

    def __init__(self, alphas: List[float]):
        self._p_homos = [self._prob_homodyne(a=current_alpha) for current_alpha in alphas]
        self._p_hels = [self._prob_helstrom(a=current_alpha) for current_alpha in alphas]
        self._p_ken_op = self._compute_probs_with_betas_optimized(alphas=alphas)

    @property
    def p_homos(self) -> Tuple[List[float], str]:
        return (self._p_homos, "pHom(a)")

    @property
    def p_hels(self) -> Tuple[List[float], str]:
        return (self._p_hels, "pHel(a)")

    @property
    def p_ken_op(self) -> Tuple[List[float], str]:
        return (self._p_ken_op, "pKenOp(a)")

    def _prob_homodyne(self, a: float) -> float:
        return (1 + math.erf(math.sqrt(2) * a)) / 2

    def _prob_helstrom(self, a: float) -> float:
        return (1 + math.sqrt(1 - math.exp(-4 * a**2))) / 2

    def _p_zero(self, a: float) -> float:
        return math.exp(-a**2)

    def _p_err(self, b: float, a: float) -> float:
        return (self._p_zero(-a + b) + 1 - self._p_zero(a + b)) / 2

    def _p_succ(self, b: float, a: float) -> float:
        return (self._p_zero(a + b) + 1 - self._p_zero(-a + b)) / 2

    def _optimize(self, alphas: List[float]) -> List[float]:
        return [minimize(self._p_err, 0, args=(alpha,), method='BFGS', tol=1e-6).x[0] for alpha in alphas]

    def _compute_probs_with_betas_optimized(self, alphas: List[float]) -> List[float]:
        opt_betas = self._optimize(alphas=alphas)
        return [self._p_succ(b=opt_beta, a=alpha) for (opt_beta, alpha) in zip(opt_betas, alphas)]