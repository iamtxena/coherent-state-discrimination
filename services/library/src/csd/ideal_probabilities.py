# ideal_probabilities.py

from abc import ABC
from typing import List, NamedTuple, Tuple
import math
from csd.codeword import CodeWord
from csd.util import filter_number_modes_from_codebook
from scipy.optimize import minimize


def compute_homodyne_probability(a: float, number_modes: int):
    return ((1 + math.erf(math.sqrt(2) * a)) / 2)**number_modes


def compute_helstrom_probability(a: float, number_modes: int) -> float:
    return ((1 + math.sqrt(1 - math.exp(-4 * a**2))) / 2)**number_modes


class IdealProbabilities(ABC):
    """ Class to compute the ideal probability

    """

    def __init__(self, alphas: List[float], number_modes: int = 1):
        self._number_modes = number_modes
        self._p_homos = [self._prob_homodyne(a=current_alpha) for current_alpha in alphas]
        self._p_hels = [self._prob_helstrom(a=current_alpha) for current_alpha in alphas]
        self._p_ken_op = self._compute_probs_with_betas_optimized(alphas=alphas)

    @property
    def p_homos(self) -> Tuple[List[float], str]:
        return (self._p_homos, f'$pHom(a)^{self._number_modes}$')

    @property
    def p_hels(self) -> Tuple[List[float], str]:
        return (self._p_hels, f'$pHel(a)^{self._number_modes}$')

    @property
    def p_ken_op(self) -> Tuple[List[float], str]:
        return (self._p_ken_op, "pKenOp(a)")

    def _prob_homodyne(self, a: float) -> float:
        return compute_homodyne_probability(a, number_modes=self._number_modes)

    def _prob_helstrom(self, a: float) -> float:
        return compute_helstrom_probability(a, number_modes=self._number_modes)

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


class IdealHomodyneProbability(NamedTuple):
    alpha: float
    number_modes: int

    @property
    def homodyne_probability(self) -> float:
        return compute_homodyne_probability(a=self.alpha, number_modes=self.number_modes)


class IdealHelstromProbability(NamedTuple):
    alpha: float
    number_modes: int

    @property
    def helstrom_probability(self) -> float:
        return compute_helstrom_probability(a=self.alpha, number_modes=self.number_modes)


class IdealLinearCodesHomodyneProbability(NamedTuple):
    codebook: List[CodeWord]

    @property
    def homodyne_probability(self) -> float:
        if len(self.codebook) <= 0:
            return 0.0
        filtered_number_modes = filter_number_modes_from_codebook(codebook=self.codebook)
        return compute_homodyne_probability(a=self.codebook[0].alpha, number_modes=filtered_number_modes)


class IdealLinearCodesHelstromProbability(NamedTuple):
    codebook: List[CodeWord]

    @property
    def helstrom_probability(self) -> float:
        if len(self.codebook) <= 0:
            return 0.0
        filtered_number_modes = filter_number_modes_from_codebook(codebook=self.codebook)
        return compute_helstrom_probability(a=self.codebook[0].alpha, number_modes=filtered_number_modes)
