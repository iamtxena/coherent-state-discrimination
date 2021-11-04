from typing import List, NamedTuple, Tuple
import json

from csd.ideal_probabilities import IdealHomodyneProbability


class GlobalResult(NamedTuple):
    alpha: float
    success_probability: float
    number_modes: int
    time_in_seconds: float

    @property
    def distance_to_homodyne_probability(self) -> float:
        return (self.success_probability - IdealHomodyneProbability(
            alpha=self.alpha,
            number_modes=self.number_modes).homodyne_probability)

    @property
    def bit_error_rate(self) -> float:
        return (1 - self.success_probability) / self.number_modes

    def header(self) -> List[str]:
        return ['alpha',
                'success_probability',
                'number_modes',
                'time_in_seconds',
                'distance_to_homodyne_probability',
                'bit_error_rate']

    def values(self) -> Tuple[float, float, int, float, float, float]:
        return (self.alpha,
                self.success_probability,
                self.number_modes,
                self.time_in_seconds,
                self.distance_to_homodyne_probability,
                self.bit_error_rate)

    def __str__(self) -> str:
        return json.dumps({
            "alpha": self.alpha,
            "success_probability": self.success_probability,
            "number_modes": self.number_modes,
            "time_in_seconds": self.time_in_seconds,
            "distance_to_homodyne_probability": self.distance_to_homodyne_probability,
            "bit_error_rate": self.bit_error_rate
        })

    def __repr__(self) -> str:
        return self.__str__()
