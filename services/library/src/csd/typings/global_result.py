from typing import List, NamedTuple, Tuple
import json

from csd.ideal_probabilities import IdealHelstromProbability


class GlobalResult(NamedTuple):
    alpha: float
    success_probability: float
    number_modes: int
    time_in_seconds: float
    squeezing: bool
    number_ancillas: int

    @property
    def distance_to_helstrom_probability(self) -> float:
        return self.success_probability - IdealHelstromProbability(
            alpha=self.alpha,
            number_modes=self.number_modes).helstrom_probability

    @property
    def bit_error_rate(self) -> float:
        return (1 - self.success_probability) / self.number_modes

    def header(self) -> List[str]:
        return ['alpha',
                'success_probability',
                'number_modes',
                'time_in_seconds',
                'distance_to_helstrom_probability',
                'bit_error_rate',
                'squeezing',
                'number_ancillas']

    def values(self) -> Tuple[float, float, int, float, float, float, bool, int]:
        return (self.alpha,
                self.success_probability,
                self.number_modes,
                self.time_in_seconds,
                self.distance_to_helstrom_probability,
                self.bit_error_rate,
                self.squeezing,
                self.number_ancillas)

    def __eq__(self, other) -> bool:
        return (self.alpha == other.alpha and
                self.success_probability == other.success_probability and
                self.number_modes == other.number_modes and
                self.time_in_seconds == other.time_in_seconds and
                self.distance_to_helstrom_probability == other.distance_to_helstrom_probability and
                self.bit_error_rate == other.bit_error_rate and
                self.squeezing == other.squeezing and
                self.number_ancillas == other.number_ancillas)

    def __str__(self) -> str:
        return json.dumps({
            "alpha": self.alpha,
            "success_probability": self.success_probability,
            "number_modes": self.number_modes,
            "time_in_seconds": self.time_in_seconds,
            "distance_to_helstrom_probability": self.distance_to_helstrom_probability,
            "bit_error_rate": self.bit_error_rate,
            "squeezing": self.squeezing,
            "number_ancillas": self.number_ancillas,
        })

    def __repr__(self) -> str:
        return self.__str__()
