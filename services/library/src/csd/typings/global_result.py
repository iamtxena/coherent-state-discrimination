from typing import List, NamedTuple, Tuple
import json


class GlobalResult(NamedTuple):
    alpha: float
    success_probability: float
    number_modes: int
    time_in_seconds: float
    squeezing: bool
    number_ancillas: int
    helstrom_probability: float
    homodyne_probability: float
    best_success_probability: float
    best_helstrom_probability: float
    best_homodyne_probability: float
    best_codebook: List[List[int]]
    best_measurements: List[List[int]]
    best_optimized_parameters: List[float]

    @property
    def distance_to_helstrom_probability(self) -> float:
        return self.success_probability - self.helstrom_probability

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
                'number_ancillas',
                'helstrom_probability',
                'homodyne_probability',
                'best_success_probability',
                'best_helstrom_probability',
                'best_homodyne_probability',
                'best_codebook',
                'best_measurements',
                'best_optimized_parameters']

    def values(self) -> Tuple[float, float, int, float, float,
                              float, bool, int, float, float,
                              float, float, float, List[List[int]], List[List[int]],
                              List[float]]:
        return (self.alpha,
                self.success_probability,
                self.number_modes,
                self.time_in_seconds,
                self.distance_to_helstrom_probability,
                self.bit_error_rate,
                self.squeezing,
                self.number_ancillas,
                self.helstrom_probability,
                self.homodyne_probability,
                self.best_success_probability,
                self.best_helstrom_probability,
                self.best_homodyne_probability,
                self.best_codebook,
                self.best_measurements,
                self.best_optimized_parameters)

    def __eq__(self, other) -> bool:
        return (self.alpha == other.alpha and
                self.success_probability == other.success_probability and
                self.number_modes == other.number_modes and
                self.time_in_seconds == other.time_in_seconds and
                self.distance_to_helstrom_probability == other.distance_to_helstrom_probability and
                self.bit_error_rate == other.bit_error_rate and
                self.squeezing == other.squeezing and
                self.number_ancillas == other.number_ancillas and
                self.helstrom_probability == other.helstrom_probability and
                self.homodyne_probability == other.homodyne_probability and
                self.best_success_probability == other.best_success_probability and
                self.best_helstrom_probability == other.best_helstrom_probability and
                self.best_homodyne_probability == other.best_homodyne_probability and
                self.best_codebook == other.best_codebook and
                self.best_measurements == other.best_measurements and
                self.best_optimized_parameters == other.best_optimized_parameters)

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
            "helstrom_probability": self.helstrom_probability,
            "homodyne_probability": self.homodyne_probability,
            "best_success_probability": self.best_success_probability,
            "best_helstrom_probability": self.best_helstrom_probability,
            "best_homodyne_probability": self.best_homodyne_probability,
            "best_codebook": self.best_codebook,
            "best_measurements": self.best_measurements,
            "best_optimized_parameters": self.best_optimized_parameters
        })

    def __repr__(self) -> str:
        return self.__str__()
