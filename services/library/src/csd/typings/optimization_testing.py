from typing import NamedTuple, Union
from csd.circuit import Circuit

from csd.engine import Engine
from csd.tf_engine import TFEngine
from csd.typings.typing import MeasuringTypes


class OptimizationTestingOptions(NamedTuple):
    engine: Union[Engine, TFEngine]
    circuit: Circuit
    backend_name: str
    measuring_type: MeasuringTypes
    shots: int
    plays: int
