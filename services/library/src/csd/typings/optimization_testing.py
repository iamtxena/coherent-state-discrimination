from typing import NamedTuple, Union
from csd.circuit import Circuit

from csd.engine import Engine
from csd.tf_engine import TFEngine


class OptimizationTestingOptions(NamedTuple):
    engine: Union[Engine, TFEngine]
    circuit: Circuit
    backend_name: str
    shots: int
    plays: int
