from typing import NamedTuple, Union
from csd.circuit import Circuit

from csd.typings.typing import MeasuringTypes
from csd.engine import Engine
from csd.tf_engine import TFEngine


class CostFunctionOptions(NamedTuple):
    engine: Union[Engine, TFEngine]
    circuit: Circuit
    backend_name: str
    measuring_type: MeasuringTypes
    shots: int
    plays: int
