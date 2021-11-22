from typing import List, NamedTuple, Union
from csd.circuit import Circuit

from csd.typings.typing import MeasuringTypes
from csd.engine import Engine
from csd.tf_engine import TFEngine
from tensorflow import Variable


class CostFunctionOptions(NamedTuple):
    engine: Union[Engine, TFEngine]
    circuit: Circuit
    backend_name: str
    measuring_type: MeasuringTypes
    shots: int
    all_counts: List[Variable]
    plays: int
