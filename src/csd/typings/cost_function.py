from typing import NamedTuple, Union

from csd.engine import Engine
from csd.tf_engine import TFEngine
from csd.typings.multi_layer_circuit import MultiLayerCircuit
from csd.typings.typing import MeasuringTypes


class CostFunctionOptions(NamedTuple):
    """Cost function options"""

    engine: Union[Engine, TFEngine]
    circuit: MultiLayerCircuit
    backend_name: str
    measuring_type: MeasuringTypes
    shots: int
    plays: int
