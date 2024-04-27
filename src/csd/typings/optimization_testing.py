from typing import NamedTuple, Union

from csd.circuit import Circuit
from csd.engine import Engine
from csd.tf_engine import TFEngine
from csd.typings.typing import MeasuringTypes, MetricTypes


class OptimizationTestingOptions(NamedTuple):
    engine: Union[Engine, TFEngine]
    circuit: Circuit
    backend_name: str
    measuring_type: MeasuringTypes
    metric_type: MetricTypes
    shots: int
    plays: int
