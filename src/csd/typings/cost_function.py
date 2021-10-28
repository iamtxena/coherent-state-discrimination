from typing import NamedTuple
import strawberryfields as sf

from csd.typings.typing import MeasuringTypes
from csd.engine import Engine


class CostFunctionOptions(NamedTuple):
    engine: Engine
    circuit: sf.Program
    backend_name: str
    measuring_type: MeasuringTypes
    shots: int
    plays: int
