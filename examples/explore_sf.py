import numpy as np
import strawberryfields as sf

# Used for displacement.
displacement = 0.5

def single_layer():
    # Creates a single mode quantum "program".
    # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.Program.html
    prog = sf.Program(1)

    with prog.context as q:
        # Phase space displacement gate.
        # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.Dgate.html
        sf.ops.Dgate(0.0, 0.0) | q[0]

        # Measures whether a mode contain zero or nonzero photons.
        # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.MeasureThreshold.html
        sf.ops.MeasureThreshold()       | q[0]

    return eng.run(prog)


if __name__ == "__main__":
    # Instantiate the Gaussian backend.
    # https://strawberryfields.readthedocs.io/en/stable/introduction/circuits.html
    eng = sf.Engine('gaussian')

    # Execute the single layer of the quantum "program".
    result = single_layer()

    # Obtain results.
    print(result.samples)
