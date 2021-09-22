import numpy as np
import strawberryfields as sf

NUM_SHOTS = 100

def single_layer(params):
    # Creates a single mode quantum "program".
    # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.Program.html
    prog = sf.Program(1)

    # Instantiate the Gaussian backend.
    # https://strawberryfields.readthedocs.io/en/stable/introduction/circuits.html
    eng = sf.Engine("tf", backend_options={"cutoff_dim": 5})

    with prog.context as q:
        # Phase space squeezing gate.
        # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.Sgate.html
        sf.ops.Dgate(params["displacement_magnitude"])  | q[0]

        # Measures whether a mode contains zero or nonzero photons.
        # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.MeasureThreshold.html
        sf.ops.MeasureFock()                       | q[0]

    return eng.run(prog, shots=NUM_SHOTS)


if __name__ == "__main__":
    # Parameters
    params = {
        "displacement_magnitude": 0.5
    }

    # Execute the single layer of the quantum "program".
    result = single_layer(params=params)

    # Obtain results.
    print(result.samples)
