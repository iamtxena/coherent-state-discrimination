import strawberryfields as sf
import numpy as np

NUM_SHOTS = 1000


def single_layer(params):
    # Creates a single mode quantum "program".
    # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.Program.html
    prog = sf.Program(1)

    # Instantiate the Gaussian backend.
    # https://strawberryfields.readthedocs.io/en/stable/introduction/circuits.html
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})

    alpha = prog.params("alpha")
    beta = prog.params("beta")

    with prog.context as q:
        sf.ops.Dgate(alpha, 0.0) | q[0]
        sf.ops.Dgate(beta, 0.0) | q[0]
        sf.ops.MeasureFock() | q[0]

    # return eng.run(prog, shots=NUM_SHOTS)
    return eng.run(prog,
                   args={
                       "alpha": params[0],
                       "beta": params[1],
                   })


if __name__ == "__main__":
    # Parameters
    alphas = [0.1, 0.7, 1.1]
    betas = list(np.arange(-5, 5, 0.3))

    results = []
    for alpha in alphas:
        probs = []
        for beta in betas:
            params = [alpha, beta]
            samples = []

            for _ in range(0, NUM_SHOTS):
                result = single_layer(params=params)
                samples.append(result.samples[0][0])
            probability = sum([1 for read_value in samples if read_value == 0]) / len(samples)
            probs.append(probability)
            print(f'alpha: {alpha}, beta: {np.round(beta, 2)}, prob_0:{probability}')
        result = {
            'alpha': alpha,
            'betas': betas,
            'probs': probs
        }
        results.append(result)
