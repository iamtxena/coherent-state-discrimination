from csd import CSD
from csd.typings import MeasuringTypes, RunConfiguration, CSDConfiguration, Backends
from csd.config import logger
import numpy as np
import json
import random
import strawberryfields as sf
import tensorflow as tf

ALPHA = 0.7
STEPS = 500
BATCH_SIZE = 10


def orig_tf() -> None:
    tf_displacement_magnitude = tf.Variable(0.1)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    steps = STEPS

    batch_size = BATCH_SIZE
    threshold = 0.5

    for step in range(steps):
        batch = [1 if random.random() > threshold else -1 for _ in range(batch_size)]
        alpha_val = ALPHA * np.array(batch)
        # alpha_val = BATCH_ALPHA

        eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": 7,
            "batch_size": len(alpha_val),
        })

        circuit = sf.Program(1)

        displacement_magnitude = circuit.params("displacement_magnitude")

        alpha = circuit.params("alpha")

        with circuit.context as q:
            sf.ops.Dgate(alpha, 0.0) | q[0]
            sf.ops.Dgate(displacement_magnitude, 0.0) | q[0]

        with tf.GradientTape() as tape:
            results = eng.run(circuit, args={
                "displacement_magnitude": tf_displacement_magnitude,
                "alpha": alpha_val
            })

            # get the probability of |0>
            p_zero = results.state.fock_prob([0])

            # get the porbability of anything by |0>
            p_one = 1 - p_zero

            loss = 0.0

            for i, mult in enumerate(batch):
                if mult == 1:
                    loss += p_one[i]
                else:
                    loss += p_zero[i]

            loss /= len(batch)

        gradients = tape.gradient(loss, [tf_displacement_magnitude])
        opt.apply_gradients(zip(gradients, [tf_displacement_magnitude]))

        if (step + 1) % 10 == 0:
            print("Learned displacement value at step {}: {}".format(step + 1, tf_displacement_magnitude.numpy()))
    logger.info(f'beta optimized: {tf_displacement_magnitude.numpy()} and loss: {loss}')


def test_tf() -> None:
    # alphas = list(np.arange(0.05, 1.5, 0.05))
    alphas = [0.0, 0.7, 1.3]

    csd_configuration = CSDConfiguration({
        'cutoff_dim': 7,
        'steps': STEPS,
        'batch_size': BATCH_SIZE
    })
    csd = CSD(csd_config=csd_configuration)
    run_configuration = RunConfiguration({
        'alphas': alphas,
        'backend': Backends.TENSORFLOW,
        'number_qumodes': 1,
        'number_layers': 1,
        'measuring_type': MeasuringTypes.PROBABILITIES
    })
    result = csd.execute(configuration=run_configuration)
    logger.info(f"beta optimized: {result['opt_betas'][0]} and loss: {result['p_err'][0]}")


def test_gaus_sampling() -> None:
    # alphas = list(np.arange(0.05, 2.1, 0.05))
    alphas = [0.7]

    csd_configuration = CSDConfiguration({
        'batch_size': 2,
        'shots': 100
    })
    csd = CSD(csd_config=csd_configuration)
    run_configuration = RunConfiguration({
        'alphas': alphas,
        'backend': Backends.GAUSSIAN,
        'number_qumodes': 1,
        'number_layers': 1,
        'measuring_type': MeasuringTypes.SAMPLING,
    })
    result = csd.execute(configuration=run_configuration)
    logger.info(result)


def test_sampling() -> None:
    # alphas = list(np.arange(0.05, 2.1, 0.05))
    alphas = [0.7]

    csd_configuration = CSDConfiguration({
        'batch_size': 10,
        'shots': 100,
        'cutoff_dim': 10
    })
    csd = CSD(csd_config=csd_configuration)
    run_configuration = RunConfiguration({
        'alphas': alphas,
        'backend': Backends.FOCK,
        'number_qumodes': 1,
        'number_layers': 1,
        'measuring_type': MeasuringTypes.SAMPLING,
    })
    result = csd.execute(configuration=run_configuration)
    logger.info(json.dumps(result, indent=2))


def test_tf_2() -> None:
    # alphas = list(np.arange(0.05, 1.1, 0.05))
    alphas = [0.7]

    csd = CSD(csd_config=CSDConfiguration({
        'steps': 500,
        'cutoff_dim': 10
    }))

    csd.plot_success_probabilities(alphas=alphas)
    csd.execute_all_backends_and_measuring_types(
        alphas=alphas,
        measuring_types=[MeasuringTypes.PROBABILITIES]
    )
    csd.plot_success_probabilities(measuring_types=[MeasuringTypes.PROBABILITIES])


if __name__ == '__main__':
    # orig_tf()
    test_tf()
    # test_gaus_sampling()
    # test_sampling()
    # test_tf_2()
