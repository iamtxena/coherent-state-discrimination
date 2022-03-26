# recurrent architecture

## making Strawberry Fields not crash erratically

The issue is documented [here](https://github.com/XanaduAI/strawberryfields/issues/670) and the fix that I made is described in [this comment](https://github.com/XanaduAI/strawberryfields/issues/670#issuecomment-1016363917).

It is a single line fix in the fock backend of the library.

## dependencies

The dependencies are as follows:

```bash
numpy == 1.19.5
loguru == 0.5.3
scipy == 1.7.1
tqdm == 4.62.3
wandb == 0.12.10
StrawberryFields == 0.21.0
scikit-learn == 1.0.1
```

You can install these by running `pip3 install -r requirements.txt` from the directory that this readme is housed in.

## project organization

`sf_recurrent.py` is the main entry point of the program, experiments are run from this file.
The training method, the evaluation method, the loss metrics are all defined in this file.

`quantum_circuit.py` defines the quantum circuit used for simulation along with the backends used and performs necessary operations to make the circuit work with the program.

`model_wrapper.py` is home to the classical machine learning model that is used to predict our circuit parameters.
It has various utility functions to facilitate interoperation between the scikit-learn model and the `scipy.optimize.minimize` method.

`util.py` has various utility functions that are required by the three previous files.

## running experiments

Modify parameters in `sf_recurrent.py` and run experiments by executing `python3 sf_recurrent.py` in your terminal.
