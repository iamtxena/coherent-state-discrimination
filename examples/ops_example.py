# ops_example.py
from typing import List, Union
import strawberryfields as sf
import numpy as np
from strawberryfields.parameters import FreeParameter
from tensorflow.python.framework.ops import EagerTensor
from csd.operations.interferometer import Interferometer
from csd.operations.universal_multimode import UniversalMultimode


def create_circuit():
    params = []
    prog = sf.Program(2)
    alpha = prog.params("alpha")
    beta = prog.params("beta")
    r = prog.params("r")
    phi_r = prog.params("phi_r")
    params.append(alpha)
    params.append(beta)
    params.append(r)
    params.append(phi_r)
    print(params)
    print(params[0])
    print(type(params[0]))
    with prog.context as q:
        sf.ops.Dgate(params[0], 0.0) | q[0]
        sf.ops.Dgate(params[1], 0.0) | q[0]
        sf.ops.Sgate(params[2], params[3]) | q[0]
        sf.ops.MeasureFock() | q

    prog.print()


def create_interferometer():
    # create a 4 mode quantum program
    number_modes = 4
    M = number_modes
    prog = sf.Program(number_modes)
    theta = np.random.rand(1, int(M * (M - 1) / 2))[0].tolist()
    print(theta)
    phi = np.random.rand(1, int(M * (M - 1) / 2))[0].tolist()
    print(phi)
    varphi = np.random.rand(1, M)[0].tolist()
    print(varphi)

    with prog.context as q:
        Interferometer(theta=theta,
                       phi=phi,
                       varphi=varphi,
                       number_modes=number_modes,
                       context=q)
        sf.ops.MeasureFock() | q

    prog.print()


def create_parametrized_interferometer():
    # create a 4 mode quantum program
    number_modes = 4
    M = number_modes
    K = int(M * (M - 1) / 2)
    prog = sf.Program(number_modes)
    theta = _create_free_parameter_list(base_name='theta', number_elems=K, circuit=prog)
    phi = _create_free_parameter_list(base_name='phi', number_elems=K, circuit=prog)
    varphi = _create_free_parameter_list(base_name='varphi', number_elems=M, circuit=prog)

    with prog.context as q:
        Interferometer(theta=theta,
                       phi=phi,
                       varphi=varphi,
                       number_modes=number_modes,
                       context=q)
        sf.ops.MeasureFock() | q

    prog.print()


def create_universal_multimode(squeezing: bool = True):
    # create a 4 mode quantum program
    number_modes = 4
    M = number_modes
    K = int(M * (M - 1) / 2)
    prog = sf.Program(number_modes)
    theta_1 = _create_random_list(K)
    phi_1 = _create_random_list(K)
    varphi_1 = _create_random_list(M)
    r = _create_random_list(M)
    phi_r = _create_random_list(M)
    theta_2 = _create_random_list(K)
    phi_2 = _create_random_list(K)
    varphi_2 = _create_random_list(M)
    a = _create_random_list(M)

    with prog.context as q:
        UniversalMultimode(theta_1=theta_1,
                           phi_1=phi_1,
                           varphi_1=varphi_1,
                           r=r,
                           phi_r=phi_r,
                           theta_2=theta_2,
                           phi_2=phi_2,
                           varphi_2=varphi_2,
                           a=a,
                           number_modes=number_modes,
                           context=q,
                           squeezing=squeezing)
        sf.ops.MeasureFock() | q

    prog.print()


def create_parametrized_universal_multimode(squeezing: bool = True):
    # create a 4 mode quantum program
    number_modes = 4
    M = number_modes
    K = int(M * (M - 1) / 2)
    prog = sf.Program(number_modes)
    theta_1 = _create_free_parameter_list(base_name='theta_1', number_elems=K, circuit=prog)
    phi_1 = _create_free_parameter_list(base_name='phi_1', number_elems=K, circuit=prog)
    varphi_1 = _create_free_parameter_list(base_name='varphi_1', number_elems=M, circuit=prog)
    r = _create_free_parameter_list(base_name='r', number_elems=M, circuit=prog)
    phi_r = _create_free_parameter_list(base_name='phi_r', number_elems=M, circuit=prog)
    theta_2 = _create_free_parameter_list(base_name='theta_2', number_elems=K, circuit=prog)
    phi_2 = _create_free_parameter_list(base_name='phi_2', number_elems=K, circuit=prog)
    varphi_2 = _create_free_parameter_list(base_name='varphi_2', number_elems=M, circuit=prog)
    a = _create_free_parameter_list(base_name='a', number_elems=M, circuit=prog)

    parameters = _count_free_parameters(
        theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, squeezing)
    print(parameters)

    with prog.context as q:
        UniversalMultimode(theta_1=theta_1,
                           phi_1=phi_1,
                           varphi_1=varphi_1,
                           r=r,
                           phi_r=phi_r,
                           theta_2=theta_2,
                           phi_2=phi_2,
                           varphi_2=varphi_2,
                           a=a,
                           number_modes=number_modes,
                           context=q,
                           squeezing=squeezing)
        sf.ops.MeasureFock() | q

    prog.print()


def _create_random_list(number_elems: int) -> List[float]:
    return np.random.rand(1, number_elems)[0].tolist()


def _create_free_parameter_list(base_name: str,
                                number_elems: int,
                                circuit: sf.Program) -> List[FreeParameter]:
    return [circuit.params(f'{base_name}_{str(elem)}') for elem in range(number_elems)]


def _count_free_parameters(theta_1: List[Union[float, EagerTensor, FreeParameter]],
                           phi_1: List[Union[float, EagerTensor, FreeParameter]],
                           varphi_1: List[Union[float, EagerTensor, FreeParameter]],
                           r: List[Union[float, EagerTensor, FreeParameter]],
                           phi_r: List[Union[float, EagerTensor, FreeParameter]],
                           theta_2: List[Union[float, EagerTensor, FreeParameter]],
                           phi_2: List[Union[float, EagerTensor, FreeParameter]],
                           varphi_2: List[Union[float, EagerTensor, FreeParameter]],
                           a: List[Union[float, EagerTensor, FreeParameter]],
                           squeezing: bool = True) -> None:
    if squeezing:
        return (len(theta_1) + len(phi_1) + len(varphi_1) + len(r) +
                len(phi_r) + len(theta_2) + len(phi_2) + len(varphi_2) + len(a))
    return (len(theta_1) + len(phi_1) + len(varphi_1) +
            len(theta_2) + len(phi_2) + len(varphi_2) + len(a))


if __name__ == '__main__':
    # create_interferometer()
    # create_universal_multimode()
    # create_universal_multimode(squeezing=False)
    # create_circuit()
    # create_parametrized_interferometer()
    create_parametrized_universal_multimode()
    # create_parametrized_universal_multimode(squeezing=False)
