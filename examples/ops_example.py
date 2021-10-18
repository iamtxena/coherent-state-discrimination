# ops_example.py
from typing import List
import strawberryfields as sf
import numpy as np
from csd.operations.interferometer import Interferometer
from csd.operations.universal_multimode import UniversalMultimode


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


def _create_random_list(number_elems: int) -> List[float]:
    return np.random.rand(1, number_elems)[0].tolist()


if __name__ == '__main__':
    # create_interferometer()
    create_universal_multimode()
    # create_universal_multimode(squeezing=False)
