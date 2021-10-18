# ops_example.py
import strawberryfields as sf
import numpy as np
from csd.operations.interferometer import Interferometer


def create_interferometer():
    # create a 3 mode quantum program
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


if __name__ == '__main__':
    create_interferometer()
