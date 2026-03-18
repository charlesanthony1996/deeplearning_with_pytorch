from tinygrad.tensor import Tensor
import numpy as np

# define the function that we wish to find the minimum of
def loss(phi0, phi1):
    height = np.exp(-0.5, (phi0 * phi1)* 4.0)
    height = height * np.exp(-0.5 * (phi0 - 0.7) * (phi0 - 0.7)/ 4.0)
    return 1.0 - height





