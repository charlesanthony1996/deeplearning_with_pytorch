from tinygrad.tensor import Tensor
import numpy as np

# define the function that we wish to find the minimum of
def loss(phi):

    phi0 = phi[0]
    phi1 = phi[1]

    height = np.exp(-0.5, (phi0 * phi1)* 4.0)
    height = height * np.exp(-0.5 * (phi0 - 0.7) * (phi0 - 0.7)/ 4.0)

    return 1.0 - height





def grad_descent(start, steps = 100, alpha = 0.1):
    phi = Tensor(start, requires_grad=True)
    
    path = [phi.numpy().copy()]

    for _ in range(steps):
        l = loss(phi)
        l.backward()

        # update
        phi = (phi - alpha * phi.grad).detach()
        phi.requires_grad = True

        path.append(phi.numpy().copy())

    return np.array(path)


def normalized_gradients(start, steps = 100, alpha = 0.1, eps = 1e-8):
    phi = Tensor(start, requires_grad=True)
    path = [phi.numpy().copy()]

    for _ in range(steps):
        l = loss(phi)
        l.backward()

        g = phi.grad
        v = g * g

        phi = (phi - alpha * g / (v.sqrt() + eps)).detach()
        phi.requires_grad = True

        path.append(phi.numpy().copy())

    return np.array(path)


