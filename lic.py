import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import h5py
import time

def get_noise(vectors):
    return np.random.rand(*(vectors.shape[0:2]))


@njit
def lic_flow(vectors, t=0, len_pix=5, noise=None):
    vectors = np.asarray(vectors)
    m, n, two = vectors.shape
    if two != 2:
        raise ValueError

    if noise is None:
        noise = np.random.rand(*(vectors.shape[0:2]))

    result = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            y = i
            x = j
            forward_sum = 0
            forward_total = 0
            # Advect forwards
            for k in range(len_pix):
                dx = vectors[int(y), int(x), 0]
                dy = vectors[int(y), int(x), 1]
                dt_x = dt_y = 0
                if dy > 0:
                    dt_y = ((np.floor(y) + 1) - y) / dy
                elif dy < 0:
                    dt_y = (y - (np.ceil(y) - 1)) / -dy
                if dx > 0:
                    dt_x = ((np.floor(x) + 1) - x) / dx
                elif dx < 0:
                    dt_x = (x - (np.ceil(x) - 1)) / -dx
                if dx == 0 and dy == 0:
                    dt = 0
                else:
                    dt = min(dt_x, dt_y)
                x = min(max(x + dx * dt, 0), n - 1)
                y = min(max(y + dy * dt, 0), m - 1)
                weight = pow(np.cos(t + 0.46 * k), 2)
                forward_sum += noise[int(y), int(x)] * weight
                forward_total += weight
            y = i
            x = j
            backward_sum = 0
            backward_total = 0
            # Advect backwards
            for k in range(1, len_pix):
                dx = vectors[int(y), int(x), 0]
                dy = vectors[int(y), int(x), 1]
                dy *= -1
                dx *= -1
                dt_x = dt_y = 0
                if dy > 0:
                    dt_y = ((np.floor(y) + 1) - y) / dy
                elif dy < 0:
                    dt_y = (y - (np.ceil(y) - 1)) / -dy
                if dx > 0:
                    dt_x = ((np.floor(x) + 1) - x) / dx
                elif dx < 0:
                    dt_x = (x - (np.ceil(x) - 1)) / -dx
                if dx == 0 and dy == 0:
                    dt = 0
                else:
                    dt = min(dt_x, dt_y)
                x = min(max(x + dx * dt, 0), n - 1)
                y = min(max(y + dy * dt, 0), m - 1)
                weight = pow(np.cos(t - 0.46 * k), 2)
                backward_sum += noise[int(y), int(x)] * weight
                backward_total += weight
            result[i, j] = (forward_sum + backward_sum) / (forward_total + backward_total)
    return result





def show_grey(tex):
    plt.figure()
    tex = tex.T
    plt.imshow(tex, origin = 'lower', cmap='Greys')

def streamlines(Vx, Vy):
    Vx, Vy = Vx.T, Vy.T
    x = np.linspace(0, Vx.shape[0]-1, Vx.shape[0])
    y = np.linspace(0, Vx.shape[1]-1, Vx.shape[1])
    x, y = np.meshgrid(y, x)
    print(x.shape, y.shape, Vx.shape)

    plt.streamplot(x, y, Vx, Vy, density=1)


if __name__ == "__main__":
    with h5py.File("D:\CUHK\Data_from_zcao\struct01\struct01_snap52.h5", 'r') as f:
        B_x = f['i_mag_field'][:,:,50]
        B_y = f['j_mag_field'][:,:,50]
        rho = f['gas_density'][:,:,50]
    start_time = time.time()
    show_grey(lic_flow(np.stack((B_y, B_x), axis=-1), t=0, len_pix=10))

    streamlines(B_x, B_y)
    print()
    print("--- %.2f seconds ---" % (time.time() - start_time))
    plt.show()
