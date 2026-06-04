import numpy as np


def coil_dimension_calc(
    N: int,
    l: float,
    w: float,
    s: float,
    z_thick: float,
) -> np.ndarray:
    """Generate vertex coordinates for a square spiral PCB emitter coil.

    Parameters
    ----------
    N : int
        Number of turns per layer. Total turn count is 2*N (top + bottom).
    l : float
        Side length of the outermost square turn (m) : float
    w : float 
        Width of each copper trace (m).
    s : float
        Edge-to-edge spacing between adjacent traces (m).
    z_thick : float
        PCB thickness (m). Sets the z-depth of the bottom layer.

    Returns
    -------
    np.ndarray, shape (4*N + 2, 3)
        Array containing vertex coordinates of the coil.
    """

    ll_s = w + s  # track pitch

    ramp = np.concatenate([np.arange(1, N), np.arange(N - 2, -1, -1)]) #length 2*N - 2
    shortening = np.concatenate([[0, 0, 0], np.repeat(ramp, 2)])  # length 4*N - 1
    seg_len = np.concatenate([l - shortening * ll_s, [l - ll_s]])  # length 4*N

    x_traj = np.array([0,-1,0,1])
    y_traj = np.array([1,0,-1,0])

    q = np.arange(4 * N) % 4  # direction index for each segment
    x = np.concatenate([[0.0], np.cumsum(x_traj[q] * seg_len)])  # shape (4*N + 1,)
    y = np.concatenate([[0.0], np.cumsum(y_traj[q] * seg_len)])
    z = np.concatenate([np.zeros(2 * N + 1), np.full(2 * N, -z_thick)])
    

    #Insert the via point
    via = 2 * N + 1
    x_out = np.insert(x, via, x[via - 1])
    y_out = np.insert(y, via, y[via - 1])
    z_out = np.insert(z, via, -z_thick)
    
    #Move coil to centre
    x_out += l / 2
    y_out -= l / 2

    return np.stack([x_out, y_out, z_out], axis=-1)
