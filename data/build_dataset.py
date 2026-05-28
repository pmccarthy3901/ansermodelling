from anser import *
import numpy as np

def build_dataset(
        N : int = 10000,
        bounds : np.ndarray = np.array([
            [-0.15,0.15],
            [-0.15,0.15],
            [0,0.5],
            [-np.pi,np.pi],
            [0,np.pi/2]
        ]),
        seed : int = 1
        ) -> tuple:
    '''
    Build training dataset using data generated from anser simulation.

    Parameters
    ----------
    N : int 
        Number of samples 
    bounds : np.ndarray shape(5,2)
        Bounds for x,y,z,theta,phi coordinates of samples.
    seed : int 
        Seed for RNG

    Returns 
    -------
    x_train, y_train : np.ndarray shape (N,num_coils,3), np.ndarray shape (N,5)
    '''


    coils_global = build_field_generator(N_turns,l,w,s,z_thick,centres,rotations)
    
    rng = np.random.default_rng(seed)

    x = rng.uniform(bounds[0,0],bounds[0,1],N)
    y = rng.uniform(bounds[1,0],bounds[1,1],N)
    z = rng.uniform(bounds[2,0],bounds[2,1],N)
    theta = rng.uniform(bounds[3,0],bounds[3,1],N)
    phi = rng.uniform(bounds[4,0],bounds[4,1],N)

    y_train = np.stack((x,y,z,theta,phi), axis = -1)
    
    x_train = np.array([field_coil_calc(I,coils_global,y_train[i,:3]) for i in range(N)])
    
    return x_train, y_train


if __name__ == "__main__":
    x_train , y_train = build_dataset()

    np.savez("data/dataset.npz", x_train=x_train, y_train=y_train)
    print(f"Saved dataset: x_train {x_train.shape}, y_train {y_train.shape} to data/dataset.npz")
