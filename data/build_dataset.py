from anser import *
import numpy as np

def build_dataset(
        N : int = 100000,
        bounds : np.ndarray = np.array([
            [-0.125,0.125],
            [-0.125,0.125],
            [0.0016,0.25],
            [-1,1],
            [0,2*np.pi]
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
    x_train, y_train : np.ndarray shape (N,num_coils), np.ndarray shape (N,6)
    '''


    coils_global = build_field_generator(N_turns,l,w,s,z_thick,centres,rotations)
    
    rng = np.random.default_rng(seed)

    x = rng.uniform(bounds[0,0],bounds[0,1],N)
    y = rng.uniform(bounds[1,0],bounds[1,1],N)
    z = rng.uniform(bounds[2,0],bounds[2,1],N)
    cos_theta = rng.uniform(bounds[3,0],bounds[3,1],N)
    phi = rng.uniform(bounds[4,0],bounds[4,1],N)

    sin_theta = np.sqrt(1 - cos_theta**2)
    #shape (N,3)
    sensor_normal = np.stack([sin_theta*np.cos(phi),
                              sin_theta*np.sin(phi),
                              cos_theta], axis = -1) 
   

    #shape (N,6)
    pos = np.stack([x,y,z], axis = -1)
    ys = np.concatenate([pos,sensor_normal], axis = -1)

    #shape (N,8,3)
    h = np.array([field_coil_calc(I,coils_global,ys[i,:3]) for i in range(N)])
    
    xs = np.sum(sensor_normal[:,None,:] * h, axis = -1)

    return xs, ys



