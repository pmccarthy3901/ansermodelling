import numpy as np 
from .normal_vector import normal_vector
from .field_coil_calc import field_coil_calc 

def forward_model(
    PO : np.ndarray,
    coils : np.ndarray,
    calibration : np.ndarray = np.ones(8)
) -> np.ndarray:
    '''
    Computes 8 flux values given a coil geometry and pose.


    Parameters
    ---------- 
    PO : np.ndarray, shape (5,)
        Pose vector (x,y,z,theta,phi)
    coils : np.ndarray, shape(num_coils,num_vertices,3)
        Locations of coil vertices for system.
    calibration : np.ndarray 
        Calibration factor.
    
    Returns
    ------- 
    fluxes : np.ndarray, shape(8,)
        Fluxes due to each of the 8 coils at a given pose
    '''
    
    sensor_normal = normal_vector(PO)
    
    h = field_coil_calc(1,coils,PO[:3]) 

    fluxes = np.sum(sensor_normal * h, axis = -1)

    return fluxes
    
