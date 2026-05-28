import numpy as np

def local_to_global(
    x_local : np.ndarray,
    d : np.ndarray,
    theta : float,
    ax : int
)   -> np.ndarray:
    '''Converts from local coordinate system to global coordinate system by applying a rotation then a translation. 

    Parameters
    ---------- 
    x_local : np.ndarray shape (...,3)
        Array containing data in local coordinates. Last index is always coordinates.
    d : np.ndarray shape (3,)
        Origin of local system in global system.
    theta : float
        Angle by which to rotate (radians)
    ax : int 
        Axis about which the rotation takes place (0,1,2) - (x,y,z) 

    Returns
    ------- 
    x_global : np.ndarray shape (...,3)
        Original x_local converted into global coordinate system 
    '''

    #ROTATION

    if ax not in (0,1,2): 
        raise ValueError("ax must be 0, 1, or 2, corresponding to the x, y, and z axes.")


    #Permutes the axes clockwise 
    d1 = int((ax + 1) % 3)
    d2 = int((ax + 2) % 3)
    

    x_global = x_local.copy()
    
    #Does the rotation using a rotation matrix in component form.
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    d1_tmp = cos_theta * x_global[...,d1] - sin_theta * x_global[...,d2]
    d2_tmp = sin_theta * x_global[...,d1] + cos_theta * x_global[...,d2]

    x_global[...,d1] = d1_tmp
    x_global[...,d2] = d2_tmp

    #TRANSLATION
    
    x_global = x_global + d

    return x_global

