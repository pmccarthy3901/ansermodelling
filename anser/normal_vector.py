import numpy as np 

def normal_vector(PO: np.ndarray) -> np.ndarray:
    ''' Generates a normal vector from a position and orientation vector 

        Parameters
        ---------- 
        PO : np.ndarray, shape(5,)
            Position and orientation vector [x,y,z,theta,phi] 

        Returns 
        ------- 
        n : np.ndarray, shape(3,)
            Normal vector for the orientation [x,y,z]
    '''

    theta = PO[3]
    phi = PO[4]

    n = np.array([np.sin(theta)*np.cos(phi),
                  np.sin(theta)*np.sin(phi),
                  np.cos(theta)])

    return n 

