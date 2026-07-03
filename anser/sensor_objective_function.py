import numpy as np
from .field_coil_calc import field_coil_calc
from .normal_vector import normal_vector 

def sensor_objective_function(
    current_PO : np.ndarray,
    flux_real : np.ndarray,
    coil_position : np.ndarray, 
    calibration: np.ndarray,
) -> np.ndarray: 
    '''
        Calculates the loss of the model comparing the model's flux to the real value.

        Parameters
        ---------- 
        current_PO: np.ndarray, shape(5,)
            Array containing current x,y,z,theta,phi of the sensor.
        flux_real: np.ndarry, shape(num_coils,)
            The real measured flux at the sensor.
        coil_position: np.ndarray, shape(num_coils,num_vertices,3)
            Position of the coil (x,y,z)
        calibration: np.ndarray, shape(num_coils,)
            Calibration constant

        Returns 
        ------- 
        loss: np.ndarray, shape(num_coils,) 
            The difference between the measured and calculated flux for each coil.


    '''
    x,y,z,theta,phi = current_PO 
    

    sensor_normal = normal_vector(current_PO)

    h = field_coil_calc(1,coil_position,np.array([x,y,z]))

    flux_model = calibration * np.sum(sensor_normal*h, axis=-1)

    return flux_model - flux_real
