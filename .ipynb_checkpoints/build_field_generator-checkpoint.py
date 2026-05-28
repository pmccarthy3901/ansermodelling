import numpy as np
from coil_dimension_calc import coil_dimension_calc
from local_to_global import local_to_global

def build_field_generator(
    N: int,
    l: float,
    w: float,
    s: float,
    z_thick: float,
    centres: np.ndarray,
    rotations: np.ndarray
) -> np.ndarray:
    '''
        Builds the coils and places them in the global coordinate system.

        Parameters
        ---------- 
        N: int 
            Number of turns total in each coil 
        l: float 
            Length of outer winding.
        w: float 
            Width of each trace.
        s: float 
            Edge to edge spacing between traces 
        z_thick : int 
            Thickness of the PCB 
        centres: np.ndarray, shape(num_coils,3) 
            Centres of each coil in the global system 
        rotations: np.ndarray, shape(num_coils,2)
            Rotation of each coil in the global system [ax,angle]
        
        Returns 
        ------- 
        coils_global: np.ndarray, shape(num_coils,num_vertices,3)
    '''
    
    #coil_local has shape (num_vertices,3)
    coil_local = coil_dimension_calc(N,l,w,s,z_thick)

    coils = []

    for i , pos in enumerate(centres):
        coils.append(local_to_global(coil_local,pos,rotations[i,1],rotations[i,0]))

    coils_global = np.stack(coils)

    return coils_global





