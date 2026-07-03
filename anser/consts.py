import numpy as np

#COIL PARAMETERS

l = 70e-3 # Side length of outer square (m) 
w = 5e-4 # Width of each trace (m)
s = 2.5e-4 #Spacing between traces (m) 
z_thick = 1.6e-3 # Thickness of emitter coil PCB (m)
N_turns = 25 # Number of total turns (bottom and top) of each coil.
I = 1 # Current carried in the coil (A)

#COIL LOCATIONS

#centre of each coil 
centres = np.array([
    [-93.543,93.543,0],
    [0,68.55,0],
    [93.543,93.543,0],
    [-68.55,0,0],
    [68.55,0,0],
    [-93.543,-93.543,0],
    [0,-68.55,0],
    [93.543,-93.543,0]
]) * 1e-3

rotations = np.array([
    [2,np.pi/2],
    [2,np.pi/4],
    [2,np.pi/2],
    [2,np.pi/4],
    [2,np.pi/4],
    [2,np.pi/2],
    [2,np.pi/4],
    [2,np.pi/2]
])


calibration = np.ones(8) # flux calibration factor. Left as ones in placeholder

