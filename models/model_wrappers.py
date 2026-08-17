import torch 
import numpy as np 
from models import lm_solve
from data.reparametrisation import normal_to_angles,angles_to_normal

class LMSolver: 
    '''
    Levenberg-Marquardt from a fixed initial guess.

    Parameters
    ---------- 
    f : callable 
        Function whose inverse is to be approximated
    x0 : np.ndarray 
        Initial guess in 6d form
    lm_kwargs
        Keyword arguments for LM solver
    '''

    def __init__(self,
                f : callable,
                x0 : np.ndarray,
                **lm_kwargs
                 ) -> None:
        
        self.f = f
        self.x0 = normal_to_angles(x0) 
        self.lm_kwargs = lm_kwargs

    def solve(self,
              measurements : np.ndarray 
              ) -> tuple[np.ndarray,np.ndarray]:
        n = len(measurements)

        pose_angle = np.empty((n,5))

        success = np.empty(n, dtype = bool)
        

        for i,m in enumerate(measurements):
            pose, _, suc = lm_solve(self.f,m,self.x0, **self.lm_kwargs)

            pose_angle[i] = pose
            success[i] = suc
    
        return angles_to_normal(pose_angle), success


class NNSolver: 
    '''
    Neural network solver. 

    Parameters 
    ---------- 
    net : torch.nn.Module 
        Trained network mapping 8 measurements to 6d pose
    '''

    def __init__(self,
                 net: torch.nn.Module, 
                 ) -> None:
        self.net = net.eval()

    def solve(self,
              measurements : np.ndarray
              ) -> tuple[np.ndarray,np.ndarray]:

        x = torch.as_tensor(measurements, dtype = torch.float32)

        with torch.no_grad():
            out = self.net(x).numpy().copy()

        n = out[...,3:]

        out[...,3:] = n / np.linalg.norm(n,axis=-1,keepdims=True)

        return out, np.ones(len(out), dtype = bool)


class HybridSolver:
    '''
    NN initialisation followed by LM refinement 

    Parameters 
    ---------- 
    net : torch.nn.Module 
        Trainied network mapping 8 measurements to 6d pose 
    f : callable 
        Funciton whose inverse is to be approximated 
    lm_kwargs
        Keyword arguments for LM solver

    '''
    
    def __init__(self, 
                 net: torch.nn.Module, 
                 f: callable, 
                 **lm_kwargs
                 ) -> None: 
        self.net = net.eval() 
        self.f = f 
        self.lm_kwargs = lm_kwargs 

    def solve(self,
              measurements : np.ndarray
              ) -> tuple[np.ndarray,np.ndarray]:
 

        x = torch.as_tensor(measurements, dtype = torch.float32)

        with torch.no_grad():
            out = self.net(x).numpy().copy()

        n = out[...,3:]

        out[...,3:] = n / np.linalg.norm(n,axis=-1,keepdims=True)
        
        l = len(measurements)
        success = np.empty(l, dtype = bool)
        pose_angle = np.empty((l,5))
        
        x0 = normal_to_angles(out)
        
        for i,m in enumerate(measurements):
            pose, _, suc = lm_solve(self.f,m,x0[i], **self.lm_kwargs)

            pose_angle[i] = pose
            success[i] = suc
    
        return angles_to_normal(pose_angle), success


