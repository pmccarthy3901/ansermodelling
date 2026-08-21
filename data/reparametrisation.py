import numpy as np 
import torch

def normal_to_angles(pose_normal : np.ndarray) -> np.ndarray:
    '''
    Converts (x,y,z,n_1,n_2,n_3) to (x,y,z,theta,phi).

    Parameters
    ---------- 
    pose_normal : np.ndarray, shape(...,6)
        Pose in form (x,y,z,n_1,n_2,n_3)

    Returns
    ------- 
    pose_angles : np.ndarray, shape(...,5)
        Pose in form (x,y,z,theta,phi)
    '''

    pose_angles = np.empty(pose_normal.shape[:-1] + (5,), dtype = float)
    pose_angles[...,:3] = pose_normal[...,:3]

    n = pose_normal[...,3:]

    n =  n / np.linalg.norm(n,axis=-1, keepdims = True)

    pose_angles[...,3] = np.arccos(np.clip(n[...,2], -1.0, 1.0))

    pose_angles[...,4] = np.mod(np.arctan2(n[...,1],n[...,0]), 2.0 * np.pi)

    return pose_angles 

def angles_to_normal(pose_angles : np.ndarray) -> np.ndarray:
    '''
    Converts pose in form (x,y,z,theta,phi) to pose in form (x,y,z,n_1,n_2,n_3)

    Parameters 
    ---------- 
    pose_angles : np.ndarray, shape(...,5)
        Pose in form (x,y,z,theta,phi)

    Returns 
    ------- 
    pose_normal : np.ndarray, shape(...,6)
        Pose in form (x,y,z,n_1,n_2,n_3)
    ''' 

    pose_normal = np.empty(pose_angles.shape[:-1] + (6,), dtype = float)
    
    pose_normal[...,:3] = pose_angles[...,:3]

    theta = pose_angles[...,3]
    phi = pose_angles[...,4]
    
    st,ct,sp,cp = np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)

    pose_normal[...,3] = st * cp 
    pose_normal[...,4] = st * sp
    pose_normal[...,5] = ct


    return pose_normal

def normal_to_angles_t(pose_normal: torch.Tensor) -> torch.Tensor:
    '''
    Converts (x,y,z,n_1,n_2,n_3) to (x,y,z,theta,phi).

    Torch equivalent of normal_to_angles, for use inside the training
    loop where targets must stay on the autograd graph.

    Parameters
    ----------
    pose_normal : torch.Tensor, shape(...,6)
        Pose in form (x,y,z,n_1,n_2,n_3)

    Returns
    -------
    pose_angles : torch.Tensor, shape(...,5)
        Pose in form (x,y,z,theta,phi), phi wrapped to [0, 2*pi)
    '''
    n = pose_normal[..., 3:]
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)

    theta = torch.arccos(n[..., 2].clamp(-1.0, 1.0))
    phi = torch.remainder(torch.atan2(n[..., 1], n[..., 0]), 2.0 * torch.pi)

    return torch.cat([pose_normal[..., :3],
                      theta.unsqueeze(-1),
                      phi.unsqueeze(-1)], dim=-1)


def angles_to_normal_t(pose_angles: torch.Tensor) -> torch.Tensor:
    '''
    Converts (x,y,z,theta,phi) to (x,y,z,n_1,n_2,n_3).

    Torch equivalent of angles_to_normal.

    Parameters
    ----------
    pose_angles : torch.Tensor, shape(...,5)
        Pose in form (x,y,z,theta,phi)

    Returns
    -------
    pose_normal : torch.Tensor, shape(...,6)
        Pose in form (x,y,z,n_1,n_2,n_3)
    '''
    theta = pose_angles[..., 3]
    phi = pose_angles[..., 4]

    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)

    return torch.cat([pose_angles[..., :3],
                      (st * cp).unsqueeze(-1),
                      (st * sp).unsqueeze(-1),
                      ct.unsqueeze(-1)], dim=-1)
