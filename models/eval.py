from .train import pose_errors
import numpy as np
import torch
def report_error_stats(pred : np.ndarray,
         true : np.ndarray,
         success : np.ndarray,
         eps_x = 1.00,
         eps_n = 1.00
         ) -> None:

    ex, en = pose_errors(torch.as_tensor(pred,dtype=torch.float64), torch.as_tensor(true,dtype=torch.float64))
    ex, en = ex.numpy(), en.numpy()

    print(f"Mean pos error: {ex.mean():.3g}, mean angle error: {en.mean():.3g}")
        
    print(f"Median pos error: {np.median(ex):.3g}, Median angle error: {np.median(en):.3g}")

    print(f"95% pos error : {np.percentile(ex,95):.3g} 95% angle error: {np.percentile(en,95):.3g}")

    print(f"LM success rate: {success.mean():.3g}")

    converged = (ex < eps_x) & (en < eps_n)

    print(f"Convergence rate: {converged.mean():.3g}")


    ex_conv = ex[converged]
    en_conv = en[converged] 

    print(f"Mean pos error of converged: {ex_conv.mean():.3g}, mean angle error of converged: {en_conv.mean():.3g} ")

