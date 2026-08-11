import numpy as np 
from data import build_dataset
if __name__ == "__main__":
    xs , ys = build_dataset(N = 100000, seed = 1)

    np.savez("data/dataset.npz", xs=xs, ys=ys)
    print(f"Saved dataset: xs {xs.shape}, ys {ys.shape} to data/dataset.npz")
