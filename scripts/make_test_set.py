from data import build_dataset
import numpy as np
if __name__ == "__main__":
    xs,ys = build_dataset(N = 20000, seed = 2026)

    np.savez("data/test_set.npz", xs = xs, ys = ys)
    print(f"Saved dataset: xs {xs.shape}, ys {ys.shape} to data/dataset.npz")
