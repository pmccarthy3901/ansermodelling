# Pose Solving Algorithms for Electromagnetic Tracking.


This repo contains a Python reimplementation of the [Anser EMT](https://osf.io/47q8q/) electromagnetic tracking system, originally written in MATLAB. It also contains a number of pose solving algorithms, including a neural network initialised Levenberg-Marquardt solver.

## Overview

Anser EMT is an open-source platform for tracking induction coil sensors in 3D space, designed for image-guided medical interventions. This Python port covers the core simulation pipeline: coil geometry generation, Biot-Savart forward modelling, and 5-DoF pose solving.

## Result 

Levenberg-Marquardt solves this problem to machine precision when it converges, but from a cold start it lands in the wrong basin of attraction on roughly 14% of poses. A neural network initialiser predicts pose to approximately 9mm error - not useful alone but when used to initialise a Levernberg-Marqurdt solver, reduces failed solves by almost a factor of three.

| Solver | Convergence rate | 95th percentile position error |
|--------|-----------------|------------------------|
| LM (cold start) | 85.7% | 61.8 mm |
| Network only | — | 20.6 mm |
| Hybrid | 94.4% | 1.67 mm |


## Structure


- `anser/` - Core modules for simulation.
- `report/` - Report chapters.
- `demos/` - Jupyter notebooks demonstrating the pipeline
- `data/` - Dataset generation and PyTorch Loading.
- `models/` - Contains solver models and training pipeline.
- `scripts/` - Entry points.
## References

- Jaeger et al. (2017), "Anser EMT: the first open-source electromagnetic tracking platform for image-guided interventions"
