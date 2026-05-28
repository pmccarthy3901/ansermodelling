# Anser EMT — Python Implementation

A Python reimplementation of the [Anser EMT](https://osf.io/47q8q/) electromagnetic tracking system, originally written in MATLAB by Alex Jaeger and Kilian O'Donoghue at UCC.

## Overview

Anser EMT is an open-source platform for tracking induction coil sensors in 3D space, designed for image-guided medical interventions. This Python port covers the core simulation pipeline: coil geometry generation, Biot-Savart forward modelling, and 5-DoF pose solving.

## Structure

- `anser/` — Core modules (coil geometry, forward model, solver, coordinate transforms)
- `demos/` — Jupyter notebooks demonstrating the pipeline

## Dependencies

- numpy
- scipy
- matplotlib

## References

- Jaeger et al. (2017), "Anser EMT: the first open-source electromagnetic tracking platform for image-guided interventions"
- Jaeger & Cantillon-Murphy (2019), "Electromagnetic Tracking Using Modular Tiled Field Generators"
