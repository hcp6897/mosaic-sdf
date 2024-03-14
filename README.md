# MosaicSDF Implementation

## Introduction
Repository contains implementation of MosaicSDF shape representation algorhithm, as described in [Mosaic-SDF for 3D Generative Models](https://arxiv.org/abs/2312.09222) paper. 

> Mosaic-SDF (M-SDF): a simple 3D shape representation that approximates the Signed Distance Function (SDF) of a given shape by using a set of local grids spread near the shape's boundary. The M-SDF representation is fast to compute for each shape individually making it readily parallelizable; it is parameter efficient as it only covers the space around the shape's boundary; and it has a simple matrix form, compatible with Transformer-based architectures.

### What you will find in sources:
- MosaicSDF algorhithm to approximate SDF using local grids: `mosaic_sdf.py`
- Optimization of grids parameters - position and scale: `optimizer.py` 
- Parameters fine-tuning using `Ray Tune`: `run_tune.py`
- Interactive use of the algorhithm: `interactive_mosaic_sdf.ipynb`
- Helper tools:
    - Sampling SDFs
    - Visualizing SDFs (using marching-cubes) and meshes

### What is not implemented:
- Farthest point sampling
- Building generative model on top of SDFs

## Getting Started
### Prerequisites and Installation
- check `setup.sh` and `requirements.txt` for list of prerequisites and modules required.

## Results
Comparison of MosaicSDF(`mosaic_meshes`) with 256 mosaic cells to ground-truth sdf(`gt_sdf_mesh`) and ground-truth mesh (`gt_mesh`):

