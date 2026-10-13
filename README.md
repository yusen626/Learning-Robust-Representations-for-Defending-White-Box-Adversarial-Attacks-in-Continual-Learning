# Learning-Robust-Representations-for-Defending-White-Box-Adversarial-Attacks-in-Continual-Learning
We have released the complete training scripts and model code for LRRF, the method proposed in our paper, *Learning Robust Representations for Defending White-Box Adversarial Attacks in Continual Learning*.

**Our training scripts include the complete training and evaluation procedures, including how the average accuracy reported in the main paper is evaluated and calculated. Our model code also provides a detailed implementation of all mechanisms of LRRF.**

## Dependencies

The reference Conda environment uses **Python 3.10.19** on **Linux**. The following tables summarize selected packages recorded in this environment.

These versions describe the recorded software environment. They do not constitute a complete environment lockfile or a validated minimal dependency specification.

### Deep Learning and Scientific Computing

| Package | Recorded Version |
| --- | --- |
| Python | 3.10.19 |
| torch | 2.10.0 |
| torchvision | 0.25.0 |
| timm | 0.9.16 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| pillow | 12.3.0 |
| einops | 0.8.1 |
| kornia | 0.8.2 |

### Adversarial Robustness and Continual Learning

| Package | Recorded Version |
| --- | --- |
| torchattacks | 3.5.1 |
| autoattack | 0.1 |
| pyautoattack | 0.2.0 |
| robustbench | 1.1.1 |
| continuum | 1.2.7 |

### Additional Model and Training Packages

| Package | Recorded Version |
| --- | --- |
| transformers | 4.53.0 |
| datasets | 2.14.6 |
| accelerate | 1.15.0 |
| clip | 1.0 |
| open-clip-torch | 3.3.0 |
| pytorch-lightning | 2.6.6 |
| torchmetrics | 1.8.2 |
| ema-pytorch | 0.7.9 |
| pot | 0.9.6.post1 |
| geomloss | 0.3.1 |
| geotorch | 0.3.0 |
| quadprog | 0.1.13 |

### Configuration, Logging, and Visualization

| Package | Recorded Version |
| --- | --- |
| pyyaml | 6.0.3 |
| hydra-core | 1.3.2 |
| omegaconf | 2.3.0 |
| easydict | 1.13 |
| yacs | 0.1.8 |
| tensorboard | 2.20.0 |
| wandb | 0.27.0 |
| tqdm | 4.67.1 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

### CUDA Environment

The environment snapshot includes the following GPU-related packages:

| Package | Recorded Version |
| --- | --- |
| cuda-toolkit | 13.0.2 |
| cuda-bindings | 12.9.4 |
| nvidia-cuda-runtime-cu12 | 12.8.90 |
| nvidia-cudnn-cu12 | 9.10.2.21 |

The snapshot contains both CUDA 12 and CUDA 13 components. These package versions alone do not establish the CUDA version used by the installed PyTorch build. In particular, the presence of `cuda-toolkit==13.0.2` should not be interpreted as a confirmed CUDA 13 requirement.

To inspect the active PyTorch installation, run:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('cuDNN:', torch.backends.cudnn.version())"
```

### Environment Notes

- The tables summarize selected packages and omit many system libraries and transitive dependencies.
- The snapshot also lists `torchaudio==2.11.0` alongside `torch==2.10.0`. This combination should be checked before including TorchAudio in installation requirements.
- Both `opencv-python==4.13.0.90` and `opencv-python-headless==5.0.0.93` are recorded. Their coexistence should not be treated as a requirement to install both packages.
- The snapshot lists the pip package `matplotlib==3.10.8` and the Conda package `matplotlib-base==3.10.9`. Verify the version imported at runtime before finalizing a reproducible environment specification.
- Package names and versions alone do not identify the source repository or commit of source-installed packages such as `autoattack` and `clip`.
