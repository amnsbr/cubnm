<div align="center">
<img src="./assets/logo_text.png" style="width:300px; margin:auto;">
</div>

The cuBNM toolbox performs efficient simulation and optimization of biophysical network models (BNM) of the brain on GPUs.

## Overview
The toolbox currently supports simulation of network nodes activity based on reduced Wong-Wang model with analytical-numerical feedback inhibition control. The Balloon-Windkessel model is used for calculation of simulated BOLD signals. It calculates goodness of fit of the simulated BOLD to the empirical BOLD based on functional connectivity (FC) and functional connectivity dynamics (FCD) matrices. The parameter optimization of the model can be done using grid search or evolutionary optimizers. Parallelization of the entire grid or each iteration of evolutionary optimizers is done at two levels: 1) simulations (across the GPU ‘blocks’), and 2) nodes (across each block’s ‘threads’). The toolbox additionally supports running the simulations on single- or multi-core CPUs, which will be used if no GPUs are detected or when requested by user (but the toolbox mainly focuses on GPUs).

This is a simplified flowchart of the different components of the program written in Python, C++ and CUDA.

![flowchart](./assets/flowchart_extended.png)

## Installation
```
git clone https://github.com/amnsbr/cuBNM.git
cd cuBNM && pip install .
```

Installation requires Python (tested with 3.9) and g++. Currently Windows, Mac and non-Nvidia GPUs are not supported. 

[GSL](https://www.gnu.org/software/gsl/) is another requirement which will be installed by the package (in `~/.cuBNM/gsl`) but it takes a rather long time and is only done if `libgsl.a` and `libgslcblas.a` are not found in `"/usr/lib", "/lib", "/usr/local/lib", $LIBRARY_PATH, $LD_LIBRARY_PATH`. If you have GSL on your system but it is installed elsewhere, please add the `libgsl.a` and `libgslcblas.a` directories to `$LIBRARY_PATH`.

## Usage
In `./examples/examples.py` you can find some examples of running a single simulation (`run_sims`), grid search (`run_grid`) or CMAES optimization (`run_cmaes_optimizer`). More comprehensive documentations and examples will be added.