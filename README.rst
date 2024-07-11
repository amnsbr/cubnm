.. raw:: html

    <div align="center">
    <img 
        src="https://raw.githubusercontent.com/amnsbr/cubnm/main/docs/_static/logo_text.png" 
        style="width:300px; margin:auto; padding-bottom:10px;" alt="cuBNM logo"
    >
    </div>

The cuBNM toolbox is designed for efficient biophysical network modeling of 
the brain on GPUs and CPUs.

Overview
--------
The toolbox simulates neuronal activity of network nodes (neural mass models) 
which are connected through the structural connectome using GPUs/CPUs. 
Currently three models (`rWW`, `rWWEx` and `Kuramoto`) are implemented, but the
modular design of the code makes it possible to include additional models in 
future. The simulated activity of model neurons is fed into the Balloon-Windkessel
model to calculate simulated BOLD signal. Functional connectivity (FC) and 
functional connectivity dynamics (FCD) from the simulated BOLD signal are 
calculated efficiently on GPUs/CPUs and compared to FC and FCD matrices 
derived from empirical BOLD signals to assess similarity (goodness-of-fit) 
of the simulated to empirical BOLD signal.

The toolbox supports parameter optimization algorithms including grid search and
evolutionary optimizers (via `pymoo`), such as the covariance matrix adaptation-evolution 
strategy (CMA-ES). Parallelization within the grid or the iterations of 
evolutionary optimization is done at the level of simulations (across the GPU
‘blocks’), and nodes (across each block’s ‘threads’). The models can incorporate 
global or regional free parameters that are fit to empirical data using the 
provided optimization algorithms. Regional parameters can be homogeneous or vary
across nodes based on a parameterized combination of fixed maps or independent 
free parameters for each node or group of nodes.

The primary focus of the toolbox is on GPU usage but it also supports running the
simulations on single or multiple cores of CPU. CPUs will be used if no GPUs are
detected or if requested by the user.

The core of the toolbox runs highly parallelized simulations using C++/CUDA, while the 
user interface, written in Python, allows for user control over simulation 
configurations:

.. image:: https://raw.githubusercontent.com/amnsbr/cubnm/main/docs/_static/flowchart_extended.png
    :alt: Flowchart of the cuBNM program

.. overview-end

Documentation
-------------
Please find the documentations on installation, usage examples and API at 
https://cubnm.readthedocs.io.