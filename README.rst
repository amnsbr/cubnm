.. image:: https://raw.githubusercontent.com/amnsbr/cubnm/main/docs/_static/logo_text.png
    :align: center
    :width: 350px

.. badges-start

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.12097797.svg
  :target: https://zenodo.org/doi/10.5281/zenodo.12097797

.. image:: https://img.shields.io/pypi/v/cubnm
  :target: https://pypi.org/project/cubnm/

.. image:: https://img.shields.io/readthedocs/cubnm
  :target: https://cubnm.readthedocs.io

.. image:: https://img.shields.io/badge/docker-amnsbr/cubnm-blue.svg?logo=docker
  :target: https://hub.docker.com/r/amnsbr/cubnm

.. image:: https://app.codacy.com/project/badge/Grade/e1af99e878bc4dbf9525d8eb610e0026
  :target: https://app.codacy.com/gh/amnsbr/cubnm/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade

.. image:: https://img.shields.io/github/license/amnsbr/cubnm
  :target: ./LICENSE

.. badges-end

cuBNM toolbox is designed for efficient brain network modeling on GPUs. 

Documentation can be found `here <https://cubnm.readthedocs.io>`_. 
Read the preprint `on bioRxiv <https://www.biorxiv.org/content/10.1101/2025.11.13.688224v1>`_.

.. overview-start

Overview
--------
cuBNM toolbox uses GPUs to efficiently run simulations of brain network models 
consisting of nodes which are connected through a connectome, 
and fit them to empirical neuroimaging data through integrated optimization algorithms.

GPU parallelization enables massive scaling of the simulations into higher number of
simulations and nodes. Below you can see how computing time varies
as a function of number of simulations and nodes on GPUs versus CPUs. For example,
running 32,768 simulations (duration: 60s, nodes: 100) would take 3.8 days on a single
CPU thread, but only 5.6 minutes on Nvidia A100 GPU, and 21.8 minutes on Nvidia 
GeForce RTX 4080 Super:

.. image:: https://raw.githubusercontent.com/amnsbr/cubnm/main/docs/_static/scaling.png
    :alt: Scaling plots

GPU usage is the primary focus of the toolbox but it also supports running the
simulations on single or multiple cores of CPU. CPUs will be used if no GPUs are
detected or if requested by the user.

Several commonly used models (e.g., reduced Wong-Wang, Jansen-Rit, Kuramoto, Wilson-Cowan) 
are implemented, and new models can be added via YAML definition files. A guide is included
on the structure of model definition YAML files to help researchers implement custom models. 

The simulated activity of model neurons is fed into the Balloon-Windkessel model 
to calculate simulated BOLD signal. Functional connectivity (FC) and  functional 
connectivity dynamics (FCD) from the simulated BOLD signal are calculated efficiently 
on GPUs/CPUs and compared to FC and FCD matrices derived from empirical BOLD signals 
to assess similarity (goodness-of-fit) of the simulated to empirical BOLD signal.
The toolbox supports parameter optimization algorithms including grid search and
evolutionary optimizers (via ``pymoo``), such as the covariance matrix adaptation-evolution 
strategy (CMA-ES). Parallelization within the grid or the iterations of 
evolutionary optimization is done at the level of simulations (across the GPU
‘blocks’), and nodes (across each block’s ‘threads’). The models can incorporate 
global or regional free parameters that are fit to empirical data using the 
provided optimization algorithms. Regional parameters can be homogeneous or vary
across nodes based on a parameterized combination of fixed maps or independent 
free parameters for each node or group of nodes.

At its core the toolbox runs highly parallelized simulations using C++/CUDA, while the 
user interface is written in Python and allows for user control over simulation 
configurations:

.. image:: https://raw.githubusercontent.com/amnsbr/cubnm/main/docs/_static/flowchart.png
    :alt: Flowchart of the cuBNM program

Citation
--------
If you use cuBNM in your work, please cite:

    Saberi et al., cuBNM: GPU-Accelerated Brain Network Modeling. bioRxiv (2025) [`link <https://www.biorxiv.org/content/10.1101/2025.11.13.688224v1>`_].

In addition, please cite the original papers for the BNMs and optimization algorithms you use.

.. overview-end

