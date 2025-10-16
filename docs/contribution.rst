Contributing a New Model
########################

Overview
********

cuBNM uses a YAML-based code generation system to automatically create C++/CUDA 
and Python implementations from model specifications. This guide explains how 
to contribute a new model by creating a YAML specification file that defines 
your model's equations, parameters, and configuration.

All model implementations (C++, CUDA, and Python) are automatically generated 
from a single YAML file during the build process. This ensures consistency 
across CPU and GPU implementations and significantly reduces code duplication.

.. note::
    Runtime definition of new models (i.e., without modifying source code) is not currently supported.
    It may be considered in future versions.


Quick Start
***********

To contribute a new model:

1. Create a YAML file in ``codegen/recipes/`` (e.g., ``mymodel.yaml``)
2. Define your model following the structure described below
3. Generate code for all models by running ``python codegen/generate_models.py``
4. Build the toolbox from source
5. Run tests
6. Make a pull request

Example Model Description
**************************

To clarify the core components of the YAML file and its structure, we use the reduced 
Wong-Wang model with excitatory populations based on 
`Deco et al. 2013 <https://doi.org/10.1523/JNEUROSCI.1091-13.2013>`_ as an example.
We abbreviate this model as ``rWWEx``.

This model is described by the following differential equations:

.. math::

    x_i = w_i J_N S_i + G J_N \sum_j C_{ij} S_j + I_{i0}

    r_i = \frac{a x_i - b}{1 - \exp(-d(a x_i - b))}

    \dot{S_i} = -\frac{S_i}{\tau_s} + (1 - S_i) \gamma r_i + {\sigma}_i v_i(t)
    

There are 3 state variables for each node: :math:`x_i` (total input current), 
:math:`r_i` (firing rate), and :math:`S_i` (synaptic gating variable). Model parameters include
:math:`G` (global coupling) and regional parameters :math:`w_i` (recurrent excitation), :math:`I_{i0}` 
(external input current), and :math:`{\sigma}_i` (noise amplitude). The structural connectivity (SC) matrix is
denoted by :math:`C_{ij}`. The model also includes fixed constants :math:`a`, :math:`b`, :math:`d`,
:math:`\tau_s`, and :math:`J_N`. 

This is a "conduction-based" model where the input from other nodes 
is additive (in contrast to "oscillatory" models such as Kuramoto where phase differences are used).
Note that :math:`C_{ij}` is multiplied by :math:`S_j`, making :math:`S` the ``conn_state_var`` - 
the state variable representing the outgoing signal from nodes to their connected targets.
:math:`S_i` is also the state variable representing node activity that is fed into the Balloon-Windkessel
model to generate simulated BOLD signals, making it the ``bold_state_var`` as well.

Now, let's implement this model in the toolbox by creating a YAML file.

YAML File Structure
*******************

A model specification YAML file contains the following main sections:

Basic Information
=================

This section includes the model's short and full name, as well as citations (optional).

.. code-block:: yaml

    model_name: rWWEx
    full_name: reduced Wong Wang (excitatory)
    citations:
      - Deco et al. 2013 Journal of Neuroscience (10.1523/JNEUROSCI.1091-13.2013)

Model Equations
===============

This section defines the model equations as pseudo-code using a syntax similar to Python.
Equations are based on variables, constants, and config parameters defined in later sections (see below), 
in addition to the special variable ``globalinput`` which represents the summed input from other nodes.

Equations must be defined for simulation initialization (``init_equations``) and for each
integration step (``step_equations``).

Initialization equations must at least include initialization of state variables that will be incrementally
updated through time steps (i.e., variables whose derivatives appear in the model equations).
For ``rWWEx``, only ``S`` needs initialization, as ``x`` and ``r`` are calculated
independently in each time step.

.. code-block:: yaml

    init_equations: |
      S = 0.001

Step equations must include the full set of model equations and must update all state variables
that are incrementally modified through time steps. The step equations for ``rWWEx`` are as follows 
(comments show the corresponding mathematical equations):

.. code-block:: yaml

    step_equations: |
      # Eq. 1
      x = w * J_N * S + G * J_N * globalinput + I0
      
      # Eq. 2
      axb = a * x - b
      r = axb / (1 - exp(-d * axb))
      
      # Eq. 3
      dSdt = dt_gamma * ((1 - S) * r) - dt_itau * S + noise * sqrt_dt * sigma
      S += dSdt
      # Clip S to valid range [0, 1]
      S = max(0.0, min(1.0, S))

On the left-hand side of equations, state variables and their updates are defined. We also
define ``intermediate`` variables such as ``axb`` and ``dSdt``, which are not state variables but
are useful for breaking down complex calculations into simpler steps and avoiding repeated computations
(thereby improving efficiency). On the right-hand side, we can use state variables,
intermediate variables, noise variables, model parameters, constants, config variables (see below), and the
special variable ``globalinput``. All common arithmetic operations as well as most math functions
(e.g., ``exp``, ``sin``, ``cos``, ``log``) are supported.

In addition to the required ``init_equations`` and ``step_equations``, there is an optional ``restart_equations``
section which, if not defined, defaults to ``init_equations``. This section is used
when simulations are restarted, which currently only occurs for the ``rWW`` model when numerical FIC is enabled
(see ``codegen/recipes/rww.yaml`` for details). Furthermore, optional hooks are available for ``post_bw_step``
and ``post_integration`` which can be defined as C++/CUDA code if needed, and are currently only used by the ``rWW`` model 
(see :ref:`advanced configurations<advanced>` below).

Connectivity Between Nodes
===========================

The ``globalinput`` variable used in step equations represents the summed input from other nodes.
We must define: (1) which state variable of the source nodes is used as the outgoing signal 
(``conn_state_var``), and (2) how inputs from other nodes are integrated.

Currently, two integration modes are available: "additive" (for conduction-based models like ``rWWEx``) and
"phase difference" (for oscillatory models like ``Kuramoto``). The default is "additive" unless
``is_osc`` is set to ``true``. For ``rWWEx``:

.. code-block:: yaml

    conn_state_var: S
    is_osc: false

This means that the state variable ``S`` of source nodes is used as the outgoing signal, following
the equation :math:`\sum_j C_{ij} S_j`, where :math:`C_{ij}` is the SC matrix. Note that delays
may be introduced, in which case ``conn_state_var`` is accessed from previous time points
based on the conduction delay between nodes. Conduction delay needs not to be explicitly defined,
as it is handled automatically (and when requested by user) in the toolbox framework.

If a combination of state variables is used as the connectivity signal, create a new state variable 
in ``step_equations`` that defines this combination, then set ``conn_state_var`` to this new variable 
(e.g., see ``codegen/recipes/jr.yaml`` for an example).

BOLD Simulation
===============

Simulated BOLD signals are generated by feeding a state variable into the
Balloon-Windkessel model. We must specify which state variable is used (``bold_state_var``). 
For ``rWWEx``, ``S`` is used:

.. code-block:: yaml

    bold_state_var: S

If a combination of state variables should be used as input to the Balloon-Windkessel model,
create a new state variable in ``step_equations`` that defines this combination, then set ``bold_state_var`` to this new variable.

Model Variables, Constants, and Config
=======================================

Each variable used in model equations (except ``globalinput``) must be defined in the
``variables``, ``constants``, or ``config`` sections.

Variables
---------

State variables, intermediate variables, parameters, noise terms, and additional variables (if any) 
are defined in the ``variables`` section. Each variable must have a ``name`` and ``type``. 
The name must match exactly the name used in model equations.
Type must be one of the following:

- ``state_var``: State variables (``x``, ``r``, and ``S`` in ``rWWEx``)
- ``intermediate_var``: Intermediate variables for computational efficiency. Note that intermediate
  variables are not saved in outputs and cannot be used as ``conn_state_var`` or ``bold_state_var``.
- ``global_param``: Global parameters, usually only the global coupling ``G``. These are
  parameters specific to a simulation but not defined per node.
- ``regional_param``: Regional parameters (``w``, ``I0``, and ``sigma`` in ``rWWEx``). These are
  node-specific parameters.
- ``noise``: Noise terms (:math:`v_i(t)` in ``rWWEx``). These are random samples from a standard Gaussian
  distribution (mean 0, standard deviation 1), unique to each node and time point.
  Models may have multiple noise terms per node (e.g., ``rWW`` has separate noise for excitatory and inhibitory populations).
  The name of noise variables need not be ``noise``.
- Additional types including ``ext_bool_shared``, ``ext_int_shared``, ``global_out_bool``,
  and ``global_out_int`` are available for advanced use cases (see ``codegen/recipes/rww.yaml``).

All variables are assumed to be double-precision floating point numbers.

Each variable may include a ``description`` field (optional but recommended) and a ``value`` field 
(for parameters with default values).

The variables section for ``rWWEx`` is:

.. code-block:: yaml

    variables:
      - name: x
        type: state_var
        description: total input current
      - name: r
        type: state_var
        description: firing rate
      - name: S
        type: state_var
        description: synaptic gating variable
      - name: axb
        type: intermediate_var
        description: a*x - b
      - name: dSdt
        type: intermediate_var
        description: derivative of S
      - name: G
        type: global_param
        description: global coupling strength
        # no default value
      - name: w
        type: regional_param
        description: local excitatory recurrence
        value: 0.9
      - name: I0
        type: regional_param
        description: external input current
        value: 0.3
      - name: sigma
        type: regional_param
        description: noise amplitude
        value: 0.001
      - name: noise
        type: noise
        description: noise for S

Constants
---------

Constants are fixed values used in model equations. They are defined in the ``constants`` section.
Each constant must have a ``type``, ``name``, and ``value``, plus optional ``description``. 
The name must match exactly the name used in model equations. Type refers to the data type
(``double``, ``int``, or ``bool``). 

Constant values can be defined directly or as functions of other constants. 
When defining derived constants, other constants must be prefixed with ``mc.`` (model constants). 
For example, ``sqrt_dt`` is defined as ``sqrt(mc.dt)``. Usual arithmetic operations and math functions
can be used (e.g., ``sqrt``, ``exp``, ``log``, ``sin``, ``cos``).
Derived constants improve performance by computing fixed values once at initialization
rather than recalculating them every time step.

The constants section for ``rWWEx`` is:

.. code-block:: yaml

    constants:
      - type: double
        name: dt
        value: dt
        description: integration step
      - type: double
        name: sqrt_dt
        value: sqrt(mc.dt)
        description: square root of integration step
      - type: double
        name: J_N
        value: "0.2609"
        description: synaptic coupling
      - type: double
        name: a
        value: "270"
        description: input-output function parameter a
      - type: double
        name: b
        value: "108"
        description: input-output function parameter b
      - type: double
        name: d
        value: "0.154"
        description: input-output function parameter d
      - type: double
        name: gamma
        value: "(double)0.641/(double)1000.0"
        description: kinetic parameter
      - type: double
        name: tau
        value: "100"
        description: synaptic time constant
      - type: double
        name: itau
        value: 1.0/mc.tau
        description: inverse of tau (1/tau)
      - type: double
        name: dt_itau
        value: mc.dt * mc.itau
        description: dt / tau
      - type: double
        name: dt_gamma
        value: mc.dt * mc.gamma
        description: dt * gamma

Config
------

Config variables define model-specific configurations that do not change during
simulations but are modifiable by the user within ``<Model>SimGroup`` in Python.
Each config variable must have ``type``, ``name``, ``value``, and ``description``. 
Derived values can also be defined but are currently not supported by the auto-generated
``set_conf()`` method, so they must be set manually in custom code (see ``codegen/recipes/rww.yaml``).

The ``rWWEx`` model does not have model-specific configurations. For an example of config usage,
see the ``rWW`` model (``codegen/recipes/rww.yaml``), which includes configurations for
feedback inhibition control (FIC):

.. code-block:: yaml

    config:
      - type: bool
        name: do_fic
        value: "true"
        description: whether to apply feedback inhibition control
      - type: int
        name: max_fic_trials
        value: 0
        description: maximum number of numerical FIC trials. If set to 0, FIC will be done only analytically
      - type: int
        name: I_SAMPLING_START
        value: 1000
        description: starting time of numerical FIC I_E sampling (msec)
      - type: int
        name: I_SAMPLING_END
        value: 10000
        description: end time of numerical FIC I_E sampling (msec)
      - type: int
        name: I_SAMPLING_DURATION
        value: "I_SAMPLING_END - I_SAMPLING_START + 1"
        description: duration of numerical FIC I_E sampling (iterations)
      - type: double
        name: init_delta
        value: 0.02
        description: initial delta for numerical FIC adjustment
      - type: double # only used in Python code
        name: fic_penalty_scale
        value: 0.5
        description: how much deviation from FIC target mean rE of 3 Hz is penalized. Set to 0 to disable FIC penalty.

.. _advanced:

Advanced Configurations and Custom Code
========================================
Some models require additional advanced configurations or custom code snippets.
None of these are needed for ``rWWEx``, but they are used in the ``rWW`` model
(see ``codegen/recipes/rww.yaml``), primarily to implement analytical and numerical 
feedback inhibition control (FIC). These can be defined in the following optional sections:

- ``custom_methods``: Custom C++/CUDA methods. Can include:

  - ``set_conf``: Code to set model-specific configurations. If defined, replaces
    the auto-generated ``set_conf()`` method.
  - ``prep_params``: Code to prepare or modify parameters before simulations.
  - ``post_bw_step``: Code executed after each Balloon-Windkessel step on GPU.
  - ``_j_post_bw_step`` and ``h_post_bw_step``: CPU equivalents of ``post_bw_step``,
    including per-node operations (``_j_post_bw_step``) and across-nodes operations (``h_post_bw_step``).
  - ``post_integration``: Code executed after integration completes on GPU.
  - ``h_post_integration``: CPU equivalent of ``post_integration``.

- ``has_prep_params``: Indicates whether the model has a ``prep_params`` custom method.
- ``has_post_bw_step``: Indicates whether the model has ``post_bw_step``, ``_j_post_bw_step``
  and ``h_post_bw_step`` custom methods.
- ``has_post_integration``: Indicates whether the model has ``post_integration`` and
  ``h_post_integration`` custom methods.
- ``external_declarations``: C++ declarations of external functions used in custom methods.
- ``cpp_includes``: Additional C++ headers to include in generated ``<model>.cpp`` code.

Python Class Generation Configuration
======================================

Code generation creates both C++/CUDA implementations and corresponding Python classes
(``<Model>SimGroup``). Python-specific configuration is defined in the ``python_config`` section.
For ``rWWEx``, only one option is required:

.. code-block:: yaml

    python_config:
      sel_state_var: r

The ``python_config`` section may include additional fields:

- ``sel_state_var``: State variable used in tests (required)
- ``labels``: Dictionary of (LaTeX) labels for state variables and parameters, used in plots
- ``modifiable_params``: List of parameters that may be modified during simulation
- ``exclude_configs``: List of config variables to exclude from Python class initialization
- ``custom_methods``: Custom Python methods to include in the generated class (can override or extend ``cubnm.sim.SimGroup`` methods)

For examples of these advanced options, see ``codegen/recipes/rww.yaml``.

Build the Toolbox from Source
******************************

Once your model is defined in a YAML file, generate code and build the toolbox:

1. Generate code for all models (after installing ``codegen/requirements.txt``):

   .. code-block:: bash

       python codegen/generate_models.py

   This creates/modifies C++/CUDA and Python files in ``src/ext/models/``,
   ``include/cubnm/models/``, and ``src/cubnm/sim/`` directories.

2. Prepare build requirements as described in :ref:`installation from source <from-source>`.
   Ideally, use a GPU-enabled device to test both GPU and CPU implementations. 
   We recommend using a container for the compilation toolchain, such as
   https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_11.8.

3. Install the modified toolbox:

   .. code-block:: bash

       cd /path/to/cubnm
       # Remove current installation
       python -m pip uninstall cubnm
       # Install build package
       python -m pip install build
       # Build and install
       python -m build . && python -m pip install $(ls -tr ./dist/*.whl | tail -n 1)

   If you encounter compilation errors, troubleshoot them or open an issue on the GitHub repository.

4. Perform an initial test:

   .. code-block:: python

       from cubnm import sim, datasets

       # instantiate a simulation group with your model
       # (replace rWWExSimGroup with your model's class name
       # and adjust parameters as needed)
       sim_group = sim.rWWExSimGroup(
           duration=60,
           TR=1,
           window_size=10,
           window_step=2,
           sc=datasets.load_sc('strength', 'schaefer-100'),
       )
       sim_group.N = 1
       # set model parameters
       sim_group.param_lists['G'] = np.array([0.5])
       sim_group._set_default_params()
       sim_group.run()

Run Tests
*********

To ensure your model works consistently:

1. Generate expected test results:

   .. code-block:: bash

       cd /path/to/cubnm
       python ./tests/sim/gen_expected.py <model_name>

2. Run all tests for your model:

   .. code-block:: bash

       python -m pytest tests/sim/test.py -k "<model_name>"

All tests except the CPU-GPU identity test (``test_identical_cpu_gpu``) must pass.

.. note::
    The CPU-GPU identity test is not expected to pass for all models and configurations.
    This is due to inherent differences in hardware-level implementations of some math functions
    [`see NVIDIA documentation <https://docs.nvidia.com/cuda/floating-point/index.html#considerations-for-a-heterogeneous-world>`_],
    which in some models (such as Kuramoto with long delays) result in numerical differences that
    accumulate over simulation time.

Once all required tests pass, you can contribute your model to the toolbox.
If tests fail and you cannot resolve the issues, open an issue on the GitHub repository.

Pull Request
************

Once your model is implemented and tests pass, create a pull request to the main repository.

Support
*******

Please don't hesitate to open a GitHub issue or contact amnsbr [at] gmail [dot] com if you have questions
or would like to add a model that doesn't fit within the current framework.