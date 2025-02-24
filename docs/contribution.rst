Contributing a new model
########################

To make this guide more intuitive, it details the steps to contribute a new model to the toolbox
by using an example model, which is already included in the toolbox. To contribute a new model, 
fork the toolbox repository and within your fork make changes and additions to the source code as 
described below, modifying it according to your new model.

Example model description
*****************

We'll take take the reduced Wong-Wang model with excitatory populations based on 
`Deco et al. 2013 <https://doi.org/10.1523/JNEUROSCI.1091-13.2013>`_ as an example.
We will abbreviate this model as ``rWWEx``. This abbreviation will come up in many places
including the class names (starting with ``rWWEx``), the model string name (``"rWWEx"``),
and the source file names (e.g. ``rwwex.hpp``).

This model is described by the following differential equations:

.. math::

    x_i = w_i J_N S_i + G J_N \sum_j C_{ij} S_j + I_{i0}

    r_i = \frac{a x_i - b}{1 - \exp(-d(a x_i - b))}

    \dot{S_i} = -\frac{S_i}{\tau_s} + (1 - S_i) \gamma r_i + {\sigma}_i v_i(t)
    

There are 3 state variables for each node, including :math:`x_i` (total input current to the node), 
:math:`r_i` (node firing rate) and :math:`S_i` (synaptic gating variable). Model free parameters include
:math:`G` (global coupling), and regional parameters :math:`w_i` (recurrent excitation), :math:`I_{i0}` 
(external input current) and :math:`{\sigma}_i` (noise amplitude). The structural connectivity (SC) matrix is
denoted by :math:`C_{ij}`. The model also includes fixed parameters :math:`a`, :math:`b`, :math:`d`,
:math:`\tau_s` and :math:`J_N`. This model is a "conduction-based" model, and the input from other nodes 
conveyed through SC is additive. In contrast there are "oscillatory" models such as Kuramoto, which are 
not covered in this guide (but have a look at the Kuramoto model included in the toolbox and see how 
it is implemented). Note that :math:`C_{ij}` is multiplied by :math:`S_i`, making it the ``conn_state_var``.
:math:`S_i` is also the state variable that represents node activity and is fed into the Balloon-Windkessel
model to generate simulated BOLD signals, making it the ``bold_state_var`` as well.

Now, let's implement this model in the toolbox. 
We will start with making changes in the Python code, and then
proceed to the CUDA/C++ code where the model calculations are performed.

Define the ``{Model}SimGroup`` class in Python
***********************************************

1.  In ``src/cubnm/sim.py`` we should add a new class called ``rWWExSimGroup`` which inherits from
    :class:`cubnm.sim.SimGroup`. We first define class attributes including the model name,
    as well as the names of the global parameters, regional parameters and state variables (noting
    that their order is important and will be used in the CUDA/C++ code). In addition we should
    define how many noise elements does each node require at any given time point. In this model
    this is just one, since we only have one excitatory neuronal population in each node and have
    a single :math:`v_i(t)` term (but it could be more than one, e.g. in ``rWW`` model which includes
    both excitatory and inhibitory neurons). Lastly, we should also define which state variable
    should be used in the tests and assign it to ``sel_state_var`` (more on tests later).

    .. code-block:: python

        class rWWExSimGroup(SimGroup):
            model_name = "rWWEx"
            global_params = ["G"]
            regional_params = ["w", "I0", "sigma"]
            state_vars = ["x", "r", "S"]
            sel_state_var = "r"
            noise_elements = 1

2.  The ``rWWExSimGroup`` class must define a method called ``_set_default_params`` which as
    its name implies, sets the default values for the model free parameters. This method
    should populate the dictionary ``self.param_lists`` with the default values for each
    simulation (and in the case of regional parameters, each node in the simulation). Note
    that it must also call the parent class method to set the default values for the other
    parameters shared between different types of models (currently only ``v``, conduction
    velocity, is shared between all models, and is used when conduction delays are applied
    which is not the default). The method should look like this:

    .. code-block:: python

        class rWWExSimGroup(SimGroup):
            ...
            def _set_default_params(self):
                super()._set_default_params()
                self.param_lists["G"] = np.repeat(0.5, self.N)
                self.param_lists["w"] = np.full((self.N, self.nodes), 0.9)
                self.param_lists["I0"] = np.full((self.N, self.nodes), 0.3)
                self.param_lists["sigma"] = np.full((self.N, self.nodes), 0.001)


    Notice that the default values for the regional parameters are set as 2D arrays, where the first
    dimension is the number of simulations and the second dimension is the number of nodes. 
    In contrast, the global parameters are set as 1D arrays, where the length of the array 
    is equal to the number of simulations.

3.  Defining ``__init__`` is not required but recommended. It may be needed for some models to
    define additional configs which are not set in the parent :class:`cubnm.sim.SimGroup` 
    (e.g. see :class:`cubnm.sim.rWWSimGroup` or :class:`cubnm.sim.KuramotoSimGroup`). If defined,
    this method must call the parent ``__init__``. Even though ``rWWEx`` model does not
    require any model-specific initializations, here we will define it to include further 
    info and examples in its documentation, and for consistency with the other models.

    .. code-block:: python
        
        class rWWExSimGroup(SimGroup):
            ...
            def __init__(self, *args, **kwargs):
                """
                Group of reduced Wong-Wang simulations (excitatory only, Deco 2013) 
                that are executed in parallel

                Parameters
                ---------
                *args, **kwargs:
                    see :class:`cubnm.sim.SimGroup` for details

                Attributes
                ----------
                param_lists: :obj:`dict` of :obj:`np.ndarray`
                    dictionary of parameter lists, including
                        - 'G': global coupling. Shape: (N_SIMS,)
                        - 'w': local excitatory self-connection strength. Shape: (N_SIMS, nodes)
                        - 'I0': local external input current. Shape: (N_SIMS, nodes)
                        - 'sigma': local noise sigma. Shape: (N_SIMS, nodes)
                        - 'v': conduction velocity. Shape: (N_SIMS,)

                Example
                -------
                To see example usage in grid search and evolutionary algorithms
                see :mod:`cubnm.optimize`.

                Here, as an example on how to use SimGroup independently, we
                will run a single simulation and save the outputs to disk. ::

                    from cubnm import sim, datasets

                    sim_group = sim.rWWExSimGroup(
                        duration=60,
                        TR=1,
                        sc=datasets.load_sc('strength', 'schaefer-100'),
                    )
                    sim_group.N = 1
                    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
                    sim_group.param_lists['w'] = np.full((N_SIMS, nodes), 0.9)
                    sim_group.param_lists['I0'] = np.full((N_SIMS, nodes), 0.3)
                    sim_group.param_lists['sigma'] = np.full((N_SIMS, nodes), 0.001)
                    sim_group.run()
                """
                super().__init__(*args, **kwargs)


Define model C++/CUDA headers
****************************

1.  Create a new file in ``include/cubnm/models/rwwex.hpp``. This file will define the
    C++ class ``rWWExModel`` derived from the ``BaseModel`` class defined in
    ``include/cubnm/models/base.hpp``. The file should include the following: (see the
    comments in the code for explanations)

    .. code-block:: cpp

        #ifndef RWWEX_HPP // Include guards (set to "MODELNAME_HPP")
        #define RWWEX_HPP
        #include "cubnm/models/base.hpp" // Include the base model class
        class rWWExModel : public BaseModel { // Define ModelNameModel class derived from BaseModel
        public:
            // define Constants struct which will include
            // model constants including dt (integration step)
            // and its square root which are required for all
            // models, as well as other model-specific constants
            // and constants derived from other constants (e.g.
            // dt_itau = dt / tau)
            // Note that here only the variables are defined, but
            // their values will be set later in "./src/ext/models/rwwex.cpp"
            struct Constants {
                u_real dt; // Must be defined for all models
                u_real sqrt_dt; // Must be defined for all models
                u_real J_N; // Start of model-specific constants
                u_real a;
                u_real b;
                u_real d;
                u_real gamma;
                u_real tau;
                u_real itau; // Start of model-specific constants
                             // which are derived from other constants
                u_real dt_itau; 
                u_real dt_gamma;
            };
            // define Config struct which will include
            // model-specific configurations. This must
            // be defined for all models, even if it is empty
            // (as in this case, but see rWW model for an example
            // of a non-empty Config struct)
            // Configurations refer not to the model parameters
            // but rather to alternative ways of implementing
            // the simulations
            // Note that here only the variables are defined, but
            // their values will be set later in "./src/ext/models/rwwex.cpp"
            struct Config {
            };

            // use the boilerplate macro to include
            // the repetitive elements of the class definitions
            DEFINE_DERIVED_MODEL(
                rWWExModel, // name of C++ class
                "rWWEx", // string name of the model
                3, // number of state variables
                   // in rWWEx model, we have x, r, S
                1, // number of intermediate variables needed for calculations
                   // (more on this below in the definition of `step`)
                1, // number of noise elements needed per node per time point
                   // in rWWEx we need a single noise element per node (v_i(t))
                1, // number of global parameters
                   // in rWWEx we have G
                3, // number of regional parameters
                   // in rWWEx we have w, I0, sigma
                2, // index of the state variable that will be used
                   // as input to the other nodes
                   // in rWWEx model given `C_{ij} S_j` term, S is the
                   // state variable that is used as input to other nodes
                2, // index of the state variable that will be used
                   // as input to the Balloon-Windkessel model
                   // in rWWEx model, this is also S
                false, // whether the model has a `post_bw_step`
                       // function
                       // in rWWEx not needed, but was needed in e.g.
                       // the rWW model to do numerical FIC calculations
                false, // whether the model has a `post_integration`
                       // function
                       // in rWWEx not needed, but was needed in e.g.
                       // the rWW model to return the final wIE value
                       // resulted from numerical FIC
                false, // whether the model is oscillatory vs conduction-based
                       // rWWEx is conduction-based
                0, // number of additional integer variables needed ber node
                   // in rWWEx (and most models) we don't need any
                0, // number of additional boolean variables needed per node
                   // in rWWEx (and most models) we don't need any
                0, // number of additional integer variables shared by all nodes
                   // in rWWEx (and most models) we don't need any
                0, // number of additional boolean variables shared by all nodes
                   // in rWWEx (and most models) we don't need any
                0, // number of additional global integer outputs
                   // in rWWEx (and most models) we don't need any
                0, // number of additional global boolean outputs
                   // in rWWEx (and most models) we don't need any
                0, // number of additionl global double/float outputs
                   // in rWWEx (and most models) we don't need any
                0, // number of additional regional integer outputs
                   // in rWWEx (and most models) we don't need any
                0, // number of additional regional boolean outputs
                   // in rWWEx (and most models) we don't need any
                0  // number of additional regional double/float outputs
                   // in rWWEx (and most models) we don't need any
            )

            // additional functions that need to be overridden
            // (in addition to h_init, h_step, _j_restart
            // which are always overriden and have to be defined)
            // None in this model (see rWW model for an example
            // in which additional functions are defined)
        };

        #endif

    .. note::

        Technical note: While the usage of a boilerplate macro may be cryptic and not very clean,
        it is necessary primarily as CUDA does not support virtual functions, and the boilerplate
        was used as a workaround to avoid having to define the same functions/kernels in every model class
        without making them virtual. The boilerplate macro is defined in ``include/cubnm/models/boilerplat.hpp``.

2.  Create a new file in ``include/cubnm/models/rwwex.cuh`` which does three things: i. includes the 
    header file for the model (the file we just created), ii. initializes an instant of model constants
    on the GPU, and iii. explicitly instanciates the template of ``_init_gpu`` and ``_run_simulations_gpu``
    functions (defined in ``src/ext/bnm.cu``) for the model. The file should include the following:

    .. code-block:: cpp

        #ifndef RWWEX_CUH // Include guards (set to "MODELNAME_CUH")
        #define RWWEX_CUH
        #include "rwwex.hpp" // Include the model C++ header file
        __constant__ rWWExModel::Constants d_rWWExc; // Initialize model constants on the GPU
        // Explicitly instanciate the template of _init_gpu and _run_simulations_gpu functions
        template void _run_simulations_gpu<rWWExModel>(
            double*, double*, double*, 
            u_real**, u_real**, u_real*, 
            u_real**, int*, u_real*, 
            BaseModel*
        );
        template void _init_gpu<rWWExModel>(BaseModel*, BWConstants, bool);
        #endif


Model calculations on GPU
*************************

Create a new file in ``src/ext/models/rwwex.cu``. This file will define the implementation
of model calculations on GPU. It first should include the required header files:

    .. code-block:: cpp

        #include "cubnm/includes.cuh"
        #include "cubnm/defines.h"
        #include "cubnm/models/rwwex.cuh"

Then, it must at least define the implementation of GPU kernels ``init``, ``step`` and 
``restart``. The kernels ``post_bw_step`` and ``post_integration`` can optionally be 
defined depending on the model.

``step`` kernel
===============

The ``step`` kernel is where most of the actual model calculations occur. It is called
in each iteration (time step) of the integration loop, and performs the calculations for
one node in one simulation. It should update the state variables of the node, given previous
states, the model parameters, noise, and the input from other nodes. Here is how we can define
it for the rWWEx model (in comments, we show pseudo-code corresonding to the equations above):

.. code-block:: cpp

    __device__ void rWWExModel::step(
            u_real* _state_vars, u_real* _intermediate_vars,
            u_real* _global_params, u_real* _regional_params,
            u_real& tmp_globalinput,
            u_real* noise, long& noise_idx
            ) {
        // x = w * J_N * S + G * J_N * tmp_globalinput + I0
        _state_vars[0] = _regional_params[0] * d_rWWExc.J_N * _state_vars[2] + _global_params[0] * d_rWWExc.J_N * tmp_globalinput + _regional_params[1] ; 
        // ax_b = a * x - b
        _intermediate_vars[0] = d_rWWExc.a * _state_vars[0] - d_rWWExc.b;
        // r = ax_b / (1 - exp(-d * ax_b))
        _state_vars[1] = _intermediate_vars[0] / (1 - exp(-d_rWWExc.d * _intermediate_vars[0]));
        // S += dt * ((gamma * (1 - S) * r) - (S / tau)) + sigma * sqrt(dt) * noise
        _state_vars[2] += d_rWWExc.dt_gamma * ((1 - _state_vars[2]) * _state_vars[1]) - d_rWWExc.dt_itau * _state_vars[2] + noise[noise_idx] * d_rWWExc.sqrt_dt * _regional_params[2];
        // clip S to 0-1
        _state_vars[2] = max(0.0f, min(1.0f, _state_vars[2]));
    }

The input arguments to this kernel are fixed and should not be changed. They include:

-   ``_state_vars``: an array of state variables for the current node and simulation. Here it
    is a 3-element array corresponding to ``x``, ``r`` and ``S``. Therefore ``_state_vars[0]``
    is ``x``, ``_state_vars[1]`` is ``r`` and ``_state_vars[2]`` is ``S``.
-   ``_intermediate_vars``: an array of intermediate variables for the current node and simulation.
    They are useful when the same term is used in multiple calculations, such as ``a * x - b`` 
    which is used twice in the calculation of firing rate ``r``. Usage of intermediate variables
    is not necessary, but can make the code more readable and efficient.
    
    .. note::
        Note that the rWWEx implementation included in the toolbox is slightly different, 
        and additionally includes ``dSdt`` as an intermediate variable, but that is not necessary
        and will cause no differences in the results.

-   ``_global_params``: an array of global parameters for the current simulation. Here it is a 1-element
    array corresponding to ``G``.
-   ``_regional_params``: an array of regional parameters for the current node and simulation. Here it
    is a 3-element array corresponding to ``w``, ``I0`` and ``sigma``.
-   ``tmp_globalinput``: this is a floating point number representing the sum of the inputs from 
    other nodes to the current node within current time point and simulation, i.e., 
    :math:`\sum_k C_{jk} S_k`. It is calculated by the core kernel ``bnm`` (in ``./src/ext/bnm.cu``)
    right before ``step`` is executed, using ``global_input_cond`` for conduction-based and
    ``global_input_osc`` for oscillatory models (both defined in ``./src/ext/bnm.cu``). In
    the case of conduction-based models such as rWWEx, ``global_input_cond`` calculates
    ``tmp_global_input`` by summing up the multiplication of ``S`` at other nodes ``k`` (
    from 1 or more time points ago depending on presence and amount of conduction delay) 
    by their connectivity strength to current node ``j`` as defined by the SC matrix.
    As this sum is calculated by the core kernel, it should be used as a given in the ``step`` kernel.
    For example the term :math:`+ G J_N \sum_j C_{ij} S_j` in the model equation translates
    to ``_global_params[0] * d_rWWExc.J_N * tmp_globalinput`` in the ``step`` kernel.
-   ``noise``: the entire precalculated noise array
-   ``noise_idx``: the (starting) index of the noise element(s) to be used in the current time point 
    and node. The calculation of ``noise_idx`` is handled by the core. If a model has a single noise
    element per node (like rWWEx), the noise at current node and time point can simply be accessed via 
    ``noise[noise_idx]``. If a model has multiple noise elements per node, additional noise elements
    can be accessed at ``noise[noise_idx+1]``, ``noise[noise_idx+2]`` and so on.

Another variable that can and should be used in the ``step`` kernel is ``d_rWWExc`` (``d_{ModelName}c``),
which contains the model constants that are on the GPU constant memory. This was initialized earlier
in the ``rwwex.cuh`` file (but we still have not set the values for the constants, which will be done
in another source file).

Arithmetics as well as CUDA Math functions (`see full list <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html>`_)
can be used in this and other custom kernels.

Let's now break down the ``step`` kernel for the rWWEx model, and see how it corresponds to the equations:

1.  :math:`x_i = w_i J_N S_i + G J_N \sum_j C_{ij} S_j + I_{i0}` translates to:

    .. code-block:: cpp

        // pseudo-code
        // x = w * J_N * S + G * J_N * tmp_globalinput + I0
        // CUDA code
        _state_vars[0] = _regional_params[0] * d_rWWExc.J_N * _state_vars[2] + _global_params[0] * d_rWWExc.J_N * tmp_globalinput + _regional_params[1];

    where ``_state_vars[0]`` is :math:`x_i`, ``_regional_params[0]`` is :math:`w_i`, ``_global_params[0]`` 
    is :math:`G`, ``_state_vars[2]`` is :math:`S_i`, ``tmp_globalinput`` is :math:`\sum_j C_{ij} S_j`, 
    and model constant :math:`J_N` can be accessed through ``d_rWWExc.J_N``.

2.  :math:`r_i = \frac{a x_i - b}{1 - \exp(-d(a x_i - b))}` translates to:

    .. code-block:: cpp

        // pseudo-code
        // ax_b = a * x - b
        // r = ax_b / (1 - exp(-d * ax_b))
        // CUDA code
        _intermediate_vars[0] = d_rWWExc.a * _state_vars[0] - d_rWWExc.b;
        _state_vars[1] = _intermediate_vars[0] / (1 - exp(-d_rWWExc.d * _intermediate_vars[0]));

    Here as :math:`a x_i - b` is used twice, we calculate and store it in an intermediate variable
    ``_intermediate_vars[0]``. Then we calculate :math:`r_i` using this intermediate variable and model
    constants :math:`a`, :math:`b` and :math:`d` which can be accessed through ``d_rWWExc``.

3.  :math:`\dot{S_i} = -\frac{S_i}{\tau_s} + (1 - S_i) \gamma r_i + {\sigma}_i v_i(t)`, using
    Euler-Maruyama method, translates to:

    .. code-block:: cpp

        // pseudo-code
        // S += dt * ((gamma * (1 - S) * r) - (S / tau)) + sigma * sqrt(dt) * noise
        // CUDA code (broken into multiple lines for readability)
        _state_vars[2] += 
            d_rWWExc.dt_gamma * ((1 - _state_vars[2]) * _state_vars[1]) 
            - d_rWWExc.dt_itau * _state_vars[2] 
            + noise[noise_idx] * d_rWWExc.sqrt_dt * _regional_params[2];

    where ``_state_vars[2]`` is :math:`S_i`, ``_state_vars[1]`` is :math:`r_i` and ``_regional_params[2]`` is
    :math:`{\sigma}_i`. Note that given ``dt``, ``gamma`` and ``tau`` are constants and fixed, ``dt * gamma``
    and ``dt / tau`` are also fixed, and rather than recalculating them in each step, we can precalculate
    them once and store them in the model constants ``d_rWWExc.dt_gamma`` and ``d_rWWExc.dt_itau``.

    Finally, as :math:`S_i` (synaptic gating variable) must be between 0 and 1 (and may go beyond
    depending on noise), we clip it to 0-1:

    .. code-block:: cpp
        
        // clip S to 0-1
        _state_vars[2] = max(0.0f, min(1.0f, _state_vars[2]));



``init`` kernel
===============
The ``init`` kernel is called once for each node before the integration loop starts.
It should initialize the state variables that need to be initialized, in addition to doing
other model-specific initializations as needed. In the case of the rWWEx model, it
simply initializes the state variable ``S`` to initial value of 0.001 (but no need to do
so for other variables, as only ``S`` is incrementally updated through time steps, 
and the other variables ``x`` and ``r`` are calculated independently in each time step).

.. code-block:: cpp

    __device__ __NOINLINE__ void rWWExModel::init(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) {
        _state_vars[2] = 0.001; // S
    }
    
Some of the input arguments to this kernel are similar to the ``step`` kernel, but there are
additional arguments to this kernel (though they are not used in the rWWEx model). They include:

-  ``_ext_int``: an array of additional integer variables pertaining to the current node
-  ``_ext_bool``: an array of additional boolean variables pertaining to the current node
-  ``_ext_int_shared``: an array of additional integer variables shared by all nodes and pertaining to the current simulation
-  ``_ext_bool_shared``: an array of additional boolean variables shared by all nodes and pertaining to the current simulation

``restart`` kernel
==================
The ``restart`` kernel is called for each node when the simulation is restarted. This case
does not happen in the rWW model, and is used currently only in the rWW model. However, it
still must be defined for all models, and should basically redo the initialization (this
requirement will be fixed in the future, making it not required in the models that do not
get restarted).
.. todo: fix this

.. code-block:: cpp

    __device__ __NOINLINE__ void rWWExModel::restart(
        u_real* _state_vars, u_real* _intermediate_vars, 
        u_real* _global_params, u_real* _regional_params,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) {
        _state_vars[2] = 0.001; // S
    }

Full code of ``rwwex.cu``
=======================
Putting all the definitions above together, the final ``./src/ext/models/rwwex.cu`` 
file will be:

    .. code-block:: cpp

        #include "cubnm/includes.cuh"
        #include "cubnm/defines.h"
        #include "cubnm/models/rwwex.cuh"

        __device__ void rWWExModel::step(
            u_real* _state_vars, u_real* _intermediate_vars,
            u_real* _global_params, u_real* _regional_params,
            u_real& tmp_globalinput,
            u_real* noise, long& noise_idx
        ) {
            _state_vars[0] = _regional_params[0] * d_rWWExc.J_N * _state_vars[2] + _global_params[0] * d_rWWExc.J_N * tmp_globalinput + _regional_params[1];
            _intermediate_vars[0] = d_rWWExc.a * _state_vars[0] - d_rWWExc.b;
            _state_vars[1] = _intermediate_vars[0] / (1 - exp(-d_rWWExc.d * _intermediate_vars[0]));
            _state_vars[2] += d_rWWExc.dt_gamma * ((1 - _state_vars[2]) * _state_vars[1]) - d_rWWExc.dt_itau * _state_vars[2] + noise[noise_idx] * d_rWWExc.sqrt_dt * _regional_params[2];
            _state_vars[2] = max(0.0f, min(1.0f, _state_vars[2]));
        }

        __device__ __NOINLINE__ void rWWExModel::init(
            u_real* _state_vars, u_real* _intermediate_vars,
            u_real* _global_params, u_real* _regional_params,
            int* _ext_int, bool* _ext_bool,
            int* _ext_int_shared, bool* _ext_bool_shared
        ) {
            _state_vars[2] = 0.001; // S
        }

        __device__ __NOINLINE__ void rWWExModel::restart(
            u_real* _state_vars, u_real* _intermediate_vars, 
            u_real* _global_params, u_real* _regional_params,
            int* _ext_int, bool* _ext_bool,
            int* _ext_int_shared, bool* _ext_bool_shared
        ) {
            _state_vars[2] = 0.001; // S
        }

Changes in ``bnm.cu``
=====================
In the ``./src/ext/bnm.cu`` file, two changes are needed:

1.  Include the header file for the rWWEx model at the top of the file after
    other includes:

    .. code-block:: cpp
        
        #include ...
        #include "cubnm/models/rwwex.cuh"

2.  At the beginning of ``_init_gpu`` function there is a segment of code
    which copies the model constants of each specific model to its corresponding
    global variable on the GPU (i.e., ``d_{ModelName}c``). Add the following case
    after the other cases to do this for the rWWEx model:

    .. code-block:: cpp

        // find this line of code
        if (strcmp(Model::name, "rWW")==0) {
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_rWWc, &Model::mc, sizeof(typename Model::Constants)));
        }
        // then add this case at the end
        // after other cases
        // ...
        // ...
        else if (strcmp(Model::name, "rWWEx")==0) {
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_rWWExc, &Model::mc, sizeof(typename Model::Constants)));
        }

.. note::
    This step should be automatized in future to avoid having to make these manual edits
    in the ``bnm.cu`` file.

Define values of constants and model calculations on CPU
*****************************************************
Next, we will make CPU-side changes in the code to i. define the constants values, and ii. define 
the implementation of model calculations on CPU. The latter for a large part involves copying 
and making slight changes to the code from the CUDA implementation.

.. warning::
    For uniformity of models across the toolbox the CPU implementation must be defined
    even if you are only interested in the GPU implementation. This is because the toolbox
    is designed to be able to run simulations on both CPU and GPU.

.. note::
    In future versions having to duplicate code between CPU and GPU should be minimized.

Create a new file in ``src/ext/models/rwwex.cpp``. We should first include its corresponding header file:

    .. code-block:: cpp

        #include "cubnm/models/rwwex.hpp"


Define values of constants
=======================

In ``src/ext/models/rwwex.cpp`` we then declare a static CPU-side copy of the model constants:

    .. code-block:: cpp

        rWWExModel::Constants rWWExModel::mc;

And define a member function ``init_constants`` that will set the
values of all constants:

    .. code-block:: cpp

        void rWWExModel::init_constants(u_real dt) {
            // based on Deco et al. 2013
            mc.dt = dt;
            mc.sqrt_dt = SQRT(mc.dt); 
            mc.J_N  = 0.2609;
            mc.a = 270;
            mc.b = 108;
            mc.d = 0.154;
            mc.gamma = (u_real)0.641/(u_real)1000.0;
            mc.tau = 100;
            mc.itau = 1.0/mc.tau;
            mc.dt_itau = mc.dt * mc.itau;
            mc.dt_gamma = mc.dt * mc.gamma;
        }

It must take the integration time step ``dt`` as an argument (this will be passed on from 
the core and by default is equal to 0.1 msec), and set the values of all constants
according to the ``dt`` and the model equations. Note that we have defined the ``Constants`` ``struct``
earlier in ``./include/cubnm/models/rwwex.hpp``.

Model calculations on CPU
=========================
We continue in ``src/ext/models/rwwex.cpp`` to add CPU implementations of the model calculations.
We must at least define the implementation of CPU functions ``h_init`` (h short for host, referring
to the CPU), ``h_step`` and ``_j_restart``. The functions ``h_post_bw_step`` and ``h_post_integration``
can optionally be defined depending on the model.

-   ``h_step`` takes the same arguments as GPU-side ``step`` kernel, and runs one integration
    step for current node at current time point and simulation. It is largely a copy of 
    ``step`` kernel, except that all instances of ``d_rWWExc`` should be replaced
    with the CPU-side copy of the constants in ``rWWExModel::mc``. The function would then
    look like:

    .. code-block:: cpp

        void rWWExModel::h_step(
                u_real* _state_vars, u_real* _intermediate_vars,
                u_real* _global_params, u_real* _regional_params,
                u_real& tmp_globalinput,
                u_real* noise, long& noise_idx
                ) {
            // x = w * J_N * S + G * J_N * tmp_globalinput + I0
            _state_vars[0] = _regional_params[0] * rWWExModel::mc.J_N * _state_vars[2] + _global_params[0] * rWWExModel::mc.J_N * tmp_globalinput + _regional_params[1] ; 
            // axb = a * x - b
            _intermediate_vars[0] = rWWExModel::mc.a * _state_vars[0] - rWWExModel::mc.b;
            // r = axb / (1 - exp(-d * axb))
            _state_vars[1] = _intermediate_vars[0] / (1 - exp(-rWWExModel::mc.d * _intermediate_vars[0]));
            // S += dt * ((gamma * (1 - S) * r) - (S / tau)) + sigma * sqrt(dt) * noise
            _state_vars[2] += rWWExModel::mc.dt_gamma * ((1 - _state_vars[2]) * _state_vars[1]) - rWWExModel::mc.dt_itau * _state_vars[2] + noise[noise_idx] * rWWExModel::mc.sqrt_dt * _regional_params[2];
            // clip S to 0-1
            _state_vars[2] = fmax(0.0f, fmin(1.0f, _state_vars[2]));
        }

-   ``h_init`` initializes the state variables of the current node and simulation. It is again
    a copy of the GPU-side ``init`` kernel, but only if constants are used (which are not in rWWEx's
    initialization) they should be accessed at ``rWWExModel::mc``. The function would be defined as:

    .. code-block:: cpp

        void rWWExModel::h_init(
            u_real* _state_vars, u_real* _intermediate_vars,
            u_real* _global_params, u_real* _regional_params,
            int* _ext_int, bool* _ext_bool,
            int* _ext_int_shared, bool* _ext_bool_shared
        ) {
            _state_vars[2] = 0.001; // S
        }

    We can see that for rWWEx model the CPU-side ``h_init`` implemnetation is identical to the 
    GPU-side ``init`` kernel.

-   ``_j_restart`` restarts the current node when simulation is restarted. Same as above, it's
    a copy of the GPU-side ``restart`` kernel, but only if constants are used (which are not in rWWEx's
    initialization) they should be accessed at ``rWWExModel::mc``. The function would be defined as:

    .. code-block:: cpp

        void rWWExModel::_j_restart(
            u_real* _state_vars, u_real* _intermediate_vars, 
            u_real* _global_params, u_real* _regional_params,
            int* _ext_int, bool* _ext_bool,
            int* _ext_int_shared, bool* _ext_bool_shared
        ) {
            _state_vars[2] = 0.001; // S
        }

Full code of ``rwwex.cpp``
=======================
Putting all the definitions above together, the final ``./src/ext/models/rwwex.cpp`` 
file will be:

.. code-block:: cpp

    #include "cubnm/models/rwwex.hpp"

    rWWExModel::Constants rWWExModel::mc;

    void rWWExModel::init_constants(u_real dt) {
        // based on Deco et al. 2013
        mc.dt = dt;
        mc.sqrt_dt = SQRT(mc.dt); 
        mc.J_N  = 0.2609;
        mc.a = 270;
        mc.b = 108;
        mc.d = 0.154;
        mc.gamma = (u_real)0.641/(u_real)1000.0;
        mc.tau = 100;
        mc.itau = 1.0/mc.tau;
        mc.dt_itau = mc.dt * mc.itau;
        mc.dt_gamma = mc.dt * mc.gamma;
    }

    void rWWExModel::h_step(
            u_real* _state_vars, u_real* _intermediate_vars,
            u_real* _global_params, u_real* _regional_params,
            u_real& tmp_globalinput,
            u_real* noise, long& noise_idx
            ) {
        // x = w * J_N * S + G * J_N * tmp_globalinput + I0
        _state_vars[0] = _regional_params[0] * rWWExModel::mc.J_N * _state_vars[2] + _global_params[0] * rWWExModel::mc.J_N * tmp_globalinput + _regional_params[1] ; 
        // axb = a * x - b
        _intermediate_vars[0] = rWWExModel::mc.a * _state_vars[0] - rWWExModel::mc.b;
        // r = axb / (1 - exp(-d * axb))
        _state_vars[1] = _intermediate_vars[0] / (1 - exp(-rWWExModel::mc.d * _intermediate_vars[0]));
        // S += dt * ((gamma * (1 - S) * r) - (S / tau)) + sigma * sqrt(dt) * noise
        _state_vars[2] += rWWExModel::mc.dt_gamma * ((1 - _state_vars[2]) * _state_vars[1]) - rWWExModel::mc.dt_itau * _state_vars[2] + noise[noise_idx] * rWWExModel::mc.sqrt_dt * _regional_params[2];
        // clip S to 0-1
        _state_vars[2] = fmax(0.0f, fmin(1.0f, _state_vars[2]));
    }

    void rWWExModel::h_init(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) {
        _state_vars[2] = 0.001; // S
    }

    void rWWExModel::_j_restart(
        u_real* _state_vars, u_real* _intermediate_vars, 
        u_real* _global_params, u_real* _regional_params,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) {
        _state_vars[2] = 0.001; // S
    }


Include the model in ``core.cpp``
**********************************

Finally, we need to include the rWWEx model in the ``core.cpp`` file. This requires the following
changes:

1.  Include the header file for the rWWEx model at the top of the file after other includes:

    .. code-block:: cpp

        #include ...
        #include "cubnm/models/rwwex.hpp"

2.  Add the following lines to the ``run_simulations`` function to initialize the `rWWExModel`
    instance in case the model name is "rWWEx":

    .. code-block:: cpp

        // find this line
        if (strcmp(model_name, "rWW")==0) {
            model = new rWWModel(
                nodes, N_SIMS, N_SCs, BOLD_TR, states_sampling, 
                time_steps, do_delay, rand_seed, dt, bw_dt
            );
        } 
        // then add this case at the end
        // after other cases
        // ...
        else if (strcmp(model_name, "rWWEx")==0) {
            model = new rWWExModel(
                nodes, N_SIMS, N_SCs, BOLD_TR, states_sampling, 
                time_steps, do_delay, rand_seed, dt, bw_dt
            );
        } 

.. note::
    This step should be automatized in future to avoid having to make these manual edits
    in the ``core.cpp`` file.


Build the toolbox from source
*****************************
1.  Prepare all the build requirements as described in :ref:`installation from source <from-source>`.
    Ideally use a GPU-enabled device to be able to test both GPU and CPU implementations. Also it's
    best (but not required) to use a container for the compliation toolchain. We recommend using
    https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_11.8. 
2.  Install the modified toolbox code which includes your new model via:

    .. code-block:: bash

        cd /path/to/cubnm
        # remove the current installation
        python -m pip uninstall cubnm
        # install build package
        python -m pip install build
        # build the wheel file and install it
        python -m build . && python -m pip install $(ls -tr ./dist/*.whl | tail -n 1)

   
    You might get compilation errors, in which case you should troubleshoot them. 
    If you are unable to resolve the errors, open an issue on the GitHub repository.

3.  Do an initial test of the model by running:

    .. code-block:: python

        from cubnm import sim, datasets

        sim_group = sim.rWWExSimGroup(
            duration=60,
            TR=1,
            window_size=10,
            window_step=2,
            sc=datasets.load_sc('strength', 'schaefer-100'),
        )
        sim_group.N = 1
        sim_group._set_default_params()
        sim_group.run()

Run tests
*********
To ensure that the model is working consistently, you should first generate expected results for the tests.
While the modified toolbox with the new model is installed, run the following: 
``cd /path/to/cubnm && python ./tests/sim/gen_expected.py rWWEx``.

Then run all the tests pertaining to this new model by running:
``cd /path/to/cubnm && python -m pytest tests/sim/test.py -k "rWWEx"``. All tests except the CPU-GPU identity
test (``test_identical_cpu_gpu``) must pass. Even with identical CPU and GPU implementations, it is likely
that the identity test of CPU and GPU does not pass depending on the model and precision of calculations.
Once all required tests pass, you can proceed to the next step and contribute the model to the toolbox.
If they fail and you are unable to resolve the issues, open an issue on the GitHub repository.

Pull request
************
Once you have successfully implemented the model and all tests pass, you can create a pull request
to the main repository.

Support
*******
Please don't hesitate to open a GitHub issue or reach out to me (amnsbr [at] gmail [dot] com) if you had issues
or questions, or would like to add a new model that does not fit within the current framework.

