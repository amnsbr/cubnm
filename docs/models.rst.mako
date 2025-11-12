Models
======

The following models are available in cuBNM. For detailed equations and model characteristics,
refer to the documentation of each model's corresponding class.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Model Name
     - Class
% for short_name, full_name in models_info.items():
   * - ${full_name.title()}
     - :class:`cubnm.sim.${short_name.lower()}.${short_name}SimGroup`
% endfor
