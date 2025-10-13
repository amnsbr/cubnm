"""
Loads model specifications from YAML files
"""

from collections import OrderedDict
import yaml
import os
import glob

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelVariable:
    """
    Model variable with its properties.
    """
    def __init__(self, name, var_type, description=None):
        self.name = name
        self.var_type = var_type
        self.description = description
        # index across variables of the same type
        # will be determined later
        self.index = None
    

class ModelSpec:
    """
    Model specification
    """
    def __init__(self, model_name):
        self.model_name = model_name
                
        self.constants = []
        self.config = []
        self.external_declarations = None
        self.additional_methods = None
        self.conn_state_var = None
        self.bold_state_var = None
        
        # equations (as strings)
        self.init_equations = []
        self.step_equations = []
        self.restart_equations = []
        
        # flags
        self.has_post_bw_step = False
        self.has_post_integration = False
        self.is_osc = False
        self.gpu_enabled = True

        self.variables = OrderedDict()
        
        # will be computed automatically
        self._var_indices = {}
        self._var_arrays = {}

    
    def add_variable(self, name, var_type, description=None):
        """
        Add a variable to the model.
        """
        var = ModelVariable(name, var_type, description)
        self.variables[name] = var
        return var
    
    def _compute_indices(self):
        """
        Compute indices for all variables based on their types.
        """
        type_counters = {}
        type_arrays = {
            'state_var': '_state_vars',
            'intermediate_var': '_intermediate_vars',
            'global_param': '_global_params',
            'regional_param': '_regional_params',
            'noise': 'noise',
            'ext_int': '_ext_int',
            'ext_bool': '_ext_bool',
            'ext_int_shared': '_ext_int_shared',
            'ext_bool_shared': '_ext_bool_shared',
            'global_out_int': 'global_out_int',
            'global_out_bool': 'global_out_bool',
            'global_out_double': 'global_out_double',
            'regional_out_int': 'regional_out_int',
            'regional_out_bool': 'regional_out_bool',
            'regional_out_double': 'regional_out_double',
        }
        
        for name, var in self.variables.items():
            var_type = var.var_type
            if var_type not in type_counters:
                type_counters[var_type] = 0
            
            var.index = type_counters[var_type]
            type_counters[var_type] += 1
            
            self._var_indices[name] = var.index
            self._var_arrays[name] = type_arrays.get(var_type, var_type)
        
        return type_counters
    
    def get_counts(self):
        """
        Get counts of each variable type.
        """
        counts = self._compute_indices()
        return {
            'n_state_vars': counts.get('state_var', 0),
            'n_intermediate_vars': counts.get('intermediate_var', 0),
            'n_noise': counts.get('noise', 0),
            'n_global_params': counts.get('global_param', 0),
            'n_regional_params': counts.get('regional_param', 0),
            'n_ext_int': counts.get('ext_int', 0),
            'n_ext_bool': counts.get('ext_bool', 0),
            'n_ext_int_shared': counts.get('ext_int_shared', 0),
            'n_ext_bool_shared': counts.get('ext_bool_shared', 0),
            'n_global_out_int': counts.get('global_out_int', 0),
            'n_global_out_bool': counts.get('global_out_bool', 0),
            'n_global_out_double': counts.get('global_out_double', 0),
            'n_regional_out_int': counts.get('regional_out_int', 0),
            'n_regional_out_bool': counts.get('regional_out_bool', 0),
            'n_regional_out_double': counts.get('regional_out_double', 0),
        }
    
    def to_dict(self):
        """
        Convert to dictionary for mako template rendering.
        """
        counts = self.get_counts()
        
        # set conn_state_var_idx and bold_state_var_idx
        assert (self.conn_state_var is not None), \
            "conn_state_var must be defined in the model specification"
        assert (self.bold_state_var is not None), \
            "bold_state_var must be defined in the model specification"
        for name, var in self.variables.items():
            if var.var_type == 'state_var':
                if self.conn_state_var == name:
                    conn_state_var_idx = var.index
                if self.bold_state_var == name:
                    bold_state_var_idx = var.index
        
        assert(len(self.config) == 0), \
            "config options are not supported yet."

        return {
            'model_name': self.model_name,
            **counts,
            'conn_state_var_idx': conn_state_var_idx,
            'bold_state_var_idx': bold_state_var_idx,
            'has_post_bw_step': self.has_post_bw_step,
            'has_post_integration': self.has_post_integration,
            'is_osc': self.is_osc,
            'gpu_enabled': self.gpu_enabled,
            'external_declarations': self.external_declarations,
            'constants': self.constants,
            'config': self.config,
            'additional_methods': self.additional_methods,
            'variables': self.variables,
            'var_indices': self._var_indices,
            'var_arrays': self._var_arrays,
            'init_equations': self.init_equations,
            'step_equations': self.step_equations,
            'restart_equations': self.restart_equations,
            'yaml_path': getattr(self, 'yaml_path', 'N/A'),
        }


def load_model_from_yaml(yaml_file):
    """
    Load a model specification from a YAML file.
    
    Parameters
    ----------
    yaml_file: :obj:`str`
        Path to the YAML file
        
    Returns
    -------
    :obj:`ModelSpec`
        The loaded model specification
    """
    # load YAML data
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # create ModelSpec
    spec = ModelSpec(data['model_name'])

    spec.yaml_path = os.path.relpath(yaml_file, PACKAGE_ROOT)
    
    # set flags
    spec.is_osc = data.get('is_osc', False)
    spec.has_post_bw_step = data.get('has_post_bw_step', False)
    spec.has_post_integration = data.get('has_post_integration', False)

    if spec.has_post_bw_step or spec.has_post_integration:
        raise NotImplementedError(
            "Code generation for BOLD hook and post "
            "integration hooks are not implemented yet."
        )
    
    # set connectivity and BOLD state variables
    spec.conn_state_var = data['conn_state_var']
    spec.bold_state_var = data['bold_state_var']
    
    # add variables
    for var in data['variables']:
        spec.add_variable(
            var['name'], 
            var['type'], 
            var.get('description', ''),
        )
    
    # add constants
    spec.constants = [
        (c['type'], c['name'], c['value'], c.get('description', ''))
        for c in data.get('constants', [])
    ]
    
    # add equations
    # split multi-line strings into lists of individual lines
    if 'init_equations' in data:
        equations = data['init_equations']
        spec.init_equations = equations.strip().split('\n') if isinstance(equations, str) else equations
    
    if 'restart_equations' in data:
        equations = data['restart_equations']
        spec.restart_equations = equations.strip().split('\n') if isinstance(equations, str) else equations
    
    if 'step_equations' in data:
        equations = data['step_equations']
        spec.step_equations = equations.strip().split('\n') if isinstance(equations, str) else equations
    
    # optional fields
    spec.config = data.get('config', [])
    spec.external_declarations = data.get('external_declarations')
    spec.additional_methods = data.get('additional_methods')
    
    return spec

ALL_MODELS = {}
for yaml_file in sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'recipes', '*.yaml'))):
    model_spec = load_model_from_yaml(yaml_file)
    ALL_MODELS[model_spec.model_name] = model_spec