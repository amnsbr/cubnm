<%!
import re

def get_var_access(var_name, var_arrays, var_indices, variables, constants_names, config_names, model_name):
    """
    Get C++ array access for a variable.
    """
    # special variables
    if var_name == 'globalinput':
        return 'tmp_globalinput'
    if var_name == 'noise_idx':
        return 'noise_idx'
    
    # other variables (states, intermediates, params, etc.)
    if var_name in variables:
        var = variables[var_name]
        array = var_arrays[var_name]
        index = var_indices[var_name]
        
        # Special handling for noise
        if var.var_type == 'noise':
            if index > 0:
                return f"{array}[noise_idx + {index}]"
            else:
                return f"{array}[noise_idx]"
        
        return f"{array}[{index}]"
    
    # constants
    if var_name in constants_names:
        return f"{model_name}Model::mc.{var_name}"

    # configs
    if var_name in config_names:
        return f"this->conf.{var_name}"
    
    # otherwise return as-is
    return var_name

def process_equation(equation, var_arrays, var_indices, variables, constants_names, config_names, model_name):
    """
    Process an equation, replacing variable names with array accesses.
    """
    equation = equation.strip()
    # skip comments and empty lines
    if not equation or equation.startswith('#'):
        return equation
    
    # split into left and right side
    if '=' not in equation:
        return equation
    
    # handle compound operators (+=, -=, etc.)
    compound_ops = ['+=', '-=', '*=', '/=']
    is_compound = any(op in equation for op in compound_ops)
    
    if is_compound:
        for op in compound_ops:
            if op in equation:
                parts = equation.split(op, 1)
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                
                lhs_code = get_var_access(lhs, var_arrays, var_indices, variables, constants_names, config_names, model_name)
                rhs_code = process_expression(rhs, var_arrays, var_indices, variables, constants_names, config_names, model_name)
                
                return f"{lhs_code} {op} {rhs_code};"
    else:
        parts = equation.split('=', 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        
        lhs_code = get_var_access(lhs, var_arrays, var_indices, variables, constants_names, config_names, model_name)
        rhs_code = process_expression(rhs, var_arrays, var_indices, variables, constants_names, config_names, model_name)
        
        return f"{lhs_code} = {rhs_code};"

def process_expression(expr, var_arrays, var_indices, variables, constants_names, config_names, model_name):
    """
    Process an expression, replacing variable names with array accesses.
    """
    # Find all identifiers in the expression
    identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    
    def replace_identifier(match):
        name = match.group(0)
        # Don't replace function names or keywords
        keywords = {'max', 'min', 'exp', 'sqrt', 'pow', 'abs', 'sin', 'cos', 'tan',
                   'log', 'floor', 'ceil', 'std', 'fabs', 'fmin', 'fmax'}
        if name in keywords:
            return name
        return get_var_access(name, var_arrays, var_indices, variables, constants_names, config_names, model_name)
    
    result = re.sub(identifier_pattern, replace_identifier, expr)
    return result

def format_equations(equations, var_arrays, var_indices, variables, constants_names, config_names, model_name, indent='    '):
    """
    Format a list of equations with proper indentation and comments.
    """
    lines = []
    for eq in equations:
        eq = eq.strip()
        if not eq:
            lines.append('')
        elif eq.startswith('#'):
            # Comment
            lines.append(f"{indent}{eq.replace('#', '//')}")
        else:
            # Equation
            processed = process_equation(eq, var_arrays, var_indices, variables, constants_names, config_names, model_name)
            lines.append(f"{indent}// {eq}")
            lines.append(f"{indent}{processed}")
    return '\n'.join(lines)

def get_constants_names(constants):
    """
    Create a list of constant names
    """
    const_list = []
    for const in constants:
        const_list.append(const[1])
    return const_list

def get_config_names(config):
    """
    Create a list of constant names
    """
    conf_list = []
    for conf in config:
        conf_list.append(conf[1])
    return conf_list

def format_constants(constants):
    """
    Format constant value assignments.
    """
    lines = []
    for const in constants:
        type_name, var_name, value, comment = const
        line = f"    mc.{var_name} = {value};"
        if comment:
            line += f" // {comment}"
        lines.append(line)
    return '\n'.join(lines)

%>\
<%
# get names of constants and configs
constants_names = get_constants_names(constants)
config_names = get_config_names(config)
%>\
/* 
    Autogenerated using:
    - Template: '${template_path}'
    - Model definition: '${yaml_path}'
    Do not modify this autogenerated code. Instead modify template
    or model definition.
*/
#include "cubnm/models/${model_name.lower()}.hpp"
% if cpp_includes:
${cpp_includes}
% endif

// Static constants instance
${model_name}Model::Constants ${model_name}Model::mc;

// Initialize constants based on dt
void ${model_name}Model::init_constants(double dt) {
${format_constants(constants)}
}
% if len(config) > 0:
% if custom_methods.get('set_conf'):
${custom_methods['set_conf']}
% else:
<%
# Auto-generate set_conf
def get_conversion(c_type):
    if c_type == 'bool':
        return '(bool)std::stoi(pair.second)'
    elif c_type == 'int':
        return 'std::stoi(pair.second)'
    elif c_type == 'double':
        return 'std::stod(pair.second)'
    else:
        return 'pair.second'

def needs_transform(conf):
    c_type, c_name, c_value, c_desc = conf
    # Check if value is an expression (contains operators)
    return isinstance(c_value, str) and any(op in c_value for op in ['+', '-', '*', '/', '(', ')'])
%>\
void ${model_name}Model::set_conf(std::map<std::string, std::string> config_map) {
    set_base_conf(config_map);
    for (const auto& pair : config_map) {
% for idx, conf in enumerate(config):
<%
    c_type, c_name, c_value, c_desc = conf
    conversion = get_conversion(c_type)
    condition = 'if' if idx == 0 else 'else if'
%>\
        ${condition} (pair.first == "${c_name}") {
            this->conf.${c_name} = ${conversion};
        }\
% endfor

    }
}
% endif
% endif
% if custom_methods.get('prep_params'):
${custom_methods['prep_params']}
% endif

void ${model_name}Model::h_init(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
${format_equations(init_equations, var_arrays, var_indices, variables, constants_names, config_names, model_name)}
}

void ${model_name}Model::_j_restart(
    double* _state_vars, double* _intermediate_vars, 
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
${format_equations(restart_equations, var_arrays, var_indices, variables, constants_names, config_names, model_name)}
}

void ${model_name}Model::h_step(
        double* _state_vars, double* _intermediate_vars,
        double* _global_params, double* _regional_params,
        double& tmp_globalinput,
        double* noise, long& noise_idx
        ) {
${format_equations(step_equations, var_arrays, var_indices, variables, constants_names, config_names, model_name)}
}
% if has_post_bw_step:
${custom_methods.get('_j_post_bw_step', '')}
${custom_methods.get('h_post_bw_step', '')}
% endif
% if has_post_integration:
${custom_methods.get('h_post_integration', '')}
% endif