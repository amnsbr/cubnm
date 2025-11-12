#!/usr/bin/env python3
"""
This script generates C++ header files (.hpp), CUDA headers (.cuh),
and implementation files (.cpp, .cu) from YAML-based model specifications.
It also creates "models.cpp" and "models.cu" files that include all models.
"""

import sys
import os
from mako.template import Template
from model_specs import ALL_MODELS 

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_from_template(model_spec_dict, template_path, output_path):
    """
    Generate code from model specification and mako template path.

    Parameters
    ----------
    model_spec_dict : dict
        Dictionary containing model specifications.
    template_path : str
        Path to the mako template file.
    output_path : str
        Path to save the generated code file.
    """
    with open(template_path, 'r') as f:
        template = Template(f.read())
    template_path_rel = os.path.relpath(template_path, start=PACKAGE_ROOT)
    model_spec_dict.update({
        'template_path': template_path_rel,
    })
    try:
        code_content = template.render(**model_spec_dict)
    except Exception as e:
        print(f"Error during template rendering: {e}", file=sys.stderr)
        from mako import exceptions
        print(exceptions.text_error_template().render(), file=sys.stderr)
        raise
    # save
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(code_content)
    print(f"Generated: {output_path}")

def main(add_examples=False):
    """
    Generate model files from templates for all available models.
    In addition, generate models.cpp, models.cu, and sim/__init__.py.

    Parameters
    ----------
    add_examples : bool, optional
        Whether to add examples to the docstring of the generated Python SimGroup class.
        Default is False.
    """
    hpp_template = os.path.join(PACKAGE_ROOT, 'include', 'cubnm', 'models', 'model.hpp.mako')
    cuh_template = os.path.join(PACKAGE_ROOT, 'include', 'cubnm', 'models', 'model.cuh.mako')
    cpp_template = os.path.join(PACKAGE_ROOT, 'src', 'ext', 'models', 'model.cpp.mako')
    cu_template = os.path.join(PACKAGE_ROOT, 'src', 'ext', 'models', 'model.cu.mako')
    simgroup_template = os.path.join(PACKAGE_ROOT, 'src', 'cubnm', 'sim', 'simgroup.py.mako')
    header_output_dir = os.path.join(PACKAGE_ROOT, 'include', 'cubnm', 'models')
    impl_output_dir = os.path.join(PACKAGE_ROOT, 'src', 'ext', 'models')
    py_output_dir = os.path.join(PACKAGE_ROOT, 'src', 'cubnm', 'sim')
    
    models_to_generate = ALL_MODELS.keys()

    templates = {
        'hpp': hpp_template,
        'cuh': cuh_template,
        'cpp': cpp_template,
        'cu': cu_template,
        'py': simgroup_template
    }

    # generate model-specific files
    for model_name in models_to_generate:
        model_spec = ALL_MODELS[model_name].to_dict().copy()
        model_spec['gpu_enabled'] = True # TODO: support no-GPU code generation
        
        for k, template in templates.items():
            if k == 'hpp' or k == 'cuh':
                output_dir = header_output_dir
            elif k == 'py':
                model_spec['add_examples'] = add_examples
                output_dir = py_output_dir
            else:
                output_dir = impl_output_dir
            output_path = os.path.join(output_dir, f"{model_name.lower()}.{k}")
            generate_from_template(model_spec, template, output_path)
    
    # generate models.cpp, models.cu and sim/__init__.py
    # which depend on list of available models
    models = list(models_to_generate)
    for k in ['cpp', 'cu', 'py']:
        if k == 'py':
            template = os.path.join(PACKAGE_ROOT, 'src', 'cubnm', 'sim', '__init__.py.mako')
        else:
            template = os.path.join(PACKAGE_ROOT, 'src', 'ext', f'models.{k}.mako')
        template_path_rel = os.path.relpath(template, start=PACKAGE_ROOT)
        context = {
            'models': models,
            'template_path': template_path_rel
        }
        if k == 'py':
            output_path = os.path.join(PACKAGE_ROOT, 'src', 'cubnm', 'sim', '__init__.py')
        else:
            output_path = os.path.join(PACKAGE_ROOT, 'src', 'ext', f'models.{k}')
        generate_from_template(
            context,
            template,
            output_path
        )
    
    # generate the list of models in the documentation
    doc_template = os.path.join(PACKAGE_ROOT, 'docs', 'models.rst.mako')
    doc_output_path = os.path.join(PACKAGE_ROOT, 'docs', 'models.rst')
    models_info = {k: v.full_name for k, v in ALL_MODELS.items()}
    context = {
        'models_info': models_info,
        'template_path': os.path.relpath(doc_template, start=PACKAGE_ROOT)
    }
    generate_from_template(
        context,
        doc_template,
        doc_output_path
    )

if __name__ == '__main__':
    main()
