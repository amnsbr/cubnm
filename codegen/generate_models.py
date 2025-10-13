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
    """
    with open(template_path, 'r') as f:
        template = Template(f.read())
    template_path_rel = os.path.relpath(template_path, start=PACKAGE_ROOT)
    model_spec_dict.update({
        'template_path': template_path_rel
    })
    code_content = template.render(**model_spec_dict)
    # save
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(code_content)
    print(f"Generated: {output_path}")

def main():
    hpp_template = os.path.join(PACKAGE_ROOT, 'include', 'cubnm', 'models', 'model.hpp.mako')
    cuh_template = os.path.join(PACKAGE_ROOT, 'include', 'cubnm', 'models', 'model.cuh.mako')
    cpp_template = os.path.join(PACKAGE_ROOT, 'src', 'ext', 'models', 'model.cpp.mako')
    cu_template = os.path.join(PACKAGE_ROOT, 'src', 'ext', 'models', 'model.cu.mako')
    header_output_dir = os.path.join(PACKAGE_ROOT, 'include', 'cubnm', 'models')
    impl_output_dir = os.path.join(PACKAGE_ROOT, 'src', 'ext', 'models')
    
    # Determine which models to generate
    models_to_generate = ALL_MODELS.keys()

    templates = {
        'hpp': hpp_template,
        'cuh': cuh_template,
        'cpp': cpp_template,
        'cu': cu_template
    }

    # Generate model-specific files
    for model_name in models_to_generate:
        model_spec = ALL_MODELS[model_name].to_dict().copy()
        model_spec['gpu_enabled'] = True # TODO: add option to disable GPU
        
        for k, template in templates.items():
            if k == 'hpp' or k == 'cuh':
                output_dir = header_output_dir
            else:
                output_dir = impl_output_dir
            output_path = os.path.join(output_dir, f"{model_name.lower()}.{k}")
            generate_from_template(model_spec, template, output_path)
    
    # Generate models.cpp and models.cu
    models = ['rWW'] + list(models_to_generate) # Add 'rWW' which is not auto-generated
    for k in ['cpp', 'cu']:
        template = os.path.join(PACKAGE_ROOT, 'src', 'ext', f'models.{k}.mako')
        template_path_rel = os.path.relpath(template, start=PACKAGE_ROOT)
        context = {
            'models': models,
            'template_path': template_path_rel
        }
        output_path = os.path.join(PACKAGE_ROOT, 'src', 'ext', f'models.{k}')
        generate_from_template(
            context,
            template,
            output_path
        )

if __name__ == '__main__':
    main()
