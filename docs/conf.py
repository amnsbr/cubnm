import sys
import os
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cubnm'
copyright = '2024, Amin Saberi'
author = 'Amin Saberi'

sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'autoapi.extension',
    # 'sphinx_rtd_theme',
    'sphinxarg.ext',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "*/staging/*", "*/venv/*", "*/.ipynb_checkpoints/*"]

# -- Options for autoapi -----------------------------------------------------
autoapi_dirs = ['../src/cubnm']
autoapi_type = "python"
autoapi_python_class_content = "init"
autodoc_inherit_docstrings = False
autoapi_ignore = ["*_core*", "*_setup_opts*", "*cli*", "*_version*", "*/.ipynb_checkpoints/*", "*/venv/*"]
autoapi_template_dir = '_autoapi_templates'
autoapi_options = [
    'members', 'undoc-members', 'show-inheritance', 'inherited-members',
    'show-module-summary', 'special-members', 'private-members'
]

def autoapi_skip(app, what, name, obj, skip, options):
    # exclusion based on name
    excluded_names = ["__slots__", "__version__"]
    for excluded_name in excluded_names:
        if excluded_name in name:
            return True
    # exclude properties
    if what == "property":
        return True
    return skip

def setup(sphinx):
   sphinx.connect("autoapi-skip-member", autoapi_skip)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo_text_black.png",
    "dark_logo": "logo_text_white.png",
}
# html_theme = 'rtd_theme'
# html_theme_options = {
#     'logo_only': True,
#     'display_version': True,
#     'style_nav_header_background': 'black',
# }
# html_logo = "_static/logo_text_white.png"
html_css_files = [
    'css/custom.css',
]
html_sourcelink_suffix = ''

# -- Options for nbsphinx
nbsphinx_execute = "never"
# TODO: add nbsphinx_prolog to launch on Kaggle
nbsphinx_epilog = r"""
{% set docname = env.doc2path(env.docname) %}

.. role:: raw-html(raw)
    :format: html

.. nbinfo::

    This page was generated from `this Jupyter Notebook <{{docname}}>`_.
"""