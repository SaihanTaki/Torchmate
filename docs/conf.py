# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information



import datetime
import sys
import os
import matplotlib


sys.path.append('..')


project = 'Torchmate'
copyright = f'{datetime.datetime.now().year}, Abdullah Saihan Taki'
author = 'Abdullah Saihan Taki'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_rtd_theme",
            "sphinx.ext.coverage",
            "sphinx.ext.napoleon",
            "sphinx.ext.viewcode",
            "sphinx.ext.mathjax",
            "autodocsumm",
            'sphinx_copybutton',
            'notfound.extension',
            'nbsphinx']


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_theme = 'faculty-sphinx-theme'
html_static_path = ['_static']

html_logo = "logo.png"

html_css_files = [
    'faculty.css', 
    # https://github.com/facultyai/faculty-sphinx-theme/blob/master/faculty_sphinx_theme/static/css/faculty.css
]

html_theme_option = {
    "style_external_link": True
}

# -- Extension configuration -------------------------------------------------

viewcode_line_numbers = True
autodoc_inherit_docstrings = False
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_numpy_docstring = False
autodoc_member_order = 'bysource'


autodoc_mock_imports = [
    "torch",
    "numpy",
    "wandb",
    "matplotlib"
    "matplotlib.pyplot"
    "torchmate"
    "torchmate.trainer"
    "torchmate.callbacks"
    "torchmate.utils"
]

autoclass_content = "both"
#autodoc_typehints = "description"

# --- Work around to make autoclass signatures not (*args, **kwargs) ----------
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/docs/conf.py



class FakeSignature:
    def __getattribute__(self, *args):
        raise ValueError


def f(app, obj, bound_method):
    if "__new__" in obj.__name__:
        obj.__signature__ = FakeSignature()


def setup(app):
    app.connect("autodoc-before-process-signature", f)


