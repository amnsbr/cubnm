"""
This imports run_simulations extension while
handling some known issues that may occur during import
in addition to initializing the extension
"""
import sys

try:
    from cubnm.core import run_simulations
except ImportError as e:
    error_msg = str(e)
    if "undefined symbol" in error_msg:
        print("To fix this error try building cuBNM from source (https://github.com/amnsbr/cuBNM)")
    raise
else:
    from cubnm.core import init, set_const
    init()
