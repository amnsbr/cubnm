"""
This imports run_simulations extension while
handling some known issues that may occur during import
in addition to initializing the extension
"""
try:
    from cuBNM.core import run_simulations
except ImportError as e:
    error_msg = str(e)
    if "GLIBC_2.29" in error_msg:
        print(error_msg)
        print("To fix this error either update `ldd` or build cuBNM from source (https://github.com/amnsbr/cuBNM)")
        exit(1)
    elif "undefined symbol" in error_msg:
        print(error_msg)
        print("To fix this error try building cuBNM from source (https://github.com/amnsbr/cuBNM)")
        exit(1)
    else:
        raise(e)
else:
    from cuBNM.core import init, set_const, set_conf
    init()