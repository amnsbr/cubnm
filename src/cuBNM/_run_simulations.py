"""
This simply imports run_simulations extension but
handles some known issues that may occur during import
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