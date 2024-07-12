# -----------------------------------------------------------------------------
# How to build:
#   $ cd <cubnm_root>
#   $ docker build -t amnsbr/cubnm:v0.0.2 -f ./docker/v0.0.2/Dockerfile .
#
# How to run:
#   $ docker run -it --entrypoint /bin/bash amnsbr/cubnm:v0.0.2 # access to bash
# In the interactive session `cubnm` is installed in `python3.10` and can be imported.
# -----------------------------------------------------------------------------
FROM --platform=linux/amd64 nvcr.io/nvidia/driver:550.90.07-ubuntu22.04

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install cubnm package using pip
# Also force numpy < 2 (this will be fixed in next releases)
RUN python3.10 -m pip install "cubnm" "numpy<2"

ENTRYPOINT [ "/bin/bash" ]