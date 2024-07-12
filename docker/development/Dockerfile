# -----------------------------------------------------------------------------
# How to build:
#   $ cd <cubnm_root>
#   $ docker build -t amnsbr/cubnm:dev -f ./docker/development/Dockerfile .
#
# How to run:
#   $ docker run amnsbr/cubnm:dev # for CLI
#   $ docker run -it --entrypoint /bin/bash amnsbr/cubnm:dev # access to bash
# In the interactive session `cubnm` is installed in `/opt/miniconda/bin/python`.
#
# Note: This container includes latest development version of the toolbox
# which is built from source. Therefore it requires CUDA Toolkit and is
# larger. It will not be updated with every commit of the development code.
#
# Note 2: Currently cross-platform reproducibility of the random number generator
# cannot be guaranteed. On the hardware that I have access to, exact same builds
# (wheels) generate different simulations (most likely due to different noise)
# and therefore the expected simulation tests will fail.
# -----------------------------------------------------------------------------

# Base image
FROM sameli/manylinux2014_x86_64_cuda_11.8

# Install mininconda (for Python3.10 and GSL)
RUN curl -s -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH $PATH:/opt/miniconda/bin

# Install Python 3.10 and make it the default python
RUN conda install -y python=3.10
RUN echo 'alias python=/opt/miniconda/bin/python' >> ~/.bashrc

# Install GSL
RUN conda install -y --no-deps -c conda-forge gsl=2.7

# Install build
RUN /opt/miniconda/bin/python -m pip install build

# Build cubnm and install it
COPY ../.. /cubnm
WORKDIR /cubnm
RUN /opt/miniconda/bin/python -m build . && \
    /opt/miniconda/bin/python -m pip install ./dist/*.whl

ENTRYPOINT ["/opt/miniconda/bin/cubnm"]