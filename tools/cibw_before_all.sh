arch=$(uname -m)
project=$1
# install miniconda to install pre-built gsl (faster than building it)
# (TODO: miniconda can also be used to install cuda toolkit for platforms
# for which cuda manylinux is not available)
curl -s -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$arch.sh
bash Miniconda3-latest-Linux-$arch.sh -b -p $HOME/miniconda
conda install -y --no-deps -c conda-forge gsl=2.7
# conda install -y -c nvidia cuda-nvcc=12.0 cuda-cccl=12.0 cuda-cudart=12.0 cuda-cudart-static=12.0 cuda-cudart-dev=12.0
# if [[ ! -e /root/miniconda/bin/nvcc ]]; then
#     echo "NVCC not found..Stopping build"
#     exit 1
# fi