arch=$(uname -m)
project=$1
# install miniforge to install pre-built gsl (faster than building it)
# (TODO: miniforge can also be used to install cuda toolkit for platforms
# for which cuda manylinux is not available)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$arch.sh"
bash Miniforge3-Linux-$arch.sh -b -p /opt/miniforge
conda install -y --no-deps -c conda-forge gsl=2.7
# get rid of "._" files created by MacOS (when calling cibuildwheel from MacOS)
find $1/ -type f -name '._*' -delete
