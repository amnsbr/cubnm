arch=$(uname -m)
project=$1
# install miniconda to install pre-built gsl (faster than building it)
# (TODO: miniconda can also be used to install cuda toolkit for platforms
# for which cuda manylinux is not available)
curl -s -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$arch.sh
bash Miniconda3-latest-Linux-$arch.sh -b -p /opt/miniconda
conda install -y --no-deps -c conda-forge gsl=2.7
# get rid of "._" files created by MacOS (when calling cibuildwheel from MacOS)
find /project -type f -name '._*' -delete
