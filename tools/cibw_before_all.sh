arch=$(uname -m)
# install miniconda: this is a faster approach for downloading nvidia components (instead of the entire toolkit) + gsl (instead of building it)
curl -s -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$arch.sh
bash Miniconda3-latest-Linux-$arch.sh -b -p $HOME/miniconda && export PATH="$HOME/miniconda/bin:$PATH"
conda install -y -c nvidia cuda-compiler=12.3 cuda-cudart-static=12.3 cuda-cudart-dev=12.3
conda install -y --no-deps -c conda-forge gsl=2.7