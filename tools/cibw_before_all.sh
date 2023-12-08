arch=$(uname -m)
project=$1
# install miniconda: this is a faster approach for downloading nvidia components (instead of the entire toolkit) + gsl (instead of building it)
# curl -s -o $project/tools/cuda_12.3.1_545.23.08_linux.run https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda_12.3.1_545.23.08_linux.run -O ${project}/tools/cuda_12.3.1_545.23.08_linux.run
# sh -s $project/tools/cuda_12.3.1_545.23.08_linux.run
# yum install -y dnf
# rpm -i $project/tools/cuda-repo-rhel9-12-3-local-12.3.1_545.23.08-1.x86_64.rpm
# dnf clean all
# dnf -y install cuda-toolkit-12-3
# yum install -y gcc-toolset-10
curl -s -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$arch.sh
bash Miniconda3-latest-Linux-$arch.sh -b -p $HOME/miniconda
conda install -y -c nvidia cuda-nvcc=12.0 cuda-cccl=12.0 cuda-cudart=12.0 cuda-cudart-static=12.0 cuda-cudart-dev=12.0
conda install -y --no-deps -c conda-forge gsl=2.7
if [[ ! -e /root/miniconda/bin/nvcc ]]; then
    echo "NVCC not found..Stopping build"
    exit 1
fi