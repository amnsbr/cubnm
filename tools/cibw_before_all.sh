project=$1
# Try to install GSL by downloading the pre-built conda-forge package directly
# (faster than installing miniforge). Falls back to miniforge if the download fails.
if [ "$(uname -m)" = "x86_64" ]; then
    GSL_URL="https://api.anaconda.org/download/conda-forge/gsl/2.7/linux-64/gsl-2.7-he838d99_0.tar.bz2"
else
    echo "Unsupported architecture: $(uname -m)" && exit 1
fi
mkdir -p /opt/miniforge
if curl -fsSL -o /tmp/gsl.tar.bz2 "$GSL_URL" && tar -xjf /tmp/gsl.tar.bz2 -C /opt/miniforge; then
    echo "GSL installed from conda-forge archive"
else
    echo "Direct download failed, falling back to miniforge"
    curl -fsSL -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge
    /opt/miniforge/bin/conda install -y --no-deps -c conda-forge gsl=2.7
fi
# get rid of "._" files created by MacOS (when calling cibuildwheel from MacOS)
find $1/ -type f -name '._*' -delete
