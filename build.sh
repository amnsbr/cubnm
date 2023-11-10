# TODO: use MakeFile
cd /data/project/ei_development/tools/cuBNM/cuBNM/cuda
nvcc -c -rdc=true --compiler-options '-fPIC' -o bnm_tmp.o bnm.cu -I/data/project/ei_development/tools/gsl_build_shared/include \
-I/data/project/ei_development/tools/cuBNM/cuBNM/cpp #-D NOISE_SEGMENT
nvcc -dlink --compiler-options '-fPIC' -o bnm.o bnm_tmp.o \
/data/project/ei_development/tools/gsl_build_shared/lib/libgsl.a \
/data/project/ei_development/tools/gsl_build_shared/lib/libgslcblas.a \
-lm -lcudart
rm -f libbnm.a
ar cru libbnm.a bnm.o bnm_tmp.o
ranlib libbnm.a
cd /data/project/ei_development/tools/cuBNM
python setup.py build --force && python setup.py install --force