# all in one
cd /data/project/ei_development/tools/pybnm
nvcc -c -rdc=true --compiler-options '-fPIC' -o bnm_tmp.o bnm.cu -I/data/project/ei_development/tools/gsl_build/include \
-I/data/project/ei_development/tools/libks/include
nvcc -dlink --compiler-options '-fPIC' -o bnm.o bnm_tmp.o \
/data/project/ei_development/tools/gsl_build_shared/lib/libgsl.a \
/data/project/ei_development/tools/gsl_build_shared/lib/libgslcblas.a \
/data/project/ei_development/tools/libks/libks.so \
-lm -lcudart
rm -f libbnm.a
ar cru libbnm.a bnm.o bnm_tmp.o
ranlib libbnm.a
python setup.py build --force && python setup.py install --force