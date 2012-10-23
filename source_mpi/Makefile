ARCH = sm_13

CUDA_ROOT = /usr/local/cuda

CC = gcc
CXX = mpicxx
NVCC = nvcc

LDLIBS = -lcudart -lgomp
LDFLAGS = -L$(CUDA_ROOT)/lib64
INCLUDES = -I${BOOST_ROOT} -I${CUDA_ROOT}/include

NVCCFLAGS = -O3 $(INCLUDES) -arch $(ARCH) --use_fast_math -Xcompiler -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP

CXXFLAGS = -O3 $(INCLUDES)

%.co : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.co : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

spin_relax.co: spin_relax.cu

rand_normal.co: rand_normal.cpp

spin_study: spin_relax.co rand_normal.co
	$(CXX) -o spin_study $(INCLUDES) $(LDLIBS) $(LDFLAGS) spin_relax.co rand_normal.co spin_study.cpp
