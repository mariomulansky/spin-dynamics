ARCH = sm_13

CUDA_ROOT = /usr/local/cuda

CC = gcc
CXX = g++
NVCC = nvcc

LDLIBS = -lcudart -lgomp
LDFLAGS = -L$(CUDA_ROOT)/lib64
INCLUDES = -I${BOOST_ROOT}

NVCCFLAGS = -O3 $(INCLUDES) -arch $(ARCH) --use_fast_math 
#-Xcompiler -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP
#--compiler-bindir=/usr/bin/g++-4.4

%.co : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

spin_dynamics.co: spin_dynamics.cu
spin_dynamics: spin_dynamics.co
	$(CC) -o spin_dynamics $(LDLIBS) $(LDFLAGS) spin_dynamics.co
